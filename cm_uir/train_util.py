import copy
import functools
import math
import os
from cm_uir.save_max import save_max
import blobfile as bf
import torch as th
import shutil
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
import PIL.Image as Image
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np
from eva.psnr_ssim import quality_assess_SM3_test
class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data_train,
        data_test,
        batch_size,
        microbatch=-1,  # -1 disables microbatches
        lr,
        ema_rate,
        log_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        LUDEN,
        sample_data_dir,
        checkpoint_dir="./Checkpoints",
    ):
        self.sample_data_dir =sample_data_dir
        self.checkpoint_dir = checkpoint_dir
        self.denoise = diffusion.denoise
        self.LUDEN = LUDEN
        self.model = model
        self.diffusion = diffusion
        self.data_train= data_train
        self.data_test = data_test
        self.batch_size = batch_size
        self.teacher_model = None
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay,decoupled_weight_decay=True
        )
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.model.parameters())
        dist_util.sync_params(self.model.buffers())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        checkpoint_dir = bf.dirname(main_checkpoint)
        opt_checkpoint = bf.join(checkpoint_dir, f"opt{self.resume_step:06}.pt")
        if not bf.exists(opt_checkpoint):
            opt_checkpoint = bf.join(checkpoint_dir, "opt.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while not self.lr_anneal_steps or self.step < self.lr_anneal_steps:
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self.step += 1
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self,warmup_steps=2000):
        if not self.lr_anneal_steps:
            return
        min_lr = 4e-5
        max_lr = self.lr
        if self.step < warmup_steps:  # 前500步线性预热
            lr = min_lr + (max_lr - min_lr) * (self.step / warmup_steps)
        else: 
            T_max = self.lr_anneal_steps-warmup_steps  # 半周期长度（总步数）
            cos_value = math.cos(math.pi * (self.step-warmup_steps) / T_max)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos_value)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(self.checkpoint_dir, filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.checkpoint_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()


class CMTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        target_model,
        training_mode,
        ema_scale_fn,
        total_training_epochs,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.training_mode = training_mode
        self.ema_scale_fn = ema_scale_fn
        self.target_model = target_model
        self.total_training_epochs = total_training_epochs
        self.device = th.device(
            'cuda' if th.cuda.is_available() is not None else 'cpu')
        if target_model:
            self._load_and_sync_target_parameters()
            self.target_model.requires_grad_(False)
            self.target_model.train()

            self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
                self.target_model.named_parameters()
            )
            self.target_model_master_params = make_master_params(
                self.target_model_param_groups_and_shapes
            )
        self.global_step = self.step
    def _load_and_sync_target_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            target_name = name
            resume_target_checkpoint = os.path.join(path, target_name)
            if bf.exists(resume_target_checkpoint) and dist.get_rank() == 0:
                logger.log(
                    "loading model from checkpoint: {resume_target_checkpoint}..."
                )
                self.target_model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_target_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.target_model.parameters())
        dist_util.sync_params(self.target_model.buffers())

    def run_loop(self):
        tbar = range(len(self.data_train))
        while (((self.global_step // len(self.data_train))+1) <= self.total_training_epochs):
            train_data = iter(self.data_train)
            self.loss_all =0
            for i in tbar:
                input,target=next(train_data)
                input = self.set_device(input)
                target = self.set_device(target)
                self.run_step(input,target)
                if self.global_step % self.log_interval == 0 :
                    logger.dumpkvs()
                if self.global_step==1:
                    print('\n')
                    print('Please wait for a moment, the model is training...')
                del input,target
                # if self.global_step ==10:
                #                 break
            psnr,ssim=self.test()
            self.save(f"{psnr:.5f}",f"{ssim:.5f}")
            save_max(self.checkpoint_dir)
            th.cuda.empty_cache()
            if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                return
        print("Training completed.")

    def run_step(self,input,target):
        self.forward_backward(input,target)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
            if self.target_model:
                self._update_target_ema()
            self.step += 1
            self.global_step += 1
        self._anneal_lr()
        self.log_step()
    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(
                self.target_model_master_params,
                self.mp_trainer.master_params,
                rate=target_ema,
            )
            master_params_to_model_params(
                self.target_model_param_groups_and_shapes,
                self.target_model_master_params,
            )

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x
    def forward_backward(self, input,target):
        self.mp_trainer.zero_grad()
    
        for j in range(1):
            last_batch = (j + self.microbatch) >=1
            t, weights = self.schedule_sampler.sample(1, dist_util.dev())
            ema, num_scales = self.ema_scale_fn(self.global_step)
            if self.training_mode == "consistency_training":
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.ddp_model,
                    num_scales,
                    LUDEN =self.LUDEN,
                    input=input,
                    target=target,
                    target_model=self.target_model,
                )
            else:
                raise ValueError(f"Unknown training mode {self.training_mode}")

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()

            self.mp_trainer.backward(loss)
            self.loss_item = loss.item()
            self.loss_all = self.loss_all + self.loss_item
            del input,target

    def save(self,psnr,ssim):
        import blobfile as bf
        # step = self.global_step               
        # def save_checkpoint(rate, params):
        #     state_dict = self.mp_trainer.master_params_to_state_dict(params)
        #     if dist.get_rank() == 0:
        #         logger.log(f"saving model {rate}...")
        #         if not rate:
        #             filename = f"model{step:06d}.pt"
        #         else:
        #             filename = f"ema_{rate}_{step:06d}.pt"

        #         #os.makedirs(bf.join(self.checkpoint_dir,f'{self.global_step}_psnr:{psnr}_ssim:{ssim}'), exist_ok=True)
        #         with bf.BlobFile(bf.join(self.checkpoint_dir,f'step:{str(self.global_step).zfill(7)}_psnr:{psnr}_ssim:{ssim}', filename), "wb") as f:
        #             th.save(state_dict, f)
        # for rate, params in zip(self.ema_rate, self.ema_params):
        #     save_checkpoint(rate, params)

        logger.log("saving optimizer state...")
        with bf.BlobFile(
            bf.join(self.checkpoint_dir, f'step:{str(self.global_step).zfill(7)}_psnr:{psnr}_ssim:{ssim}',f"opt.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

        if self.target_model:
            logger.log("saving SDGDN model state")
            filename = f"SDGDN.pt"
            #os.makedirs(bf.join(self.checkpoint_dir,f'{self.global_step}_psnr:{psnr}_ssim:{ssim}'), exist_ok=True)
            with bf.BlobFile(bf.join(self.checkpoint_dir,f'step:{str(self.global_step).zfill(7)}_psnr:{psnr}_ssim:{ssim}', filename), "wb") as f:
                th.save(self.target_model.state_dict(), f)
            
        result_file = bf.join(self.checkpoint_dir,'result')
        save_path = bf.join(self.checkpoint_dir,f'step:{str(self.global_step).zfill(7)}_psnr:{psnr}_ssim:{ssim}','result')
        shutil.copytree(result_file, save_path)  
        print(" ✅ save result")
        dist.barrier()
    def denoiser(self,x_t, sigmas,depth):
        model_kwargs = {}
        _, denoised = self.denoise(self.target_model, x_t,sigmas, depth ,**model_kwargs)
        return denoised
    def log_step(self):
        step = self.global_step
        logger.logkv("step", step)
        logger.logkv("loss", self.loss_item)
        #logger.logkv("samples", (step + 1) * self.global_batch)
        logger.logkv("epoch", step//len(self.data_train)+1)
    def test(self):
        idx = 1
        self.target_model.eval()
        print("testing...")
        for _,test_data in enumerate(self.data_test):
            with th.no_grad():
                data =self.set_device(test_data[0])
                noise = th.randn_like(test_data[0])
                noise=self.set_device(noise)
                mid_out = data
                prediction = self.LUDEN(mid_out)
                depth=prediction.expand(-1, 3, -1, -1)
                #depth= -(depth-depth.min()) / (depth.max() - depth.min())*2
                depth = depth.to(th.float64)
                x_Tn = mid_out+noise* 80.00000
                x_T  = th.cat([mid_out,x_Tn], dim=1)
                s_in = mid_out.new_ones([mid_out.shape[0]])*80.00000
                out = self.denoiser(x_T,s_in,depth)
                out = ((out + 1) * 127.5).clamp(0, 255).to(th.uint8)
                out = out.permute(0, 2, 3, 1)
                out = out.contiguous()
                out = out.cpu().numpy().squeeze(0)
                path_test_save = f'{self.checkpoint_dir}/result'
                psnr_path = f'{self.checkpoint_dir}' 
                if not os.path.exists(path_test_save):
                    os.makedirs(path_test_save)  # 递归创建多级目录
                    print(f"文件夹 '{path_test_save}' 已创建")
                Image.fromarray(out, 'RGB').save(f'{path_test_save}/{str(idx).zfill(4)}.png')
                idx += 1
        print('Test img finished!')
        self.target_model.train()
        now_lr = self.opt.param_groups[0]['lr']
        psnr,ssim= quality_assess_SM3_test(inputdir=path_test_save, gtdir=f'{self.sample_data_dir}/target_256',resultdir = psnr_path,num = self.global_step,now_lr=now_lr,loss_all=self.loss_all)#计算psnr
        # with open(f'{psnr_path}/max_avg.txt', 'r',encoding='gbk') as file:
        #     file.seek(0)
        #     b = float(file.read().strip())#获得以往最大psnr值
        # if psnr>b:
        #     fc = open(os.path.join(psnr_path,f'max_avg.txt'), 'w')
        #     fc.write('%f'%psnr)
        #     fc.close()
        return psnr,ssim


def parse_resume_step_from_filename(filename):
    if "step:" in filename:
        step_str = filename.split("step:", 1)[1].split("_", 1)[0]
        try:
            return int(step_str)
        except ValueError:
            return 0

    # 旧格式(文件): "model001234.pt"
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    #return
    for key, values in losses.items():
        #logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
