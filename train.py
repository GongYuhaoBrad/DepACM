import argparse
import data as DATA
from cm_uir import dist_util, logger
from cm_uir.resample import create_named_schedule_sampler
from cm_uir.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
    replace_bn_with_gn
)
from DepthNet.net import LUDEN
from cm_uir.train_util import CMTrainLoop
import torch.distributed as dist
import torch as th
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.log("creating model and diffusion...")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    depth_model = LUDEN(None, features=64, backbone="efficientnet_lite3", exportable=True,
                            non_negative=True, blocks={'expand': True}).to(device)
    replace_bn_with_gn(depth_model)
    depth_model.load_state_dict(th.load('./Pretrain_models/model_LUDEN/LUDEN_model.pth',map_location=device))
    depth_model.eval()
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_epochs*len(DATA.create_dataset(args,phase='train')),
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")

    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs,start_scales=args.start_scales)
    if th.cuda.is_available():
        num_gpus = th.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        if num_gpus < 1:
            raise Exception("At least two GPUs are required for this training.")
    else:
        raise Exception("No GPU available.")
    def get_param_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_count = get_param_count(model)
    print(f"Model Parameter Count: {param_count}")
    model.to(device)
    model.train()
    if args.use_fp16:
        model.convert_to_fp16()
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    batch_size = args.batch_size
    data_set = DATA.create_dataset(args,phase='train')
    data_train_loader = DATA.create_dataloader(data_set, args,phase='train')
    data_sample = DATA.create_datasetforsample(data_dir=args.sample_data_dir,phase='val')#数据集地址
    data_test_loader = DATA.create_dataloader(data_sample,args,phase='val')


    # load the target model
    logger.log("creating the target model")
    target_model, _ = create_model_and_diffusion(
        **model_and_diffusion_kwargs,start_scales=args.start_scales
    )
    target_model.to(device)
    target_model.train()
    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())
    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)
    if args.use_fp16:
        target_model.convert_to_fp16()
    logger.log("training...")
    CMTrainLoop(model=model,target_model=target_model,
        training_mode=args.training_mode,ema_scale_fn=ema_scale_fn,total_training_epochs=args.total_training_epochs,
        diffusion=diffusion,data_train=data_train_loader,LUDEN = depth_model,data_test=data_test_loader,batch_size=batch_size,lr=args.lr,
        ema_rate=args.ema_rate,log_interval=args.log_interval,resume_checkpoint=args.resume_checkpoint,use_fp16=args.use_fp16,fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,weight_decay=args.weight_decay,lr_anneal_steps=args.lr_anneal_steps,sample_data_dir = args.sample_data_dir,
        checkpoint_dir=args.checkpoint_dir,
    ).run_loop()
def create_argparser():
    defaults = dict(
        data_dir="./UIEB/TRAIN",#"/home/customer4/Documents/semi_new_test_flash/new_UIEB/UIEB_data_hk"
        schedule_sampler="uniform",
        lr=4e-4,
        weight_decay=4e-5,
        batch_size=8,
        ema_rate="0.999943",
        log_interval=5,
        total_training_epochs=2000,
        resume_checkpoint="",#./Checkpoints/step:0122863_psnr:29.63155_ssim:0.98453/SDGDN.pt
        use_fp16=True,
        fp16_scale_growth=1e-3,
        data_len=-1,
        sample_data_dir = "./dataset/UIEB/VAL",
        checkpoint_dir="./Checkpoints_test",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()


