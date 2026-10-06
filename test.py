import argparse
import os
import numpy as np
import torch as th
import cv2
import time
from cm_uir import  logger
from cm_uir.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    replace_bn_with_gn,
    cm_train_defaults
)
from eva.psnr_ssim import quality_assess_SM3
import data as DATA
from concurrent.futures import ThreadPoolExecutor, as_completed
from DepthNet.net import LUDEN
from cm_uir.karras_diffusion import karras_sample
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def get_param_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
def main(eval):
    args = create_argparser().parse_args()
    if args.seed is not None:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    depth_model = LUDEN(None, features=64, backbone="efficientnet_lite3", exportable=True,
                            non_negative=True, blocks={'expand': True}).to(device)
    replace_bn_with_gn(depth_model)
    depth_model.load_state_dict(th.load('./Pretrain_models/model_LUDEN/LUDEN_model.pth',map_location=device))
    depth_model.eval()
    param_count = get_param_count(depth_model)
    print(f"Model Parameter Count: {param_count}")
    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()),start_scales=args.start_scales,
    distillation=distillation,)
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    param_count = get_param_count(model)
    print(f"Model Parameter Count: {param_count}")
    logger.log("sampling...")
    if args.sampler == "multistep":
        ts="0,1,3"
        ts = tuple(int(x) for x in ts.split(","))
    else:
        ts = None
    data_set = DATA.create_datasetforsample(data_dir=args.val_data,phase='val')
    data = DATA.create_dataloader(data_set,args,phase='val')
    model_kwargs = {}
    total_samples = len(data)
    _, _, h, w = next(iter(data))[0].shape
    image_shape = (h, w, 3)
    all_images = np.empty((total_samples, *image_shape), dtype=np.uint8)
    path = args.model_path.split('/')[-1]
    name = f'{args.sample_save_path}/{path[:-4]}_{args.sampler}'
    _ = karras_sample(diffusion=diffusion,model=model,steps=args.steps,depth_model = depth_model,
                                model_kwargs=model_kwargs,device=device,sampler=args.sampler,sigma_min=args.sigma_min,
                                sigma_max=args.sigma_max,ts=ts,data=th.randn(1,1, 3, h, w))
    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)
    start.record()
    for k ,data in enumerate(data):
        sample = karras_sample(diffusion=diffusion,model=model,steps=args.steps,depth_model = depth_model,
                            model_kwargs=model_kwargs,device=device,sampler=args.sampler,sigma_min=args.sigma_min,
                            sigma_max=args.sigma_max,ts=ts,data=data)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        all_images[k] = sample.to('cpu').numpy()[0]
        if (k+1)%20==0 or k==total_samples-1:
            print(f"created {k+1} samples")
    end.record()
    def save_image(img_idx):
        img_rgb = all_images[img_idx]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_path = f'{name}/{str(img_idx+1).zfill(4)}.png'
        cv2.imwrite(img_path, img_bgr)
    os.makedirs(name, exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(save_image, i) for i in range(len(all_images))]
        for _ in as_completed(futures):
            pass
    print(f"FPS:{1000/(start.elapsed_time(end)/total_samples)}")
    print(f"save path is {name}")
    logger.log("sampling complete")
    if eval == 'True':
        gtdir = args.val_data + '/target_256'
        resultdir = name
        quality_assess_SM3(name, gtdir,resultdir,0)
    del all_images
def create_argparser():
    defaults = dict(
        training_mode="consistency_distillation",
        batch_size=1,
        sampler="onestep",#onestep or multistep
        steps=4,
        sample_save_path = './result',
        val_data = './dataset/UIEB/VAL',
        model_path=f"./Pretrain_models/model_SDGDN/SDGDN_model.pth",
        seed=88,# or None
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
        eval = 'True'#False or True
        main(eval)
    
