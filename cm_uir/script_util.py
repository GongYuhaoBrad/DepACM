import argparse
import cv2
from .karras_diffusion import KarrasDenoiser
from .SDGDN import SDGDN
import numpy as np
import torch.nn as nn
NUM_CLASSES = 1000
def save_image(img_idx,all_images,name):
        img_rgb = all_images[img_idx]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_path = f'{name}/{str(img_idx+1).zfill(4)}.png'
        cv2.imwrite(img_path, img_bgr)
def replace_bn_with_gn(module, num_groups=8):
    """
    将模块中的所有 BatchNorm2d 层替换为 GroupNorm 层。
    num_groups: GroupNorm 的组数，常见选择有 8, 16, 32。通常通道数需要能被 num_groups 整除。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # 计算 GroupNorm 所需的组数，确保通道数可被整除
            num_channels = child.num_features
            # 如果指定组数大于通道数，则使用通道数（即退化为 InstanceNorm）
            groups = min(num_groups, num_channels)
            # 创建新的 GroupNorm 层，保持类似的超参数
            new_layer = nn.GroupNorm(
                num_groups=groups,
                num_channels=num_channels,
                eps=child.eps,
                affine=child.affine  # 保留原有的可学习参数（gamma 和 beta）
            )
            # 如果需要，可以将 BN 的运行均值和方差拷贝到 GN，但 GN 无需这些统计量，通常忽略
            # 将新层移动到原模型的设备上
            new_layer.to(next(child.parameters()).device)
            setattr(module, name, new_layer)
        else:
            # 递归遍历子模块
            replace_bn_with_gn(child, num_groups)
def cm_train_defaults():
    return dict(
        teacher_model_path="",
        teacher_dropout=0.0,
        training_mode="consistency_training",
        target_ema_mode="fixed",
        scale_mode="fixed",
        global_batch_size=-1,
        microbatch=-1,
        lr_anneal_steps=500000,
        start_ema=0.95,
        start_scales=4,
        end_scales=18,
        distill_steps_per_iter=50000,
        loss_norm="l2",
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        sigma_min=0.002,
        sigma_max=80.0,
        image_size=256,
        num_channels=64,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=32,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=False, 
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
        learn_sigma=False,
        weight_schedule="uniform",
    )
    return res


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    weight_schedule,
    start_scales = 4,
    sigma_min=0.002,
    sigma_max=80.0,
    distillation=False,
    
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = KarrasDenoiser(
        sigma_data=0.5,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        distillation=distillation,
        weight_schedule=weight_schedule,
        num_scales=start_scales,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0.1,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1,2,2.5)#(1,2,4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return SDGDN(
        image_size=image_size,
        in_channels=6,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1

        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        elif target_ema_mode == "fixed" and scale_mode == "progdist":
            distill_stage = step // distill_steps_per_iter
            scales = start_scales // (2**distill_stage)
            scales = np.maximum(scales, 2)

            sub_stage = np.maximum(
                step - distill_steps_per_iter * (np.log2(start_scales) - 1),
                0,
            )
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = 2 // (2**sub_stage)
            sub_scales = np.maximum(sub_scales, 1)

            scales = np.where(scales == 2, sub_scales, scales)

            target_ema = 1.0
        else:
            raise NotImplementedError

        return float(target_ema), int(scales)

    return ema_and_scales_fn


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
