import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision import models
from .nn import mean_flat, append_dims, append_zero
def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


class KarrasDenoiser:
    def __init__(
        self,
        sigma_data: float = 1.0,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="l2",
        num_scales=4.0,
    ):
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.eval().cuda()
        self.loss_per = PerpetualLoss(vgg_model).cuda()
        self.device = th.device(
            'cuda' if th.cuda.is_available() is not None else 'cpu')
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        self.rho = rho
        self.num_timesteps = 40
        self.device0 = th.device(
            'cuda' if th.cuda.is_available() is not None else 'cpu')
        self.weight = {}
        for n in range(int(num_scales)-1):
            self.weight[n]=1.0+((3.0-1.0)/(num_scales-1))*n # You can change the 3.0 and 1.0 to adjust the weight range.
    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def training_losses(self, model, x_start, sigmas, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        terms = {}
        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        model_output, denoised = self.denoise(model, x_t, sigmas, **model_kwargs)
        snrs = self.get_snr(sigmas)
        weights = append_dims(
            get_weightings(self.weight_schedule, snrs, self.sigma_data), dims
        )
        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
        terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms
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
    def consistency_losses(
        self,
        model,
        num_scales,
        input,
        target,
        LUDEN,
        model_kwargs=None,
        target_model=None,
        noise=None,
    ):  
        if model_kwargs is None:
            model_kwargs = {}
        x_start = target
        if noise is None:
            noise = th.randn_like(x_start)
            noise=set_device(self.device,noise)
            with th.no_grad():
                prediction = LUDEN(input)
                depth=prediction.expand(-1, 3, -1, -1)
                #depth= -(depth-depth.min()) / (depth.max() - depth.min())*2
                depth = depth.to(th.float64)
        dims = x_start.ndim

        def denoise_fn(x, t,depth):
            return self.denoise(model, x, t, depth,**model_kwargs)[1]
        
        if target_model:
            @th.no_grad()
            def target_denoise_fn(x, t,depth):
                return self.denoise(target_model, x, t,depth ,**model_kwargs)[1]
        else:
            raise NotImplementedError("Must have a target model")

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            return samples
        
        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t=x_start + noise * append_dims(t, dims)
        x_tmix = th.cat([input,x_t], dim=1)
        dropout_state = (th.get_rng_state(), th.cuda.get_rng_state())
        distiller = denoise_fn(x_tmix, t,depth)
        x_t2 = euler_solver(x_t, t, t2, x_start).detach()
        x_t2mix=th.cat([input,x_t2], dim=1)
        th.set_rng_state(dropout_state[0])
        th.cuda.set_rng_state(dropout_state[1])
        distiller_target = target_denoise_fn(x_t2mix, t2,depth)
        distiller_target = distiller_target.detach()

        #loss function
        if self.loss_norm == "l1":
            diffs = th.abs(distiller - distiller_target)
            loss = mean_flat(diffs) 
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) 
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")
        perpetual_loss = self.loss_per(distiller,distiller_target)
        loss_sub = 1.0*loss+0.3*perpetual_loss
        weights_list = th.tensor([self.weight[val.item()] for val in indices],device=self.device)
        total_loss = loss_sub*weights_list
        terms = {}
        terms["loss"] = total_loss.mean()

        return terms

    def denoise(self, model, x_t, sigmas, depth, **model_kwargs):
        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim)
                for x in self.get_scalings_for_boundary_condition(sigmas)
            ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, depth,**model_kwargs)
        x_t1=x_t[:, 3:, :, :]
        denoised = c_out * model_output + c_skip * x_t1
        return model_output, denoised
    
def set_device(device,x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(device)
        else:
            x = x.to(device)
        return x

def karras_sample(
    diffusion,
    model,
    steps,
    depth_model,
    clip_denoised=True,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80.000, 
    rho=7.0,
    sampler="heun",
    ts=None,
    data=None,
):
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)
    input_de = set_device(device,data[0])
    noise= th.randn_like(input_de,device=device)
    with th.no_grad():
        prediction = depth_model(input_de)
        depth=prediction.expand(-1, 3, -1, -1)
        #depth= (depth-depth.min()) / (depth.max() - depth.min())*5
        depth = depth.to(th.float64)
    #noise=set_device(device,noise)
    depth=set_device(device,depth)
    x_Tn =input_de+noise * sigma_max
    x_T  = th.cat([input_de,x_Tn], dim=1)
    sample_fn = {
        "onestep": sample_onestep,
        "multistep": stochastic_iterative_sampler,
    }[sampler]
    if sampler == "multistep":
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=diffusion.rho, steps=steps
        )
    else:
        sampler_args = {}
    def denoiser(x_t, sigma,depth1):
        _, denoised = diffusion.denoise(model, x_t, sigma, depth1 ,**model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised
    if sampler=='onestep':
        x_0 = sample_fn(
            denoiser,
            x_T,
            sigmas,
            depth,
        )
    else:
        x_0 = sample_fn(
            input_de,
            denoiser,
            x_T,
            noise,
            depth,
            **sampler_args,
        )
    return x_0.clamp(-1, 1)
def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

@th.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    depth,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    
    return distiller(x, sigmas[0] * s_in, depth)

@th.no_grad()
def stochastic_iterative_sampler(
    input_de,
    distiller,
    x,
    noise,
    depth,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t* s_in,depth)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        noise = th.randn_like(x0).to('cuda')
        x = x0+ noise*(np.sqrt(next_t**2 - t_min**2))
        x_cat = th.cat([input_de,x], dim=1)
        x = x_cat
        debug = x0
    return debug
class PerpetualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerpetualLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            #x1 = x.double
            x = module(x.to(th.float))
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss_all = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss=F.mse_loss(dehaze_feature, gt_feature,reduction='none')
            dims = tuple(range(1, loss.dim()))
            loss_all.append(loss.mean(dim=dims))
        return sum(loss_all)/len(loss_all)
