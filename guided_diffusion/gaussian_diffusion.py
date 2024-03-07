import math
import os
import random
from functools import partial
from math import exp
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from util.img_utils import clear_color, Blurkernel
from .posterior_mean_variance import get_mean_processor, get_var_processor
from .svd_replacement import SuperResolution, Deblurring, Deblurring2D



__SAMPLER__ = {}

def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!") 
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(sampler,
                   steps,
                   noise_schedule,
                   model_mean_type,
                   model_var_type,
                   dynamic_threshold,
                   clip_denoised,
                   rescale_timesteps,
                   c_rate,
                   particle_size,
                   timestep_respacing=""
                   ):
    
    sampler = get_sampler(name=sampler)
    
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
         
    return sampler(use_timesteps=space_timesteps(steps, timestep_respacing),
                   betas=betas,
                   model_mean_type=model_mean_type,
                   model_var_type=model_var_type,
                   dynamic_threshold=dynamic_threshold,
                   clip_denoised=clip_denoised, 
                   rescale_timesteps=rescale_timesteps,
                   c_rate=c_rate,
                   particle_size=particle_size)


class GaussianDiffusion:
    def __init__(self,
                 betas,
                 model_mean_type,
                 model_var_type,
                 dynamic_threshold,
                 clip_denoised,
                 rescale_timesteps,
                 c_rate,
                 particle_size
                 ):

        # use float64 for accuracy.
        self.sigma = 0.05
        self.c_rate = c_rate
        self.M = particle_size    # Subsampling size
        
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <=1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.variance_diff = self.alphas_cumprod * self.sigma * self.sigma
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        self.posterior_variance = (1.0 - self.alphas_cumprod_prev) * self.c_rate
        
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
        self.ex_values = self.variance_diff / (self.posterior_variance + 1.0e-6)

        self.mean_processor = get_mean_processor(model_mean_type,
                                                 betas=betas,
                                                 c_rate=c_rate,
                                                 dynamic_threshold=dynamic_threshold,
                                                 clip_denoised=clip_denoised)    
    
        # self.var_processor = get_var_processor(model_var_type,
        #                                       betas=betas)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        
        mean = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start) * x_start
        variance = extract_and_expand(1.0 - self.alphas_cumprod, t, x_start)
        log_variance = extract_and_expand(self.log_one_minus_alphas_cumprod, t, x_start)

        return mean, variance, log_variance

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return coef1 * x_start + coef2 * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = extract_and_expand(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = extract_and_expand(self.posterior_log_variance_clipped, t, x_t)

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_sample_loop(self,
                      model,
                      x_start,
                      measurement,
                      measurement_cond_fn,
                      operator,
                      op,
                      mask,
                      record,
                      save_root):
        """
        The function used for sampling from noise.
        """ 
        img = x_start
        device = measurement.device
        batch_size = 1
        
        if op == 'super_resolution':
            self.task_Svd = SuperResolution(channels=3, img_dim=256, ratio=4, device=device)
        
        elif op == 'gaussian_blur':
            kernel = operator.get_kernel().type(torch.float64).reshape(61,61)
            kernel = kernel[30,:] / torch.sqrt(kernel[30,30])
            self.task_Svd = Deblurring(kernel=kernel, channels=3, img_dim=256, device=device)
        
        elif op == 'motion_blur':
            kernel = operator.get_kernel().type(torch.float64).reshape(61,61)
            kernel1 = kernel[30,:] / torch.sum(kernel[30,:])
            kernel1 = torch.tensor([0.0] * 30 + [1.0] + [0.0] * 30)
            conv2 = Blurkernel(blur_type='gaussian',
                              kernel_size=61,
                              std=0.5,
                              device=device).to(device)
            kernel2 = conv2.get_kernel().view(1,1,61,61).type(torch.float64).reshape(61,61)
            kernel2 = kernel2[30,:] / torch.sum(kernel2[30,:])
            self.task_Svd = Deblurring2D(kernel1=kernel1, kernel2=kernel2, channels=3, img_dim=256, device=device)
        
        
        # Backward DDIM Generation of y-sequence
        
        if op == 'inpainting':
            y_last = operator.forward(torch.randn_like(img), mask=mask)
        elif op == 'super_resolution':
            y_last = operator.forward(torch.randn_like(img))
        else:
            y_last = self.task_Svd.forward(torch.randn_like(img))

        ys = [y_last]
        coef_1 = np.sqrt((1-self.c_rate) * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        coef_2 = np.sqrt(self.alphas_cumprod_prev)
        coef_3 = np.sqrt(self.alphas_cumprod)
        coef_4 = np.sqrt(self.c_rate) * np.sqrt(1.0 - self.alphas_cumprod_prev)
        for _ in range(self.num_timesteps - 1, 0, -1):
            noise = torch.randn_like(img)
            c1 = extract_and_expand(coef_1, _, measurement)
            c2 = extract_and_expand(coef_2, _, measurement)
            c3 = extract_and_expand(coef_3, _, measurement)
            c4 = extract_and_expand(coef_4, _, measurement)
            
            if op == 'inpainting':
                noise_ = operator.forward(noise, mask=mask)
            elif op == 'super_resolution':
                noise_ = operator.forward(noise)
            else:
                noise_ = self.task_Svd.forward(noise)
            new_y = c2 * measurement + c1 * (ys[-1] - c3 * measurement) + c4 * noise_
            ys.append(new_y)
            
        ys = ys[::-1]

        starting = True
        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx], device=device)
            y = ys[idx]
            y = torch.tile(y, (self.M, 1, 1, 1))
            
            if starting:
                if op == 'inpainting':
                    noise = torch.randn_like(y)
                    img = operator.forward(y, mask=mask)
                    img = img + noise - operator.forward(noise, mask=mask)
                    starting = False
                    continue
                else:
                    noise = torch.randn_like(img)
                    tmp = self.task_Svd.transpose(y)
                    img = self.task_Svd.get_mean(tmp, self.variance_diff[idx])
                    img = img + self.task_Svd.get_noise(noise, self.variance_diff[idx])
                    starting = False
                    img = img.reshape(batch_size, 3, 256, 256)
                    continue
            
            img = img.requires_grad_()
            out = self.p_sample(x=img, t=time, model=model, operator = operator, op = op, mask = mask, Y = y)
            img = out['sample']
            img = img.detach().reshape(self.M, 3, 256, 256)
            if record:
                if idx % 10 == 0:
                    file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))
                    file_path = os.path.join(save_root, f"progress/y_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(y))

        return torch.unsqueeze(img[0], 0)      
        
    def p_sample(self, model, x, t):
        raise NotImplementedError
        
    def p_mean_variance(self, model, x, t, operator, op, mask, Y):
        
        model_output = model(x, self._scale_timesteps(t))

        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x.shape[1]:
            model_output, _ = torch.split(model_output, x.shape[1], dim=1)

        model_mean_back, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_mean = model_mean_back
        
        if t.data[0] != 0:
            coef = extract_and_expand(1/self.ex_values, t, x)
            if op == 'inpainting':
                model_mean = model_mean_back + coef * operator.forward(Y, mask=mask)
                model_mean = model_mean - (coef / (coef+1.0)) * operator.forward(model_mean, mask=mask)
            else:
                model_mean = model_mean + coef * self.task_Svd.transpose(Y)
                model_mean = self.task_Svd.get_mean(model_mean, self.ex_values[t.data[0]])
            
        model_variance = self.sigma * self.sigma * extract_and_expand(self.alphas_cumprod_prev, t, Y)
        model_log_variance = torch.log(model_variance)


        return {'mean_back': model_mean_back,
                'mean': model_mean,
                'variance': model_variance,
                'log_variance': model_log_variance,
                'pred_xstart': pred_xstart}

    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


@register_sampler(name='ddim')
class DDIM(SpacedDiffusion):
    def p_sample(self, model, x, t, operator, op, mask, Y):  # ZD: Add operator and mask.
        out = self.p_mean_variance(model, x, t, operator, op, mask, Y)  # ZD: Add operator and mask.
        sample = out['mean']
        sample_back = out['mean_back']
        
        if t.data[0] != 0:  # no noise when t == 0
            M = self.M
            samples = []
            model_variance = self.sigma * self.sigma * self.alphas_cumprod_prev[t.data[0]]
            sample_list = [torch.unsqueeze(sample[i], 0) for i in range(M)]
            sample_back_list = [torch.unsqueeze(sample_back[i], 0) for i in range(M)]
            e_x = sample_list[0]
            e_y = torch.unsqueeze(Y[0], 0)
            a_coef = self.variance_diff
            c1_coef = np.sqrt(self.posterior_variance)
            a = extract_and_expand(a_coef, t, e_x)
            c1 = extract_and_expand(c1_coef, t, e_x)
            
            # ZD: Inpainting -------------------
            
            if op == 'inpainting':
                self.union_variance = a_coef * self.posterior_variance / (a_coef + self.posterior_variance)
                c2_coef = np.sqrt(self.union_variance)
                c2 = extract_and_expand(c2_coef, t, Y)
                for _ in range(M):
                    noise = torch.randn_like(e_x)
                    noise = c1 * noise - (c1 - c2) * operator.forward(noise, mask=mask)
                    samples.append(sample_list[_] + noise)
                prob_y = [-torch.linalg.norm(e_y - operator.forward(samples[i], mask=mask)).item() ** 2 / (2*model_variance) for i in range(M)]
                prob_x = [-torch.linalg.norm(samples[i] - sample_back_list[i]).item() ** 2 / (2 * self.posterior_variance[t.data[0]]) for i in range(M)]
                prob_prev = [-torch.linalg.norm(samples[i] - sample_list[i]).item() ** 2 / (2 * self.posterior_variance[t.data[0]]) + torch.linalg.norm(operator.forward(samples[i] - sample_list[i], mask=mask)).item() ** 2 * (1/(2*self.posterior_variance[t.data[0]])-1/(2*self.union_variance[t.data[0]])) for i in range(M)]
                prob = [prob_y[_] + prob_x[_] - prob_prev[_] for _ in range(M)]
                exp_prob = [exp(x-max(prob)) for x in prob]
                sample_id = random.choices(list(range(M)), weights=exp_prob, k=M)
                samples = [samples[i] for i in sample_id]
                sample = torch.stack(samples, 0)[:,0,:,:,:]
                
            elif op == 'super_resolution':
                for _ in range(M):
                    noise = torch.randn_like(e_x)
                    noise = c1 * noise
                    noise = self.task_Svd.get_noise(noise, self.ex_values[t.data[0]])
                    samples.append(sample_list[_] + noise)
                prob_y = [-torch.linalg.norm(e_y - operator.forward(samples[i])).item() ** 2 / (2*model_variance) for i in range(M)]
                prob_x = [-torch.linalg.norm(samples[i] - sample_back_list[i]).item() ** 2 / (2 * c1 * c1) for i in range(M)]
                prob_prev = [-torch.linalg.norm(samples[i] - sample_list[i]).item() ** 2 / (2 * c1 * c1) - torch.linalg.norm(operator.forward(samples[i]-sample_list[i])).item() ** 2 / (2 * a) for i in range(M)]
                prob = [prob_y[_] + prob_x[_] - prob_prev[_] for _ in range(M)]
                exp_prob = [exp(x-max(prob)) for x in prob]
                sample_id = random.choices(list(range(M)), weights=exp_prob, k=M)
                samples = [samples[i] for i in sample_id]
                sample = torch.stack(samples, 0)[:,0,:,:,:]
            
            else:
                for _ in range(M):
                    noise = torch.randn_like(e_x)
                    noise = c1 * noise
                    noise = self.task_Svd.get_noise(noise, self.ex_values[t.data[0]])
                    samples.append(sample_list[_] + noise)
                prob_y = [-torch.linalg.norm(e_y - self.task_Svd.forward(samples[i])).item() ** 2 / (2*model_variance) for i in range(M)]
                prob_x = [-torch.linalg.norm(samples[i] - sample_back_list[i]).item() ** 2 / (2 * c1 * c1) for i in range(M)]
                prob_prev = [-torch.linalg.norm(samples[i] - sample_list[i]).item() ** 2 / (2 * c1 * c1) - torch.linalg.norm(operator.forward(samples[i]-sample_list[i])).item() ** 2 / (2 * a) for i in range(M)]
                prob = [prob_y[_] + prob_x[_] - prob_prev[_] for _ in range(M)]
                exp_prob = [exp(x-max(prob)) for x in prob]
                sample_id = random.choices(list(range(M)), weights=exp_prob, k=M)
                samples = [samples[i] for i in sample_id]
                sample = torch.stack(samples, 0)[:,0,:,:,:]

        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

# =================
# Helper functions
# =================

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# ================
# Helper function
# ================

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])
   
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
