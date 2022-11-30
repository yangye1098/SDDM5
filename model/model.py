from torch import nn
from .diffusion import GaussianDiffusion
from model.diffusion.sde import OUVESDE
from .sampling import *


class SDDM(nn.Module):
    def __init__(self, diffusion:GaussianDiffusion, noise_estimate_model:nn.Module,
                 noise_condition='sqrt_alpha_bar', p_transition='original', q_transition='original'):
        super().__init__()
        self.diffusion = diffusion
        self.noise_estimate_model = noise_estimate_model
        self.num_timesteps = self.diffusion.num_timesteps
        self.noise_condition = noise_condition
        self.p_transition = p_transition
        self.q_transition = q_transition
        if noise_condition != 'sqrt_alpha_bar' and noise_condition != 'time_step' \
            and noise_condition != 'normalized_time_step':
            raise NotImplementedError

        if p_transition != 'original' and p_transition != 'supportive' \
                and p_transition != 'sr3' and p_transition != 'conditional'\
                and p_transition != 'condition_in':
            raise NotImplementedError

        if q_transition != 'original' and q_transition != 'conditional':
            raise NotImplementedError

    # train step
    def forward(self, clean_audio, noisy_audio, extra_condition=None):
        """
        clean_audio is the clean_audio source
        condition is the noisy conditional input
        """

        # generate noise
        if self.q_transition == 'original':
            noise = torch.randn_like(clean_audio, device=clean_audio.device)
            x_t, noise_level, t = self.diffusion.q_stochastic(clean_audio, noise)
            if self.noise_condition == 'sqrt_alpha_bar':
                predicted = self.noise_estimate_model(x_t, noise_level, noisy_audio)
            elif self.noise_condition == 'time_step':
                predicted = self.noise_estimate_model(x_t, t, noisy_audio)
            elif self.noise_condition == 'normalized_time_step':
                t = t/self.num_timesteps
                predicted = self.noise_estimate_model(x_t, t, noisy_audio)
            else:
                raise ValueError
        elif self.q_transition == 'conditional':
            raise ValueError
            # noise = torch.randn_like(clean_audio, device=clean_audio.device)
            # x_t, noise, t = self.diffusion.q_stochastic_conditional(clean_audio, noisy_audio, noise)
            # predicted = self.noise_estimate_model(x_t, t, extra_condition)
        else:
            raise ValueError

        return predicted, noise

    @torch.no_grad()
    def infer(self, noisy_audio, extra_condition=None, continuous=False):
        # initial input

        # TODO: predict noise level to reduce computation cost

        if self.p_transition == 'conditional':
            # start from conditional input, conditional diffusion process
            x_t = self.diffusion.get_x_T_conditional(noisy_audio)
        elif self.p_transition == 'condition_in':
            # start from conditional input + gaussian noise, original diffusion process
            x_t = self.diffusion.get_x_T(noisy_audio)
        elif self.p_transition == 'supportive':
            # start from conditional input + gaussian noise, original diffusion process
            x_t = noisy_audio
        else:
            # start from total noise
            x_t = torch.randn_like(noisy_audio, device=noisy_audio.device)


        num_timesteps = self.diffusion.num_timesteps
        sample_inter = (1 | (num_timesteps // 100))

        batch_size = noisy_audio.shape[0]
        b = noisy_audio.shape[0]
        noise_level_sample_shape = torch.ones(noisy_audio.ndim, dtype=torch.int)
        noise_level_sample_shape[0] = b
        # iterative refinement

        samples = [noisy_audio]
        if continuous:
            assert batch_size==1, 'Batch size must be 1 to do continuous sampling'

        for t in reversed(range(1, self.num_timesteps+1)):

            if self.p_transition == 'original' or self.p_transition == 'condition_in':

                if self.noise_condition == 'sqrt_alpha_bar':
                    noise_level = self.diffusion.get_noise_level(t) * torch.ones(tuple(noise_level_sample_shape),
                                                                                 device=noisy_audio.device)
                    predicted = self.noise_estimate_model(x_t, noise_level, noisy_audio)
                elif self.noise_condition == 'time_step':
                    time_steps = t * torch.ones(tuple(noise_level_sample_shape), device=noisy_audio.device)
                    predicted = self.noise_estimate_model(x_t, time_steps, noisy_audio)

                elif self.noise_condition == 'normalized_time_step':
                    t_normalized = t/self.num_timesteps
                    t_normalized = t_normalized * torch.ones(tuple(noise_level_sample_shape), device=noisy_audio.device)
                    predicted = self.noise_estimate_model(x_t, t_normalized, noisy_audio)

                else:
                    raise ValueError

                x_t = self.diffusion.p_transition(x_t, t, predicted)
            elif self.p_transition == 'sr3':

                raise ValueError
                # if self.noise_condition == 'sqrt_alpha_bar':
                #     noise_level = self.diffusion.get_noise_level(t) * torch.ones(tuple(noise_level_sample_shape),
                #                                                                  device=noisy_audio.device)
                #     predicted = self.noise_estimate_model(x_t, noisy_audio, noisy_spec, noise_level)
                # elif self.noise_condition == 'time_step':
                #     time_steps = t * torch.ones(tuple(noise_level_sample_shape), device=noisy_audio.device)
                #     predicted = self.noise_estimate_model(x_t, noisy_audio, noisy_spec, time_steps)
                # else:
                #     raise ValueError

                # x_t = self.diffusion.p_transition_sr3(x_t, t, predicted)
            elif self.p_transition == 'conditional':
                raise ValueError

                # if self.noise_condition == 'sqrt_alpha_bar':
                #     noise_level = self.diffusion.get_noise_level(t) * torch.ones(tuple(noise_level_sample_shape),
                #                                                                  device=noisy_audio.device)
                #     predicted = self.noise_estimate_model(noisy_spec, x_t, noise_level)
                # elif self.noise_condition == 'time_step':
                #     time_steps = t * torch.ones(tuple(noise_level_sample_shape), device=noisy_audio.device)
                #     predicted = self.noise_estimate_model(noisy_spec, x_t, time_steps)
                # else:
                #     raise ValueError
                # x_t = self.diffusion.p_transition_conditional(x_t, t, predicted, noisy_audio)
            else:
                raise ValueError

            if continuous and t % sample_inter == 0:
                samples.append(x_t)

        if continuous:
            return samples
        else:
            return x_t


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())

        params = sum([p.numel() for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)





class SDDM_SDE(nn.Module):
    def __init__(self, diffusion:OUVESDE, noise_estimate_model:nn.Module, t_eps=3e-2,
                 predictor_name="reverse_diffusion", corrector_name="ald", reverse_sample_steps=30,
                 corrector_steps=1, snr = 0.5 ):
        super().__init__()
        self.sde = diffusion
        self.noise_estimate_model = noise_estimate_model
        self.t_eps = t_eps
        self.sampler_type = "pc"
        self.predictor_name = predictor_name
        self.corrector_name = corrector_name
        self.reverse_sample_steps = reverse_sample_steps
        self.corrector_steps = corrector_steps
        self.snr = snr


    def forward(self, clean, noisy, extra_condition=None):
        t = torch.rand(clean.shape[0], device=clean.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(clean, t, noisy)
        z = torch.randn_like(clean)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z
        score = -self.noise_estimate_model(perturbed_data, t, noisy)
        return score, z/sigmas

    @torch.no_grad()
    def infer(self, noisy_spec, extra_condition=None, denoise:bool=True, probability_flow: bool =False):
        """
        One-call speech enhancement of noisy speech `y`,
        """

        # get a copy of the sde and change reverse sample steps
        N = self.sde.N if self.reverse_sample_steps is None else self.reverse_sample_steps
        sde = self.sde.copy()
        sde.N = N

        if self.predictor_name == 'reverse_diffusion':
            predictor = ReverseDiffusionPredictor(sde, self.noise_estimate_model, probability_flow=probability_flow)
        elif self.predictor_name == 'euler_maruyama':
            predictor = EulerMaruyamaPredictor(sde, self.noise_estimate_model, probability_flow=probability_flow)
        else:
            raise NotImplementedError

        if self.corrector_name == 'langevin':
            corrector = LangevinCorrector(sde, self.noise_estimate_model, snr=self.snr, n_steps=self.corrector_steps)
        elif self.corrector_name == 'ald':
            corrector = AnnealedLangevinDynamics(sde, self.noise_estimate_model, snr=self.snr, n_steps=self.corrector_steps)
        else:
            raise NotImplementedError

        with torch.no_grad():
            xt = sde.prior_sampling(noisy_spec.shape, noisy_spec).to(noisy_spec.device)
            timesteps = torch.linspace(sde.T, self.t_eps, sde.N, device=noisy_spec.device)
            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(noisy_spec.shape[0], device=noisy_spec.device) * t
                xt, xt_mean = corrector.update_fn(xt, vec_t, noisy_spec)
                xt, xt_mean = predictor.update_fn(xt, vec_t, noisy_spec)
            x_result = xt_mean if denoise else xt
            ns = sde.N * (corrector.n_steps + 1)

        return x_result


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())

        params = sum([p.numel() for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
