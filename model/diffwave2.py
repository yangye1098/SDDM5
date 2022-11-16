import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt, log

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, noise_emb_channels, scale):
        super().__init__()
        half_dim = dim //2
        step = torch.arange(half_dim)
        #self.embedding_vector = torch.exp(-log(1e4) * step.unsqueeze(0))
        # 1e-4 ** (step/half_dim)
        self.scale = scale
        self.embedding_vector = 10.0 ** (-step * 4.0/half_dim)
        self.embedding_vector = scale * self.embedding_vector

        self.projection1 = Linear(dim, noise_emb_channels)
        self.projection2 = Linear(noise_emb_channels, noise_emb_channels)

    def forward(self, diffusion_step):
        # TODO: fast sampling
        x = self._build_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _build_embedding(self, diffusion_step):
        diffusion_step = diffusion_step.view(-1, 1)
        self.embedding_vector = self.embedding_vector.to(diffusion_step.device)
        encoding = diffusion_step * self.embedding_vector
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1) # [B, self.dim]
        return encoding


class SpectrogramUpsampler(nn.Module):
    def __init__(self, win_length, hop_samples):
        super().__init__()
        assert hop_samples == 256
        # up sample 16 * 16 = 256 in time
        # only work if hop_samples == 256
        self.conv1 = ConvTranspose2d(1, 1, (3, 32), stride=(1, 16), padding=(1, 8))
        self.conv2 = ConvTranspose2d(1, 1, (3, 32), stride=(1, 16), padding=(1, 8))

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = silu(x)
        x = self.conv2(x)
        x = silu(x)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, freq_bins, residual_channels, dilation, noise_emb_channels):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels,
                                   2 * residual_channels,
                                   3,
                                   padding=dilation,
                                   dilation=dilation)
        self.audio_condition_projection = Conv1d(residual_channels,
                                   2 * residual_channels,
                                   3,
                                   padding=dilation,
                                   dilation=dilation)

        self.diffusion_projection = Linear(noise_emb_channels, residual_channels)
        self.spec_condition_projection = Conv1d(freq_bins, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_residual = Conv1d(residual_channels, residual_channels, 1)

    def forward(self, x, audio_condition, spec_condition, noise_level):
        noise_level = self.diffusion_projection(noise_level).unsqueeze(-1)
        audio_conditioner = self.audio_condition_projection(audio_condition)
        spec_conditioner = self.spec_condition_projection(spec_condition)

        y = x + noise_level
        y = self.dilated_conv(y) + audio_conditioner + spec_conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        residual = self.output_residual(y)
        skip = self.output_projection(y)

        return (x + residual) / sqrt(2.0), skip


class DiffWave2(nn.Module):
    def __init__(self,
                 spec_config,
                 num_timesteps,
                 residual_channels=64,
                 residual_layers=30,
                 dilation_cycle_length=10,
                 noise_emb_dim = 128,
                 noise_emb_channels= 512,
                 noise_emb_scale = 50000
                 ):
        super().__init__()

        hop_samples = spec_config['hop_samples']
        win_length = spec_config['window_length']
        if spec_config['is_mel']:
            freq_bins = spec_config['n_mel']
        else:
            freq_bins = win_length // 2 + 1

        self.input_projection = Conv1d(1, residual_channels, 1)
        self.audio_condition_projection = Conv1d(1, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(noise_emb_dim, noise_emb_channels, noise_emb_scale)
        self.spectrogram_upsampler = SpectrogramUpsampler(win_length, hop_samples)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(freq_bins, residual_channels, 2 ** (i % dilation_cycle_length), noise_emb_channels)
            for i in range(residual_layers)
        ])
        self.len_res = len(self.residual_layers)
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 1, 1)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, noisy_audio, noisy_spec, noise_level):
        """
            spectrogram: [B, 1, n_freq, n_time]
            audio: [B, 1, T]
            noise_level: [B, 1, 1]
        """
        x = self.input_projection(x)
        x = silu(x)
        audio_condition = self.audio_condition_projection(noisy_audio)
        audio_condition = silu(audio_condition)
        noise_level = self.diffusion_embedding(noise_level)
        spec_condition = self.spectrogram_upsampler(noisy_spec)

        skip = 0.
        for layer in self.residual_layers:
            x, skip_connection = layer(x, audio_condition, spec_condition, noise_level)
            skip += skip_connection

        x = x + skip / sqrt(self.len_res)
        x = self.skip_projection(x)
        x = silu(x)
        x = self.output_projection(x)

        return x

if __name__ == '__main__':
    pass