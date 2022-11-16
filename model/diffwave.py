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
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        step = torch.arange(self.dim//2)
        #self.embedding_vector = torch.exp(-log(1e4) * step.unsqueeze(0))
        self.embedding_vector = 10.0 ** (step * 4.0/63)

        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        # TODO: fast sampling
        x = self._build_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _build_embedding(self, diffusion_step):
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
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, freq_bins, residual_channels, dilation, keep_residual):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(freq_bins, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, residual_channels, 1)
        if keep_residual:
            self.output_residual = Conv1d(residual_channels, residual_channels, 1)
        self.keep_residual = keep_residual

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        skip = self.output_projection(y)

        if self.keep_residual:
            residual = self.output_residual(y)
            return (x + residual) / sqrt(2.0), skip
        else:
            return None, skip



class DiffWave(nn.Module):
    def __init__(self,
                 spec_config,
                 num_timesteps,
                 residual_channels=64,
                 residual_layers=30,
                 dilation_cycle_length=10,
                 ):
        super().__init__()

        hop_samples = spec_config['hop_samples']
        win_length = spec_config['window_length']
        if spec_config['is_mel']:
            freq_bins = spec_config['n_mel']
        else:
            freq_bins = win_length // 2 + 1

        self.input_projection = Conv1d(1, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding()
        self.spectrogram_upsampler = SpectrogramUpsampler(win_length, hop_samples)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(freq_bins, residual_channels, 2 ** (i % dilation_cycle_length), keep_residual = (i<residual_layers-1))
            for i in range(residual_layers)
        ])
        self.len_res = len(self.residual_layers)
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spectrogram, audio, diffusion_step):
        """
            spectrogram: [B, 1, n_freq, n_time]
            audio: [B, 1, T]
            diffusion_step [B, 1, 1]
        """
        diffusion_step = diffusion_step.squeeze(-1)
        x = self.input_projection(audio)
        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        spectrogram = self.spectrogram_upsampler(spectrogram)


        skip = 0.
        for layer in self.residual_layers:
            x, skip_connection = layer(x, spectrogram, diffusion_step)
            skip += skip_connection

        x = skip / sqrt(self.len_res)

        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)

        return x

if __name__ == '__main__':
    emb = DiffusionEmbedding(128)
    embedding = emb._build_embedding(torch.ones(1))
    print(embedding)
