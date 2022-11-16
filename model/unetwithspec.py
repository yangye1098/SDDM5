import math
import torch
from torch import nn



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def silu(x):
    return x * torch.sigmoid(x)

class PositionalEncoding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        half_dim = self.dim //2
        step = torch.arange(half_dim)
        self.embedding_vector = 10.0 ** (step * 4.0/half_dim)

        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)

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


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level

        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x



class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim, dropout=0, norm_groups=32, use_affine_level=False):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)

class WaveformEncoder(nn.Module):
    def __init__(self, spec_config):
        super().__init__()

        window_length = spec_config['window_length']
        hop_samples = spec_config['hop_samples']

        is_mel = spec_config['is_mel']
        if is_mel:
            freq_channels = spec_config['n_mel']
        else:
            freq_channels = window_length // 2

        self.encoder_1 = nn.Conv1d(1, freq_channels, kernel_size=window_length, stride=hop_samples)
        self.encoder_2 = nn.Conv1d(1, freq_channels, kernel_size=window_length, stride=hop_samples)


    def forward(self, x, noisy_audio, noisy_spec):
        x_enc = self.encoder_1(x)
        x_enc = x_enc.unsqueeze(1)
        audio_enc = self.encoder_2(noisy_audio)
        audio_enc = audio_enc.unsqueeze(1)
        noise_spec = noisy_spec.unsqueeze(1)
        encoded = torch.cat((x_enc, audio_enc, noise_spec), dim=1)

        return encoded

class WaveformDecoder(nn.Module):
    def __init__(self, spec_config):
        super().__init__()

        window_length = spec_config['window_length']
        hop_samples = spec_config['hop_samples']

        is_mel = spec_config['is_mel']
        if is_mel:
            freq_channels = spec_config['n_mel']
        else:
            freq_channels = window_length // 2 

        self.decoder = nn.ConvTranspose1d(in_channels=freq_channels, out_channels=1, kernel_size=window_length, stride=hop_samples)


    def forward(self, x):
        x = x.squeeze(1)
        decoded = self.decoder(x)

        return decoded



class UNetWithSpec(nn.Module):
    def __init__(
                self,
                spec_config,
                num_timesteps,
                in_channel=3,
                out_channel=1,
                inner_channel=32,
                norm_groups=32,
                channel_mults=(1, 2, 3, 4, 5),
                res_blocks=3,
                dropout=0,
                ):
        super().__init__()

        # first conv raise # channels to inner_channel


        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(noise_level_channel),
            nn.Linear(noise_level_channel, noise_level_channel * 4),
            Swish(),
            nn.Linear(noise_level_channel * 4, noise_level_channel)
        )

        self.encoder = WaveformEncoder(spec_config)

        self.downs = nn.ModuleList([nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)])

        # record the number of output channels
        feat_channels = [inner_channel]

        num_mults = len(channel_mults)

        n_channel_in = inner_channel
        for ind in range(num_mults):

            n_channel_out = inner_channel * channel_mults[ind]

            for _ in range(0, res_blocks):
                self.downs.append(ResnetBlock(
                    n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(n_channel_out)
                n_channel_in = n_channel_out

            # doesn't change # channels
            self.downs.append(Downsample(n_channel_out))
            feat_channels.append(n_channel_out)

        n_channel_out = n_channel_in
        self.mid = nn.ModuleList([
                ResnetBlock(n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                   dropout=dropout),
        ])
        self.ups = nn.ModuleList([])


        for ind in reversed(range(num_mults)):

            n_channel_in = inner_channel * channel_mults[ind]
            n_channel_out = n_channel_in

                # combine down sample layer skip connection
            self.ups.append(ResnetBlock(
                    n_channel_in + feat_channels.pop(), n_channel_out, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout))

            # up sample
            self.ups.append(Upsample(n_channel_out))

            if ind == 0:
                n_channel_out = inner_channel
            else:
                n_channel_out = inner_channel * channel_mults[ind-1]

            # combine resnet block skip connection
            for _ in range(0, res_blocks):
                self.ups.append(ResnetBlock(
                    n_channel_in+feat_channels.pop(), n_channel_out, noise_level_emb_dim=noise_level_channel , norm_groups=norm_groups,
                    dropout=dropout))
                n_channel_in = n_channel_out

        n_channel_in = n_channel_out
        self.final_conv = Block(n_channel_in, out_channel, groups=norm_groups)
        self.decoder = WaveformDecoder(spec_config)




    def forward(self, x_t, noisy_audio, noisy_spec, noise_level):
        """
            x_t: [B, 1, T]
            noisy_audio: [B, 1, T]
            noisy_spec: [B, freq_channels, n_timebins]
            time: [B, 1, 1]
        """
        # expand to 4d
        noise_level = noise_level.unsqueeze(dim=-1)
        input = self.encoder(x_t, noisy_audio, noisy_spec)

        t = self.noise_level_mlp(noise_level)

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlock):
                input = layer(input, t)
            else:
                input = layer(input)
            feats.append(input)
        for layer in self.mid:
            input = layer(input, t)

        for layer in self.ups:
            if isinstance(layer, ResnetBlock):
                input = layer(torch.cat((input, feats.pop()), dim=1), t)
            else:
                input = layer(input)

        output = self.final_conv(input)
        output = self.decoder(output)
        return output
