import math
import torch
from torch import nn



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def silu(x):
    return x * torch.sigmoid(x)

class NoiseEmbedding(nn.Module):
    def __init__(self, dim=128, scale=1000):
        super().__init__()
        self.dim = dim
        half_dim = self.dim // 2
        step = torch.arange(half_dim)/(half_dim-1)
        self.embedding_vector = scale*10.0 ** (-step * 4.0)

        self.projection1 = nn.Linear(dim, 512)
        self.projection2 = nn.Linear(512, dim)

    def forward(self, t):
        # TODO: fast sampling
        x = self._build_embedding(t)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _build_embedding(self, t):
        self.embedding_vector = self.embedding_vector.to(t.device)
        encoding = t.unsqueeze(-1) * self.embedding_vector
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


class UNet(nn.Module):
    def __init__(
                self,
                in_channel=4,
                out_channel=2,
                inner_channel=32,
                norm_groups=32,
                channel_mults=(1, 2, 3, 4, 5),
                res_blocks=3,
                dropout=0,
                noise_emb_scale=1000,
                ):
        super().__init__()

        # first conv raise # channels to inner_channel


        noise_level_channels = 128
        self.noise_level_emb = NoiseEmbedding(dim=noise_level_channels, scale=noise_emb_scale)

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
                    n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channels, norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(n_channel_out)
                n_channel_in = n_channel_out

            # doesn't change # channels
            self.downs.append(Downsample(n_channel_out))
            feat_channels.append(n_channel_out)

        n_channel_out = n_channel_in
        self.mid = nn.ModuleList([
                ResnetBlock(n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channels, norm_groups=norm_groups,
                                   dropout=dropout),
        ])
        self.ups = nn.ModuleList([])


        for ind in reversed(range(num_mults)):

            n_channel_in = inner_channel * channel_mults[ind]
            n_channel_out = n_channel_in

                # combine down sample layer skip connection
            self.ups.append(ResnetBlock(
                    n_channel_in + feat_channels.pop(), n_channel_out, noise_level_emb_dim=noise_level_channels,
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
                    n_channel_in+feat_channels.pop(), n_channel_out, noise_level_emb_dim=noise_level_channels , norm_groups=norm_groups,
                    dropout=dropout))
                n_channel_in = n_channel_out

        n_channel_in = n_channel_out
        self.final_conv = Block(n_channel_in, out_channel, groups=norm_groups)



    def forward(self, x_t, time, noisy_condition):
        """
            x_t: [B, 1, N, L]
            time: [B]
            noisy_condition: same size as x_t
        """
        # expand to 4d
        input = torch.cat([x_t.real, x_t.imag, noisy_condition.real, noisy_condition.imag], dim=1)

        # time  is in [t_eps, 1]

        t = self.noise_level_emb(time)

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
        output = torch.permute(output, (0, 2, 3, 1)).contiguous()
        output = torch.view_as_complex(output)[:, None, :, :] # same shape as x_t

        return output
