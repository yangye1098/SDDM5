import math
import torch
from torch import nn
import math



def conv3x3(in_planes, out_planes) :
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(1,1), padding=(1,1),
                     dilation=(1,1), bias=True)
    return conv

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

        self.projection1 = nn.Linear(dim, dim*4)
        self.projection2 = nn.Linear(dim*4, dim*4)

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

class UpsampleFIR(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

class UpsampleSP(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

class DownsampleFIR(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass




class AttnBlock(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return (out + input)/math.sqrt(2.)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        groups = min(dim//4, 32)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0. else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, noise_level_emb_dim, dropout=0., with_attn=False, with_down=False, with_up=False):
        super().__init__()
        self.noise_emb = nn.Linear(noise_level_emb_dim, dim_out)
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out or with_down or with_up else nn.Identity()
        self.with_attn = with_attn

        if with_attn:
            self.attn = AttnBlock(dim_out)
        self.with_down = with_down
        # regular down sample
        if with_down:
            self.down_x = Downsample(dim_out)
            self.down_h = Downsample(dim_out)

        self.with_up = with_up
        if with_up:
            self.up_x = Upsample(dim_out)
            self.up_h = Upsample(dim_out)


    def forward(self, x, time_emb):
        # time_emb: [B, noise_level_emb_dim]
        # x: [B, dim, N, N]
        b = x.shape[0]
        h = x
        if self.with_up:
            h = self.up_h(h)
            x = self.up_x(x)
        elif self.with_down:
            h = self.down_h(h)
            x = self.down_x(x)

        h = self.block1(h)
        h = h + self.noise_emb(time_emb).view((b, -1, 1, 1))
        h = self.block2(h)
        h = (h + self.res_conv(x))/math.sqrt(2.)

        if self.with_attn:
            return self.attn(h)
        else:
            return h


class NCSNPP_REP(nn.Module):
    def __init__(
                self,
                in_channel=4,
                out_channel=2,
                inner_channel=128,
                channel_mults=(1, 1, 2, 2, 2, 2, 2),
                attn_layers=(4,),
                res_blocks=2,
                dropout=0.,
                noise_emb_scale=1000,
                ):
        super().__init__()

        # first conv raise # channels to inner_channel


        noise_level_channels = 128
        self.noise_level_emb = NoiseEmbedding(dim=noise_level_channels, scale=noise_emb_scale)
        noise_level_channels = noise_level_channels * 4

        self.first_conv = nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])

        self.progressive_downs = nn.ModuleList([])
        self.progressive_down_branches = nn.ModuleList([])
        self.combiners = nn.ModuleList([])

        # record the number of output channels
        feat_channels = [inner_channel]

        num_mults = len(channel_mults)

        n_channel_in = inner_channel
        for ind in range(num_mults):

            n_channel_out = inner_channel * channel_mults[ind]
            use_attn = ind in attn_layers
            for _ in range(0, res_blocks):
                self.downs.append(ResnetBlock(
                    n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channels, dropout=dropout, with_attn=use_attn))

                feat_channels.append(n_channel_out)
                n_channel_in = n_channel_out
            if ind != num_mults-1:
                # doesn't change # channels
                self.downs.append(ResnetBlock(
                    n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channels, dropout=dropout, with_attn=False, with_down=True))
                self.progressive_downs.append(Downsample(in_channel))
                self.progressive_down_branches.append(nn.Conv2d(in_channel, n_channel_out, kernel_size=1))
                feat_channels.append(n_channel_out)

        n_channel_out = n_channel_in

        self.mid = nn.ModuleList([])

        self.mid.append(ResnetBlock(
                n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channels, dropout=dropout, with_attn=False))
        self.mid.append(ResnetBlock(
            n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channels, dropout=dropout, with_attn=True))
        self.mid.append(ResnetBlock(
                n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channels, dropout=dropout, with_attn=False))

        self.ups = nn.ModuleList([])
        self.progressive_up_branches = nn.ModuleList([])
        self.progressive_ups = nn.ModuleList([])

        n_channel_in = n_channel_out
        for ind in reversed(range(num_mults)):

            n_channel_out = inner_channel * channel_mults[ind]

            # combine resnet block skip connection
            use_attn = ind in attn_layers
            for _ in range(0, res_blocks+1):
                self.ups.append(ResnetBlock(
                    n_channel_in+feat_channels.pop(), n_channel_out, noise_level_emb_dim=noise_level_channels,
                    dropout=dropout, with_attn=use_attn))

                n_channel_in = n_channel_out

            # combine skip connection from down sample layers

            # up sample
            self.progressive_up_branches.append(nn.Sequential(
                    nn.GroupNorm(num_groups=min(n_channel_out // 4, 32), num_channels=n_channel_out),
                    Swish(),
                    nn.Conv2d(n_channel_out, in_channel, kernel_size=3, padding=1),
            ))
            if ind != 0:
                self.progressive_ups.append(Upsample(in_channel))
                self.ups.append(ResnetBlock(
                n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channels,
                dropout=dropout, with_attn=False, with_up=True))


        self.final_conv = nn.Conv2d(in_channel, out_channel,kernel_size=1)
        #self.final_conv = Block(n_channel_in, out_channel, groups=norm_groups)



    def forward(self, x_t, time, noisy_condition):
        """
            x_t: [B, 1, N, L]
            time: [B]
            noisy_condition: same size as x_t
        """
        # expand to 4d
        input = torch.cat([x_t.real, x_t.imag, noisy_condition.real, noisy_condition.imag], dim=1)
        progressive_input = input
        input = self.first_conv(input)
        # time  is in [t_eps, 1]

        t = self.noise_level_emb(time)

        n_down = 0
        feats = [input]
        for layer in self.downs:
            if isinstance(layer, ResnetBlock):
                input = layer(input, t)
                if layer.with_down:
                    # downsample layer
                    progressive_input = self.progressive_downs[n_down](progressive_input)
                    input = input + self.progressive_down_branches[n_down](progressive_input) # sum
                    n_down = n_down + 1

                else:
                    pass

            else:
                raise ValueError

            feats.append(input)

        for layer in self.mid:
            if isinstance(layer, ResnetBlock):
                input = layer(input, t)
            else:
                raise ValueError

        n_up = 0
        progressive_input = 0
        for layer in self.ups:
            if isinstance(layer, ResnetBlock):
                if layer.with_up:
                    progressive_input = progressive_input + self.progressive_up_branches[n_up](input)
                    # upsample layer
                    progressive_input = self.progressive_ups[n_up](progressive_input)
                    n_up = n_up+1

                    input = layer(input, t)
                else:
                    input = layer(torch.cat((input, feats.pop()), dim=1), t)
            else:
                raise ValueError

        # last progressive up
        progressive_input = progressive_input + self.progressive_up_branches[n_up](input)

        output = self.final_conv(progressive_input)
        output = torch.permute(output, (0, 2, 3, 1)).contiguous()
        output = torch.view_as_complex(output)[:, None, :, :] # same shape as x_t

        return output
