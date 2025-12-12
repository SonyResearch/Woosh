import torch 
from torch import nn

from .blocks import ResnetBlock, make_attn, Normalize, nonlinearity, Upsample1d, UpsampleTimeStride4, Downsample1d, DownsampleTimeStride4

class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        downsample_time_stride4_levels=[],
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.downsample_time_stride4_levels = downsample_time_stride4_levels

        if len(self.downsample_time_stride4_levels) > 0:
            assert max(self.downsample_time_stride4_levels) < self.num_resolutions, (
                "The level to perform downsample 4 operation need to be smaller than the total resolution number %s"
                % str(self.num_resolutions)
            )

        # downsampling
        self.conv_in = torch.nn.Conv1d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult) # Channel multiplicator 
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = 0
        for i_level in range(self.num_resolutions):
            print("resolution:", i_level)
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level] 
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out

                # Attention layers
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if i_level in self.downsample_time_stride4_levels:
                    down.downsample = DownsampleTimeStride4(block_in, resamp_with_conv)
                else:
                    #down.downsample = Downsample(block_in, resamp_with_conv)
                    down.downsample = Downsample1d()
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        hs = [self.conv_in(x)] # returns none
        

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        downsample_time_stride4_levels=[],
        attn_type="vanilla",
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.downsample_time_stride4_levels = downsample_time_stride4_levels

        if len(self.downsample_time_stride4_levels) > 0:
            assert max(self.downsample_time_stride4_levels) < self.num_resolutions, (
                "The level to perform downsample 4 operation need to be smaller than the total resolution number %s"
                % str(self.num_resolutions)
            )

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(
        # self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv1d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level - 1 in self.downsample_time_stride4_levels:
                    up.upsample = UpsampleTimeStride4(block_in, resamp_with_conv)
                else:
                    #up.upsample = Upsample(block_in, resamp_with_conv)
                    up.upsample = Upsample1d()
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h