from typing import List
import torch
from torch import nn

from sfxfm.module.model.diffusion_blocks import DBlockL, NormedConv1d, UBlockL


class UnetEncoder(nn.Module):

    def __init__(
        self,
        channels_in,
        base_dim,
        time_embed_dim,
        resblock,  # partial
        downsample,  # partial
        first_kernel_size: int,  # 3 in noise2music
        downsamplings: List[int],
        channels_mult: List[int],
        num_resblocks: List[int],
        self_attentions: List[bool],
        dropouts: List[float],
        return_skip=False,
    ) -> None:
        super().__init__()
        self.return_skip = return_skip

        self.first_conv = NormedConv1d(
            dim_in=channels_in,
            dim_out=base_dim,
            kernel_size=first_kernel_size,
            bias=True,
            wn=False,
            gn=False,
            num_groups=channels_in,
        )

        assert len(downsamplings) == len(channels_mult) == len(num_resblocks)

        self.dblocks = nn.ModuleList([])
        current_dim = base_dim
        for downsampling, mult, num_res, self_attention, dropout in zip(
            downsamplings, channels_mult, num_resblocks, self_attentions, dropouts
        ):

            self.dblocks.append(
                DBlockL(
                    downsample=downsample,
                    resblock=resblock,
                    dim_in=current_dim,
                    dim_out=base_dim * mult,
                    downsampling=downsampling,
                    num_resblocks=num_res,
                    time_embed_dim=time_embed_dim,
                    self_attention=self_attention,
                    dropout=dropout,
                )
            )
            current_dim = base_dim * mult

    #     self.ff = FourierFeatures(1, base_dim)

    # def first_conv(self, x):
    #     y = rearrange(x, 'b c t -> b t c')
    #     y = self.ff(y)
    #     y = rearrange(y, 'b t c -> b c t')
    #     return y

    def forward(self, x, t_embed=None):
        """
        x: (b c t)
        t: (b)
        """
        y = x
        y = self.first_conv(y)

        skips = []
        # skips are of the same size as y (before downsampling)

        for dblock in self.dblocks:
            y, skip = dblock(y, t_embed)
            skips.append(skip)

        if self.return_skip:
            return y, skips
        else:
            return y


class Dbranch(nn.Module):

    def __init__(
        self,
        channels_in,
        base_dim,
        time_embed_dim,
        resblock,  # partial
        downsample,  # partial
        first_kernel_size: int,  # 3 in noise2music
        downsamplings: List[int],
        channels_mult: List[int],
        num_resblocks: List[int],
        self_attentions: List[bool],
        dropouts: List[float],
        return_skip=False,
        padding_mode="reflect",
        skip_rescale=True,
    ) -> None:
        super().__init__()
        self.return_skip = return_skip

        # self.first_conv = NormedConv1d(dim_in=channels_in,
        #                                dim_out=base_dim,
        #                                kernel_size=first_kernel_size,
        #                                bias=True)

        assert len(downsamplings) == len(channels_mult) == len(num_resblocks)

        self.dblocks = nn.ModuleList([])
        current_dim = base_dim
        for downsampling, mult, num_res, self_attention, dropout in zip(
            downsamplings, channels_mult, num_resblocks, self_attentions, dropouts
        ):

            self.dblocks.append(
                DBlockL(
                    downsample=downsample,
                    resblock=resblock,
                    dim_in=current_dim,
                    dim_out=base_dim * mult,
                    downsampling=downsampling,
                    num_resblocks=num_res,
                    time_embed_dim=time_embed_dim,
                    self_attention=self_attention,
                    dropout=dropout,
                    skip_rescale=skip_rescale,
                    padding_mode=padding_mode,
                )
            )
            current_dim = base_dim * mult

    def forward(self, x, skips=None, t_embed=None):
        """
        x: (b c t)
        t: (b)
        """
        y = x
        # y = self.first_conv(y)

        # handles case where skip_in is None
        # creates on skip_in=None per layer
        if skips is None:
            skips = [None] * len(self.dblocks)

        skips_out = []
        # skips_out are of the same size as y (before downsampling)

        for dblock, skip_in in zip(self.dblocks, skips):
            y, skip_out = dblock(y, t_embed, skip=skip_in)
            skips_out.append(skip_out)

        if self.return_skip:
            return y, skips_out
        else:
            return y


class Ubranch(nn.Module):

    def __init__(
        self,
        channels_in,
        base_dim,
        time_embed_dim,
        resblock,  # partial
        upsample,  # partial
        first_kernel_size: int,  # 3 in noise2music
        downsamplings: List[int],
        channels_mult: List[int],
        num_resblocks: List[int],
        self_attentions: List[bool],
        dropouts: List[bool],
        return_skip=False,
        skip_rescale=True,
        padding_mode="reflect",
    ) -> None:
        super().__init__()
        self.return_skip = return_skip

        # self.first_conv = NormedConv1d(dim_in=channels_in,
        #                                dim_out=base_dim,
        #                                kernel_size=first_kernel_size,
        #                                bias=True)

        assert len(downsamplings) == len(channels_mult) == len(num_resblocks)

        self.ublocks = nn.ModuleList([])
        channels_mid = channels_mult[-1] * base_dim
        current_dim = channels_mid

        channels_mult = list(reversed(channels_mult))[1:] + [1]
        for upsampling, mult, num_res, self_attention, dropout in zip(
            reversed(downsamplings),
            channels_mult,
            reversed(num_resblocks),
            reversed(self_attentions),
            reversed(dropouts),
        ):

            self.ublocks.append(
                UBlockL(
                    upsample=upsample,
                    resblock=resblock,
                    dim_in=current_dim,
                    dim_out=base_dim * mult,
                    upsampling=upsampling,
                    num_resblocks=num_res,
                    time_embed_dim=time_embed_dim,
                    self_attention=self_attention,
                    dropout=dropout,
                    skip_rescale=skip_rescale,
                    padding_mode=padding_mode,
                )
            )
            current_dim = base_dim * mult

    def forward(self, x, skips=None, t_embed=None):
        """
        x: (b c t)
        t: (b)
        """
        y = x

        # y = self.first_conv(y)

        # handles case where skip_in is None
        # creates on skip_in=None per layer
        if skips is None:
            skips = [None] * len(self.ublocks)

        skips_out = []
        # skips_out are of the same size as y (before downsampling)

        for ublock, skip_in in zip(self.ublocks, skips):
            y = ublock(y, t_embed, skip=skip_in)
            skips_out.append(y)

        if self.return_skip:
            return y, skips_out
        else:
            return y


class Wnet(nn.Module):

    def __init__(
        self,
        channels_in,
        channels_cond,
        channels_out,
        time_embed_dim,
        base_dim,
        resblock,  # partial
        downsample,  # partial
        upsample,  # partial
        time_embedding,  # partial
        first_kernel_size: int,  # 3 in noise2music
        last_kernel_size: int,
        downsamplings: List[int],
        channels_mult: List[int],
        num_resblocks: List[int],
        self_attentions: List[bool],
        dropouts: List[float],
        aggregate_mode: str = "sum",
        padding_mode: str = "reflect",
        skip_rescale: bool = True,
    ) -> None:
        super().__init__()
        """
        Last downsampling is dropped
        First upsampling is used to upscale cond
        """
        self.time_embedding = time_embedding(time_embed_dim=time_embed_dim)
        self.n_q_embedding = nn.Embedding(
            num_embeddings=13, embedding_dim=time_embed_dim
        )

        # === embed condition
        channels_mid = channels_mult[-1] * base_dim
        if channels_cond == channels_mid:
            self.embed_cond = None
        else:
            # TODO check?! shoiuld
            self.embed_cond = NormedConv1d(
                dim_in=channels_cond,
                dim_out=base_dim,
                kernel_size=first_kernel_size,
                bias=True,
                wn=False,
                gn=False,
                padding_mode=padding_mode,
            )
            raise NotImplementedError

        self.ubranch_1 = Ubranch(
            channels_in=channels_in,
            base_dim=base_dim,
            # TODO t_embed in first ubranch?
            time_embed_dim=0,
            # time_embed_dim=time_embed_dim,
            resblock=resblock,
            upsample=upsample,
            first_kernel_size=first_kernel_size,
            downsamplings=downsamplings,
            channels_mult=channels_mult,
            num_resblocks=num_resblocks,
            self_attentions=self_attentions,
            dropouts=dropouts,
            skip_rescale=skip_rescale,
            padding_mode=padding_mode,
            return_skip=True,
        )

        self.first_conv = NormedConv1d(
            dim_in=channels_in,
            dim_out=base_dim,
            kernel_size=first_kernel_size,
            bias=True,
            wn=False,
            gn=False,
            num_groups=channels_in,
            padding_mode=padding_mode,
        )
        self.dbranch = Dbranch(
            channels_in=channels_in,
            base_dim=base_dim,
            time_embed_dim=time_embed_dim,
            resblock=resblock,
            downsample=downsample,
            first_kernel_size=first_kernel_size,
            downsamplings=downsamplings,
            channels_mult=channels_mult,
            num_resblocks=num_resblocks,
            self_attentions=self_attentions,
            dropouts=dropouts,
            skip_rescale=skip_rescale,
            padding_mode=padding_mode,
            return_skip=True,
        )

        self.ubranch_2 = Ubranch(
            channels_in=channels_in,
            base_dim=base_dim,
            time_embed_dim=time_embed_dim,
            resblock=resblock,
            upsample=upsample,
            first_kernel_size=first_kernel_size,
            downsamplings=downsamplings,
            channels_mult=channels_mult,
            num_resblocks=num_resblocks,
            self_attentions=self_attentions,
            dropouts=dropouts,
            skip_rescale=skip_rescale,
            padding_mode=padding_mode,
            return_skip=False,
        )

        assert len(downsamplings) == len(channels_mult) == len(num_resblocks)

        self.last_conv = NormedConv1d(
            dim_in=base_dim,
            dim_out=channels_out,
            kernel_size=last_kernel_size,
            bias=False,
            wn=False,
            gn=False,
            padding_mode=padding_mode,
        )

    def forward(self, x, t, cond=None, n_q=None):
        """
        x: (b c t)
        t: (b)
        """
        batch_size, _, _ = x.size()
        if cond is None:
            raise NotImplementedError

        # first upsample cond
        if cond is not None:
            if self.embed_cond is not None:
                c = self.embed_cond(cond)
            else:
                # no need to upscale
                c = cond
        else:
            c = None
            raise NotImplementedError

        if t is not None and self.time_embedding is not None:
            if len(t.size()) == 0:
                t = t.unsqueeze(0).repeat((batch_size,))
            t_embed = self.time_embedding(t)
        else:
            t_embed = None

        # TODO n_q is not embedded anymore!
        # if n_q is not None:
        # n_q is now already a LongTensor
        # n_q = torch.LongTensor([n_q]).to(x.device)
        # n_q = self.n_q_embedding(n_q)

        # we stop concatenating
        # t_embed = torch.cat([t_embed, n_q], dim=1)
        # t_embed = t_embed +  n_q

        # TODO t_embed here?
        # _, skips_cond = self.ubranch_1(x=c, t_embed=t_embed, skips=None)
        _, skips_cond = self.ubranch_1(x=c, t_embed=None, skips=None)

        y = self.first_conv(x)
        y, skips = self.dbranch(x=y, t_embed=t_embed, skips=reversed(skips_cond))

        y = self.ubranch_2(x=y, skips=reversed(skips), t_embed=t_embed)

        # TODO activation? last convblock
        y = self.last_conv(y)
        return y


class DiffusionUnet(nn.Module):

    def __init__(
        self,
        channels_in,
        channels_cond,
        channels_out,
        time_embed_dim,
        base_dim,
        resblock,  # partial
        downsample,  # partial
        upsample,  # partial
        time_embedding,  # partial
        first_kernel_size: int,  # 3 in noise2music
        last_kernel_size: int,
        downsamplings: List[int],
        channels_mult: List[int],
        num_resblocks: List[int],
        self_attentions: List[bool],
        dropouts: List[float],
        aggregate_mode: str = "sum",
        padding_mode: str = "reflect",
        skip_rescale: bool = True,
    ) -> None:
        super().__init__()
        """
        Last downsampling is dropped
        First upsampling is used to upscale cond
        """
        self.time_embedding = time_embedding(time_embed_dim=time_embed_dim)

        self.first_conv = NormedConv1d(
            dim_in=channels_in,
            dim_out=base_dim,
            kernel_size=first_kernel_size,
            bias=True,
            wn=False,
            gn=False,
            num_groups=channels_in,
            padding_mode=padding_mode,
        )
        self.dbranch = Dbranch(
            channels_in=channels_in,
            base_dim=base_dim,
            time_embed_dim=time_embed_dim,
            resblock=resblock,
            downsample=downsample,
            first_kernel_size=first_kernel_size,
            downsamplings=downsamplings,
            channels_mult=channels_mult,
            num_resblocks=num_resblocks,
            self_attentions=self_attentions,
            dropouts=dropouts,
            skip_rescale=skip_rescale,
            padding_mode=padding_mode,
            return_skip=True,
        )

        self.ubranch = Ubranch(
            channels_in=channels_in,
            base_dim=base_dim,
            time_embed_dim=time_embed_dim,
            resblock=resblock,
            upsample=upsample,
            first_kernel_size=first_kernel_size,
            downsamplings=downsamplings,
            channels_mult=channels_mult,
            num_resblocks=num_resblocks,
            self_attentions=self_attentions,
            dropouts=dropouts,
            skip_rescale=skip_rescale,
            padding_mode=padding_mode,
            return_skip=False,
        )

        assert len(downsamplings) == len(channels_mult) == len(num_resblocks)

        self.last_conv = NormedConv1d(
            dim_in=base_dim,
            dim_out=channels_out,
            kernel_size=last_kernel_size,
            bias=False,
            wn=False,
            gn=False,
            padding_mode=padding_mode,
        )

    def forward(self, x, t, cond=None):
        """
        x: (b c t)
        t: (b)
        """
        assert cond is None
        batch_size = x.size(0)
        if t is not None and self.time_embedding is not None:
            if len(t.size()) == 0:
                t = t.unsqueeze(0).repeat((batch_size,))
            t_embed = self.time_embedding(t)
        else:
            t_embed = None

        y = self.first_conv(x)
        y, skips = self.dbranch(x=y, t_embed=t_embed, skips=None)
        y = self.ubranch(x=y, skips=reversed(skips), t_embed=t_embed)

        # TODO activation? last convblock
        y = self.last_conv(y)
        return y
