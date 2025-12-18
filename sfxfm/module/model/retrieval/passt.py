import logging
import torch
from torch import nn, einsum
import torch.nn.functional as F

import sfxfm.utils.loading

log = logging.getLogger(__name__)

from hear21passt.base import get_model_passt
from hear21passt.models.passt import Attention, Block, PaSST, PatchEmbed
from hear21passt.models.preprocess import AugmentMelSTFT
from torchaudio.compliance.kaldi import (
    mel_scale,
    inverse_mel_scale,
    vtln_warp_mel_freq,
)


def get_mel_banks_compile_compatible(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq,
    high_freq,
    vtln_low: float,
    vtln_high: float,
    vtln_warp_factor: float,
):
    assert num_bins > 3, "Must have at least 3 mel bins"
    assert window_length_padded % 2 == 0
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq

    if high_freq <= 0.0:
        high_freq += nyquist

    assert (
        (0.0 <= low_freq < nyquist)
        and (0.0 < high_freq <= nyquist)
        and (low_freq < high_freq)
    ), "Bad values in options: low-freq {} and high-freq {} vs. nyquist {}".format(
        low_freq, high_freq, nyquist
    )

    # fft-bin width [think of it as Nyquist-freq / half-window-length]
    fft_bin_width = sample_freq / window_length_padded

    mel_low_freq = mel_scale(low_freq)
    mel_high_freq = mel_scale(high_freq)

    # divide by num_bins+1 in next line because of end-effects where the bins
    # spread out to the sides.
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    if vtln_high < 0.0:
        vtln_high += nyquist

    assert vtln_warp_factor == 1.0 or (
        (low_freq < vtln_low < high_freq)
        and (0.0 < vtln_high < high_freq)
        and (vtln_low < vtln_high)
    ), (
        "Bad values in options: vtln-low {} and vtln-high {}, versus "
        "low-freq {} and high-freq {}".format(vtln_low, vtln_high, low_freq, high_freq)
    )

    bin = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + bin * mel_freq_delta  # size(num_bins, 1)
    center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta  # size(num_bins, 1)
    right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta  # size(num_bins, 1)

    if vtln_warp_factor != 1.0:
        left_mel = vtln_warp_mel_freq(
            vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, left_mel
        )
        center_mel = vtln_warp_mel_freq(
            vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, center_mel
        )
        right_mel = vtln_warp_mel_freq(
            vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, right_mel
        )

    center_freqs = inverse_mel_scale(center_mel)  # size (num_bins)
    # size(1, num_fft_bins)
    mel = mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)

    # size (num_bins, num_fft_bins)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)

    if vtln_warp_factor == 1.0:
        # left_mel < center_mel < right_mel so we can min the two slopes and clamp negative values
        bins = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))
    else:
        # warping can move the order of left_mel, center_mel, right_mel anywhere
        bins = torch.zeros_like(up_slope)
        up_idx = torch.gt(mel, left_mel) & torch.le(
            mel, center_mel
        )  # left_mel < mel <= center_mel
        down_idx = torch.gt(mel, center_mel) & torch.lt(
            mel, right_mel
        )  # center_mel < mel < right_mel
        bins[up_idx] = up_slope[up_idx]
        bins[down_idx] = down_slope[down_idx]

    return bins, center_freqs


class MaskedAttention(nn.Module):
    def __init__(self, old_attention_module):
        super().__init__()
        self.__dict__ = old_attention_module.__dict__  # .copy()

    def forward(self, x):
        x, attention_mask = x
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        i, j, dtype = *dots.shape[-2:], dots.dtype

        if attention_mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            attention_mask = attention_mask.unsqueeze(1)
            dots = dots.masked_fill(~attention_mask, mask_value)

        attn = F.softmax(dots, dim=-1, dtype=torch.float32)
        attn = attn.type(dtype)
        attn = self.attn_drop(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        x = out.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MaskedBlock(nn.Module):
    def __init__(self, old_block_module):
        super().__init__()
        self.__dict__ = old_block_module.__dict__  # .copy()

    def forward(self, x):
        x, attention_mask = x
        x = x + self.drop_path(self.attn((self.norm1(x), attention_mask)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return (x, attention_mask)


class MaskedPatchEmbed(nn.Module):
    def __init__(self, old_patch_module):
        super().__init__()
        self.__dict__ = old_patch_module.__dict__  # .copy()

    def forward(self, x):
        # to do maybe replace weights
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class MaskedPaSST(nn.Module):
    def __init__(self, old_passt_module):
        super().__init__()
        self.__dict__ = old_passt_module.__dict__  # .copy()

    def interpolate_attention(self, attention_mask, size):
        return F.interpolate(
            attention_mask.float(),
            size,
            mode="linear",
        ).bool()

    def forward_features(self, x, attention_mask):
        x = self.patch_embed(x)  # [b, e, f, t]
        if attention_mask is not None:
            attention_mask = (
                self.interpolate_attention(attention_mask, x.size(-1))
                .unsqueeze(2)
                .expand(x.size(0), 1, x.size(2), x.size(3))
            )
        B_dim, E_dim, F_dim, T_dim = x.shape  # slow
        time_new_pos_embed = self.time_new_pos_embed
        if x.shape[-1] != time_new_pos_embed.shape[-1]:
            time_new_pos_embed = time_new_pos_embed[:, :, :, : x.shape[-1]]
        x = x + time_new_pos_embed
        x = x + self.freq_new_pos_embed

        # Structured Patchout https://arxiv.org/abs/2110.05069 Section 2.2
        if self.training and self.s_patchout_t:
            random_indices = (
                torch.randperm(T_dim)[: T_dim - self.s_patchout_t].sort().values
            )
            x = x[:, :, :, random_indices]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :, random_indices]
        if self.training and self.s_patchout_f:
            random_indices = (
                torch.randperm(F_dim)[: F_dim - self.s_patchout_f].sort().values
            )
            x = x[:, :, random_indices, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, random_indices, :]
        ###
        # Flatten the sequence
        x = x.flatten(2).transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attention_mask.flatten(2).transpose(1, 2)
        # Unstructured Patchout
        if self.training and self.u_patchout:
            seq_len = x.shape[1]
            random_indices = (
                torch.randperm(seq_len)[: seq_len - self.u_patchout].sort().values
            )
            x = x[:, random_indices, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, random_indices, :]
        ####
        # Add the C/D tokens
        cls_tokens = self.cls_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, :1, :]
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)
            if attention_mask is not None:
                attention_pad = torch.ones(
                    (attention_mask.shape[0], 1, 1),
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (
                        attention_pad,
                        attention_mask,
                    ),
                    dim=1,
                )
        else:
            dist_token = (
                self.dist_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, 1:, :]
            )
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
            if attention_mask is not None:
                attention_pad = torch.ones(
                    (attention_mask.shape[0], 2, 1),
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (
                        attention_pad,
                        attention_mask,
                    ),
                    dim=1,
                )

        x = self.pos_drop(x)
        x, _ = self.blocks((x, attention_mask))
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, attention_mask=None):
        x = self.forward_features(x, attention_mask)

        if self.head_dist is not None:
            features = (x[0] + x[1]) / 2
            x = self.head(features)
            return x, features
        else:
            features = x
            x = self.head(x)
        return x, features


class AugmentMelSTFTCompile(nn.Module):
    def __init__(self, old_mel_module):
        super().__init__()
        self.__dict__ = old_mel_module.__dict__  # .copy()

    def forward(self, x):
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(
            1
        )
        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=False,
        )
        x = (x**2).sum(dim=-1)  # power mag
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,))
        fmax = (
            self.fmax
            + self.fmax_aug_range // 2
            - torch.randint(self.fmax_aug_range, (1,))
        )
        # don't augment eval data
        if not self.training:
            fmin = torch.tensor(self.fmin)
            fmax = torch.tensor(self.fmax)
        mel_basis, _ = get_mel_banks_compile_compatible(
            self.n_mels,
            self.n_fft,
            self.sr,
            fmin,
            fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = torch.as_tensor(
            torch.nn.functional.pad(mel_basis, (0, 1), mode="constant", value=0),
            device=x.device,
        )
        with torch.amp.autocast("cuda", enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.0  # fast normalization

        return melspec


class Wrapper(torch.nn.Module):
    def __init__(self, model, mask_attention):
        super().__init__()
        self.model = model
        self.mel = AugmentMelSTFT(
            n_mels=128,
            sr=32000,
            win_length=800,
            hopsize=320,
            n_fft=1024,
            freqm=48,
            timem=192,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000,
        )
        self.mask_attention = mask_attention

    def interpolate_attention(self, attention_mask):
        attention_mask = F.interpolate(
            attention_mask.unsqueeze(1).float(),
            attention_mask.size(1) // self.mel.hopsize,
            mode="linear",
        ).bool()
        return attention_mask

    def forward(self, x, padding_mask, **kwargs):
        attention_mask = ~padding_mask if self.mask_attention else None
        # with torch.no_grad():
        mel = self.mel(x)
        if attention_mask is not None:
            # downsample padding mask
            attention_mask = self.interpolate_attention(attention_mask)
        return self.model(mel[:, None], attention_mask=attention_mask)[-1]


def create_passt_model(audio_config):
    assert audio_config.sample_rate == 32000, (
        "Sample rate must be 32kHz for passt models"
    )
    s_patchout_t = audio_config.s_patchout_t
    s_patchout_f = audio_config.s_patchout_f
    pretrained = not sfxfm.utils.loading.lazy_loading_enabled
    # get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
    if "passt_s" == audio_config.name:
        print("#### Using PaSST-S ap486 model with no overlap ####\n")
        model = get_model_passt(
            "passt_s_kd_p16_128_ap486",
            input_tdim=998,
            fstride=10,
            tstride=10,
            s_patchout_t=s_patchout_t,
            s_patchout_f=s_patchout_f,
            pretrained=pretrained,
        )
    elif "passt_20" == audio_config.name:
        print("#### Using PaSST-S  train on `20` seconds ####\n")
        model = get_model_passt(
            arch="passt_20sec",
            input_tdim=2000,
            fstride=10,
            tstride=10,
            s_patchout_t=s_patchout_t,
            s_patchout_f=s_patchout_f,
            pretrained=pretrained,
        )
    elif "passt_l" == audio_config.name:
        print("#### Using PaSST-L  ####\n")
        model = get_model_passt(
            arch="passt_l_kd_p16_128_ap47",
            input_tdim=998,
            fstride=10,
            tstride=10,
            s_patchout_t=s_patchout_t,
            s_patchout_f=s_patchout_f,
            pretrained=pretrained,
        )
    elif "passt_no" == audio_config.name:
        print("#### Using PaSST model with no overlap ####")
        model = get_model_passt(
            "passt_s_p16_s16_128_ap468",
            input_tdim=1000,
            fstride=16,
            tstride=16,
            s_patchout_t=s_patchout_t,
            s_patchout_f=s_patchout_f,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unknown audio model {audio_config.name}")

    model = Wrapper(model, audio_config.get("mask_attention", False))
    model = patch_passt_model_attention(model)
    model = patch_melstft_compilable(model)

    return model, 768


def patch_passt_model_attention(model):
    """
    Monkey-patching attention masking into pre-trained PaSST model.
    Removes first_RUN prints, due to limited access to global variable.
    """

    def replace_module(module, target_module, new_module):
        for child_name, child_module in module.named_children():
            if isinstance(child_module, target_module):
                setattr(module, child_name, new_module(child_module))
            else:
                replace_module(child_module, target_module, new_module)

    log.info(
        "Patching attention masking into pre-trained PaSST model. Replacing PaSST, PatchEmbed, Block and Attention modules with patched variant."
    )
    replace_module(model, PaSST, MaskedPaSST)
    replace_module(model, PatchEmbed, MaskedPatchEmbed)
    replace_module(model, Block, MaskedBlock)
    replace_module(model, Attention, MaskedAttention)

    return model


def patch_melstft_compilable(model):
    """
    Monkey-patching mel spectrogram computation to make it compatible with torch.compile.
    Original torchaudio kaldi implementation uses a scalar version to compute mel_scale.
    This variant is not compatible with torch jit compilation.
    """

    def replace_module(module, target_module, new_module):
        for child_name, child_module in module.named_children():
            if isinstance(child_module, target_module):
                setattr(module, child_name, new_module(child_module))
            else:
                replace_module(child_module, target_module, new_module)

    log.info(
        "Patching mel spectrogram computation to be compatible with torch.compile."
    )
    replace_module(model, AugmentMelSTFT, AugmentMelSTFTCompile)
    return model
