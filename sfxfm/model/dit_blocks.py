"""
Basic building blocks used to construct DiTs

Note that in this file the keys on which the blocks operate are defined at init time!
"""

import math
import contextlib
from typing import Dict, List, Optional, Tuple
from einops import rearrange

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from sfxfm.model.dit_types import DiTArgs, DictTensor


def cast_v_context(cast_v) -> contextlib.AbstractContextManager:
    """
    Returns the context for autocasting v in mixed precision.
    """
    if cast_v:
        return torch.autocast(device_type="cuda", enabled=False)
    else:
        return contextlib.nullcontext()


def precompute_freqs_cis(args: DiTArgs, to_audio_fps_multiplier=None) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.
    dim = max(args.qk_rope_head_dim, args.qkv_head_dim)

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.
        to_audio_fps_multiplier (Optional[Float]): if we are computing freqs_cis based on a different fps (e.g. video) compared to audio. to_audio_fps_multiplier =  audio_fps / video_fps
    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = max(args.qk_rope_head_dim, args.qkv_head_dim)
    seqlen = math.ceil(args.max_seq_len / args.patch_size)
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    original_seq_len = math.ceil(args.original_seq_len / args.patch_size)
    if seqlen > original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    if to_audio_fps_multiplier is not None:
        t = t * to_audio_fps_multiplier

    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    # truncate freqs_cis if it has more frequencies than x
    #  also truncate shorter sequences x.shape = [5, 551, 8, 64]
    # TODO: maybe for shorter sequences we should sample a subsequence?
    freqs_cis = freqs_cis[: x.size(1), : x.size(-1)]
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class FourierFeaturesTime(nn.Module):
    """
    FourierFeatures from stable audio
    for time embedding only
    (batch, in_features)
    """

    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class FixedFourierFeaturesTime(nn.Module):
    """
    Fixed Fourier Features from Flux
    Our input is log(sigma)/4 which is roughly in [-1, 1]

    """

    def __init__(
        self, in_features, out_features, max_period=10000, time_factor: float = 1000.0
    ):
        super().__init__()
        assert out_features % 2 == 0
        assert in_features == 1

        half = out_features // 2
        self.register_buffer(
            "freqs",
            torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32)
                / half
            ),
        )
        self.time_factor = time_factor

    def forward(self, input):
        t = self.time_factor * input
        args = t.float() * self.freqs[None]  # type: ignore
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer with modulation.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
    """

    def __init__(
        self, args: DiTArgs, main_key: str = "x", mod_key: Optional[str] = None
    ):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()

        self.norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=1e-6)

        self.w1 = nn.Linear(args.dim, args.inter_dim)
        self.w2 = nn.Linear(args.inter_dim, args.dim)

        self.mlp_act_type = args.mlp_act
        if self.mlp_act_type == "gelu":
            self.act = nn.GELU(approximate="tanh")
        elif self.mlp_act_type == "swiglu":
            self.act = nn.SiLU()
            self.w3 = nn.Linear(args.dim, args.inter_dim)
        else:
            raise NotImplementedError(
                f"MLP activation {self.mlp_act_type} is not implemented. Choose between 'gelu' or 'swiglu'"
            )

        self.use_modulation = mod_key is not None
        if self.use_modulation:
            self.mod_proj = nn.Linear(args.dim, args.dim * 3)
        self.main_key = main_key
        self.mod_key = mod_key

    def forward(self, d: DictTensor) -> DictTensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        x = d[self.main_key]
        residual = x
        x = self.norm(x)
        if self.use_modulation:
            assert self.mod_key is not None
            bias, scale, gate = self.mod_proj(d[self.mod_key].unsqueeze(1)).split(
                d[self.mod_key].size(-1), dim=-1
            )
            x = (1 + scale) * x + bias

        if self.mlp_act_type == "gelu":
            x = self.w2(self.act(self.w1(x)))
        elif self.mlp_act_type == "swiglu":
            x = self.w2(self.act(self.w1(x)) * self.w3(x))
        else:
            raise NotImplementedError(
                f"MLP activation {self.mlp_act_type} is not implemented. Choose between 'gelu' or 'swiglu'"
            )

        if self.use_modulation:
            x = x * gate  # type: ignore
        x = residual + x
        d[self.main_key] = x
        return d


class ModalityAttention(nn.Module):
    """

    SelfAttention for one modality
    with modulation and rotary embeddings.

    """

    def __init__(
        self,
        args: DiTArgs,
        x_key: str,
        mod_key: Optional[str] = None,
        freqs_cis_key: Optional[str] = None,
        mask_key: Optional[str] = None,
        cast_v: bool = False,
    ):
        super().__init__()
        self.x_key = x_key
        self.mod_key = mod_key
        self.freqs_cis_key = freqs_cis_key
        self.mask_key = mask_key

        self.dim = args.dim
        self.n_heads = args.n_heads

        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.head_dim = args.qk_rope_head_dim + args.qk_nope_head_dim
        self.out_proj = nn.Linear(self.head_dim * args.n_heads, self.dim)

        self.use_modulation = self.mod_key is not None

        self.qkv = nn.Linear(self.dim, self.head_dim * args.n_heads * 3)
        # norm used before modulation
        self.mod_norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=1e-6)
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

        if self.use_modulation:
            self.mod_proj = nn.Linear(args.dim, args.dim * 3)

        self.use_rotary = self.freqs_cis_key is not None

        if args.symmetric_attention_init:
            self.qkv.weight.data[: self.head_dim * args.n_heads] = self.qkv.weight.data[
                self.head_dim * args.n_heads : 2 * self.head_dim * args.n_heads
            ].clone()
        self.cast_v = cast_v

    def modulate(self, x, d):
        if self.use_modulation:
            assert self.mod_key is not None
            mod: Tensor = d[self.mod_key]
            bias, scale, gate = self.mod_proj(mod.unsqueeze(1)).split(
                mod.size(-1), dim=-1
            )
            x = (1 + scale) * x + bias

        else:
            bias, scale, gate = None, None, None
        return x, gate

    def precompute(
        self,
        d: DictTensor,
        apply_rope: bool = True,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
    ]:
        """
        Returns all essential elements for the forward pass of the attention layer.

        Does not go through the scaled_dot_product_attention nor update the dictionary.
        """
        d = d.copy()  # for checkpointing
        # extract elements from compute dictionary

        x: Tensor = d[self.x_key]
        bsz, seqlen, _ = x.size()

        # we always normalize
        x = self.mod_norm(x)
        x, gate = self.modulate(x, d)

        # compute q and k
        q, k, v = self.qkv(x).split(self.head_dim * self.n_heads, dim=-1)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)
        if self.use_rotary and apply_rope:
            assert self.freqs_cis_key is not None
            # split between rope and nope
            q_rope, q_nope = q.split(
                [self.qk_rope_head_dim, self.qk_nope_head_dim],
                dim=-1,
            )
            k_rope, k_nope = k.split(
                [self.qk_rope_head_dim, self.qk_nope_head_dim],
                dim=-1,
            )
            q_rope = apply_rotary_emb(q_rope, d[self.freqs_cis_key])
            k_rope = apply_rotary_emb(k_rope, d[self.freqs_cis_key])

            # stack
            q_ropenope = torch.cat([q_rope, q_nope], dim=-1)
            k_ropenope = torch.cat([k_rope, k_nope], dim=-1)
        else:
            # returns unrotated q and k as default
            q_ropenope = q
            k_ropenope = k
        q_ropenope = rearrange(q_ropenope, "b s h d->b h s d", h=self.n_heads, s=seqlen)
        k_ropenope = rearrange(k_ropenope, "b s h d->b h s d", h=self.n_heads, s=seqlen)

        q: Tensor = rearrange(q, "b s h d->b h s d", h=self.n_heads, s=seqlen)
        k: Tensor = rearrange(k, "b s h d->b h s d", h=self.n_heads, s=seqlen)
        v: Tensor = rearrange(v, "b s h d->b h s d", h=self.n_heads, s=seqlen)

        # mask is on kv_keys
        # set it to one if key is None
        mask = (
            d[self.mask_key]
            if self.mask_key
            else torch.ones(bsz, seqlen, device=x.device)
        )
        attn_mask = mask.bool().unsqueeze(1).unsqueeze(1)

        return q, k, v, q_ropenope, k_ropenope, attn_mask, gate

    def forward(
        self,
        d: DictTensor,
    ) -> DictTensor:
        d = d.copy()
        x = d[self.x_key]
        res = x
        q, k, v, q_ropenope, k_ropenope, attn_mask, gate = self.precompute(d)

        if self.use_rotary:
            assert q_ropenope is not None and k_ropenope is not None
            q, k = q_ropenope, k_ropenope

        if attn_mask.bool().all():
            # issue with JVP in mixed precision, v bf16 but q,k float32
            with cast_v_context(self.cast_v):
                x = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v.to(q.dtype) if self.cast_v else v, attn_mask=None
                )

        else:
            # issue with JVP in mixed precision, v bf16 but q,k float32
            with cast_v_context(self.cast_v):
                x = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v.to(q.dtype) if self.cast_v else v, attn_mask=attn_mask
                )

        x = self.out_proj(rearrange(x, "b h s d -> b s (h d)"))

        if self.use_modulation:
            x = x * gate  # type: ignore

        x = x + res

        d[self.x_key] = x
        return d


class ModalityAttentionWrapper(nn.Module):
    def __init__(self, attn: ModalityAttention):
        super().__init__()
        self.attn = attn

    def modulate(self, x, d):
        return self.attn.modulate(x, d)

    def precompute(
        self,
        d: DictTensor,
        apply_rope: bool = True,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
    ]:
        return self.attn.precompute(d, apply_rope=apply_rope)

    def forward(
        self,
        d: DictTensor,
    ) -> DictTensor:
        return self.attn.forward(d)

    @property
    def x_key(self):
        return self.attn.x_key

    @x_key.setter
    def x_key(self, value):
        self.attn.x_key = value

    @property
    def mod_key(self):
        return self.attn.mod_key

    @mod_key.setter
    def mod_key(self, value):
        self.attn.mod_key = value

    @property
    def freqs_cis_key(self):
        return self.attn.freqs_cis_key

    @freqs_cis_key.setter
    def freqs_cis_key(self, value):
        self.attn.freqs_cis_key = value

    @property
    def mask_key(self):
        return self.attn.mask_key

    @mask_key.setter
    def mask_key(self, value):
        self.attn.mask_key = value

    @property
    def dim(self):
        return self.attn.dim

    @property
    def n_heads(self):
        return self.attn.n_heads

    @property
    def qk_rope_head_dim(self):
        return self.attn.qk_rope_head_dim

    @property
    def qk_nope_head_dim(self):
        return self.attn.qk_nope_head_dim

    @property
    def head_dim(self):
        return self.attn.head_dim

    @property
    def use_modulation(self):
        return self.attn.use_modulation

    @property
    def use_rotary(self):
        return self.attn.use_rotary

    @property
    def cast_v(self):
        return self.attn.cast_v

    @cast_v.setter
    def cast_v(self, value):
        self.attn.cast_v = value

    @property
    def out_proj(self):
        return self.attn.out_proj


class ModalityAttentionMFWrapper(ModalityAttentionWrapper):
    """
    Adds extra conditioning to ModalityAttention
    """

    def __init__(self, attn: ModalityAttention, extra_mod_key):
        super().__init__(attn)
        self.extra_mod_key = extra_mod_key
        # Test with bias = False
        # self.extra_mod_proj = nn.Linear(self.attn.dim, self.attn.dim * 3)
        self.extra_mod_proj = nn.Linear(self.attn.dim, self.attn.dim * 3, bias=False)
        self.attn.modulate = self.modulate

        # small init?
        # nn.init.uniform_(self.extra_mod_proj.weight, a=-1e-5, b=1e-5)
        # nn.init.zeros_(self.extra_mod_proj.bias)

    def modulate(self, x, d):
        if self.use_modulation:
            assert self.mod_key is not None
            mod: Tensor = d[self.mod_key]
            extra_mod: Tensor = d[self.extra_mod_key]
            mod_proj = self.attn.mod_proj(mod.unsqueeze(1)) + self.extra_mod_proj(
                extra_mod.unsqueeze(1)
            )

            bias, scale, gate = mod_proj.split(mod.size(-1), dim=-1)
            x = (1 + scale) * x + bias

        else:
            bias, scale, gate = None, None, None
        return x, gate


class BaseModalityBlock(nn.Module):
    """
    Base class for modality blocks containing a self-attention and an MLP,
    with shared logic for forward and key-setting methods.
    """

    def __init__(self, attn: ModalityAttention, ffn: MLP | nn.Identity):
        super().__init__()
        self.attn = attn
        self.ffn = ffn

    def forward(self, d: DictTensor) -> DictTensor:
        d = d.copy()  # for checkpointing
        d = self.attn(d)
        d = self.ffn(d)
        return d

    def set_x_key(self, x_key: str):
        """
        Set the x_key for the attention and MLP layers.
        This is useful for changing the key after initialization.
        """
        if hasattr(self.attn, "x_key"):
            self.attn.x_key = x_key
        if hasattr(self.ffn, "main_key"):
            self.ffn.main_key = x_key

    def set_freqs_cis_key(self, freqs_cis_key: str):
        """
        Set the freqs_cis_key for the attention layer.
        This is useful for changing the key after initialization.
        """
        if hasattr(self.attn, "freqs_cis_key"):
            self.attn.freqs_cis_key = freqs_cis_key

    def set_cast_v(self, cast_v: bool):
        """
        Set whether to cast v in the attention layer.
        This is useful for changing the behavior after initialization.
        """
        self.attn.cast_v = cast_v


class ModalityBlock(BaseModalityBlock):
    """
    A block that contains a self-attention and an MLP
    with modulation and rotary embeddings.

    This is a standard Unimodal Transformer block
    """

    def __init__(
        self,
        args: DiTArgs,
        x_key: str,
        mod_key: Optional[str] = None,
        freqs_cis_key: Optional[str] = None,
        mask_key: Optional[str] = None,
    ):
        attn = ModalityAttention(
            args,
            x_key=x_key,
            mod_key=mod_key,
            freqs_cis_key=freqs_cis_key,
            mask_key=mask_key,
        )
        ffn = MLP(args, main_key=x_key, mod_key=mod_key)
        super().__init__(attn, ffn)


class ModalityBlockFromAttnMLP(BaseModalityBlock):
    """
    A block constructed from provided attention and MLP modules.
    """

    def __init__(
        self,
        attn: ModalityAttention,
        ffn: MLP,
    ):
        super().__init__(attn, ffn)


class MMMAttention(nn.Module):
    """
    General cross attention
    on DictTensors
    with
    rotary embeddings
    QK norm
    layernorm modulation on q & kv
    uses Pytorch scaled_dot_product_attention

    PURELY content-based, only nope
    attention, no rope



    """

    def __init__(
        self,
        modality_attn_dict: Dict[str, ModalityAttention],
        cast_v: bool = False,  # whether to cast v to float32 in mixed precision
    ):
        super().__init__()

        self.modalities = nn.ModuleDict(modality_attn_dict)
        self.cast_v = cast_v

    @property
    def modality_list(self) -> List[ModalityAttention]:
        return list(self.modalities.values())  # type: ignore

    def forward(
        self,
        d: DictTensor,
    ) -> DictTensor:
        d = d.copy()

        x_keys: List[str] = [
            modality.x_key
            for modality in self.modality_list
            if hasattr(modality, "x_key")
        ]  # type: ignore
        x = [d[key] for key in x_keys]
        res = [res for res in x]

        # precompute all qkv
        precomputations = [modality.precompute(d) for modality in self.modality_list]  # type: ignore

        # precomputations is a list of tuples
        # (q, k, v, q_ropenope, k_ropenope, attn_mask, gate)
        # qkv are before any rope
        (
            q_list,
            k_list,
            v_list,
            q_ropenope_list,
            k_ropenope_list,
            attn_mask_list,
            gate_list,
        ) = zip(*precomputations)

        seqlen_list = [q.size(2) for q in q_list]
        # qkv are (b h s d)

        # concat all q, k, v along time
        # We ALWAYS use the rope embeddings
        q = torch.cat(q_ropenope_list, dim=2)
        k = torch.cat(k_ropenope_list, dim=2)
        v = torch.cat(v_list, dim=2)

        # force flash_attention if no mask is present
        # Make sure mask is None or all ones, we NEVER use mask in this case
        for attn_mask in attn_mask_list:
            if attn_mask is not None and not attn_mask.bool().all():
                raise ValueError(
                    "MMMAttention expects all masks to be None or all ones, "
                    "but found a mask with non-ones."
                )

        # issue with JVP in mixed precision, v bf16 but q,k float32
        with cast_v_context(self.cast_v):
            z = torch.nn.functional.scaled_dot_product_attention(
                q, k, v.to(q.dtype) if self.cast_v else v, attn_mask=None
            )

        # get x
        # split z into the different modality_list
        x = [
            z[:, :, sum(seqlen_list[:i]) : sum(seqlen_list[: i + 1])]
            for i in range(len(seqlen_list))
        ]
        assert len(x) == len(self.modality_list)
        assert q.size(2) == sum(seqlen_list)

        # apply out_proj and modulation
        x = [
            modality.out_proj(rearrange(x_i, "b h s d -> b s (h d)"))  # type: ignore
            for modality, x_i in zip(self.modality_list, x)
            if hasattr(modality, "out_proj")
        ]
        x = [
            x_i * gate_i if modality.use_modulation else x_i
            for modality, x_i, gate_i in zip(self.modality_list, x, gate_list)
        ]
        # add residuals
        x = [x_i + res_i for x_i, res_i in zip(x, res)]
        # update the dictionary
        for x_key, x_i in zip(x_keys, x):
            d[x_key] = x_i

        # store attentions:
        # store_attentions = True
        store_attentions = False
        if store_attentions:
            # find_layer_id
            layer_id = 0
            while f"attn_{layer_id}" in d:
                layer_id += 1

            if self.training:
                print("WARNING: recomputing attention scores, this is not efficient")
            # since we have audio then text, the mask starts at attn_mask.sum()
            # only works if all elements have the same batch size

            q = q[:, :, :, :]
            k = k[:, :, :, :]
            d[f"attn_{layer_id}"] = torch.einsum("b h l d, b h s d -> b h l s", q, k)

        return d


class MMMBlock(nn.Module):
    """
    Merges different modality blocks together
    Fuses the attention
    """

    def __init__(
        self,
        layer_id: int,
        modality_block_dict: Dict[str, ModalityBlock],
        cast_v: bool = False,
    ):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        # attn
        modality_attn_dict = {
            key: modality.attn for key, modality in modality_block_dict.items()
        }
        self.attn = MMMAttention(modality_attn_dict=modality_attn_dict, cast_v=cast_v)

        # ffn
        self.ffns = nn.ModuleDict(
            {key: modality.ffn for key, modality in modality_block_dict.items()}
        )

        self.layer_id = layer_id

    def set_cast_v(self, cast_v: bool):
        self.attn.cast_v = cast_v

    def get_modality_block_dict(
        self,
    ) -> Dict[str, ModalityBlock]:
        """
        Returns the modality block dictionary.
        This is useful for accessing the individual modality blocks.
        """
        return {
            key: ModalityBlockFromAttnMLP(
                attn=self.attn.modalities[key],
                ffn=self.ffns[key],
            )
            for key in self.ffns.keys()
        }

    def forward(
        self,
        d: DictTensor,
    ) -> DictTensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        d = d.copy()  # for checkpointing
        d = self.attn.forward(d)

        # apply all ffns
        for key, ffn in self.ffns.items():
            d = ffn.forward(d)

        return d


class MultimodalitySingleStreamBlock(nn.Module):
    """
    A transformer block for multimodal single-stream processing.

    This block combines multiple modalities into a single sequence, applies joint self-attention and a feed-forward MLP in parallel
    x <- x + mlp(x) + attn(x)
    and updates each modality's representation. Unlike other blocks that process modalities separately or use
    cross-attention, this block concatenates all modalities along the sequence dimension and applies shared attention
    and MLP transformations. Key features include:
    - Single-stream attention over concatenated modalities.
    - Combined rotary and non-rotary attention heads.
    - Modulation support for conditioning.
    - Efficient joint processing for tightly coupled multimodal data.
    """

    def __init__(
        self,
        layer_id: int,
        args: DiTArgs,
        x_keys: List[str],
        mod_key: Optional[str] = None,
        freqs_cis_keys: List[Optional[str]] = [],
        mask_key: Optional[str] = None,
        cast_v: bool = False,
    ):
        """
        Initializes the Transformer block.
        Not using sinks for the moment

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()

        self.layer_id = layer_id

        self.x_keys = x_keys
        self.mod_key = mod_key
        self.freqs_cis_keys = freqs_cis_keys
        self.mask_key = mask_key
        assert mask_key is None  # we don't mask anymore

        self.dim = args.dim
        self.inter_dim = args.inter_dim
        self.n_heads = args.n_heads

        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.head_dim = args.qk_rope_head_dim + args.qk_nope_head_dim
        assert self.head_dim * args.n_heads == args.dim, (
            f"Head dim {self.head_dim} * n_heads {args.n_heads} (= {self.head_dim * args.n_heads}) != dim {args.dim}"
        )

        self.use_modulation = self.mod_key is not None

        self.mlp_act_type = args.mlp_act
        if self.mlp_act_type == "gelu":
            self.num_mlp_input_linear = 1
            self.mlp_act = nn.GELU(approximate="tanh")
        elif self.mlp_act_type == "swiglu":
            self.num_mlp_input_linear = 2
            self.mlp_act = nn.SiLU()
        else:
            raise NotImplementedError(
                f"MLP activation {self.mlp_act_type} is not implemented. Choose between 'gelu' or 'swiglu'"
            )

        self.qkv_mlp = nn.Linear(
            self.dim,
            self.head_dim * args.n_heads * 3
            + args.inter_dim * self.num_mlp_input_linear,
        )
        self.out_proj = nn.Linear(
            self.head_dim * args.n_heads + args.inter_dim, self.dim
        )
        # norm used before modulation
        self.mod_norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=1e-6)
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

        if self.use_modulation:
            self.mod_proj = nn.Linear(args.dim, args.dim * 3)
        # always use rope
        self.use_rotary = True
        self.cast_v = cast_v

    def forward(
        self,
        d: DictTensor,
    ) -> DictTensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        d = d.copy()  # for checkpointing
        x = torch.cat([d[x_key] for x_key in self.x_keys], dim=1)
        bsz, seqlen, _ = x.size()

        x = self.mod_norm(x)

        assert self.mod_key is not None
        mod: Tensor = d[self.mod_key]
        bias, scale, gate = self.mod_proj(mod.unsqueeze(1)).chunk(3, dim=-1)
        x = (1 + scale) * x + bias

        qkv, mlp = torch.split(
            self.qkv_mlp(x),
            [3 * self.dim, self.inter_dim * self.num_mlp_input_linear],
            dim=-1,
        )

        q, k, v = rearrange(qkv, "b s (k h d) -> k b s h d", h=self.n_heads, k=3)
        # q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q = self.norm_q(q)
        k = self.norm_k(k)

        # ==== Rope
        freqs_cis = torch.cat(
            [d[freqs_cis_key] for freqs_cis_key in self.freqs_cis_keys], dim=0
        )
        # split between rope and nope
        q_rope, q_nope = q.split(
            [self.qk_rope_head_dim, self.qk_nope_head_dim],
            dim=-1,
        )
        k_rope, k_nope = k.split(
            [self.qk_rope_head_dim, self.qk_nope_head_dim],
            dim=-1,
        )
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        k_rope = apply_rotary_emb(k_rope, freqs_cis)

        # stack
        q = torch.cat([q_rope, q_nope], dim=-1)
        k = torch.cat([k_rope, k_nope], dim=-1)
        q = rearrange(q, "b s h d->b h s d", h=self.n_heads, s=seqlen)
        k = rearrange(k, "b s h d->b h s d", h=self.n_heads, s=seqlen)
        v = rearrange(v, "b s h d->b h s d", h=self.n_heads, s=seqlen)

        # issue with JVP in mixed precision, v bf16 but q,k float32
        with cast_v_context(self.cast_v):
            # Attention
            z = torch.nn.functional.scaled_dot_product_attention(
                q, k, v.to(q.dtype) if self.cast_v else v, attn_mask=None
            )
        z = rearrange(z, "b h s d -> b s (h d)")

        # compute mlp(x) + attn(x); mlp can be swiglu or gelu
        if self.mlp_act_type == "gelu":
            x = self.out_proj(torch.cat([z, self.mlp_act(mlp)], dim=-1))
        elif self.mlp_act_type == "swiglu":
            w1, w2 = mlp.chunk(2, dim=-1)
            x = self.out_proj(torch.cat([z, self.mlp_act(w1) * w2], dim=-1))
        else:
            raise NotImplementedError

        # resplit and gate
        xs = x.split(split_size=[d[x_key].size(1) for x_key in self.x_keys], dim=1)

        for x_key, x in zip(self.x_keys, xs):
            d[x_key] = d[x_key] + gate * x

        return d

    def set_cast_v(self, cast_v: bool):
        """
        Set whether to cast v in the attention layer.
        This is useful for changing the behavior after initialization.
        """
        self.cast_v = cast_v
