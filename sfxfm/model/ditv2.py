import logging
from collections.abc import Callable
from typing import Dict, Optional
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F

from sfxfm.model.dit_pipeline import DiTPipeline, DictTensor
from sfxfm.model.dit_types import DiTArgs, DiTv2Args, MMDiTArgs
from sfxfm.model.dit_blocks import (
    FixedFourierFeaturesTime,
    FourierFeaturesTime,
    RMSNorm,
    apply_rotary_emb,
    precompute_freqs_cis,
)

log = logging.getLogger(__name__)


class CrossAttention(nn.Module):
    """
    General cross attention
    on DictTensors
    with
    rotary embeddings
    QK norm
    layernorm modulation on q & kv
    uses Pytorch scaled_dot_product_attention
    """

    def __init__(
        self,
        args: DiTArgs,
        use_modulation=True,
    ):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.qk_rope_head_dim
        self.q = nn.Linear(self.dim, args.qk_rope_head_dim * args.n_heads)
        self.kv = nn.Linear(self.dim, args.qk_rope_head_dim * args.n_heads * 2)
        self.out_proj = nn.Linear(args.qk_rope_head_dim * args.n_heads, self.dim)

        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

        self.mod_norm_q = nn.LayerNorm(args.dim, elementwise_affine=False, eps=1e-6)
        self.mod_norm_kv = nn.LayerNorm(args.dim, elementwise_affine=False, eps=1e-6)
        self.use_modulation = use_modulation
        if use_modulation:
            self.mod_proj_q = nn.Linear(args.dim, args.dim * 3)
            self.mod_proj_kv = nn.Linear(args.dim, args.dim * 2)

    def forward(
        self,
        d: DictTensor,
        freqs_cis_q: Optional[torch.Tensor],
        freqs_cis_kv: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        q_key: str = "x",
        kv_key: str = "x",
        mod_key: Optional[str] = None,
        use_rotary: bool = True,
    ) -> DictTensor:
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen_q, _ = d[q_key].size()
        bsz, seqlen_kv, _ = d[kv_key].size()

        x = d[q_key]
        y = d[kv_key]
        x_res = x
        # always norm
        x = self.mod_norm_q(x)
        y = self.mod_norm_kv(y)
        # modulate
        if mod_key is not None:
            assert self.use_modulation
            bias_q, scale_q, gate_q = self.mod_proj_q(d[mod_key].unsqueeze(1)).split(
                d[mod_key].size(-1), dim=-1
            )
            x = (1 + scale_q) * x + bias_q

            bias_kv, scale_kv = self.mod_proj_kv(d[mod_key].unsqueeze(1)).split(
                d[mod_key].size(-1), dim=-1
            )
            y = (1 + scale_kv) * y + bias_kv

        q = self.q(x)
        k, v = self.kv(y).split(self.head_dim * self.n_heads, dim=-1)

        q = q.view(bsz, seqlen_q, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen_kv, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen_kv, self.n_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if use_rotary:
            assert freqs_cis_q is not None
            assert freqs_cis_kv is not None
            q = apply_rotary_emb(q, freqs_cis_q)
            k = apply_rotary_emb(k, freqs_cis_kv)

        q = rearrange(q, "b s h d->b h s d", h=self.n_heads, s=seqlen_q)
        k = rearrange(k, "b s h d->b h s d", h=self.n_heads, s=seqlen_kv)
        v = rearrange(v, "b s h d->b h s d", h=self.n_heads, s=seqlen_kv)
        # should be broadcastable
        # mask is on kv_keys
        attn_mask = mask.bool().unsqueeze(1).unsqueeze(1) if mask is not None else None

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        x = self.out_proj(rearrange(x, "b h s d -> b s (h d)"))

        if mod_key is not None:
            x = x * gate_q  # type: ignore

        x = x + x_res

        d[q_key] = x
        return d


class SelfAttention(nn.Module):
    """
    SelfAttention
    on DictTensors
    with
    rotary embeddings
    QK norm
    layernorm modulation on qkv
    uses Pytorch scaled_dot_product_attention

    head dimension is defined by args.qkv_head_dim
    """

    def __init__(self, args: DiTArgs, use_modulation=True):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.qkv_head_dim

        self.qkv = nn.Linear(self.dim, self.head_dim * args.n_heads * 3)
        self.out_proj = nn.Linear(self.head_dim * args.n_heads, self.dim)

        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

        self.mod_norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=1e-6)
        self.use_modulation = use_modulation
        if use_modulation:
            self.mod_proj = nn.Linear(args.dim, args.dim * 3)

    def forward(
        self,
        d: DictTensor,
        freqs_cis: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        qkv_key: str = "x",
        mod_key: Optional[str] = None,
        use_rotary: bool = True,
    ) -> DictTensor:
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = d[qkv_key].size()

        x = d[qkv_key]
        x_res = x

        # always norm
        x = self.mod_norm(x)

        # modulate
        if mod_key is not None:
            assert self.use_modulation
            bias, scale, gate = self.mod_proj(d[mod_key].unsqueeze(1)).split(
                d[mod_key].size(-1), dim=-1
            )
            x = (1 + scale) * x + bias

        q, k, v = self.qkv(x).split(self.head_dim * self.n_heads, dim=-1)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if use_rotary:
            assert freqs_cis is not None
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        q = rearrange(q, "b s h d->b h s d", h=self.n_heads, s=seqlen)
        k = rearrange(k, "b s h d->b h s d", h=self.n_heads, s=seqlen)
        v = rearrange(v, "b s h d->b h s d", h=self.n_heads, s=seqlen)
        # should be broadcastable
        # mask is on kv_keys
        attn_mask = mask.bool().unsqueeze(1).unsqueeze(1) if mask is not None else None

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        x = self.out_proj(rearrange(x, "b h s d -> b s (h d)"))

        if mod_key is not None:
            x = x * gate  # type: ignore

        x = x + x_res

        d[qkv_key] = x
        return d


class MultiModalCrossAttention(nn.Module):
    """
    General cross attention
    on DictTensors
    with
    rotary embeddings
    QK norm
    layernorm modulation on q & kv
    uses Pytorch scaled_dot_product_attention

    head_dim is TWICE the size of qk_rope_head_dim
    """

    def __init__(self, args: MMDiTArgs, use_modulation=True):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.head_dim = args.qk_rope_head_dim + args.qk_nope_head_dim

        self.qkv_x = nn.Linear(self.dim, self.head_dim * args.n_heads * 3)
        self.qkv_y = nn.Linear(self.dim, self.head_dim * args.n_heads * 3)

        self.out_proj = nn.Linear(self.head_dim * args.n_heads, self.dim)

        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

        self.mod_norm_x = nn.LayerNorm(args.dim, elementwise_affine=False, eps=1e-6)
        self.mod_norm_y = nn.LayerNorm(args.dim, elementwise_affine=False, eps=1e-6)
        self.use_modulation = use_modulation
        if use_modulation:
            self.mod_proj_q = nn.Linear(args.dim, args.dim * 3)
            self.mod_proj_kv = nn.Linear(args.dim, args.dim * 3)

    def forward(
        self,
        d: DictTensor,
        freqs_cis_x: Optional[torch.Tensor],
        freqs_cis_y: Optional[torch.Tensor],
        mask_x: Optional[torch.Tensor],
        mask_y: Optional[torch.Tensor],
        x_key: str = "x",
        y_key: str = "x",
        mod_key: Optional[str] = None,
        use_rotary: bool = True,
    ) -> DictTensor:
        """
        Args:


        Returns:

        """
        d = d.copy()
        bsz, seqlen_x, _ = d[x_key].size()
        bsz, seqlen_y, _ = d[y_key].size()
        seqlen_z = seqlen_x + seqlen_y

        x = d[x_key]
        y = d[y_key]
        # store residuals
        x_res, y_res = x, y

        # always norm
        x = self.mod_norm_x(x)
        y = self.mod_norm_y(y)
        # modulate
        if mod_key is not None:
            assert self.use_modulation
            bias_x, scale_x, gate_x = self.mod_proj_q(d[mod_key].unsqueeze(1)).split(
                d[mod_key].size(-1), dim=-1
            )
            x = (1 + scale_x) * x + bias_x

            bias_y, scale_y, gate_y = self.mod_proj_kv(d[mod_key].unsqueeze(1)).split(
                d[mod_key].size(-1), dim=-1
            )
            y = (1 + scale_y) * y + bias_y

        # compute q and k
        q_x, k_x, v_x = self.qkv_x(x).split(self.head_dim * self.n_heads, dim=-1)

        q_y, k_y, v_y = self.qkv_y(y).split(self.head_dim * self.n_heads, dim=-1)

        # Concat x and y along time
        q = torch.cat([q_x, q_y], dim=1)
        k = torch.cat([k_x, k_y], dim=1)
        v = torch.cat([v_x, v_y], dim=1)

        q = q.view(bsz, seqlen_z, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen_z, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen_z, self.n_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if use_rotary:
            # Fuse embeddings
            # freqs embeddings are (length, head_dim // 2)
            assert freqs_cis_x is not None
            assert freqs_cis_y is not None
            # split between rope and nope
            q_rope, q_nope = q.split(
                [self.qk_rope_head_dim, self.qk_nope_head_dim], dim=-1
            )
            k_rope, k_nope = k.split(
                [self.qk_rope_head_dim, self.qk_nope_head_dim], dim=-1
            )

            # extend freqs with zeros
            freqs_cis_xx = torch.cat(
                [freqs_cis_x, torch.zeros_like(freqs_cis_y)], dim=0
            )
            freqs_cis_yy = torch.cat(
                [torch.zeros_like(freqs_cis_x), freqs_cis_y], dim=0
            )
            q_rope_x = apply_rotary_emb(q_rope, freqs_cis_xx)
            q_rope_y = apply_rotary_emb(q_rope, freqs_cis_yy)
            k_rope_x = apply_rotary_emb(k_rope, freqs_cis_xx)
            k_rope_y = apply_rotary_emb(k_rope, freqs_cis_yy)

            # stack
            q = torch.cat([q_rope_x, q_rope_y, q_nope], dim=-1)
            k = torch.cat([k_rope_x, k_rope_y, k_nope], dim=-1)

        q = rearrange(q, "b s h d->b h s d", h=self.n_heads, s=seqlen_z)
        k = rearrange(k, "b s h d->b h s d", h=self.n_heads, s=seqlen_z)
        v = rearrange(v, "b s h d->b h s d", h=self.n_heads, s=seqlen_z)

        # should be broadcastable
        # mask is on kv_keys
        if mask_x is None:
            mask_x = torch.ones(bsz, seqlen_x, device=x.device)
        if mask_y is None:
            mask_y = torch.ones(bsz, seqlen_y, device=y.device)
        mask = torch.cat([mask_x, mask_y], dim=1)

        attn_mask = mask.bool().unsqueeze(1).unsqueeze(1) if mask is not None else None

        z = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        z = self.out_proj(rearrange(z, "b h s d -> b s (h d)"))

        # get x and y back
        x, y = z[:, :seqlen_x], z[:, seqlen_x:]

        if mod_key is not None:
            x = x * gate_x  # type: ignore
            y = y * gate_y  # type: ignore

        x, y = x + x_res, y + y_res

        d[x_key], d[y_key] = x, y
        return d


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer with modulation.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
    """

    def __init__(self, args: DiTArgs, use_modulation=True):
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

        self.gelu = nn.GELU(approximate="tanh")

        self.use_modulation = use_modulation
        if use_modulation:
            self.mod_proj = nn.Linear(args.dim, args.dim * 3)

    def forward(
        self, d: DictTensor, mod_key: Optional[str] = None, main_key: str = "x"
    ) -> DictTensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        x = d[main_key]
        residual = x
        x = self.norm(x)
        if self.use_modulation:
            assert mod_key is not None
            bias, scale, gate = self.mod_proj(d[mod_key].unsqueeze(1)).split(
                d[mod_key].size(-1), dim=-1
            )
            x = (1 + scale) * x + bias

        x = self.w2(F.gelu(self.w1(x)))

        if self.use_modulation:
            x = x * gate  # type: ignore
        x = residual + x
        d[main_key] = x
        return d


class On(nn.Module):
    """
    Used to transform a
    Tensor -> Tensor module
    into a
    DictTensor -> DictTensor module
    """

    def __init__(self, module: Callable[[torch.Tensor], torch.Tensor], key):
        super().__init__()
        self.module = module
        self.key = key

    def forward(self, d: DictTensor) -> DictTensor:
        d[self.key] = self.module(d[self.key])
        return d


class MMBlock(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, layer_id: int, args: MMDiTArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MultiModalCrossAttention(args)
        self.ffn = MLP(args)

        self.layer_id = layer_id

    def forward(
        self,
        d: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
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
        d = self.attn.forward(
            d,
            x_key="x",
            y_key="description",
            mod_key="t",
            mask_x=None,
            mask_y=d["description_mask"],
            use_rotary=True,
            freqs_cis_x=d["freqs_cis"],
            freqs_cis_y=d["freqs_cis_description"],
        )
        d = self.ffn.forward(d, main_key="x", mod_key="t")
        return d


class UMBlock(nn.Module):
    """ """

    def __init__(
        self,
        layer_id: int,
        args: DiTArgs,
        qkv_key: str = "x",
        mod_key: Optional[str] = "t",
        freqs_cis_key: Optional[str] = "freqs_cis",
    ):
        """
        Initializes the Transformer block.

        Default behaviour is
        Modulated SelfAttention with rope
        followed by
        Modulated FFN

        with main key 'x' and modulation key 't'

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
            qkv_key: main key, used for self attention and ffn
            mod_key (optional str): modulation key used to modulate layer norms.
            no modulation if None
            freqs_cis_key: key for the rotary embeddings, if None,


        """
        super().__init__()
        use_modulation = mod_key is not None
        self.attn = SelfAttention(args, use_modulation=use_modulation)
        self.ffn = MLP(args, use_modulation=use_modulation)

        self.layer_id = layer_id

        self.qkv_key = qkv_key
        self.mod_key = mod_key
        self.freqs_cis_key = freqs_cis_key

    def forward(
        self,
        d: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
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

        use_rotary = self.freqs_cis_key is not None
        d = self.attn.forward(
            d,
            qkv_key=self.qkv_key,
            mod_key=self.mod_key,
            mask=None,
            use_rotary=use_rotary,
            freqs_cis=None if self.freqs_cis_key is None else d[self.freqs_cis_key],
        )
        d = self.ffn.forward(d, main_key="x", mod_key="t")
        return d


class xBlock(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, layer_id: int, args: DiTArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = SelfAttention(args)
        self.ffn = MLP(args)
        self.cross_attn = CrossAttention(args)

        self.layer_id = layer_id

    def forward(
        self,
        d: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
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

        d = self.attn.forward(
            d,
            qkv_key="x",
            mod_key="t",
            mask=None,
            use_rotary=True,
            freqs_cis=d["freqs_cis"],
        )
        if d.get("description") is not None:
            d = self.cross_attn(
                d,
                q_key="x",
                kv_key="description",
                mod_key="t",
                mask=d["description_mask"],
                use_rotary=False,
                freqs_cis_q=None,
                freqs_cis_kv=None,
            )
        d = self.ffn(d, main_key="x", mod_key="t")
        return d


class InputProcessing(nn.Module):
    """
    sends to DictTensor
    """

    def __init__(self, args: DiTArgs):
        super().__init__()
        cond_token_dim = args.cond_token_dim

        input_padding_size = (-args.max_seq_len) % args.patch_size
        self.patch_size = args.patch_size
        if input_padding_size > 0:
            self.input_padding = nn.Parameter(
                torch.randn(args.io_channels, input_padding_size), requires_grad=True
            )

        # === Timestep
        self.timestep_features = (
            FixedFourierFeaturesTime(1, args.timestep_features_dim)
            if args.fixed_timestep_features
            else FourierFeaturesTime(1, args.timestep_features_dim)
        )
        # Try embedding with simple MLP
        # self.timestep_features = nn.Sequential(
        #     nn.Linear(1, args.inter_dim, bias=True),
        #     nn.SiLU(),
        #     nn.Linear(args.inter_dim, args.timestep_features_dim, bias=True),
        #     nn.SiLU(),
        # )

        # For TokenVerse compatibility we split to_timestep_embed  into
        # to_timestep_embed and post_timestep_embed. This leaves the t embedding
        # as is and allows TokenVerse to use the Pre-SiLU tensor
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(
                args.timestep_features_dim,
                args.inter_dim,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(args.inter_dim, args.dim, bias=True),
            # last SiLU is included in post_timestep_embed
            # to_timestep_embed results in the TokenVerse M space
            # nn.SiLU(),
        )

        self.post_timestep_embed = nn.SiLU()

        # === x
        self.project_in = nn.Linear(
            args.io_channels * args.patch_size, args.dim, bias=True
        )

        # === condition
        self.to_cond_embed = nn.Sequential(
            nn.Linear(cond_token_dim, args.inter_dim, bias=True),
            nn.SiLU(),
            nn.Linear(args.inter_dim, args.dim, bias=True),
        )

        # === memory tokens

        # Only one memory token; like relative positional embeddings:
        self.n_memory_tokens_rope = args.n_memory_tokens_rope
        self.memory_tokens_rope = nn.Parameter(
            torch.randn(1, 1, args.dim), requires_grad=True
        )
        # === precompute rope embeddings
        freqs_cis = precompute_freqs_cis(args)
        # concat downsampled frequencies of memory tokens rope to freqs_cis
        if args.n_memory_tokens_rope > 0:
            downsampling_factor = freqs_cis.size(0) // args.n_memory_tokens_rope
            freqs_cis = torch.cat(
                [freqs_cis[downsampling_factor // 2 :: downsampling_factor], freqs_cis],
                dim=0,
            )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # === Memory tokens for description

        self.n_memory_tokens_description = args.n_memory_tokens_description
        self.memory_tokens_description = nn.Parameter(
            torch.randn(1, args.n_memory_tokens_description, args.dim),
            requires_grad=True,
        )

        if args.n_multimodal_layers > 0:
            # normal freqs_cis
            # self.register_buffer(
            #     "freqs_cis_description",
            #     precompute_freqs_cis(
            #         args.model_copy(
            #             update={
            #                 "max_seq_len": args.max_description_length
            #                 + args.n_memory_tokens_description,
            #             }
            #         )
            #     ),
            #     persistent=False,
            # )
            # constant freqs_cis for text
            self.register_buffer(
                "freqs_cis_description",
                precompute_freqs_cis(
                    args.model_copy(
                        update={
                            "max_seq_len": args.max_description_length
                            + args.n_memory_tokens_description,
                        }
                    )
                )[:1, :].expand(
                    args.max_description_length + args.n_memory_tokens_description, -1
                ),
                persistent=False,
            )
        else:
            self.register_buffer("freqs_cis_description", None, persistent=False)

        # === Estimation of logvar(t)
        self.estimate_logvar = args.estimate_logvar
        if args.estimate_logvar:
            self.timestep_logvar = FourierFeaturesTime(1, args.timestep_features_dim)
            self.to_logvar = nn.Sequential(
                nn.Linear(args.timestep_features_dim, 128, bias=True),
                nn.SiLU(),
                nn.Linear(128, 1, bias=True),
            )

        self.no_description_mask = args.no_description_mask
        if self.no_description_mask:
            # if no description mask, we use replace the masked tokens with a learnable parameter
            self.description_pad = nn.Parameter(
                torch.randn(args.max_description_length, args.cond_token_dim),
            )

    def pad_description(
        self, description: torch.Tensor, description_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pads the description with a learnable parameter if no_description_mask is True.
        """
        if self.no_description_mask:
            # replace masked tokens with a learnable parameter
            description = torch.where(
                description_mask.unsqueeze(-1).bool(),
                description,
                self.description_pad[None, :],
            )
        return description

    def embed_x(self, x):
        """
        Embeds the input tensor x by rearranging it into patches and projecting it.
        If input_padding is defined, pads the input with learnable parameters.
        """
        if hasattr(self, "input_padding"):
            # pad the input with learnable parameters
            x = torch.cat(
                [
                    self.input_padding[None, :, :].expand(x.size(0), -1, -1),
                    x,
                ],
                dim=2,
            )
        # rearrange x into patches
        x = rearrange(x, "b c (t p) -> b t (p c)", p=self.patch_size)
        return self.project_in(x)

    # Copies signature from forward method of dit.DiffusionTransformer
    def forward(
        self,
        x: torch.Tensor,
        t,
        cond: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> DictTensor:
        batch_size = x.size(0)
        m_plus = self.to_timestep_embed(self.timestep_features(t[:, None]))  # (b, c)
        d = dict(
            # memory tokens + embed(x)
            x=torch.cat(
                [
                    self.memory_tokens_rope.expand(
                        batch_size, self.n_memory_tokens_rope, -1
                    ),
                    self.embed_x(x),
                ],
                dim=1,
            ),
            x_mask=mask,
            m_plus=m_plus,
            t=self.post_timestep_embed(m_plus),
            description=(
                torch.cat(
                    [
                        self.memory_tokens_description.expand(x.size(0), -1, -1),
                        (
                            self.to_cond_embed(cond["cross_attn_cond"])
                            if not self.no_description_mask
                            else self.to_cond_embed(
                                self.pad_description(
                                    cond["cross_attn_cond"],
                                    cond["cross_attn_cond_mask"],
                                )
                            )
                        ),
                    ],
                    dim=1,
                )
                if "cross_attn_cond" in cond
                else None
            ),
            description_tids=cond["text_tids"] if "text_tids" in cond else None,
            description_tembs=cond["text_tembs"] if "text_tembs" in cond else None,
            description_mask=(
                torch.cat(
                    [
                        torch.ones(batch_size, self.n_memory_tokens_description).to(
                            x.device
                        ),
                        (
                            torch.ones_like(cond["cross_attn_cond_mask"])
                            if self.no_description_mask
                            else cond["cross_attn_cond_mask"]
                        ),
                    ],
                    dim=1,
                )
                if "cross_attn_cond" in cond
                else None
            ),
            freqs_cis=self.freqs_cis,
            freqs_cis_description=self.freqs_cis_description,
            logvar=(
                self.to_logvar(self.timestep_logvar(t[:, None]))[:, 0]  # (b,)
                if self.estimate_logvar
                else None
            ),
        )

        # DictTensor is not supposed to contain None values
        return d  # type: ignore


class PostProcessing(nn.Module):
    """
    Simple linear preceded by AdaLN if adaln_last_layer
    """

    def __init__(self, args: DiTArgs):
        super().__init__()
        self.patch_size = args.patch_size

        self.adaln_last_layer = args.adaln_last_layer
        self.adaln_last_layer_nomod = args.adaln_last_layer_nomod
        if self.adaln_last_layer_nomod:
            print("\nadaln_last_layer_nomod is True\n")
        if self.adaln_last_layer:
            self.norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=1e-6)
            if not self.adaln_last_layer_nomod:
                self.mod_proj = nn.Linear(args.dim, args.dim * 2, bias=True)

        self.linear = nn.Linear(args.dim, args.io_channels * self.patch_size, bias=True)
        self.n_memory_tokens_rope = args.n_memory_tokens_rope
        self.input_padding_size = (-args.max_seq_len) % args.patch_size

    def forward(self, d: DictTensor) -> DictTensor:
        main_key = "x"
        mod_key = "t"
        x = d[main_key]
        # Strip memory tokens rope
        x = x[:, self.n_memory_tokens_rope :]

        # Last layer
        if self.adaln_last_layer:
            if self.adaln_last_layer_nomod:
                x = self.norm(x)
            else:
                bias, scale = self.mod_proj(d[mod_key].unsqueeze(1)).chunk(2, dim=-1)
                x = (1 + scale) * self.norm(x) + bias

        x = self.linear(x)
        x = rearrange(x, "b t (p c) -> b c (t p)", p=self.patch_size)
        # and eventually remove padding tokens after depatching
        d[main_key] = x[:, :, self.input_padding_size :]
        return d


class DiTv2(DiTPipeline):
    """
    Base class to implement different Diffusion Transformers
    """

    def __init__(self, args: DiTv2Args, **kwargs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
            kwargs: Overrides to args hanled by BaseComponent
        """

        preprocessing = InputProcessing(args)
        postprocessing = PostProcessing(args)

        # add all blocks with checkpoints every args.checkpoint_every
        layers = torch.nn.Sequential()
        for layer_id in range(args.n_layers):
            layers.append(xBlock(layer_id, args))

        super().__init__(
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            layers=layers,
            non_checkpoint_layers=args.non_checkpoint_layers,
            mask_out_before=args.mask_out_before,
        )


class MMDiT(DiTPipeline):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    def __init__(self, args: MMDiTArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
            kwargs: Overrides to args handled by BaseComponent
        """
        preprocessing = InputProcessing(args)
        postprocessing = PostProcessing(args)

        # add all blocks with checkpoints every args.checkpoint_every
        layers = torch.nn.Sequential()
        for layer_id in range(args.n_layers):
            if layer_id < args.n_multimodal_layers:
                layers.append(MMBlock(layer_id, args))
            else:
                layers.append(UMBlock(layer_id, args))

        super().__init__(
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            layers=layers,
            non_checkpoint_layers=args.non_checkpoint_layers,
            mask_out_before=args.mask_out_before,
        )


# ------------------------------------
# -- For MeanFlow with 2nd timestep --
# ------------------------------------

class InputProcessingMeanFlow(InputProcessing):
    """
    InputProcessing adapted for MeanFlow with second timestep arg r.
    """

    def __init__(self, args: DiTArgs):
        super().__init__(args)

        timestep_embed_dim = args.timestep_features_dim * 2

        # Timestep encoding of (t, r)
        self.timestep_features_t = FourierFeaturesTime(1, args.timestep_features_dim)
        self.timestep_features_r = FourierFeaturesTime(1, args.timestep_features_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_embed_dim, args.inter_dim, bias=True),
            nn.SiLU(),
            nn.Linear(args.inter_dim, args.dim, bias=True),
            nn.SiLU(),
        )

        # Estimation of logvar from (t, r)
        self.estimate_logvar = args.estimate_logvar
        if args.estimate_logvar:
            self.timestep_logvar_t = FourierFeaturesTime(1, args.timestep_features_dim)
            self.timestep_logvar_r = FourierFeaturesTime(1, args.timestep_features_dim)
            self.to_logvar = nn.Sequential(
                nn.Linear(timestep_embed_dim, 128, bias=True),
                nn.SiLU(),
                nn.Linear(128, 1, bias=True),
            )

    # Copies signature from forward method of dit.DiffusionTransformer
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> DictTensor:
        batch_size = x.size(0)

        # Encode t_embed as a MLP applied t and r Fourier feats concatenated
        t_embed = self.to_timestep_embed(  # (b, c)
            torch.cat(
                [
                    self.timestep_features_t(t[:, None]),
                    self.timestep_features_r(r[:, None]),
                ],
                dim=-1,
            )
        )
        # Encode t_logvar as a MLP applied t and r Fourier feats concatenated
        t_logvar = (
            self.to_logvar(  # (b,)
                torch.cat(
                    [
                        self.timestep_logvar_t(t[:, None]),
                        self.timestep_logvar_r(r[:, None]),
                    ],
                    dim=-1,
                )
            )[:, 0]
            if self.estimate_logvar
            else None
        )
        d = dict(
            # memory tokens + embed(x)
            x=torch.cat(
                [
                    self.memory_tokens_rope.expand(
                        batch_size, self.n_memory_tokens_rope, -1
                    ),
                    self.embed_x(x),
                ],
                dim=1,
            ),
            x_mask=mask,
            t=t_embed,
            description=torch.cat(
                [
                    self.memory_tokens_description.expand(x.size(0), -1, -1),
                    self.to_cond_embed(cond["cross_attn_cond"])
                    if not self.no_description_mask
                    else self.to_cond_embed(
                        self.pad_description(
                            cond["cross_attn_cond"], cond["cross_attn_cond_mask"]
                        )
                    ),
                ],
                dim=1,
            )
            if "cross_attn_cond" in cond
            else None,
            description_mask=torch.cat(
                [
                    torch.ones(batch_size, self.n_memory_tokens_description).to(
                        x.device
                    ),
                    (
                        torch.ones_like(cond["cross_attn_cond_mask"])
                        if self.no_description_mask
                        else cond["cross_attn_cond_mask"]
                    ),
                ],
                dim=1,
            )
            if "cross_attn_cond" in cond
            else None,
            freqs_cis=self.freqs_cis,
            freqs_cis_description=self.freqs_cis_description,
            logvar=t_logvar,
        )

        # DictTensor is not supposed to contain None values
        return d  # type: ignore
