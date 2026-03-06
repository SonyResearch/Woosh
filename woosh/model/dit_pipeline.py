from typing import Dict

import torch
from torch import nn

from woosh.model.dit_types import DictTensor


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)  # type: ignore


def mask_out(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: (b, t, c)
    mask: (b, t)

    Returns: x truncated (b, truncated_length, dim)
    """
    batch_size, length = mask.size()
    batch_size, length, dim = x.size()
    return x[mask.unsqueeze(2).expand(*x.size()) == 1].view(batch_size, -1, dim)


def unmask_out(x: torch.Tensor, mask: torch.Tensor, fill_tensor=None) -> torch.Tensor:
    """
    Inverse operation of mask_out. Restores the original tensor shape by filling masked positions with zeros.

    Args:
        x: (b, truncated_length, c) Tensor after mask_out.
        mask: (b, t) Boolean mask used in mask_out.
        original_length: Original sequence length before mask_out.
        fill_tensor, (c): Optional tensor to fill the masked positions (mask == 0). If None, zeros are used.

    Returns:
        Restored tensor of shape (b, original_length, c).
    """
    batch_size, _, dim = x.size()
    original_length = mask.size(1)

    if fill_tensor is not None:
        restored = fill_tensor[None, None, :].expand(batch_size, original_length, dim)
    else:
        restored = torch.zeros(
            batch_size, original_length, dim, device=x.device, dtype=x.dtype
        )

    restored[mask.unsqueeze(2).expand(-1, -1, dim) == 1] = x.view(-1)
    return restored


def mask_out_freqs(freqs_cis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    f: (t, c)
    mask: (b, t)

    Returns: x truncated (b, truncated_length, dim)
    """
    batch_size, length = mask.size()
    length, dim = freqs_cis.size()

    # WARNING
    # for now, mask is supposed to be identical for all elements in the batch
    mask = mask[0]
    return freqs_cis[mask.unsqueeze(1).expand(*freqs_cis.size()) == 1].view(-1, dim)


class DiTPipeline(torch.nn.Module):
    def __init__(
        self,
        preprocessing: nn.Module,
        postprocessing: nn.Module,
        layers: nn.ModuleList | nn.Sequential,
        non_checkpoint_layers,
        mask_out_before,
    ):
        """
        A DiT must implement DiTPipeline
        and define the following modules:
        """
        super().__init__()
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.layers = layers

        self.non_checkpoint_layers = non_checkpoint_layers
        self.mask_out = mask_out_before

    def forward(
        self,
        x: torch.Tensor,
        t,
        cond: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> DictTensor:
        """ """
        # Creates dict
        d = self.preprocessing(x, t, cond, mask)

        # iterate over block with checkpoint
        # skip non_checkpoint_layers layers for speed up
        for layer_id, layer in enumerate(self.layers):
            # MASK OUT
            if layer_id == self.mask_out:
                d["x"] = mask_out(d["x"], mask)
                d["freqs_cis"] = mask_out_freqs(d["freqs_cis"], mask)

            # LAYERS with CHECKPOINT
            if layer_id >= self.non_checkpoint_layers:
                d = checkpoint(layer, d.copy())
            else:
                d = layer(d)

        d = self.postprocessing(d)
        return d

    # TODO remove set_cast_v?!
    def set_cast_v(self, cast_v):
        """
        Disable cast and use qkv in float32 for JVP compatibility.
        """
        for _, layer in enumerate(self.layers):
            if hasattr(layer, "set_cast_v"):
                layer.set_cast_v(cast_v)


class DiTFlowMapPipeline(torch.nn.Module):
    def __init__(
        self,
        preprocessing: nn.Module,
        postprocessing: nn.Module,
        layers: nn.ModuleList | nn.Sequential,
        non_checkpoint_layers,
        mask_out_before,
    ):
        """
        A DiT used for FlowMap must implement DiTMFPipeline
        and define the following modules.
        The only difference with DiTPipeline is that it uses r
        as an extra timestep argument
        """
        super().__init__()
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.layers = layers

        self.non_checkpoint_layers = non_checkpoint_layers
        self.mask_out = mask_out_before

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> DictTensor:
        """ """
        # Creates dict
        d = self.preprocessing(x, t, r=r, cond=cond, mask=mask)

        # iterate over block with checkpoint
        # skip non_checkpoint_layers layers for speed up
        for layer_id, layer in enumerate(self.layers):
            # MASK OUT
            if layer_id == self.mask_out:
                d["x"] = mask_out(d["x"], mask)
                d["freqs_cis"] = mask_out_freqs(d["freqs_cis"], mask)

            # LAYERS with CHECKPOINT
            if layer_id >= self.non_checkpoint_layers:
                d = checkpoint(layer, d.copy())
            else:
                d = layer(d)

        d = self.postprocessing(d)
        return d

    def set_cast_v(self, cast_v):
        """
        Disable cast and use qkv in float32 for JVP compatibility.
        """
        for _, layer in enumerate(self.layers):
            if hasattr(layer, "set_cast_v"):
                layer.set_cast_v(cast_v)
