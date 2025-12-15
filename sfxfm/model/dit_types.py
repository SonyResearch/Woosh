from typing import Annotated, Dict, Literal, Union
from pydantic import BaseModel, ConfigDict, Field
import torch

type DictTensor = Dict[str, torch.Tensor]


class MMDiTArgs(BaseModel):
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
        io_channels (int): Number of channels of the inut/output tensor.
    """

    model_config = ConfigDict(extra="forbid")

    model_type: Literal["mmmssflux",] = "mmmssflux"
    max_description_length: int = 77
    max_seq_len: int = 501

    dim: int = 2048
    inter_dim: int = 10944
    fixed_timestep_features: bool = False
    timestep_features_dim: int = 256
    n_layers: int = 27
    n_heads: int = 16
    n_multimodal_layers: int = 27
    # mla
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    qkv_head_dim: int = 128

    # memory tokens
    n_memory_tokens_rope: int = 0
    n_memory_tokens_description: int = 0

    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1

    # IO
    io_channels: int = 128
    cond_token_dim: int = 1024
    adaln_last_layer: bool = False
    adaln_last_layer_nomod: bool = False  # if adaln_last_layer, do not modulate

    # Optim
    non_checkpoint_layers: int = 0  # checkpoint all layers
    mask_out_before: int = -1  # mask out before layer #n: -1 for no masking

    #
    estimate_logvar: bool = False
    no_description_mask: bool = False
    symmetric_attention_init: bool = False

    patch_size: int = 1

    num_sinks: int = 0
    mlp_act: str = "gelu"  # gelu # swiglu


# DiTConfig can be any of the two configs
# Configs in the union are discriminated based on their model_type
DiTArgs = Annotated[Union[MMDiTArgs], Field(discriminator="model_type")]
