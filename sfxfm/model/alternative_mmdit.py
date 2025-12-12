import torch

from sfxfm.model.dit_pipeline import DiTPipeline, DiTMeanFlowPipeline
from sfxfm.model.dit_types import MMDiTArgs

from sfxfm.model.dit_blocks import (
    MMBlock,
    MMMBlock,
    ModalityAttention,
    ModalityBlock,
    MultimodalitySingleStreamBlock,
    SinkModalityBlock,
    UMBlock,
)
from sfxfm.model.ditv2 import (
    InputProcessing,
    PostProcessing,
    InputProcessingMeanFlow,
)


class newMMDiT(DiTPipeline):
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
            # Alternate between MM and UM blocks
            if layer_id % 2 == 1 and layer_id < args.n_multimodal_layers:
                layers.append(
                    MMBlock(
                        layer_id,
                        args,
                        modality_1=ModalityAttention(
                            args,
                            x_key="x",
                            mod_key="t",
                            freqs_cis_key="freqs_cis",
                            mask_key=None,
                        ),
                        modality_2=ModalityAttention(
                            args,
                            x_key="description",
                            mod_key="t",
                            mask_key="description_mask",
                            freqs_cis_key="freqs_cis_description",
                        ),
                    )
                )
            else:
                layers.append(
                    UMBlock(
                        layer_id,
                        args,
                        modality=ModalityAttention(
                            args,
                            x_key="x",
                            mod_key="t",
                            freqs_cis_key="freqs_cis",
                            mask_key=None,
                        ),
                    )
                )

        super().__init__(
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            layers=layers,
            non_checkpoint_layers=args.non_checkpoint_layers,
            mask_out_before=args.mask_out_before,
        )


class newMMDiTnottext(DiTPipeline):
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
            if layer_id % 2 == 1:
                layers.append(
                    MMBlock(
                        layer_id,
                        args,
                        modality_1=ModalityAttention(
                            args,
                            x_key="x",
                            mod_key="t",
                            freqs_cis_key="freqs_cis",
                            mask_key=None,
                        ),
                        modality_2=ModalityAttention(
                            args,
                            x_key="description",
                            mod_key=None,
                            mask_key="description_mask",
                            freqs_cis_key="freqs_cis_description",
                        ),
                    )
                )
            else:
                layers.append(
                    UMBlock(
                        layer_id,
                        args,
                        modality=ModalityAttention(
                            args,
                            x_key="x",
                            mod_key="t",
                            freqs_cis_key="freqs_cis",
                            mask_key=None,
                        ),
                    )
                )

        super().__init__(
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            layers=layers,
            non_checkpoint_layers=args.non_checkpoint_layers,
            mask_out_before=args.mask_out_before,
        )


class newMMDiTFluxattn(DiTPipeline):
    """
    Same as newMMDiT, but uses
    ropenope on x only in MM blocks; nope on text
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
            if layer_id % 2 == 1:
                layers.append(
                    MMBlock(
                        layer_id,
                        args,
                        modality_1=ModalityAttention(
                            args,
                            x_key="x",
                            mod_key="t",
                            freqs_cis_key="freqs_cis",
                            mask_key=None,
                        ),
                        apply_rope_1=True,
                        modality_2=ModalityAttention(
                            args,
                            x_key="description",
                            mod_key="t",
                            mask_key="description_mask",
                            freqs_cis_key="freqs_cis_description",
                        ),
                        apply_rope_2=False,
                    )
                )
            else:
                layers.append(
                    UMBlock(
                        layer_id,
                        args,
                        modality=ModalityAttention(
                            args,
                            x_key="x",
                            mod_key="t",
                            freqs_cis_key="freqs_cis",
                            mask_key=None,
                        ),
                    )
                )

        super().__init__(
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            layers=layers,
            non_checkpoint_layers=args.non_checkpoint_layers,
            mask_out_before=args.mask_out_before,
        )


class Flux(DiTPipeline):
    """
    Same as newMMDiT, but uses
    ropenope on x only in MM blocks; nope on text

    Tries to be as close to Flux.
    So first MM blocks use ropenope on x, then nope on text.
    then UM blocks use ropenope on x
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
                layers.append(
                    MMBlock(
                        layer_id,
                        args,
                        modality_1=ModalityAttention(
                            args,
                            x_key="x",
                            mod_key="t",
                            freqs_cis_key="freqs_cis",
                            mask_key=None,
                        ),
                        apply_rope_1=True,
                        modality_2=ModalityAttention(
                            args,
                            x_key="description",
                            mod_key="t",
                            mask_key="description_mask",
                            freqs_cis_key="freqs_cis_description",
                        ),
                        apply_rope_2=False,
                    )
                )
            else:
                layers.append(
                    UMBlock(
                        layer_id,
                        args,
                        modality=ModalityAttention(
                            args,
                            x_key="x",
                            mod_key="t",
                            freqs_cis_key="freqs_cis",
                            mask_key=None,
                        ),
                    )
                )

        super().__init__(
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            layers=layers,
            non_checkpoint_layers=args.non_checkpoint_layers,
            mask_out_before=args.mask_out_before,
        )


class MMMFlux(DiTPipeline):
    """
    Same as Flux, but uses MMMBlocks only

    Does not rely on apply_rope argument
    """

    def __init__(self, args: MMDiTArgs):
        """ """
        assert args.no_description_mask, "MMMFlux requires no description mask"
        preprocessing = InputProcessing(args)
        postprocessing = PostProcessing(args)

        layers = torch.nn.Sequential()
        for layer_id in range(args.n_layers):
            if layer_id < args.n_multimodal_layers:
                modality_block_dict = dict(
                    x=ModalityBlock(
                        args,
                        x_key="x",
                        mod_key="t",
                        freqs_cis_key="freqs_cis",
                        mask_key=None,
                    ),
                    description=ModalityBlock(
                        args,
                        x_key="description",
                        mod_key="t",
                        mask_key=None,
                        freqs_cis_key=None,
                    ),
                )
                if args.num_sinks > 0:
                    modality_block_dict["sinks"] = SinkModalityBlock(
                        args,
                        other_modality_key="x",
                    )

                layers.append(
                    MMMBlock(
                        layer_id,
                        modality_block_dict=modality_block_dict,
                    ),
                )

            else:
                if args.num_sinks > 0:
                    layers.append(
                        MMMBlock(
                            layer_id,
                            modality_block_dict=dict(
                                sinks=SinkModalityBlock(
                                    args,
                                    other_modality_key="x",
                                ),
                                x=ModalityBlock(
                                    args,
                                    x_key="x",
                                    mod_key="t",
                                    freqs_cis_key="freqs_cis",
                                    mask_key=None,
                                ),
                            ),
                        )
                    )
                else:
                    layers.append(
                        ModalityBlock(
                            args,
                            x_key="x",
                            mod_key="t",
                            freqs_cis_key="freqs_cis",
                            mask_key=None,
                        ),
                    )

        super().__init__(
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            layers=layers,
            non_checkpoint_layers=args.non_checkpoint_layers,
            mask_out_before=args.mask_out_before,
        )


# Old implem without sinks. Clearer
# class MMMFlux(DiTPipeline):
#     """
#     Same as Flux, but uses MMMBlocks only

#     Does not rely on apply_rope argument
#     """

#     def __init__(self, args: MMDiTArgs):
#         """ """
#         assert args.no_description_mask, "MMMFlux requires no description mask"
#         preprocessing = InputProcessing(args)
#         postprocessing = PostProcessing(args)

#         layers = torch.nn.Sequential()
#         for layer_id in range(args.n_layers):
#             if layer_id < args.n_multimodal_layers:
#                 layers.append(
#                     MMMBlock(
#                         layer_id,
#                         modality_block_dict=dict(
#                             x=ModalityBlock(
#                                 args,
#                                 x_key="x",
#                                 mod_key="t",
#                                 freqs_cis_key="freqs_cis",
#                                 mask_key=None,
#                             ),
#                             description=ModalityBlock(
#                                 args,
#                                 x_key="description",
#                                 mod_key="t",
#                                 mask_key=None,
#                                 freqs_cis_key=None,
#                             ),
#                         ),
#                     )
#                 )
#             else:
#                 layers.append(
#                     ModalityBlock(
#                         args,
#                         x_key="x",
#                         mod_key="t",
#                         freqs_cis_key="freqs_cis",
#                         mask_key=None,
#                     ),
#                 )

#         super().__init__(
#             preprocessing=preprocessing,
#             postprocessing=postprocessing,
#             layers=layers,
#             non_checkpoint_layers=args.non_checkpoint_layers,
#             mask_out_before=args.mask_out_before,
#         )


class MMMSSFlux(DiTPipeline):
    """
    Same as Flux, but uses MMMBlocks only
    Adds singlestream blocks

    Does not rely on apply_rope argument
    """

    def __init__(self, args: MMDiTArgs):
        """ """
        assert args.no_description_mask, "MMMFlux requires no description mask"
        preprocessing = InputProcessing(args)
        postprocessing = PostProcessing(args)

        layers = torch.nn.Sequential()
        assert args.num_sinks == 0, "MMMSSFlux requires num_sinks to be 0"
        for layer_id in range(args.n_layers):
            if layer_id < args.n_multimodal_layers:
                layers.append(
                    MMMBlock(
                        layer_id,
                        modality_block_dict=dict(
                            x=ModalityBlock(
                                args,
                                x_key="x",
                                mod_key="t",
                                freqs_cis_key="freqs_cis",
                                mask_key=None,
                            ),
                            description=ModalityBlock(
                                args,
                                x_key="description",
                                mod_key="t",
                                mask_key=None,
                                freqs_cis_key="freqs_cis_description",
                            ),
                        ),
                    ),
                )

            else:
                layers.append(
                    MultimodalitySingleStreamBlock(
                        layer_id=layer_id,
                        args=args,
                        x_keys=["x", "description"],
                        mod_key="t",
                        freqs_cis_keys=["freqs_cis", "freqs_cis_description"],
                        mask_key=None,
                    ),
                )

        super().__init__(
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            layers=layers,
            non_checkpoint_layers=args.non_checkpoint_layers,
            mask_out_before=args.mask_out_before,
        )


# -------------------------------------------
# -- For MeanFlow models with 2nd timestep --
# -------------------------------------------


class MMMFluxMeanFlow(DiTMeanFlowPipeline):
    """
    Adapted to 2nd timestep arg (meanflow).
    """

    def __init__(self, args: MMDiTArgs):
        """ """
        assert args.no_description_mask, "MMMFlux requires no description mask"
        preprocessing = InputProcessingMeanFlow(args)
        postprocessing = PostProcessing(args)

        layers = torch.nn.Sequential()
        for layer_id in range(args.n_layers):
            if layer_id < args.n_multimodal_layers:
                layers.append(
                    MMMBlock(
                        layer_id,
                        modality_block_dict=dict(
                            x=ModalityBlock(
                                args,
                                x_key="x",
                                mod_key="t",
                                freqs_cis_key="freqs_cis",
                                mask_key=None,
                            ),
                            description=ModalityBlock(
                                args,
                                x_key="description",
                                mod_key="t",
                                mask_key=None,
                                freqs_cis_key=None,
                            ),
                        ),
                    )
                )
            else:
                layers.append(
                    ModalityBlock(
                        args,
                        x_key="x",
                        mod_key="t",
                        freqs_cis_key="freqs_cis",
                        mask_key=None,
                    ),
                )

        super().__init__(
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            layers=layers,
            non_checkpoint_layers=args.non_checkpoint_layers,
            mask_out_before=args.mask_out_before,
        )
