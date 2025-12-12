import torch

from sfxfm.model.dit_pipeline import DiTPipeline, DiTMeanFlowPipeline
from sfxfm.model.dit_types import MMDiTArgs

from sfxfm.model.dit_blocks import (
    MMMBlock,
    ModalityBlock,
    MultimodalitySingleStreamBlock,
)
from sfxfm.model.ditv2 import (
    InputProcessing,
    PostProcessing,
    InputProcessingMeanFlow,
)


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
