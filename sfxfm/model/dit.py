from omegaconf import DictConfig
from pydantic import ValidationError
from sfxfm.model.alternative_mmdit import (
    MMMSSFlux,
    MMMFluxMeanFlow,
)
from sfxfm.model.dit_pipeline import DiTPipeline
from sfxfm.model.dit_types import DiTArgs, MMDiTArgs


class DiT(DiTPipeline):
    """
    Main DiT class,
    allows to choose between many implementations
    """

    def __new__(cls, args: DiTArgs):
        dict_args = args
        if isinstance(args, DictConfig):
            dict_args = dict(args)  # type: ignore
        try:
            if args.model_type == "mmmflux-meanflow":
                args = MMDiTArgs.model_validate(dict_args, strict=True)
                return MMMFluxMeanFlow(args)
            elif args.model_type == "mmmssflux":
                args = MMDiTArgs.model_validate(dict_args, strict=True)
                return MMMSSFlux(args)
        except ValidationError as e:
            print("DIT args error ", e)
            print("Model args: ", args)
            print("errors: ", e.errors())
            raise e
        raise ValueError(f"Unknown DiT model {args.model_type}")
