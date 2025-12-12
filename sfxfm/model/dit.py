from omegaconf import DictConfig
from pydantic import ValidationError
from sfxfm.model.alternative_mmdit import (
    Flux,
    MMMFlux,
    MMMSSFlux,
    newMMDiT,
    newMMDiTFluxattn,
    newMMDiTnottext,
    MMMFluxMeanFlow,
)
from sfxfm.model.dit_pipeline import DiTPipeline
from sfxfm.model.dit_types import DiTArgs, DiTv2Args, MMDiTArgs, PIOArgs
from sfxfm.model.ditv2 import DiTv2, MMDiT
# from sfxfm.model.pio import PIO


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
            if args.model_type == "mmdit":
                args = MMDiTArgs.model_validate(dict_args, strict=True)
                return MMDiT(args)
            elif args.model_type == "newmmdit":
                args = MMDiTArgs.model_validate(dict_args, strict=True)
                return newMMDiT(args)
            elif args.model_type == "newmmditnottext":
                args = MMDiTArgs.model_validate(dict_args, strict=True)
                return newMMDiTnottext(args)
            elif args.model_type == "newmmditflux":
                args = MMDiTArgs.model_validate(dict_args, strict=True)
                return newMMDiTFluxattn(args)
            elif args.model_type == "flux":
                args = MMDiTArgs.model_validate(dict_args, strict=True)
                return Flux(args)
            elif args.model_type == "mmmflux":
                args = MMDiTArgs.model_validate(dict_args, strict=True)
                return MMMFlux(args)
            elif args.model_type == "mmmflux-meanflow":
                args = MMDiTArgs.model_validate(dict_args, strict=True)
                return MMMFluxMeanFlow(args)
            elif args.model_type == "mmmssflux":
                args = MMDiTArgs.model_validate(dict_args, strict=True)
                return MMMSSFlux(args)
            elif args.model_type == "ditv2":
                args = DiTv2Args.model_validate(dict_args, strict=True)
                return DiTv2(args)
            # elif args.model_type == "pio":
            #     args = PIOArgs.model_validate(dict_args, strict=True)
            #     return PIO(args)
        except ValidationError as e:
            print("DIT args error ", e)
            print("Model args: ", args)
            print("errors: ", e.errors())
            raise e
        raise ValueError(f"Unknown DiT model {args.model_type}")
