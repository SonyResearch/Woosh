from abc import ABC, abstractmethod
import logging
import os
from typing import Mapping, Optional

from pydantic import BaseModel, ConfigDict
import torch
from enum import Enum
from .utils import CLAPCaptionPostprocessTransform
from sfxfm.inference.flowmap_sampler import sample_euler
from sfxfm.model.ldm import LatentDiffusionModel
from sfxfm.model.flowmap_from_pretrained import FlowMapFromPretrained
from sfxfm.components.base import LoadConfig
from sfxfm.utils.loading import catchtime

log = logging.getLogger(__name__)


# enum allows api verifications
class NoiseSchedulerEnum(str, Enum):
    karras = "karras"
    linear = "linear"
    sigmoid = "sigmoid"
    cosine = "cosine"


class SamplersEnum(str, Enum):
    heun = "heun"
    cfgpp = "cfgpp"


class GenerateArgs(BaseModel):
    # special field to force not having extra arguments
    model_config = ConfigDict(extra="forbid")

    # request arguments:
    prompt: str = ""

    # general args
    cfg: float = 1
    sampler: SamplersEnum = SamplersEnum.heun
    num_steps: int = 100

    sigma_min: float = 1e-5
    sigma_max: float = 80
    rho: float = 7
    S_churn: float = 1
    S_min: float = 0
    S_max: float = float("inf")
    S_noise: float = 1
    guidance_scale: float = 7.5
    noise_scheduler: NoiseSchedulerEnum = NoiseSchedulerEnum.karras

    seed: Optional[int] = None

    model: str = "Woosh-DFlow"


class GenerateAgentInterface(ABC):
    @abstractmethod
    def load_model(self):
        """Loads the model into memory."""
        pass

    @abstractmethod
    def generate(self, *args, **kwargs) -> dict:
        """Generates audio from the model.

        Returns:
            dict: {"audio": waveform, "sample_rate": sample_rate}
        """
        pass

    @abstractmethod
    def cpu(self, *args, **kwargs):
        """moves the model to cpu, freeing gpu memory"""
        pass

    @abstractmethod
    def gpu(self, *args, **kwargs):
        """moves the model to gpu"""
        pass

    @abstractmethod
    def ready(self) -> bool:
        """Whether the model is loaded, on device, and ready for inference.

        Returns:
            bool: the agent is ready for inference, by calling generate.
        """
        pass


normalize_transform = CLAPCaptionPostprocessTransform(remove_punctuation=False)


def normalize_text(text):
    # Normalize the text by removing special characters and replacing spaces with underscores
    res = normalize_transform({"captions": [text]})
    text = res["captions"][0]
    print("normalized text:", text)
    return text


class GenerateBasicAgent(GenerateAgentInterface):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        components_path="checkpoints/",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.components_path = components_path

    def load_model(self):
        model_name = self.model_name.strip()
        log.info(f"Loading `{model_name}` to  {self.device}")

        # TODO use path
        model_path = os.path.join(self.components_path, model_name)

        ldm = LatentDiffusionModel(config=LoadConfig(path=model_path))
        ldm._component_summary()
        ldm.eval()
        ldm = ldm.to(self.device)
        self.ldm = ldm

    def cpu(self):
        self.ldm = self.ldm.cpu()

    def gpu(self):
        self.ldm = self.ldm.to(self.device)

    def ready(self) -> bool:
        ldm_device = next(self.ldm.parameters()).device
        print(
            f"model {self.model_name} device: {ldm_device}, self.device: {torch.device(self.device)}"
        )
        return ldm_device != torch.device("cpu")

    def generate(self, args: GenerateArgs):
        raise NotImplementedError


class FlowMapGenerateAgent(GenerateBasicAgent):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        components_path="checkpoints/",
    ) -> None:
        super().__init__(model_name, device, components_path)

    def load_model(self):
        model_name = self.model_name.strip()
        log.info(f"Loading `{model_name}` to  {self.device}")

        # TODO use path
        model_path = os.path.join(self.components_path, model_name)
        ldm = FlowMapFromPretrained(LoadConfig(path=model_path))
        ldm._component_summary()
        ldm.eval()
        ldm = ldm.to(self.device)
        self.ldm = ldm

    def generate(self, args: GenerateArgs):
        rng_gen = torch.Generator()
        if args.seed is not None:
            rng_gen.manual_seed(args.seed)
        # @TODO do the length
        batch_size = 1
        length = 501
        dim = 128
        noise = torch.normal(
            0, 1, size=(batch_size, dim, length), generator=rng_gen
        ).to(self.device)
        # noise = torch.randn(size=(batch_size, dim, length)).to(self.device)

        description = args.prompt
        description = normalize_text(description)
        batch = {
            "description": [description] * batch_size,
            "audio": noise,
            # "x_original": reference,
            # "x2": reference,
            # "mask": mask,
        }

        cond = self.ldm.get_cond(
            batch,
            no_dropout=True,
            device=self.device,  # we must provide device if we don't provide x
        )
        cond["cfg"] = torch.ones((batch_size,), device=self.device) * args.cfg
        log.info(f"Generating with condition: {cond.keys()}")

        with torch.inference_mode():
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = sample_euler(
                self.ldm, noise, cond=cond, num_steps=args.num_steps, renoise=0.5
            )
            x = self.ldm.autoencoder.inverse(z)
            audio_max = torch.max(torch.abs(x)).item()
            if audio_max > 1.0:
                x = x / audio_max
            x = x.cpu()
        # TODO add optional sample_rate arg
        sample_rate = 48000
        # if os.environ.get("WOOSH_RETURN_FP16") == "1":
        #     pass
        # else:
        #     x = x.half()
        log.info(f"Generated audio shape: {x.shape}, sample rate: {sample_rate}")
        x = x[0]
        audio = {"audio": x.cpu(), "sample_rate": sample_rate}
        return audio


class MultimodelGenerateAgent(GenerateAgentInterface):
    def __init__(
        self,
        models: Mapping[str, str],
        device: str = "cuda",
        components_path="checkpoints/",
    ) -> None:
        """Multimodal generation agent for handling multiple models.

        Args:
            models (Mapping[str, str]): model_name to type (e.g. {"dfix-diffv3-ft2-ds3-b96": "ldm})
            device (str, optional): inference device. Defaults to "cuda".
            components_path (str, optional): components path. Defaults to "cache_dir/components/".
        """
        super().__init__()
        self.agents = {}
        for model_name, model_type in models.items():
            model_type = model_type.lower().strip()
            if model_type == "ldm":
                agent = GenerateBasicAgent(
                    model_name=model_name,
                    device=device,
                    components_path=components_path,
                )
            elif model_type == "flowmap":
                agent = FlowMapGenerateAgent(
                    model_name=model_name,
                    device=device,
                    components_path=components_path,
                )
            else:
                raise ValueError(
                    f"Unknown model type: {model_type} for model {model_name}"
                )
            self.agents[model_name] = agent

    def load_model(self):
        for name, agent in self.agents.items():
            log.info(f"Loading model: {name}")
            with catchtime(f"Loaded model {name}"):
                agent.load_model()
                agent.cpu()

    def cpu(self):
        for agent in self.agents.values():
            agent.cpu()

    def gpu(self):
        for agent in self.agents.values():
            agent.gpu()

    def ready(self) -> bool:
        # does not need to be on gpu for inference
        return True

    def generate(self, args: GenerateArgs):
        model_name = args.model
        if model_name not in self.agents:
            raise ValueError(f"Unknown model: {model_name}")
        agent = self.agents[model_name]
        if not agent.ready():
            log.info(f"Moving model `{model_name}` to {agent.device} for inference")
            self.cpu()
            agent.gpu()

        audio = agent.generate(args)
        return audio
