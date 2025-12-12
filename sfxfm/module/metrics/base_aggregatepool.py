from copy import deepcopy
import torch
import torchmetrics
from einops import rearrange
from typing import Optional, Union

from sfxfm.utils.dist.distrib import rank

###################################################################################################

rank = rank()


def param_to_buffer(module):
    """Turns all parameters of a module into buffers."""
    # https://discuss.pytorch.org/t/how-to-turn-parameter-to-buffer/144670
    modules = module.modules()
    module = next(modules)
    for name, param in dict(module.named_parameters(recurse=False)).items():
        delattr(module, name)  # Unregister parameter
        module.register_buffer(name, param.detach().clone(), persistent=False)
    for module in modules:
        param_to_buffer(module)


class AggregatePoolMetric(torchmetrics.Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        frontend,
        patchify: Optional[int] = None,
        temporal_pooling: Union[str, None] = "linspace22",
        store_feats_in_cpu: bool = False,
    ):
        """
        patchify - Number of consecutive vectors to form a patch.
        temporal_pooling - One in (flatten, average, sampleXXX, firstXXX, linspaceXXX)
        store_feats_in_cpu - Note: for multiGPU + storing in CPU, aggregation wil be
                            the average across GPUs; proper aggregation across
                            multiple GPUs will only happen when =False.
        """
        super().__init__()
        self.patchify = patchify
        assert (
            temporal_pooling is None
            or temporal_pooling in ("flatten", "average")
            or temporal_pooling.startswith("sample")
            or temporal_pooling.startswith("first")
            or temporal_pooling.startswith("linspace")
        )
        self.temporal_pooling = temporal_pooling
        self.store_feats_in_cpu = store_feats_in_cpu
        # Get frontend
        self.frontend = frontend
        # Init
        if self.store_feats_in_cpu:
            self.h_real = []
            self.h_fake = []
        else:
            self.add_state("h_real", default=[], dist_reduce_fx="cat", persistent=False)
            self.add_state("h_fake", default=[], dist_reduce_fx="cat", persistent=False)

        # In metrics the parameters don't change
        # converting them to buffers removes them from the optimizer state dict
        # converting to buffers generates a bug for code in the models similar to:
        # device = next(self.parameters()).deviceexport
        # param_to_buffer(self.frontend)
        # instead we just parameters' requires_grad to false
        self.frontend.eval()
        for param in self.frontend.parameters():
            param.requires_grad = False

    def __deepcopy__(self, memo):
        """
        don't duplicate the front-end on the gpu,
        the state are created in the __init__ and the frontend is not modified
        """
        frontend = self.frontend
        del self.frontend
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None  # type: ignore
        new = deepcopy(self)
        self.frontend = frontend
        new.frontend = frontend
        self.__deepcopy__ = deepcopy_method
        return new

    def reset(self):
        """Copied from the reset() in torchmetrics.Metric itself and
        hard-coding the erase part"""
        self._update_count = 0
        self._forward_cache = None
        self._computed = None
        # erase
        self.h_real = []
        self.h_fake = []
        # reset internal states
        self._cache = None
        self._is_synced = False

    def update(
        self,
        audio_fake: torch.Tensor,  # prediction: (B,T) or (B,1,T)
        audio_real: torch.Tensor,  # target: (B',T') or (B',1,T')
        # batch: Optional[dict],
    ):
        assert audio_real.ndim == audio_fake.ndim
        assert audio_real.ndim == 2 or (
            audio_real.ndim == 3 and audio_real.size(1) == 1 and audio_fake.size(1) == 1
        )
        if audio_real.ndim == 3:
            audio_real = audio_real.squeeze(1)
            audio_fake = audio_fake.squeeze(1)
        # Compute features
        hf = self.frontend(audio_fake)
        hr = self.frontend(audio_real)
        assert (
            hf.shape == hr.shape
        ), f"Batch size mismatch after frontend, hf={hf.shape}, hr={hr.shape}"
        if (
            self.patchify is not None
            and self.patchify > 1
            and hr.size(-1) > self.patchify + 1
        ):
            auxr = hr[:, :, : -self.patchify].clone()
            auxf = hf[:, :, : -self.patchify].clone()
            for i in range(1, self.patchify):
                auxr = torch.cat([auxr, hr[:, :, i : i + auxr.size(2)]], dim=1)
                auxf = torch.cat([auxf, hf[:, :, i : i + auxf.size(2)]], dim=1)
            hr = auxr
            hf = auxf
        # Temporal pooling
        if self.temporal_pooling is None:
            pass
        elif self.temporal_pooling == "flatten":
            hr = rearrange(hr, "b c t -> (b t) c")
            hf = rearrange(hf, "b c t -> (b t) c")
        elif self.temporal_pooling == "average":
            hr = hr.mean(2)
            hf = hf.mean(2)
        elif self.temporal_pooling.startswith("sample"):
            n = int(self.temporal_pooling.replace("sample", ""))
            idx = torch.randint(0, hr.size(2), (n,))
            hr = hr[:, :, idx]
            idx = torch.randint(0, hf.size(2), (n,))
            hf = hf[:, :, idx]
            hr = rearrange(hr, "b c t -> (b t) c")
            hf = rearrange(hf, "b c t -> (b t) c")
        elif self.temporal_pooling.startswith("first"):
            n = int(self.temporal_pooling.replace("first", ""))
            if n > hr.size(2):
                hr = hr[:, :, :n]
            if n > hf.size(2):
                hf = hf[:, :, :n]
            hr = rearrange(hr, "b c t -> (b t) c")
            hf = rearrange(hf, "b c t -> (b t) c")
        elif self.temporal_pooling.startswith("linspace"):
            n = int(self.temporal_pooling.replace("linspace", ""))
            idx = torch.linspace(0, hr.size(2) - 1, n).long()
            hr = hr[:, :, idx]
            idx = torch.linspace(0, hf.size(2) - 1, n).long()
            hf = hf[:, :, idx]
            hr = rearrange(hr, "b c t -> (b t) c")
            hf = rearrange(hf, "b c t -> (b t) c")
        # Append to state
        if self.store_feats_in_cpu:
            hr = hr.cpu()
            hf = hf.cpu()
        assert (
            hf.shape == hr.shape
        ), f"Batch size mismatch after temporal_pooling, hf={hf.shape}, hr={hr.shape}"
        self.h_real.append(hr)
        self.h_fake.append(hf)

    def _get_collected_features(self):
        # Get features
        h_real, h_fake = self.h_real, self.h_fake
        if isinstance(h_real, list):
            h_real = torch.cat(self.h_real, dim=0)
            h_fake = torch.cat(self.h_fake, dim=0)

        return h_real, h_fake

    def _return_value(self, val):
        # Return
        if self.store_feats_in_cpu:
            val = val.item()
        return val

    def _h_empty(self) -> bool:
        # returns true if any of the h is empty
        return not ((len(self.h_real) > 0) and (len(self.h_fake) > 0))

    def compute(self):
        """Should be overrided by the inheriting class, but including these lines"""
        h_real, h_fake = self._get_collected_features()
        # <your code here>
        raise NotImplementedError(
            "compute should be overrided by the inheriting class."
        )


class BatchBasedAggregatePoolMetric(AggregatePoolMetric):
    def __init__(
        self,
        frontend,
        patchify: Optional[int] = None,
        temporal_pooling: str = "flatten",
        store_feats_in_cpu: bool = False,
    ):
        super().__init__(
            frontend=frontend,
            patchify=patchify,
            temporal_pooling=temporal_pooling,
            store_feats_in_cpu=store_feats_in_cpu,
        )

    def update(
        self,
        audio_fake: torch.Tensor,  # prediction: (B,T) or (B,1,T)
        batch: dict,
    ):
        if audio_fake.ndim == 3:
            audio_fake = audio_fake.squeeze(1)
        hf, hr = self.frontend(audio_fake, batch)
        assert hf.size(0) == hr.size(0), "Batch size mismatch"
        if (
            self.patchify is not None
            and self.patchify > 1
            and hr.size(-1) > self.patchify + 1
        ):
            auxr = hr[:, :, : -self.patchify].clone()
            auxf = hf[:, :, : -self.patchify].clone()
            for i in range(1, self.patchify):
                auxr = torch.cat([auxr, hr[:, :, i : i + auxr.size(2)]], dim=1)
                auxf = torch.cat([auxf, hf[:, :, i : i + auxf.size(2)]], dim=1)
            hr = auxr
            hf = auxf
        # Temporal pooling
        if hr.ndim == 2:
            pass  # current CLAP embeddings got shape [1,512]
        elif self.temporal_pooling == "flatten":
            hr = rearrange(hr, "b c t -> (b t) c")
            hf = rearrange(hf, "b c t -> (b t) c")
        elif self.temporal_pooling == "average":
            hr = hr.mean(2)
            hf = hf.mean(2)
        elif self.temporal_pooling.startswith("sample"):
            n = int(self.temporal_pooling.replace("sample", ""))
            idx = torch.randint(0, hr.size(2), (n,))
            hr = hr[:, :, idx]
            idx = torch.randint(0, hf.size(2), (n,))
            hf = hf[:, :, idx]
            hr = rearrange(hr, "b c t -> (b t) c")
            hf = rearrange(hf, "b c t -> (b t) c")
        elif self.temporal_pooling.startswith("first"):
            n = int(self.temporal_pooling.replace("first", ""))
            if n > hr.size(2):
                hr = hr[:, :, :n]
            if n > hf.size(2):
                hf = hf[:, :, :n]
            hr = rearrange(hr, "b c t -> (b t) c")
            hf = rearrange(hf, "b c t -> (b t) c")
        elif self.temporal_pooling.startswith("linspace"):
            n = int(self.temporal_pooling.replace("linspace", ""))
            idx = torch.linspace(0, hr.size(2) - 1, n).long()
            hr = hr[:, :, idx]
            idx = torch.linspace(0, hf.size(2) - 1, n).long()
            hf = hf[:, :, idx]
            hr = rearrange(hr, "b c t -> (b t) c")
            hf = rearrange(hf, "b c t -> (b t) c")
        # Append to state
        if self.store_feats_in_cpu:
            hr = hr.cpu()
            hf = hf.cpu()
        assert hf.size(0) == hr.size(0), "Batch size mismatch after temporal_pooling"
        self.h_real.append(hr)
        self.h_fake.append(hf)


###################################################################################################
