"""
EDMSampler class is a class used to add new sampling method to EDMModule via the
register_edm_samplers decorator defined in edm_samplers/__init__.py

To register a new sampler, one only has to create subclasses using the template below


=== Template for EDMSampler subclasses ===
Put the following in a new .py file in edm_samplers/ folder


import torch
from . import EDMSampler

class NameOfTheSamplerWillNotBeUsed(EDMSampler):
    sampler_method_name = 'my_method_name_to_be_added'

    @torch.no_grad()
    def sample(
        self,
        noise,
        ...
    ):

    return torch.Tensor
"""

import torch
import typing as tp


class ForceHavingName(type):
    """
    Heritating from this class forces the subclass to define
    the sampler_method_name class attribute
    """

    def __call__(cls, *args, **kwargs):
        class_object = type.__call__(cls, *args, **kwargs)
        assert class_object.sampler_method_name is not None
        return class_object


class EDMSampler(ForceHavingName):
    """
    EDMSampler base class
    Used to define samplers EDMModule outside edm_module.py
    All EDMSampler subclasses placed in the sfxfm/module/edm_samplers folder
    are dynamically added to EDMModule and can be considered as methods of this class

    The EDMSampler.sample method is available in EDMModule under the name defined by sampler_method_name

    Only
    sample()
    and
    sampler_method_name
    MUST be defined by subclasses

    Each sampler has a callback method that is called after call_every steps if the callback function is defined
    """

    sampler_method_name = None

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    # The following defines the signature of methods in EDMModule
    # that may be used in the EDMSampler
    # Note that they MUST NOT be reimplemented in the EDMSampler
    def round_sigma(self, value: float) -> torch.Tensor:
        raise NotImplementedError

    def denoise(self, x_t, t, cond=None, ema=False) -> torch.Tensor:
        raise NotImplementedError

    def _batched_cond_denoise(
        self, x_t, t, ema, cond=None, cond_batched=None
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _batch_cond_nocond(self, x_t, cond) -> torch.Tensor:
        raise NotImplementedError

    def linear_scheduler(
        self, num_steps: int, sigma_min: float, sigma_max: float, device
    ) -> torch.Tensor:
        raise NotImplementedError

    def karras_scheduler(
        self,
        num_steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        device,
        add_zero: float = True,
    ) -> torch.Tensor:
        raise NotImplementedError

    def cosine_scheduler(
        self,
        num_steps: int,
        sigma_min: float,
        sigma_max: float,
        device,
        add_zero: float = True,
    ) -> torch.Tensor:
        raise NotImplementedError
