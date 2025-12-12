import typing as tp
import torch

import torch.nn as nn

FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.Dict[str, LogitsType], tp.Dict[str, FeatureMapType]]

class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, x: torch.Tensor) -> tp.Tuple[LogitsType, FeatureMapType]:
        """
        Discriminator forward method must return a tuple (logit, feature_map)
        each with a batch_size in the first dim and time as the last dimension
        Args:
            x (torch.Tensor): Waveform

        Returns:
            tp.Tuple[LogitsType, FeatureMapType]: 
            
            LogitsType:
            (batch_size, 1, time)
            or
            (batch_size, 1, frequency, time)
            
            FeatureMapType:
            (batch_size, feature, time)
            or
            (batch_size, feature, frequency, time)
        """
        raise NotImplementedError

class DiscriminatorCollection(nn.Module):

    def __init__(self, discs_dict: tp.Dict[str, nn.Module]):
        super().__init__()
        self.discs_dict = nn.ModuleDict(discs_dict)

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        logits = {}
        fmaps = {}
        for n_disc, disc in self.discs_dict.items():
            logit, fmap = disc(x)
            logits[n_disc] = logit
            fmaps[n_disc] = fmap
        return logits, fmaps

    @property
    def n_discs(self):
        return self.discs_dict.keys()
