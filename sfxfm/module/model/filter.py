# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import typing as tp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import lfilter
import math
import logging

import numpy as np
from scipy.signal import bilinear, freqz

# get logger
log = logging.getLogger(__name__)

if 'sinc' in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    #   LICENSE is in incl_licenses directory.
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(x == 0,
                           torch.tensor(1., device=x.device, dtype=x.dtype),
                           torch.sin(math.pi * x) / math.pi / x)


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size): # return filter [1,1,kernel_size]
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2

    #For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else:
        beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = (torch.arange(-half_size, half_size) + 0.5)
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'replicate',
                 kernel_size: int = 12):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = (kernel_size % 2 == 0)
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    #input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right),
                      mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1),
                       stride=self.stride, groups=C)

        return out


def A_weighting(
        fs: float,
        n_fft: int = None,
        min_gain_db=-96.):
    """
    Design of an A-weighting filter.

    If `n_fft` is None, the coefficients of a 5-th order IIR filter implementing 
    A-weighting are returned, as `(b, a)`.

    If `n_fft` is given, the magnitude of the frequence response of the
    A-weighting filter is returned. `n_fft//2+1` values are returned in this
    case, including the Nyquist frequency point. Working with the frequency 
    response allows setting a lower threshold for the magnitude response
    with the `min_gain_db` argument. Note that A-weighting has 0 gain for
    the DC component.

    Filter design code taken from https://gist.github.com/endolith/148112.

    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.

    """

    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2*np.pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
    DENs = np.polymul([1, 4*np.pi * f4, (2*np.pi * f4)**2],
                   [1, 4*np.pi * f1, (2*np.pi * f1)**2])
    DENs = np.polymul(np.polymul(DENs, [1, 2*np.pi * f3]),
                                 [1, 2*np.pi * f2])

    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    b, a  = bilinear(NUMs, DENs, fs)

    if n_fft is None:
        # float32 may be unstable for certain IIR filters
        b = torch.tensor(b, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.float32)
        return b, a
    else:
        w, h = freqz(
                    b,
                    a,
                    worN=n_fft//2+1,
                    whole=False,
                    include_nyquist=True,
                    fs=fs
               )
        h = torch.tensor(np.abs(h), dtype=torch.float32)
        if min_gain_db is not None:
            min_gain = np.power(10, min_gain_db/20.)
            h = torch.clamp(h, min=min_gain)
        return h


class TimePreemphasis(nn.Module):
    def __init__(self,
            taps: tp.List = None,
            template: str = None,
            learnable: bool = False,
            sample_rate: int = 44100,
        ):
        super().__init__()
        self.template = template
        self.learnable = learnable
        self.sample_rate = sample_rate

        if template is None:
            self.type = 'fir'
            self.taps = taps
            self.n_taps = len(taps)
            # filter specified by tap values
            self.alpha = None
            self.b = torch.tensor(self.taps, dtype=torch.float32)
            self.b = torch.flip(self.b, dims=[0])
            self.b = self.b.view(1, 1, self.n_taps)
            self.a = None

            if self.n_taps % 2 == 1:
                self.pad_left = (self.n_taps - 1) // 2
                self.pad_right = (self.n_taps - 1) // 2
            else:
                self.pad_left = (self.n_taps - 1) / 2
                self.pad_right = (self.n_taps - 1) / 2
                self.pad_right_int = (self.n_taps - 1) // 2
                remainder = self.pad_right - self.pad_right_int
                self.pad_left = int (self.pad_left + remainder)
                self.pad_right = self.pad_right_int

            self.padding = True
            self.padding_mode = 'replicate'
            if self.learnable:
                self.b = nn.Parameter(self.b, requires_grad=False)

        elif template=='a-weighting':
            self.type = 'iir'
            self.learnable = False
            self.b, self.a = A_weighting(self.sample_rate)


    #input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.type=='fir':
            if self.b.device != x.device:
                self.b = self.b.to(device=x.device)
            # pad
            if self.padding:
                x = F.pad(x, (self.pad_left, self.pad_right),
                          mode=self.padding_mode)
            # norm zeroth tap to 1, force no delay and unit gain
            # with torch.no_grad():
            #     self.b.data = self.b.data / self.b.clone().detach()[0,0,0]
            # apply filter
            out = F.conv1d(x, self.b.expand(C, -1, -1), groups=C)

        elif self.type=='iir':
            if self.b.device != x.device:
                self.b = self.b.to(device=x.device)
                self.a = self.a.to(device=x.device)
            out = lfilter(x, self.a, self.b, clamp=True)

        return out


class FreqPreemphasis(nn.Module):
    def __init__(self,
            n_fft: int,
            template: str = None,
            sample_rate: int = 44100,
            min_gain_db: float = -96.,
        ):
        super().__init__()
        self.n_fft = n_fft
        self.template = template
        self.sample_rate = sample_rate
        self.min_gain_db = min_gain_db

        self.spec_weights = None 
        if template=='a-weighting':
            self.spec_weights = A_weighting(
                                self.sample_rate,
                                n_fft=self.n_fft,
                                min_gain_db=self.min_gain_db,
                            ).unsqueeze(-1)
        else:
            raise ValueError(f"only 'a-weighting' allowed as spectral preemphasis")

    def forward(self, z):
        # z is [B, C, F, T]

        if self.spec_weights is None:
            return z

        if self.spec_weights.device != z.device:
            self.spec_weights = self.spec_weights.to(device=z.device)

        z *= self.spec_weights

        return z
