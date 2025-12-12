import torch

from sfxfm.utils.dist import rank
import torch.signal
from torch import nn, view_as_real, view_as_complex
from torchaudio.functional.functional import _hz_to_mel, _mel_to_hz

from typing import Optional, Tuple
from torch.nn.utils.parametrizations import weight_norm
import torchaudio
import numpy as np
from sfxfm.module.model.autoencoder import AutoEncoder, VariationalAutoEncoder
from sfxfm.module.model.vq_autoencoder import VQAutoEncoder

# from sfxfm.module.model.descript import Encoder as DACEncoder
from sfxfm.module.model.vocos_blocks import STFTEmbedding, safe_log, symexp
# from vector_quantize_pytorch import ResidualVQ

import logging

# get logger
log = logging.getLogger(__name__)


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class IMDCT(nn.Module):
    """
    Inverse Modified Discrete Cosine Transform (IMDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.frame_len = frame_len
        N = frame_len // 2
        n0 = (N + 1) / 2
        window = torch.signal.windows.cosine(frame_len).float()
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(1j * torch.pi * n0 * torch.arange(N * 2) / N)
        post_twiddle = torch.exp(1j * torch.pi * (torch.arange(N * 2) + n0) / (N * 2))
        self.register_buffer("pre_twiddle", view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", view_as_real(post_twiddle))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the Inverse Modified Discrete Cosine Transform (IMDCT) to the input MDCT coefficients.

        Args:
            X (Tensor): Input MDCT coefficients of shape (B, L, N), where B is the batch size,
                L is the number of frames, and N is the number of frequency bins.

        Returns:
            Tensor: Reconstructed audio waveform of shape (B, T), where T is the length of the audio.
        """
        B, L, N = X.shape
        Y = torch.zeros((B, L, N * 2), dtype=X.dtype, device=X.device)
        Y[..., :N] = X
        Y[..., N:] = -1 * torch.conj(torch.flip(X, dims=(-1,)))
        y = torch.fft.ifft(
            Y * view_as_complex(self.pre_twiddle).expand(Y.shape), dim=-1
        )
        y = (
            torch.real(y * view_as_complex(self.post_twiddle).expand(y.shape))
            * np.sqrt(N)
            * np.sqrt(2)
        )
        result = y * self.window.expand(y.shape)
        output_size = (1, (L + 1) * N)
        audio = torch.nn.functional.fold(
            result.transpose(1, 2),
            output_size=output_size,
            kernel_size=(1, self.frame_len),
            stride=(1, self.frame_len // 2),
        )[:, 0, 0, :]

        if self.padding == "center":
            pad = self.frame_len // 2
        elif self.padding == "same":
            pad = self.frame_len // 4
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        audio = audio[:, pad:-pad]
        return audio.contiguous()


class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class ISTFTCosSinHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.
    This is the original one from the Vocos repository
    (only changed Linear to Conv1d)

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                        the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
        clip: Optional[float] = None,
        softclip=False,
        **kwargs,
    ):
        super().__init__()
        out_dim = n_fft + 2
        # self.out = torch.nn.Linear(dim, out_dim)
        self.out = torch.nn.Conv1d(dim, out_dim, kernel_size=1)

        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )
        self.n_fft = n_fft
        self.softclip = softclip
        self.clip = clip
        if self.clip is None:
            self.clip = self.n_fft * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        L is the sequence length, and C denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x)

        mag, p = x.chunk(2, dim=1)
        if self.softclip == "resig":
            # soft clip uses rectified sigmoid (easier 0) instead of exp+clip
            mag = torch.relu((torch.sigmoid(mag) - 0.00001) * self.clip)
        if self.softclip == "softplus":
            mag = torch.nn.functional.softplus(mag)
        elif self.softclip:
            # soft clip uses sigmoid instead of exp+clip
            mag = torch.sigmoid(mag) * self.clip
        else:
            mag = torch.exp(mag)

            # Can be useful for debugging
            # if (mag > self.n_fft * 2).any():
            #     print(f"CLIPPING ISSUE: {mag.max().item()}")

            mag = torch.clip(
                mag, max=self.clip
            )  # safeguard to prevent excessively large magnitudes

        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio


class ISTFTCircleHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.
    Changed phase prediction exp(j * p) = (x + j * y)
    from
    x, y = cos(p), sin(p)
    to
    x, y = x / (x^2 + y^2), y / (x^2 + y^2)

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
        conv_pad: str = "same",
        conv_kernel: int = 1,
        clip: Optional[float] = None,
        softclip=False,
        **kwargs,
    ):
        super().__init__()
        out_dim = ((n_fft + 2) * 3) // 2
        self.n_fft = n_fft
        self.out = torch.nn.Conv1d(
            dim, out_dim, kernel_size=conv_kernel, padding=conv_pad
        )
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )
        self.softclip = softclip
        self.clip = clip
        if self.clip is None:
            self.clip = self.n_fft * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        L is the sequence length, and C denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x)

        mag, x, y = x.chunk(3, dim=1)

        if self.softclip == "softplus":
            mag = torch.nn.functional.softplus(mag)
        elif self.softclip:
            # soft clip uses sigmoid instead of exp+clip
            mag = torch.sigmoid(mag) * self.clip
        else:
            mag = torch.exp(mag)
            # can be useful for debugging
            # if (mag > self.n_fft * 2).any():
            #     print(f"HIGH CLIPPING ISSUE: {mag.max().item()}")

            mag = torch.clip(
                mag, max=self.clip
            )  # safeguard to prevent excessively large magnitudes
            # wrapping happens here. These two lines produce real and imaginary value
            # We clip at 2 * n_fft
        p_mag = torch.sqrt((x**2 + y**2).clamp(1e-8, 10e2))

        # can be useful for debugging
        if (p_mag < 1e-4).any():
            print(f"PHASE LOW CLIPPING ISSUE: {p_mag.min().item()}")
        if (p_mag > 32).any():
            print(f"PHASE HIGH CLIPPING ISSUE: {p_mag.max().item()}")

        x = x / p_mag
        y = y / p_mag

        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio


class ISTFTCircleHeadV2(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.
    Changed phase prediction exp(j * p) = (x + j * y)
    from
    x, y = cos(p), sin(p)
    to
    x, y = x / (x^2 + y^2), y / (x^2 + y^2)

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
        conv_pad: str = "same",
        conv_kernel: int = 1,
        clip: Optional[float] = None,
        softclip=False,
        **kwargs,
    ):
        super().__init__()
        out_dim = ((n_fft + 2) * 3) // 2
        self.n_fft = n_fft
        self.out = torch.nn.Conv1d(
            dim, out_dim, kernel_size=conv_kernel, padding=conv_pad
        )
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )
        self.softclip = softclip
        self.clip = clip
        if self.clip is None:
            self.clip = self.n_fft * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        L is the sequence length, and C denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x)

        mag, x, y = x.chunk(3, dim=1)

        if self.softclip == "softplus":
            mag = torch.nn.functional.softplus(mag)
        elif self.softclip:
            # soft clip uses sigmoid instead of exp+clip
            mag = torch.sigmoid(mag) * self.clip
        else:
            mag = torch.exp(mag)
            # can be useful for debugging
            # if (mag > self.n_fft * 2).any():
            #     print(f"HIGH CLIPPING ISSUE: {mag.max().item()}")

            mag = torch.clip(
                mag, max=self.clip
            )  # safeguard to prevent excessively large magnitudes
            # wrapping happens here. These two lines produce real and imaginary value
            # We clip at 2 * n_fft

        p = x.clamp(min=1e-8) + 1j * y.clamp(min=1e-8)
        p = p / p.abs().clamp(min=1e-8)
        S = mag * p
        audio = self.istft(S)
        return audio


class ISTFTUnormalizedCircleHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.
    Changed phase prediction exp(j * p) = (x + j * y)
    from
    x, y = cos(p), sin(p)
    to
    x, y = x / (x^2 + y^2), y / (x^2 + y^2)

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
        conv_pad: str = "same",
        conv_kernel: int = 1,
        clip: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        out_dim = n_fft + 2
        self.n_fft = n_fft
        self.out = torch.nn.Conv1d(
            dim, out_dim, kernel_size=conv_kernel, padding=conv_pad
        )
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )
        self.clip = clip
        if self.clip is None:
            self.clip = self.n_fft * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        L is the sequence length, and C denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x)

        x, y = x.chunk(2, dim=1)

        # can be useful for debugging
        # if (mag > self.n_fft * 2).any():
        #     print(f"HIGH CLIPPING ISSUE: {mag.max().item()}")

        S = x + 1j * y
        if self.clip > 0:  # type: ignore
            S = S.sgn() * S.abs().clamp(max=self.clip)
        audio = self.istft(S)

        return audio


class IMDCTSymExpHead(FourierHead):
    """
    IMDCT Head module for predicting MDCT coefficients with symmetric exponential function

    Args:
        dim (int): Hidden dimension of the model.
        mdct_frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
        sample_rate (int, optional): The sample rate of the audio. If provided, the last layer will be initialized
                                    based on perceptual scaling. Defaults to None.
        clip_audio (bool, optional): Whether to clip the audio output within the range of [-1.0, 1.0]. Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        mdct_frame_len: int,
        padding: str = "same",
        sample_rate: Optional[int] = None,
        clip_audio: bool = False,
        **kwargs,
    ):
        super().__init__()
        out_dim = mdct_frame_len // 2
        self.out = torch.nn.Conv1d(dim, out_dim, kernel_size=1)
        self.imdct = IMDCT(frame_len=mdct_frame_len, padding=padding)
        self.clip_audio = clip_audio
        self.mdct_frame_len = mdct_frame_len

        if sample_rate is not None:
            # optionally init the last layer following mel-scale
            m_max = _hz_to_mel(sample_rate // 2)
            m_pts = torch.linspace(0, m_max, out_dim)
            f_pts = _mel_to_hz(m_pts)
            scale = 1 - (f_pts / f_pts.max())

            with torch.no_grad():
                self.out.weight.mul_(scale.view(-1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the IMDCTSymExpHead module.

        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x)
        x = symexp(x)
        x = torch.clip(
            x, min=-2 * self.mdct_frame_len, max=2 * self.mdct_frame_len
        )  # safeguard to prevent excessively large magnitudes
        x = x.transpose(1, 2)
        audio = self.imdct(x)
        if self.clip_audio:
            audio = torch.clip(x, min=-1.0, max=1.0)

        return audio


class IMDCTCosHead(FourierHead):
    """
    IMDCT Head module for predicting MDCT coefficients with parametrizing MDCT = exp(m) · cos(p)
    This is the original one from the Vocos repository
    (only changed Linear to Conv1d)
    Args:
        dim (int): Hidden dimension of the model.
        mdct_frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
        clip_audio (bool, optional): Whether to clip the audio output within the range of [-1.0, 1.0]. Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        mdct_frame_len: int,
        padding: str = "same",
        clip_audio: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.clip_audio = clip_audio
        self.out = torch.nn.Conv1d(dim, mdct_frame_len, kernel_size=1)
        self.imdct = IMDCT(frame_len=mdct_frame_len, padding=padding)
        self.mdct_frame_len = mdct_frame_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the IMDCTCosHead module.

        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        L is the sequence length, and C denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x)
        x = x.transpose(1, 2).contiguous()
        m, p = x.chunk(2, dim=2)
        m = torch.exp(m).clip(
            max=2 * self.mdct_frame_len
        )  # safeguard to prevent excessively large magnitudes
        audio = self.imdct(m * torch.cos(p))
        if self.clip_audio:
            audio = torch.clip(x, min=-1.0, max=1.0)
        return audio


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(
        self,
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )
        self.num_features_out = n_mels

    def forward(self, audio, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features.squeeze(1)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
            # self.norm = nn.Identity()
            # self.norm = EMANormalization(dim)

        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.shift = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale + shift
        return x


class ResBlock1(nn.Module):
    """
    ResBlock adapted from HiFi-GAN V1 (https://github.com/jik876/hifi-gan) with dilated 1D convolutions,
    but without upsampling layers.

    Args:
        dim (int): Number of input channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        dilation (tuple[int], optional): Dilation factors for the dilated convolutions.
            Defaults to (1, 3, 5).
        lrelu_slope (float, optional): Negative slope of the LeakyReLU activation function.
            Defaults to 0.1.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=self.get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=self.get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=self.get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
            ]
        )

        self.gamma = nn.ParameterList(
            [
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
            xt = torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, negative_slope=self.lrelu_slope)
            xt = c2(xt)
            if gamma is not None:
                xt = gamma * xt
            x = xt + x
        return x

    def remove_weight_norm(self):
        # for l in self.convs1:
        #     remove_weight_norm(l)
        # for l in self.convs2:
        #     remove_weight_norm(l)
        pass

    @staticmethod
    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)


class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, H, L), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
        input_layer_norm: bool = True,
        final_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
            if not input_layer_norm:
                raise NotImplementedError
        else:
            self.norm = (
                nn.LayerNorm(dim, eps=1e-6) if input_layer_norm else nn.Identity()
            )

        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = (
            nn.LayerNorm(dim, eps=1e-6) if final_layer_norm else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get("bandwidth_id", None)
        with torch.autocast("cuda", enabled=False):
            x = self.embed(x.float())
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self,
        input_channels,
        dim,
        num_blocks,
        layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = weight_norm(
            nn.Conv1d(input_channels, dim, kernel_size=3, padding=1)
        )
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(
            *[
                ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            x = self.embed(x.float())
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x


class ZeroDropoutTransform(nn.Module):
    def __init__(
        self,
        size,
        p=1.0,
    ):
        """Applies random zero masking during training on the 3D input over 2nd dim.
            excluding the first (p*size) items

        Args:
            size (int): the channels size
            p (float, optional): The minmum number of items to keep. Defaults to 1.0.
        """
        super().__init__()
        self.p = p
        assert p <= 1.0 and p >= 0.0
        self.size = size

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            assert C == self.size
            start = int(self.size * self.p)
            dropout_cut = start + torch.randint(0, self.size - start + 1, (1,)).item()
            mask = torch.arange(self.size, device=x.device) < dropout_cut
            mask = mask.type(x.dtype).view(1, self.size, 1).expand_as(x)
            x = x * mask
        return x


class ParamDropoutTransform(nn.Module):
    def __init__(
        self,
        size,
        p=1.0,
    ):
        """Applies random zero masking during training on the 3D input over 2nd dim.
            excluding the first (p*size) items

        Args:
            size (int): the channels size
            p (float, optional): The minmum number of items to keep. Defaults to 1.0.
        """
        super().__init__()
        self.p = p
        assert p <= 1.0 and p >= 0.0
        self.size = size
        self.learned_const = torch.nn.Parameter(torch.zeros(1, size, 1))
        self.learned_const.requires_grad = True
        log.info(
            f"ParamDropoutTransform: Created parameter with shape={self.learned_const.shape}"
        )

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            assert C == self.size
            start = int(self.size * self.p)
            dropout_cut = start + torch.randint(0, self.size - start + 1, (1,)).item()
            mask1 = torch.arange(self.size, device=x.device) < dropout_cut
            mask2 = torch.arange(self.size, device=x.device) >= dropout_cut
            mask1 = mask1.type(x.dtype).view(1, self.size, 1).expand_as(x)
            const_embed = (
                mask2.type(x.dtype).view(1, self.size, 1) * self.learned_const
            ).expand_as(x)
            x = x * mask1 + const_embed
        return x


class VocosEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        intermediate_dim=1536,
        output_channels=64,
        num_layers=8,
        n_fft=1024,
        hop_length=256,
        input_layer_norm: bool = True,
        final_layer_norm: bool = True,
        ztransform=ZeroDropoutTransform,
        stft_normalized=False,
        spec_embed: str = "stft-complex",
        n_mels: int = 100,
        sample_rate: int = 44100,
    ):
        super().__init__()
        assert spec_embed in [
            "stft-complex",
            "stft-magnitude",
            "stft-gain-shape",
            "mel",
        ]

        if "stft" in spec_embed:
            repr_type = spec_embed.replace("stft-", "")
            self.spec_embed = STFTEmbedding(
                n_fft=n_fft,
                hop_length=hop_length,
                repr_type=repr_type,
                stft_normalized=stft_normalized,
            )

        elif spec_embed == "mel":
            self.spec_embed = MelSpectrogramFeatures(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                padding="center",
            )

        self.backbone = VocosBackbone(
            input_channels=self.spec_embed.num_features_out,
            intermediate_dim=intermediate_dim,
            dim=d_model,
            num_layers=num_layers,
            input_layer_norm=input_layer_norm,
            final_layer_norm=final_layer_norm,
        )

        self.proj = nn.Conv1d(d_model, output_channels, kernel_size=3, padding=1)

        self.bn_transformer = ztransform(size=output_channels)  # bottleneck transform
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.epsilon = 1e-8

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            y = self.spec_embed(x.float())

        y = self.backbone(y)

        with torch.autocast("cuda", enabled=False):
            y = self.proj(y.float())
            y = self.bn_transformer(y)
        return y


class VocosDecoder(nn.Module):
    def __init__(
        self,
        input_channels,
        d_model=512,
        intermediate_dim=1536,
        num_layers=8,
        n_fft=1024,
        hop_length=256,
        input_layer_norm: bool = True,
        final_layer_norm: bool = True,
        istft_head: Optional[FourierHead] = None,
    ):
        super().__init__()
        self.backbone = VocosBackbone(
            input_channels=input_channels,
            dim=d_model,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            input_layer_norm=input_layer_norm,
            final_layer_norm=final_layer_norm,
        )

        if istft_head is None:
            self.head = ISTFTCircleHead(
                d_model, n_fft=n_fft, hop_length=hop_length, padding="center"
            )
        else:
            # istft is a _partial_
            self.head = istft_head(d_model, n_fft=n_fft, hop_length=hop_length)

        self.n_fft = n_fft
        self.epsilon = 1e-8

    def forward(self, x):
        y = self.backbone(x)
        with torch.autocast("cuda", enabled=False):
            y = self.head(y.float())
        return y.unsqueeze(1)


class VocosAutoEncoder(AutoEncoder):
    def __init__(
        self,
        channels: int = 1,
        z_dim: int = 64,
        d_model: int = 1024,
        intermediate_dim: int = 1536,
        n_fft: int = 1024,
        hop_length: int = 256,
        num_layers: int = 8,
        enc_num_layers: Optional[int] = None,
        input_layer_norm: bool = True,
        final_layer_norm: bool = True,
        istft_head: Optional[FourierHead] = None,
        ztransform=ZeroDropoutTransform,
        stft_normalized=False,
        spec_embed: str = "stft-complex",
        n_mels: int = 100,
        sample_rate: int = 44100,
    ) -> None:
        """AutoEncoder using Vocos_like encoders and decoders

        Args:
            channels (int, optional): number of input channels.
            Defaults to 1.
            encoder_dim (int, optional): base dim for encoder.
            Defaults to 64.
            encoder_rates (List[int], optional): list of downscaling rates.
            Defaults to [2, 4, 8, 8].
            z_dim (Optional[int], optional): list of upscaling rates.. Defaults to None.
            decoder_dim (int, optional): base dim for decoder (max dim at the beginning, right after latent dim) Defaults to 1536.
            decoder_rates (List[int], optional): _description_. Defaults to [8, 8, 4, 2].
        """
        assert channels == 1
        self.hop_length = hop_length
        if enc_num_layers is None:
            enc_num_layers = num_layers
        encoder = VocosEncoder(
            d_model=d_model,
            output_channels=z_dim,
            num_layers=enc_num_layers,
            intermediate_dim=intermediate_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            input_layer_norm=input_layer_norm,
            final_layer_norm=final_layer_norm,
            ztransform=ztransform,
            stft_normalized=stft_normalized,
            spec_embed=spec_embed,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )

        decoder = VocosDecoder(
            input_channels=z_dim,
            d_model=d_model,
            num_layers=num_layers,
            intermediate_dim=intermediate_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            istft_head=istft_head,
            input_layer_norm=input_layer_norm,
            final_layer_norm=final_layer_norm,
        )

        super().__init__(encoder=encoder, decoder=decoder)
        if rank == 0:
            log.info(
                f"""Using VocosAutoEncoder:
                    time_downscaling: {hop_length}
                    num_channels_in: {channels}
                    z_dim: {z_dim}
                    Total compression factor {hop_length / z_dim}                     """
            )

    def fix_input_length(self, x):
        """
        in vocos, the input samples should be a multiple of hopsize
        """
        assert len(x.shape) == 3, "VocosAutoEncoder expect input of the shape B,C,T"
        x = x[
            :, :, : x.shape[2] - (x.shape[2] % self.hop_length)
        ]  # make sure we can divide by hopsize, for the waveform loss
        return x.contiguous()


class DACVocosAutoEncoder(AutoEncoder):
    def __init__(
        self,
        channels: int = 1,
        z_dim: int = 64,
        d_model: int = 1024,
        intermediate_dim: int = 1536,
        n_fft: int = 1024,
        num_layers: int = 8,
        encoder_dim: int = 64,
        encoder_rates: list = [2, 4, 8, 8],
        bias: bool = True,
        use_fourier_features: bool = False,
        istft_head: Optional[FourierHead] = None,
    ) -> None:
        """AutoEncoder using DAC encoder and Vocos_like decoders

        Args:
            channels (int, optional): number of input channels.
            Defaults to 1.
            encoder_dim (int, optional): base dim for encoder.
            Defaults to 64.
            encoder_rates (List[int], optional): list of downscaling rates.
            Defaults to [2, 4, 8, 8].
            z_dim (Optional[int], optional): list of upscaling rates.. Defaults to None.
            decoder_dim (int, optional): base dim for decoder (max dim at the beginning, right after latent dim) Defaults to 1536.
            decoder_rates (List[int], optional): _description_. Defaults to [8, 8, 4, 2].
        """
        assert channels == 1
        self.hop_length = np.prod(encoder_rates)

        encoder = DACEncoder(
            d_model=encoder_dim,
            strides=encoder_rates,
            d_latent=z_dim,
            bias=bias,
            use_fourier_features=use_fourier_features,
        )

        decoder = VocosDecoder(
            input_channels=z_dim,
            d_model=d_model,
            num_layers=num_layers,
            intermediate_dim=intermediate_dim,
            n_fft=n_fft,
            hop_length=self.hop_length,
            istft_head=istft_head,
        )

        super().__init__(encoder=encoder, decoder=decoder)
        if rank == 0:
            log.info(
                f"""Using VocosAutoEncoder:
                    time_downscaling: {hop_length}
                    num_channels_in: {channels}
                    z_dim: {z_dim}
                    Total compression factor {hop_length / z_dim}                     """
            )

    def fix_input_length(self, x):
        """
        in vocos, the input samples should be a multiple of hopsize
        """
        assert len(x.shape) == 3, "VocosAutoEncoder expect input of the shape B,C,T"
        x = x[
            :, :, : x.shape[2] - (x.shape[2] % self.hop_length)
        ]  # make sure we can divide by hopsize, for the waveform loss
        return x.contiguous()


class VocosVariationalAutoEncoder(VariationalAutoEncoder):
    def __init__(
        self,
        channels: int = 1,
        z_dim: int = 64,
        d_model: int = 1024,
        intermediate_dim: int = 1536,
        n_fft: int = 1024,
        hop_length: int = 256,
        num_layers: int = 8,
        istft_head: Optional[FourierHead] = None,
    ) -> None:
        """AutoEncoder using Vocos_like encoders and decoders

        Args:
            channels (int, optional): number of input channels.
            Defaults to 1.
            encoder_dim (int, optional): base dim for encoder.
            Defaults to 64.
            encoder_rates (List[int], optional): list of downscaling rates.
            Defaults to [2, 4, 8, 8].
            z_dim (Optional[int], optional): list of upscaling rates.. Defaults to None.
            decoder_dim (int, optional): base dim for decoder (max dim at the beginning, right after latent dim) Defaults to 1536.
            decoder_rates (List[int], optional): _description_. Defaults to [8, 8, 4, 2].
        """
        assert channels == 1

        encoder = VocosEncoder(
            d_model=d_model,
            output_channels=2 * z_dim,
            num_layers=num_layers,
            intermediate_dim=intermediate_dim,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        decoder = VocosDecoder(
            input_channels=z_dim,
            d_model=d_model,
            num_layers=num_layers,
            intermediate_dim=intermediate_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            istft_head=istft_head,
        )

        super().__init__(encoder=encoder, decoder=decoder)
        if rank == 0:
            log.info(
                f"""Using VocosAutoEncoder:
                    time_downscaling: {hop_length}
                    num_channels_in: {channels}
                    z_dim: {z_dim}
                    Total compression factor {hop_length / z_dim}                     """
            )

    def fix_input_length(self, x):
        """
        in vocos, the input samples should be a multiple of hopsize
        """
        assert len(x.shape) == 3, (
            "VocosVariationalAutoEncoder expect input of the shape B,C,T"
        )
        x = x[
            :, :, : x.shape[2] - (x.shape[2] % self.hop_length)
        ]  # make sure we can divide by hopsize, for the waveform loss
        return x.contiguous()


class VQVocosAutoEncoder(VQAutoEncoder):
    def __init__(
        self,
        channels: int = 1,
        z_dim: int = 64,
        d_model: int = 1024,
        intermediate_dim: int = 1536,
        n_fft: int = 1024,
        hop_length: int = 256,
        num_layers: int = 8,
        enc_num_layers: Optional[int] = None,
        input_layer_norm: bool = True,
        final_layer_norm: bool = True,
        istft_head: Optional[FourierHead] = None,
        ztransform=ZeroDropoutTransform,
        stft_normalized=False,
        spec_embed: str = "stft-complex",
        n_mels: int = 100,
        sample_rate: int = 44100,
        commitment_loss: float = 1.0,
        vq_layer_config: dict = {"num_quantizers": 10, "codebook_size": 1024},
    ) -> None:
        assert channels == 1
        self.hop_length = hop_length
        self.commitment_loss = commitment_loss
        if enc_num_layers is None:
            enc_num_layers = num_layers
        encoder = VocosEncoder(
            d_model=d_model,
            output_channels=z_dim,
            num_layers=enc_num_layers,
            intermediate_dim=intermediate_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            input_layer_norm=input_layer_norm,
            final_layer_norm=final_layer_norm,
            ztransform=ztransform,
            stft_normalized=stft_normalized,
            spec_embed=spec_embed,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        quantizer = ResidualVQ(dim=z_dim, **vq_layer_config)

        decoder = VocosDecoder(
            input_channels=z_dim,
            d_model=d_model,
            num_layers=num_layers,
            intermediate_dim=intermediate_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            istft_head=istft_head,
            input_layer_norm=input_layer_norm,
            final_layer_norm=final_layer_norm,
        )

        super().__init__(encoder=encoder, decoder=decoder, quantizer=quantizer)

        if rank == 0:
            log.info(
                f"""Using VQVocosAutoEncoder:
                    time_downscaling: {hop_length}
                    num_channels_in: {channels}
                    z_dim: {z_dim}
                    Total compression factor {hop_length / z_dim}                     """
            )

    def fix_input_length(self, x):
        """
        in vocos, the input samples should be a multiple of hopsize
        """
        assert len(x.shape) == 3, "VocosAutoEncoder expect input of the shape B,C,T"
        x = x[
            :, :, : x.shape[2] - (x.shape[2] % self.hop_length)
        ]  # make sure we can divide by hopsize, for the waveform loss
        return x.contiguous()
