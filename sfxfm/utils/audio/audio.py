""" Audio utils """

import torch
import torchaudio
from typing import List
import warnings
import logging

log = logging.getLogger()


def find_max_mel_bands(
    n_ffts: List[int],
    sample_rate: float,
    mel_scale: str = "htk",
) -> List[int]:
    """
    Find maximum values for n_mel for each FFT size in n_ffts list.
    These maximum values are those that make every mel filterbank band
    have non-zero energy, resulting in non-zero spectrogram bins.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # make sure to always use a list of n_fft
        if not isinstance(n_ffts, list):
            n_ffts_fixed = [n_ffts]
        else:
            n_ffts_fixed = n_ffts

        n_mels = []
        for n_fft in n_ffts_fixed:

            # absolute search bounds
            n_mel_min = 1
            n_mel_max = n_fft // 2
            # do binary search
            while True:

                # test middle way n_mel
                n_mel = (n_mel_min + n_mel_max) // 2

                # generate mel_scale for current n_fft and n_mel
                mel_filterbank = torchaudio.functional.melscale_fbanks(
                    n_freqs=n_fft // 2 + 1,
                    f_min=0,
                    f_max=sample_rate // 2,
                    n_mels=n_mel,
                    sample_rate=sample_rate,
                    mel_scale=mel_scale,
                )

                # check if at least one band has zero energy
                fbank_ok = ((mel_filterbank - 0.0).abs().sum(dim=0) == 0.0).sum() == 0
                if fbank_ok:
                    if n_mel_max == n_mel_min + 1:
                        n_mels.append(n_mel_min)
                        break
                    n_mel_min = n_mel
                else:
                    n_mel_max = n_mel

        # return a list or integer depending on given input
        if isinstance(n_ffts, list):
            return n_mels
        else:
            return n_mels[0]


def find_max_bark_bands(
    n_ffts: List[int],
    sample_rate: float,
    bark_scale: str = "traunmuller",
) -> List[int]:
    """
    Find maximum values for n_mel for each FFT size in n_ffts list.
    These maximum values are those that make every mel filterbank band
    have non-zero energy, resulting in non-zero spectrogram bins.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # make sure to always use a list of n_fft
        if not isinstance(n_ffts, list):
            n_ffts_fixed = [n_ffts]
        else:
            n_ffts_fixed = n_ffts

        n_barks = []
        for n_fft in n_ffts_fixed:

            # absolute search bounds
            n_bark_min = 1
            n_bark_max = n_fft // 2
            # do binary search
            while True:

                # test middle way n_mel
                n_bark = (n_bark_min + n_bark_max) // 2

                # generate mel_scale for current n_fft and n_mel
                bark_filterbank = barkscale_fbanks(
                    n_freqs=n_fft // 2 + 1,
                    f_min=0,
                    f_max=sample_rate // 2,
                    n_barks=n_bark,
                    sample_rate=sample_rate,
                    bark_scale=bark_scale,
                )

                # check if at least one band has zero energy
                fbank_ok = ((bark_filterbank - 0.0).abs().sum(dim=0) == 0.0).sum() == 0
                if fbank_ok:
                    if n_bark_max == n_bark_min + 1:
                        n_barks.append(n_bark_min)
                        break
                    n_bark_min = n_bark
                else:
                    n_bark_max = n_bark

        # return a list or integer depending on given input
        if isinstance(n_ffts, list):
            return n_barks
        else:
            return n_barks[0]


#
# barkscale_fbanks is taken from a pytorch experimental branch
#
def barkscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_barks: int,
    sample_rate: int,
    bark_scale: str = "traunmuller",
) -> torch.Tensor:
    r"""Create a frequency bin conversion matrix.

    .. devices:: CPU

    .. properties:: TorchScript

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_barks (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_barks``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * barkscale_fbanks(A.size(-1), ...)``.

    """

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate bark freq bins
    m_min = _hz_to_bark(f_min, bark_scale=bark_scale)
    m_max = _hz_to_bark(f_max, bark_scale=bark_scale)

    m_pts = torch.linspace(m_min, m_max, n_barks + 2)
    f_pts = _bark_to_hz(m_pts, bark_scale=bark_scale)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one bark filterbank has all zero values. "
            f"The value for `n_barks` ({n_barks}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb


def _bark_to_hz(barks: torch.Tensor, bark_scale: str = "traunmuller") -> torch.Tensor:
    """Convert bark bin numbers to frequencies.

    Args:
        barks (Tensor): Bark frequencies
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        freqs (Tensor): Barks converted in Hz
    """

    if bark_scale not in ["schroeder", "traunmuller", "wang"]:
        raise ValueError(
            'bark_scale should be one of "traunmuller", "schroeder" or "wang".'
        )

    if bark_scale == "wang":
        return 600.0 * torch.sinh(barks / 6.0)
    elif bark_scale == "schroeder":
        return 650.0 * torch.sinh(barks / 7.0)
    # Bark value correction
    if any(barks < 2):
        idx = barks < 2
        barks[idx] = (barks[idx] - 0.3) / 0.85
    elif any(barks > 20.1):
        idx = barks > 20.1
        barks[idx] = (barks[idx] + 4.422) / 1.22

    # Traunmuller Bark scale
    freqs = 1960 * ((barks + 0.53) / (26.28 - barks))

    return freqs


def _hz_to_bark(freqs: float, bark_scale: str = "traunmuller") -> float:
    r"""Convert Hz to Barks.

    Args:
        freqs (float): Frequencies in Hz
        bark_scale (str, optional): Scale to use: ``traunmuller``, ``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        barks (float): Frequency in Barks
    """

    if bark_scale not in ["schroeder", "traunmuller", "wang"]:
        raise ValueError(
            'bark_scale should be one of "schroeder", "traunmuller" or "wang".'
        )

    if bark_scale == "wang":
        return 6.0 * math.asinh(freqs / 600.0)
    elif bark_scale == "schroeder":
        return 7.0 * math.asinh(freqs / 650.0)
    # Traunmuller Bark scale
    barks = ((26.81 * freqs) / (1960.0 + freqs)) - 0.53
    # Bark value correction
    if barks < 2:
        barks += 0.15 * (2 - barks)
    elif barks > 20.1:
        barks += 0.22 * (barks - 20.1)

    return barks


def _create_triangular_filterbank(
    all_freqs: torch.Tensor,
    f_pts: torch.Tensor,
) -> torch.Tensor:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb


def strip_silence(
    audio: torch.Tensor,
    sample_rate: int = 44100,
    rate: int = 10,
    threshold: float = -60.0,
    normalize: bool = False,
    output: str = "audio",
):
    """
    Strip head and tail silence in audio file being very conservative, i.e.
    taking max over chunks of audio at a given rate, then threshold.
    """

    samples_per_chunk = int(sample_rate / rate)
    # chunk audio energy
    if normalize:
        max_rms = torch.sqrt((audio[0, :] * audio[0, :]).mean())

    block_size = 10 * samples_per_chunk
    # find start sample
    start_sample_strip = None
    for start_sample in range(0, audio.size(-1), block_size):
        audio_block = audio[0, start_sample : start_sample + block_size]
        sq_chunks = torch.chunk(
            audio_block * audio_block,
            int((audio_block.size(-1) + samples_per_chunk - 1) / samples_per_chunk),
        )
        # sqrt(max(energy)) for each chunk
        # rms = torch.tensor([ torch.sqrt(c.max()) for c in sq_chunks ])
        rms = torch.tensor([torch.sqrt(c.topk(10)[0].mean()) for c in sq_chunks])
        if normalize:
            rms_dbfs = 20.0 * torch.log10(rms / (max_rms + 1e-8))
        else:
            rms_dbfs = 20.0 * torch.log10(rms)
        # threshold to find first and last non-silence endpoints
        vad = rms_dbfs > threshold

        # convert non-silence chunk indices to samples
        idxs = torch.nonzero(vad)
        if len(idxs) > 0:
            start_sample_strip = max(
                int(start_sample + (idxs[0].item() - 1) * samples_per_chunk), 0
            )
            break
    start_sample_strip = (
        start_sample_strip
        if start_sample_strip is not None and start_sample_strip > 0
        else 0
    )

    # find end sample
    end_sample_strip = None
    for end_sample in range(audio.size(-1) - 1, -1, -block_size):
        audio_block = audio[0, end_sample - block_size + 1 : end_sample + 1].flip(
            dims=(0,)
        )
        sq_chunks = torch.chunk(
            audio_block * audio_block,
            int((audio_block.size(-1) + samples_per_chunk - 1) / samples_per_chunk),
        )
        # sqrt(max(energy)) for each chunk
        # rms = torch.tensor([ torch.sqrt(c.max()) for c in sq_chunks ])
        rms = torch.tensor([torch.sqrt(c.topk(10)[0].mean()) for c in sq_chunks])
        if normalize:
            if max_rms > 0:
                rms_dbfs = 20.0 * torch.log10(rms / max_rms)
            else:
                rms_dbfs = 20.0 * torch.log10(rms)
        else:
            rms_dbfs = 20.0 * torch.log10(rms)
        # threshold to find first and last non-silence endpoints
        vad = rms_dbfs > threshold

        # convert non-silence chunk indices to samples
        idxs = torch.nonzero(vad)
        if len(idxs) > 0:
            end_sample_strip = min(
                int(end_sample - (idxs[0].item() - 1) * samples_per_chunk),
                audio.size(-1) - 1,
            )
            break
    end_sample_strip = (
        end_sample_strip
        if end_sample_strip is not None and end_sample_strip > 0
        else audio.size(-1) - 1
    )

    if end_sample_strip < start_sample_strip:
        # something went wrong, no stripping
        start_sample_strip = 0
        end_sample_strip = audio.size(-1) - 1

    if output == "audio":
        return audio[:, start_sample_strip : end_sample_strip + 1]
    elif output == "segment_sample":
        return start_sample_strip, end_sample_strip
    elif output == "segment_sec":
        return start_sample_strip / sample_rate, end_sample_strip / sample_rate
    else:
        return start_sample_strip / sample_rate, end_sample_strip / sample_rate


def threshold_hyst(x, th_lo, th_hi, initial=False):
    """Apply hysteresis thresholding with lo and hi thresholds"""
    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi
    ind = torch.nonzero(lo_or_hi, as_tuple=True)[0]
    if ind.size(0) == 0:  # prevent index error if ind is empty
        return torch.zeros_like(x, dtype=bool) | initial
    cnt = torch.cumsum(lo_or_hi, dim=0)  # from 0 to len(ind)
    return torch.where(cnt > 0, hi[ind[cnt - 1]], initial)


def segment(
    act: torch.Tensor,
    join_segments: bool = False,
    max_audio_len: int = 1e9,
    max_silence_len: int = 1e9,
):
    """
    Convert boolean activity tensor into a list of segments.

    If join_segments==True, segments will be joined if the
    silence gap is shorter than max_silence_len and the joined
    length (with silence gap included) is shorter than max_audio_len.

    `len` units are in act samples
    """

    # convert audio boolean activity to segments
    act_diff = torch.diff(act.type(torch.int))
    if not torch.any(act_diff > 0):
        # return single segment
        return [(0, act.size(-1))]

    start = torch.nonzero(act_diff > 0, as_tuple=True)[0].tolist()
    end = torch.nonzero(act_diff < 0, as_tuple=True)[0].tolist()

    # adjust start as it was based on act_diff, 1 sample less
    start = [n + 1 for n in start]
    # sort start and end times
    if len(start) > 0:
        if len(end) > 0:
            if end[0] < start[0]:
                start.insert(0, 0)
        else:
            # no end and one start
            end.append(act_diff.size(-1))

    # truncate start,end to be perfectly paired
    n = min(len(start), len(end))
    start = start[:n]
    end = end[:n]
    # store as list of (start,end) segments
    segs = list(zip(start, end))

    if join_segments:
        # join segments with gap under maximum silence length and
        # do not exceed maximum length after joining
        start = None
        segs_out = []
        seg_pending = False
        last_start = None
        last_end = None
        for s, e in segs:
            if start is None:
                start = s
                len_silence = None
            else:
                len_silence = s - end - 1
            end = e
            len_appended_audio = end - start + 1

            # append segment to existing segment ?
            if len_silence is not None:
                if len_silence <= max_silence_len:
                    if len_appended_audio < max_audio_len:
                        # append segment, it's within length limits
                        seg_pending = True
                    else:
                        # do not append segment, but use last end
                        if last_end is not None:
                            segs_out.append((start, last_end))
                            seg_pending = True
                            start = s
                else:
                    if last_start is not None and last_end is not None:
                        segs_out.append((last_start, last_end))
                        seg_pending = True
                        start = s
                        end = e
            else:
                seg_pending = True

            # update last segment end
            last_start = start
            last_end = end

        if seg_pending:
            segs_out.append((start, end))

        return segs_out

    return segs


def remove_silence(
    audio: torch.Tensor,
    sample_rate: int = 44100,
    rate: int = 10,
    threshold_lo: float = -30.0,
    threshold_hi: float = -10.0,
    normalize: bool = False,
    join_segments: bool = False,
    max_join_segment_time: float = 10.0,
    max_join_silence_time: float = 2.0,
):
    """
    Get a list of non-silence segments in an audio file
    """

    # compute VAD activity for chunks of lenght samples_per_chunk
    samples_per_chunk = int(sample_rate / rate)
    # chunk audio energy
    if normalize:
        max_rms = torch.sqrt((audio[0, :] * audio[0, :]).mean())

    # processing will be for 10 chunks per iteration
    block_size = 10 * samples_per_chunk
    # find start sample
    start_sample_strip = None
    initial = False
    vad = []
    n_blocks = 0
    for start_sample in range(0, audio.size(-1), block_size):
        # extract 1 block of audio
        audio_block = audio[0, start_sample : start_sample + block_size]
        # split into a list of chunks
        sq_chunks = torch.chunk(
            audio_block * audio_block,
            int((audio_block.size(-1) + samples_per_chunk - 1) / samples_per_chunk),
        )
        # sqrt(max(energy)) for each chunk
        # rms = torch.tensor([ torch.sqrt(c.max()) for c in sq_chunks ])
        rms = torch.tensor(
            [torch.sqrt(c.topk(min(len(c), 10))[0].mean()) for c in sq_chunks]
        )
        if normalize:
            rms_dbfs = 20.0 * torch.log10(rms / (max_rms + 1e-8))
        else:
            rms_dbfs = 20.0 * torch.log10(rms)

        # threshold to find audio activity
        vad_block = threshold_hyst(
            rms_dbfs,
            th_lo=threshold_lo,
            th_hi=threshold_hi,
            initial=initial,
        )

        # append vad for this block
        vad.append(vad_block)

        n_blocks += 1

        # initial value for next blocks hysteresis
        initial = vad_block[-1]

    vad = torch.cat(vad, dim=0)

    # convert to segments, potentially joining segments
    max_audio_len = int(max_join_segment_time * rate)
    max_silence_len = int(max_join_silence_time * rate)
    segs = segment(
        vad,
        join_segments=join_segments,
        max_audio_len=max_audio_len,
        max_silence_len=max_silence_len,
    )

    # convert time-stamps to time (seconds)
    #   start rounded to the left
    #   end rounded to the right
    segs = [
        (
            max((s - 1) / rate, 0),
            min((e + 1) / rate, audio.size(-1) / sample_rate),
        )
        for s, e in segs
    ]

    return segs
