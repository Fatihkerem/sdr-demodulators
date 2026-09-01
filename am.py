"""Amplitude modulation receiver operating on raw IQ samples.

The chain is: load IQ -> mix the wanted station to baseband -> decimate to
audio rate through an anti-alias filter -> envelope detect -> normalise.

Envelope detection is simply ``|x|``: for AM the message rides on the
amplitude of the carrier, so taking the magnitude of the complex baseband
signal recovers it directly, with no phase tracking needed.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from iqio import load_rtlsdr, load_usrp, mix_to_baseband

RTLSDR_RATE = 2_048_000
USRP_RATE = 256_000
AUDIO_RATE = 48_000


def demodulate(
    iq: np.ndarray,
    sample_rate: float,
    freq_shift: float = 0.0,
    decimation: int = 64,
    audio_rate: int = AUDIO_RATE,
) -> np.ndarray:
    """Recover the audio message from a complex baseband AM signal.

    Parameters
    ----------
    iq:
        Complex baseband samples.
    sample_rate:
        Sample rate of ``iq`` in Hz.
    freq_shift:
        Offset of the wanted station from the tuned centre frequency, in Hz.
        Pass 0 for the station at the centre.
    decimation:
        Integer decimation factor applied through an FIR anti-alias filter.
        Chosen so the surviving bandwidth comfortably covers the AM channel
        while dropping the rate close to audio.
    audio_rate:
        Output sample rate.

    Returns
    -------
    Real-valued audio in the range [-1, 1].
    """
    centred = mix_to_baseband(iq, sample_rate, freq_shift)
    decimated = signal.decimate(centred, decimation, ftype="fir")

    envelope = np.abs(decimated)
    envelope = envelope - np.mean(envelope)  # drop the carrier's DC term

    intermediate_rate = sample_rate / decimation
    n_out = int(len(envelope) * audio_rate / intermediate_rate)
    audio = signal.resample(envelope, n_out)

    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio


def from_rtlsdr(path: str, freq_shift: float = 0.0, **kwargs) -> np.ndarray:
    """Demodulate an AM station from an RTL-SDR capture."""
    return demodulate(load_rtlsdr(path), RTLSDR_RATE, freq_shift, **kwargs)


def from_usrp(path: str, freq_shift: float = 0.0, decimation: int = 10, **kwargs) -> np.ndarray:
    """Demodulate an AM station from a USRP capture."""
    return demodulate(load_usrp(path), USRP_RATE, freq_shift, decimation=decimation, **kwargs)


def strongest_carriers(iq: np.ndarray, sample_rate: float, count: int = 3) -> np.ndarray:
    """Return the frequencies of the ``count`` strongest carriers in a capture.

    Uses Welch's method to estimate the power spectral density, then reports
    the frequency bins with the most power. This is how the other stations
    present in a single recording were located before demodulating them.
    """
    freqs, psd = signal.welch(iq, fs=sample_rate, return_onesided=False)
    order = np.argsort(psd)[::-1]
    return freqs[order[:count]]
