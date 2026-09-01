"""Wideband FM receiver using quadrature (differentiator) demodulation.

For FM the message is carried in the *instantaneous frequency*, which is the
derivative of the signal's phase. Rather than unwrapping the phase and
differencing it — numerically fragile near wrap points — this implementation
uses the standard quadrature identity:

    m(t)  proportional to  Im{ x'(t) * conj(x(t)) }

The signal is first hard-limited (divided by its own magnitude) so that any
residual amplitude variation cannot leak into the recovered audio, then
differentiated by a linear-phase FIR filter designed with the Parks-McClellan
(Remez) algorithm.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from iqio import load_rtlsdr, mix_to_baseband

RTLSDR_RATE = 2_048_000
AUDIO_RATE = 48_000

# Rate after the first decimation stage; wide enough for the ~200 kHz
# occupied bandwidth of a broadcast FM channel.
IF_RATE = 256_000


def design_differentiator(
    num_taps: int = 31,
    if_rate: float = IF_RATE,
    passband_edge: float = 105_000.0,
    stopband_edge: float = 120_000.0,
) -> np.ndarray:
    """Design the FIR differentiator used for quadrature demodulation.

    A differentiator has a magnitude response proportional to frequency. The
    Remez exchange algorithm is asked for that response across the passband
    and for rejection above it, giving a linear-phase filter that
    differentiates the baseband signal without adding phase distortion.
    """
    return signal.remez(
        num_taps,
        [0.0, passband_edge, stopband_edge, if_rate / 2],
        [passband_edge / stopband_edge, 0.0],
        fs=if_rate,
        type="differentiator",
    )


def demodulate(
    iq: np.ndarray,
    sample_rate: float = RTLSDR_RATE,
    freq_shift: float = 0.0,
    audio_rate: int = AUDIO_RATE,
) -> np.ndarray:
    """Recover mono audio from a complex baseband WBFM signal.

    Parameters
    ----------
    iq:
        Complex baseband samples.
    sample_rate:
        Sample rate of ``iq`` in Hz.
    freq_shift:
        Offset of the wanted station from the tuned centre, in Hz. Broadcast
        stations in the captures used here sit at 0 and at +/-400 kHz and
        +/-800 kHz from centre.
    audio_rate:
        Output sample rate.

    Returns
    -------
    Real-valued audio in the range [-1, 1].
    """
    centred = mix_to_baseband(iq, sample_rate, freq_shift)

    first_decimation = int(round(sample_rate / IF_RATE))
    at_if = signal.decimate(centred, first_decimation, ftype="fir")

    # Hard limiter: strip amplitude, keep phase.
    limited = at_if / (np.abs(at_if) + 1e-6)

    h_diff = design_differentiator(if_rate=sample_rate / first_decimation)
    differentiated = np.convolve(limited, h_diff, mode="same")
    message = (differentiated * np.conj(limited)).imag

    n_out = int(len(message) * audio_rate / (sample_rate / first_decimation))
    audio = signal.resample(message, n_out)

    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio


def from_rtlsdr(path: str, freq_shift: float = 0.0, **kwargs) -> np.ndarray:
    """Demodulate a WBFM station from an RTL-SDR capture."""
    return demodulate(load_rtlsdr(path), RTLSDR_RATE, freq_shift, **kwargs)
