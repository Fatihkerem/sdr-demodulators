"""Loading raw IQ recordings from SDR hardware.

Two on-disk formats are supported, because the two receivers used in this
project store samples differently:

* **RTL-SDR** writes interleaved unsigned 8-bit samples with a DC bias of 127
  (or 128, depending on the tuner). Values must be re-centred on zero before
  the stream can be treated as a complex baseband signal.
* **USRP** writes interleaved 32-bit floats that are already zero-centred.
"""

from __future__ import annotations

import numpy as np


def load_rtlsdr(path: str, dc_offset: float = 128.0) -> np.ndarray:
    """Read an RTL-SDR ``.dat`` capture as complex baseband samples.

    The file is a flat stream of ``uint8`` values ordered I, Q, I, Q, ...
    Interleaving is undone by slicing, then the DC bias introduced by the
    receiver's ADC is removed so the constellation is centred on the origin.

    Parameters
    ----------
    path:
        Path to the raw capture produced by ``rtl_sdr``.
    dc_offset:
        Bias to subtract from both branches. 128 matches most captures;
        127 is correct for some tuners (the ADS-B captures here used 127).

    Returns
    -------
    Complex-valued array of baseband samples.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    iq = raw[::2] + 1j * raw[1::2]
    return iq - (dc_offset + 1j * dc_offset)


def load_usrp(path: str) -> np.ndarray:
    """Read a USRP ``.dat`` capture as complex baseband samples.

    Samples are interleaved ``float32`` and already centred on zero, so no
    DC correction is applied.
    """
    raw = np.fromfile(path, dtype=np.float32)
    return raw[::2] + 1j * raw[1::2]


def duration_seconds(iq: np.ndarray, sample_rate: float) -> float:
    """Length of a capture in seconds."""
    return len(iq) / sample_rate


def mix_to_baseband(iq: np.ndarray, sample_rate: float, freq_shift: float) -> np.ndarray:
    """Shift a station sitting at ``freq_shift`` Hz down to 0 Hz.

    Multiplying by a complex exponential rotates the whole spectrum, so a
    carrier at ``+freq_shift`` ends up at DC where the demodulators expect it.
    The time vector is derived from the actual sample count rather than being
    hard-coded to a fixed recording length.
    """
    n = np.arange(len(iq))
    return iq * np.exp(-1j * 2 * np.pi * freq_shift * n / sample_rate)
