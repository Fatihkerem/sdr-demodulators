"""ADS-B (Mode S) decoder for 1090 MHz aircraft transponder transmissions.

ADS-B is pulse-position modulated at 1 Mbit/s: each bit occupies 1 us and is
encoded by *where* the pulse sits inside that window — energy in the first
half means 1, energy in the second half means 0. Because the information is
in amplitude timing rather than phase, the receiver only needs sample
magnitudes.

Pipeline:

1. Take ``|IQ|`` — phase is irrelevant for PPM.
2. Resample to an integer multiple of the 1 Mbit/s symbol rate so each bit
   lands on a whole number of samples.
3. Threshold to a boolean pulse train.
4. Slide along looking for the fixed 8 us preamble.
5. Read the 5-bit downlink format (DF) field to learn the frame length.
6. Assemble 56- or 112-bit frames, convert to hex, and hand them to
   ``pyModeS`` for field extraction (ICAO address, typecode, callsign).

The detection threshold trades sensitivity against false decodes: lowering it
finds more distant aircraft but admits noise that fails the preamble test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pyModeS as pms
from scipy import signal

CAPTURE_RATE = 3_200_000
RESAMPLED_RATE = 4_000_000

# 8 us preamble: pulses at 0, 1.0, 3.5 and 4.5 us, sampled at 2 MHz.
PREAMBLE = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]

SHORT_FRAME_BITS = 112   # 56 data bits, PPM-encoded at 2 samples per bit
LONG_FRAME_BITS = 240    # 112 data bits (DF-17 extended squitter)


@dataclass
class DecodeResult:
    """Summary of one pass over a capture."""

    total_frames: int = 0
    df17_frames: int = 0
    callsigns: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"frames={self.total_frames} "
            f"df17={self.df17_frames} "
            f"callsigns={sorted(set(self.callsigns))}"
        )


def bits_to_hex(bits) -> str:
    """Pack a boolean bit sequence into a hex string for pyModeS."""
    return hex(int("".join("1" if b else "0" for b in bits), 2))[2:]


def _read_df(pulses) -> int:
    """Decode the 5-bit downlink format field from PPM pulse pairs."""
    df = 0
    weight = 4
    for i in range(0, 9, 2):
        if pulses[i] and not pulses[i + 1]:
            df += 1 << weight
        weight -= 1
    return df


def decode(
    path: str,
    threshold: float,
    capture_rate: int = CAPTURE_RATE,
    verbose: bool = False,
) -> DecodeResult:
    """Decode ADS-B frames from a raw RTL-SDR capture at 1090 MHz.

    Parameters
    ----------
    path:
        Raw ``.dat`` capture produced by ``rtl_sdr -f 1090000000``.
    threshold:
        Magnitude above which a sample counts as a pulse. Typical values are
        3-10; lower finds weaker aircraft but yields more failed preambles.
    capture_rate:
        Sample rate the file was recorded at.
    verbose:
        Print each decoded DF-17 frame as it is found.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    iq = (raw[::2] + 1j * raw[1::2]) - (127 + 1j * 127)

    magnitude = np.abs(iq)
    n_resampled = int(len(magnitude) * RESAMPLED_RATE / capture_rate)
    resampled = signal.resample(magnitude, n_resampled)

    # Two samples per PPM half-bit -> decimate by 2 to one sample per half-bit.
    pulses = (resampled > threshold)[::2]

    result = DecodeResult()

    while len(pulses) > LONG_FRAME_BITS:
        candidates = np.flatnonzero(pulses)
        if candidates.size == 0:
            break
        start = candidates[0]

        window = pulses[start : start + len(PREAMBLE)]
        if len(window) < len(PREAMBLE):
            break

        if not np.array_equal(window, PREAMBLE):
            pulses = pulses[start + 1 :]
            continue

        df = _read_df(pulses[start + 16 : start + 26])

        if df in (0, 4, 5, 11):
            frame_bits = SHORT_FRAME_BITS
        elif df in (16, 17, 18, 19, 20, 21, 24):
            frame_bits = LONG_FRAME_BITS
        else:
            pulses = pulses[start + SHORT_FRAME_BITS :]
            continue

        if len(pulses[start:]) < frame_bits:
            break

        message = bits_to_hex(pulses[start + 16 : start + frame_bits])
        result.total_frames += 1
        result.messages.append(message)

        if df == 17:
            result.df17_frames += 1
            try:
                callsign = pms.adsb.callsign(message)
                if callsign:
                    result.callsigns.append(callsign.strip("_"))
                if verbose:
                    print(
                        f"DF={pms.df(message)} "
                        f"ICAO={pms.adsb.icao(message)} "
                        f"TC={pms.adsb.typecode(message)} "
                        f"callsign={callsign}"
                    )
            except Exception as exc:  # malformed frame; keep scanning
                if verbose:
                    print(f"undecodable frame {message}: {exc}")

        pulses = pulses[start + frame_bits :]

    return result
