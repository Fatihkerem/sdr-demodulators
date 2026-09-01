# SDR Demodulators

Receivers written from scratch in Python, operating directly on raw IQ samples
captured with an RTL-SDR — no GNU Radio blocks, no `rtl_fm`, no demodulation
library. Each one takes a file of interleaved ADC samples and returns audio or
decoded data.

Built for the Communication Systems Laboratory (EEE 361L) at TOBB University of
Economics and Technology.

| Module | Signal | What it recovers |
|---|---|---|
| [`am.py`](am.py) | Broadcast AM | Audio, via envelope detection |
| [`wbfm.py`](wbfm.py) | Broadcast FM | Audio, via quadrature demodulation |
| [`adsb.py`](adsb.py) | ADS-B @ 1090 MHz | Aircraft ICAO address, typecode, callsign |

---

## Why write these by hand

Every step that a black-box demodulator hides is explicit here: undoing the
interleaving of the ADC stream, removing the receiver's DC bias, mixing a
station down to baseband, choosing a decimation factor that preserves the
channel while dropping the sample rate, and only then recovering the message.
The point was to be able to explain each choice, not just to hear audio.

## Signal chains

### AM — envelope detection

```
raw uint8 IQ → de-interleave → remove DC bias → mix to baseband
             → FIR decimate → |·| → remove carrier DC → resample → audio
```

For AM the message rides on the carrier's amplitude, so the magnitude of the
complex baseband signal *is* the message. The mixing step is what lets a single
recording yield several stations: multiplying by `exp(-j2πft)` rotates the
spectrum so any chosen carrier lands at 0 Hz.

Both receiver formats are supported, because they store samples differently —
RTL-SDR writes interleaved `uint8` biased around 127/128, USRP writes
interleaved zero-centred `float32`.

`strongest_carriers()` estimates the power spectral density with Welch's method
and returns the loudest carriers, which is how the other stations present in a
single capture were located before demodulating them.

### WBFM — quadrature demodulation

```
raw IQ → mix to baseband → FIR decimate to 256 kHz → hard limiter
       → Parks–McClellan differentiator → Im{x′·conj(x)} → resample → audio
```

FM carries the message in instantaneous frequency, the derivative of phase.
Unwrapping phase and differencing it is fragile near wrap points, so this uses
the quadrature identity instead:

$$m(t) \propto \mathrm{Im}\{ x'(t)\,\overline{x(t)} \}$$

The **hard limiter** (`x / |x|`) strips residual amplitude variation so it
cannot leak into the audio. The derivative is taken by a 31-tap linear-phase
FIR differentiator designed with the **Parks–McClellan (Remez) exchange
algorithm**, specified with a magnitude response proportional to frequency
across the passband and rejection above it.

Applied to a single capture, the same function recovers the station at the
tuned centre and the stations offset by ±400 kHz and ±800 kHz.

### ADS-B — pulse-position demodulation and Mode S framing

```
raw IQ → |·| → resample to 4 MHz → threshold → find 8 µs preamble
       → read 5-bit DF field → assemble 56/112-bit frame → hex → pyModeS
```

ADS-B is pulse-position modulated at 1 Mbit/s: each bit occupies 1 µs and is
encoded by *where* the pulse sits inside that window. Energy in the first half
is a 1, energy in the second half is a 0. Since the information is in amplitude
timing rather than phase, only sample magnitudes are needed.

Frames are located by sliding along the thresholded pulse train looking for the
fixed preamble (pulses at 0, 1.0, 3.5 and 4.5 µs). The **downlink format (DF)**
field is then read to decide whether a short (56-bit) or extended-squitter
(112-bit) frame follows. Assembled frames are converted to hex and handed to
[`pyModeS`](https://github.com/junzis/pyModeS) for field extraction.

The **detection threshold** is the interesting parameter: lowering it finds
more distant aircraft but admits noise that then fails the preamble test.
Recovered callsigns were cross-checked against live FlightRadar24 data.

## Usage

```python
import am, wbfm, adsb
import soundfile as sf

# AM station at the tuned centre
audio = am.from_rtlsdr("rtl_am.dat")
sf.write("am.wav", audio, 48000)

# A different AM station in the same recording, 230 kHz off centre
other = am.from_rtlsdr("rtl_am.dat", freq_shift=230_000)

# FM station 400 kHz above the tuned centre
audio = wbfm.from_rtlsdr("rtl_wbfm.dat", freq_shift=400_000)
sf.write("fm.wav", audio, 48000)

# Aircraft transponders
result = adsb.decode("rtl_adsb.dat", threshold=5, verbose=True)
print(result)          # frames=… df17=… callsigns=[…]
```

Capturing your own IQ, using the Osmocom command-line tools:

```bash
rtl_sdr -f 88000000  -s 2048000 -n 20480000 -g 20  rtl_wbfm.dat   # FM broadcast
rtl_sdr -f 1090000000 -s 3200000 -n 32000000 -g 50 rtl_adsb.dat   # ADS-B
```

## Install

```bash
pip install -r requirements.txt
```

Raw captures are excluded from the repository — they are hundreds of megabytes
and not mine to redistribute. Record your own with the commands above.

## Also covered in the lab

Work that informed these receivers but does not live in this repository:

- **FIR filter characterisation** — driving filters with a noise source to read
  their response, and deriving the relationship between transition width and
  filter length numerically (385, 193 and 129 taps for 100, 200 and 300 Hz
  transition widths at 16 kSPS).
- **Aliasing** — demonstrated directly by sweeping tones past Nyquist in a
  decimating chain with and without the anti-alias filter.
- **QPSK transceiver in GNU Radio** — pulse shaping, symbol timing recovery,
  and a Costas loop correcting deliberately injected phase and frequency
  offsets; occupied bandwidth measured over the air in the 433 MHz ISM band.

## Notes

Course material, assignment text and instructor-authored content are
deliberately not included here; this repository contains only the receivers I
wrote. `pyModeS` is used for Mode S field extraction once frames have been
demodulated and assembled — the demodulation, preamble search and framing are
implemented here.

## Licence

MIT
