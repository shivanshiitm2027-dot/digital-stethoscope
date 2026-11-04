import numpy as np
import streamlit as st
from scipy.signal import butter, sosfilt

import io
import wave

def make_wav(samples, sample_rate):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm.tobytes())
    return buffer.getvalue()


st.set_page_config(
    page_title="Digital Stethoscope",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Digital Stethoscope Prototype")
st.write("Shivanshu Yadav | IIT Madras | Electronics & Communication Engineering")
st.info("Software simulation — hardware implementation is planned.")

st.header("Compare Both Stethoscopes")

st.table([
    {
        "Feature": "Working",
        "Normal / Acoustic": "Diaphragm → air in tube → ears",
        "Electronic": "Sensor → circuit → audio output",
        "My Proposed Design": "Sensor → ADC → DSP → audio output",
    },
    {
        "Feature": "Battery",
        "Normal / Acoustic": "Not required",
        "Electronic": "Required",
        "My Proposed Design": "Required",
    },
    {
        "Feature": "Electronic gain",
        "Normal / Acoustic": "No",
        "Electronic": "Yes",
        "My Proposed Design": "Simulated below",
    },
    {
        "Feature": "Digital filtering",
        "Normal / Acoustic": "No",
        "Electronic": "Some models",
        "My Proposed Design": "Simulated below",
    },
    {
        "Feature": "Recording",
        "Normal / Acoustic": "Not built in",
        "Electronic": "Some models",
        "My Proposed Design": "Planned",
    },
])

st.header("DSP Simulation")
st.caption(
    "Synthetic signals only, not patient recordings. "
    "Filter ranges are illustrative."
)

mode = st.selectbox(
    "Select signal",
    ["Heart-like pulses", "Breath-like noise"],
)

noise_level = st.slider("Added noise", 0.0, 1.0, 0.35)
gain = st.slider("Output gain", 0.5, 4.0, 1.0, 0.1)

fs = 4000
t = np.arange(fs * 4) / fs
rng = np.random.default_rng(42)

if mode == "Heart-like pulses":
    phase = t % 0.8
    first = np.exp(-((phase - 0.12) / 0.035) ** 2)
    second = 0.65 * np.exp(-((phase - 0.38) / 0.025) ** 2)
    clean = (first + second) * np.sin(2 * np.pi * 75 * t)
    low, high = 25, 180
else:
    envelope = 0.25 + 0.55 * np.sin(np.pi * t / 2) ** 2
    clean = envelope * rng.normal(0, 0.35, len(t))
    low, high = 100, 1000

noise = noise_level * (
    0.4 * np.sin(2 * np.pi * 1400 * t)
    + rng.normal(0, 0.18, len(t))
)
raw = clean + noise

sos = butter(
    4,
    [low, high],
    btype="bandpass",
    fs=fs,
    output="sos",
)
output = gain * sosfilt(sos, raw)

st.write(f"**Band-pass filter:** {low}–{high} Hz")

# Show a short window so oscillations remain visible.
start, end = int(0.05 * fs), int(0.55 * fs)

st.line_chart(
    {
        "Time (s)": t[start:end],
        "Input with noise": raw[start:end],
        "Filtered output": output[start:end],
    },
    x="Time (s)",
    y=["Input with noise", "Filtered output"],
)

st.subheader("Listen to the Simulation")
st.caption("Both previews use the same fixed playback scale.")

left, right = st.columns(2)

with left:
    st.write("Input with noise")
    st.audio(
        make_wav(0.2 * raw, fs),
        sample_rate=fs,
    )

with right:
    st.write("Filtered output")
    st.audio(
        make_wav(0.2 * output, fs),
        sample_rate=fs,
    )

st.subheader("My ECE Connection")
st.table([
    {"Subject": "Analog circuits", "Use": "Sensor + preamplifier"},
    {"Subject": "DSP", "Use": "Filtering + sound analysis"},
    {"Subject": "Embedded systems", "Use": "MCU + sampling + storage"},
    {"Subject": "Communication", "Use": "Optional Bluetooth"},
])

st.success(
    "Implemented: comparison, synthetic signals, "
    "filtering, gain control and audio preview."
)
st.write(
    "**Planned hardware:** sensor → preamplifier → ADC "
    "→ MCU/DSP → DAC + headphone amplifier."
)
st.caption("Educational simulation, not a clinical diagnostic device.")

# ---------- Advanced audio lab ----------
from scipy.io import wavfile

st.divider()
st.header("🔬 Advanced Audio Lab")

uploaded = st.file_uploader(
    "Upload a WAV recording, or use the synthetic input",
    type=["wav"],
)

lab_signal = raw.copy()
lab_fs = fs

if uploaded is not None:
    try:
        lab_fs, samples = wavfile.read(uploaded)
        original_dtype = samples.dtype

        # Convert integer PCM to floating-point amplitude.
        if original_dtype == np.uint8:
            samples = (samples.astype(np.float64) - 128) / 128
        elif np.issubdtype(original_dtype, np.signedinteger):
            scale = float(
                max(
                    abs(int(np.iinfo(original_dtype).min)),
                    int(np.iinfo(original_dtype).max),
                )
            )
            samples = samples.astype(np.float64) / scale
        elif np.issubdtype(original_dtype, np.floating):
            samples = samples.astype(np.float64)
        else:
            raise ValueError("Unsupported WAV sample format.")

        if samples.ndim == 2:
            samples = samples.mean(axis=1)

        if samples.size < 32 or not np.isfinite(samples).all():
            raise ValueError("Audio is too short or contains invalid samples.")

        # Limit processing to the first 30 seconds.
        lab_signal = samples[:int(lab_fs * 30)]
        st.success("WAV loaded. Processing up to the first 30 seconds.")
    except Exception as error:
        st.error(f"Could not read WAV: {error}")
        st.stop()

nyquist = lab_fs / 2
default_low = min(25.0, nyquist * 0.1)
default_high = min(180.0, nyquist * 0.8)

c1, c2, c3 = st.columns(3)

with c1:
    cutoff_low = st.number_input(
        "Lower cutoff (Hz)",
        min_value=0.1,
        max_value=float(nyquist - 0.1),
        value=float(default_low),
    )

with c2:
    cutoff_high = st.number_input(
        "Upper cutoff (Hz)",
        min_value=0.1,
        max_value=float(nyquist - 0.1),
        value=float(default_high),
    )

with c3:
    lab_gain = st.slider(
        "Advanced output gain",
        0.1, 5.0, 1.0, 0.1,
    )

order = st.select_slider(
    "Butterworth prototype order",
    options=[2, 4, 6],
    value=4,
)

if cutoff_low >= cutoff_high:
    st.warning("Lower cutoff must be below upper cutoff.")
    st.stop()

lab_sos = butter(
    order,
    [cutoff_low, cutoff_high],
    btype="bandpass",
    fs=lab_fs,
    output="sos",
)
lab_output = lab_gain * sosfilt(lab_sos, lab_signal)

m1, m2, m3 = st.columns(3)
m1.metric("Sample rate", f"{lab_fs:,} Hz")
m2.metric("Duration", f"{len(lab_signal) / lab_fs:.2f} s")
m3.metric("Output peak", f"{np.max(np.abs(lab_output)):.3f}")

clipping = np.mean(np.abs(lab_output) > 1) * 100
if clipping > 0:
    st.warning(
        f"{clipping:.2f}% of samples exceed the WAV playback range. "
        "Reduce gain to avoid clipping."
    )

wave_tab, spectrum_tab = st.tabs(
    ["Waveform", "Frequency spectrum"]
)

with wave_tab:
    # Display up to half a second, with a bounded point count.
    n_view = min(len(lab_signal), int(lab_fs * 0.5))
    step = max(1, n_view // 4000)
    view = np.arange(0, n_view, step)

    st.line_chart(
        {
            "Time (s)": view / lab_fs,
            "Input": lab_signal[view],
            "Output": lab_output[view],
        },
        x="Time (s)",
        y=["Input", "Output"],
    )

with spectrum_tab:
    n_fft = min(len(lab_signal), 65536)
    window = np.hanning(n_fft)
    frequencies = np.fft.rfftfreq(n_fft, 1 / lab_fs)

    def spectrum_db(signal):
        magnitude = (
            2 * np.abs(np.fft.rfft(signal[:n_fft] * window))
            / max(window.sum(), 1e-12)
        )
        return 20 * np.log10(np.maximum(magnitude, 1e-8))

    visible = frequencies <= min(2000, nyquist)

    st.line_chart(
        {
            "Frequency (Hz)": frequencies[visible],
            "Input (dB)": spectrum_db(lab_signal)[visible],
            "Output (dB)": spectrum_db(lab_output)[visible],
        },
        x="Frequency (Hz)",
        y=["Input (dB)", "Output (dB)"],
    )
    st.caption(
        "Amplitude spectrum relative to digital amplitude 1; "
        "not a calibrated sound-pressure measurement."
    )

st.subheader("Audio Comparison")
st.caption("Both previews use the same scale; WAV output clips at ±1.")

p1, p2 = st.columns(2)
with p1:
    st.write("Original")
    st.audio(
        make_wav(lab_signal, lab_fs),
        format="audio/wav",
    )
with p2:
    st.write("Processed")
    st.audio(
        make_wav(lab_output, lab_fs),
        format="audio/wav",
    )

st.download_button(
    "⬇ Download processed WAV",
    data=make_wav(lab_output, lab_fs),
    file_name="stethoscope_processed.wav",
    mime="audio/wav",
)

st.caption(
    "Filtering can remove useful sound as well as noise. "
    "This lab processes audio; it does not diagnose conditions."
)
