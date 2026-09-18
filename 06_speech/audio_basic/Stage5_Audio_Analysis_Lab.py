#!/usr/bin/env python3
"""
================================================================================
 Stage 5 -- Computer Audio Basics: Interactive Audio Analysis Lab
 (Standalone .py script version)
================================================================================

This is a plain-Python companion to `Stage5_Audio_Analysis_Lab.ipynb`.
It performs the exact same beginner-friendly audio analysis, but runs from
the command line instead of Google Colab, so there is no upload widget --
you pass the path to your audio file instead.

USAGE
-----
    python Stage5_Audio_Analysis_Lab.py path/to/your_audio.wav

    # or, run with no arguments and you'll be prompted for a path:
    python Stage5_Audio_Analysis_Lab.py

WHAT IT DOES
------------
Every plot is both shown (if you have a display) AND saved as a PNG into an
`audio_lab_outputs/` folder next to this script, so nothing is lost when you
run this headlessly (e.g. over SSH or in a plain terminal).

All numbers in this script are computed live from YOUR uploaded audio file --
nothing is faked or hard-coded, exactly like the notebook version.

Topics covered (Stage 5): digital audio, sampling, sampling rate, Nyquist
frequency, amplitude, frequency, bit depth, channels, mono/stereo, WAV vs
MP3, waveform, RMS energy, silence detection, noise, speech signals,
duration -- plus a labeled *preview* of FFT / pitch (Stage 6 topics).
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend automatically if there's no display
# (e.g. running over SSH), but still try an interactive one first.
try:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.close()
except Exception:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

import librosa
import librosa.display
import soundfile as sf

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_lab_outputs")


def show_and_save(fig_name):
    """Save the current matplotlib figure and try to display it."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{fig_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"  [saved figure -> {path}]")
    try:
        plt.show()
    except Exception:
        pass
    plt.close()


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def explain(text):
    print(text.strip())


def try_play_audio(path):
    """Best-effort audio playback; silently skipped if no audio backend/display."""
    try:
        from IPython.display import Audio, display
        display(Audio(path))
    except Exception:
        print("  (Audio playback widget only works inside a Jupyter/IPython session.")
        print(f"   Open '{path}' with your regular media player to listen.)")


def main():
    parser = argparse.ArgumentParser(description="Stage 5 Computer Audio Basics Lab")
    parser.add_argument("audio_file", nargs="?", help="Path to a .wav/.mp3/.flac/.m4a file")
    args = parser.parse_args()

    plt.rcParams["figure.figsize"] = (12, 4)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["font.size"] = 11

    # =====================================================================
    # SECTION 2 -- LOAD FILE PATH (script equivalent of "Upload")
    # =====================================================================
    section("SECTION 2 -- Select Audio File")
    audio_file = args.audio_file or input("Enter the path to your audio file: ").strip()

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Could not find the file '{audio_file}'.")

    file_ext = os.path.splitext(audio_file)[1].lower()
    file_size_bytes = os.path.getsize(audio_file)
    file_size_kb = file_size_bytes / 1024
    supported_exts = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

    print(f"Filename       : {audio_file}")
    print(f"File extension : {file_ext}")
    print(f"File size      : {file_size_kb:.2f} KB ({file_size_bytes} bytes)")
    if file_ext not in supported_exts:
        print(f"Warning: '{file_ext}' is not in the commonly-supported list {supported_exts}.")
        print("If loading fails below, convert the file to WAV first, e.g.:")
        print("    ffmpeg -i your_file INPUT_EXT your_file.wav")

    # =====================================================================
    # SECTION 3 -- PLAY AUDIO
    # =====================================================================
    section("SECTION 3 -- Play the Audio")
    explain("The waveform is a visual representation; this step lets you (try to) hear the signal.")
    try_play_audio(audio_file)

    # =====================================================================
    # SECTION 4 -- LOAD AUDIO
    # =====================================================================
    section("SECTION 4 -- Load the Audio (Preserving Channels & Sample Rate)")
    try:
        audio, sr = librosa.load(audio_file, sr=None, mono=False)
    except Exception as e:
        raise RuntimeError(
            f"Could not decode '{audio_file}' with librosa. "
            f"Try converting it to WAV first (e.g. with ffmpeg). Original error: {e}"
        )

    is_stereo = (audio.ndim == 2)
    n_channels = audio.shape[0] if is_stereo else 1
    n_samples = audio.shape[-1]
    duration_sec = n_samples / sr

    print(f"Sampling rate     : {sr} Hz")
    print(f"Number of samples : {n_samples}")
    print(f"Number of channels: {n_channels} ({'Stereo' if is_stereo else 'Mono'})")
    print(f"Duration          : {duration_sec:.3f} seconds")

    # =====================================================================
    # SECTION 5 -- SUMMARY DASHBOARD
    # =====================================================================
    section("SECTION 5 -- Audio Summary Dashboard")
    bit_depth_str = "Bit depth could not be reliably determined from this decoding pipeline."
    file_format = file_ext.replace(".", "").upper()
    try:
        info = sf.info(audio_file)
        file_format = info.format
        if info.subtype:
            bit_depth_str = info.subtype
    except Exception:
        pass

    min_amp = float(np.min(audio))
    max_amp = float(np.max(audio))
    mean_amp = float(np.mean(audio))
    rms_amp = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    summary_rows = [
        ("Filename", audio_file, "Name of the uploaded file"),
        ("File format", file_format, "Container/codec of the audio file"),
        ("File size", f"{file_size_kb:.2f} KB", "Size of the file on disk"),
        ("Sampling rate", f"{sr} Hz", "Samples captured per second"),
        ("Duration", f"{duration_sec:.3f} s", "Length of the audio"),
        ("Number of channels", n_channels, "1 = mono, 2 = stereo"),
        ("Number of samples", n_samples, "Total digital samples per channel"),
        ("Bit depth / subtype", bit_depth_str, "Precision used to store each sample"),
        ("Minimum amplitude", f"{min_amp:.5f}", "Smallest (most negative) sample value"),
        ("Maximum amplitude", f"{max_amp:.5f}", "Largest (most positive) sample value"),
        ("Mean amplitude", f"{mean_amp:.6f}", "Average sample value (near 0 for most audio)"),
        ("RMS amplitude", f"{rms_amp:.5f}", "Root-mean-square -- a measure of signal energy"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Property", "Value", "Meaning"])
    print(summary_df.to_string(index=False))

    # =====================================================================
    # SECTION 6 -- WHAT IS DIGITAL AUDIO
    # =====================================================================
    section("SECTION 6 -- What Is Digital Audio?")
    explain("""
    Real sound: Real-world sound -> Microphone -> Analog signal -> Sampling
    -> Digital samples -> Audio file. The computer only ever stores numbers.
    """)
    first_channel = audio[0] if is_stereo else audio
    n_preview = min(50, len(first_channel))
    print(f"First {n_preview} raw sample values of the uploaded audio:")
    print(np.round(first_channel[:n_preview], 5).tolist())

    # =====================================================================
    # SECTION 7 -- SHAPE / CHANNELS
    # =====================================================================
    section("SECTION 7 -- Audio Shape & Channels")
    print(f"Array shape: {audio.shape}")
    if is_stereo:
        print(f"-> STEREO audio: {n_channels} channels, {audio.shape[1]} samples each.")
        left, right = audio[0], (audio[1] if n_channels > 1 else audio[0])
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        librosa.display.waveshow(left, sr=sr, ax=axes[0])
        axes[0].set_title("Left Channel Waveform")
        axes[0].set_ylabel("Amplitude")
        librosa.display.waveshow(right, sr=sr, ax=axes[1])
        axes[1].set_title("Right Channel Waveform")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Amplitude")
        show_and_save("07_channels")
    else:
        print("-> MONO audio: 1 channel.")
        plt.figure(figsize=(12, 3))
        librosa.display.waveshow(audio, sr=sr)
        plt.title("Mono Waveform")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        show_and_save("07_channels")

    # =====================================================================
    # SECTION 8 -- WAVEFORM
    # =====================================================================
    section("SECTION 8 -- Waveform Visualization")
    wave_for_plot = audio[0] if is_stereo else audio
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(wave_for_plot, sr=sr)
    plt.title("Waveform of Uploaded Audio")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.axhline(0, linewidth=0.8)
    show_and_save("08_waveform")

    # =====================================================================
    # SECTION 9 -- ZOOM INTO SAMPLES
    # =====================================================================
    section("SECTION 9 -- Zoom Into a Small Segment (Discrete Samples)")
    zoom_duration = 0.05
    zoom_samples = int(min(zoom_duration * sr, len(wave_for_plot)))
    zoom_samples = max(zoom_samples, min(50, len(wave_for_plot)))
    segment = wave_for_plot[:zoom_samples]
    t = np.arange(len(segment)) / sr

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(t, segment, linewidth=1)
    axes[0].set_title("Looks Continuous (line plot)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(t, segment, linewidth=1, alpha=0.5)
    axes[1].scatter(t, segment, s=12)
    axes[1].set_title(f"Actually Discrete: {len(segment)} Samples")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    show_and_save("09_zoom_samples")
    print(f"Displayed {len(segment)} discrete samples spanning {len(segment)/sr*1000:.2f} ms.")

    # =====================================================================
    # SECTION 10 -- SAMPLING RATE / NYQUIST
    # =====================================================================
    section("SECTION 10 -- Sampling Rate & Nyquist Frequency")
    samples_per_ms = sr / 1000
    nyquist = sr / 2
    print(f"Sampling rate           : {sr} Hz")
    print(f"Samples per millisecond : {samples_per_ms:.2f}")
    print(f"Samples per second      : {sr}")
    print(f"Samples per minute      : {sr * 60}")
    print(f"Nyquist frequency       : {nyquist} Hz")
    print(f"-> This audio can faithfully represent frequencies up to about {nyquist:.0f} Hz.")

    # =====================================================================
    # SECTION 11 -- NYQUIST / ALIASING DEMO (SYNTHETIC)
    # =====================================================================
    section("SECTION 11 -- Nyquist / Aliasing Demonstration (Synthetic Example)")
    explain("NOTE: this section uses synthetic sine waves, NOT your uploaded audio.")
    demo_sr = 100
    demo_nyquist = demo_sr / 2
    t_continuous = np.linspace(0, 1, 5000)
    freqs_demo = {
        "Low frequency (5 Hz) - well below Nyquist": 5,
        f"Near Nyquist ({demo_nyquist:.0f} Hz)": demo_nyquist,
        "Above Nyquist (80 Hz) - will ALIAS": 80,
    }
    fig, axes = plt.subplots(len(freqs_demo), 1, figsize=(12, 9))
    for ax, (label, f) in zip(axes, freqs_demo.items()):
        continuous_wave = np.sin(2 * np.pi * f * t_continuous)
        ax.plot(t_continuous, continuous_wave, alpha=0.4, label="'True' continuous signal")
        t_sampled = np.arange(0, 1, 1 / demo_sr)
        sampled_wave = np.sin(2 * np.pi * f * t_sampled)
        ax.plot(t_sampled, sampled_wave, "o-", label=f"Sampled at {demo_sr} Hz")
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=8)
    show_and_save("11_nyquist_demo")

    # =====================================================================
    # SECTION 12 -- AMPLITUDE
    # =====================================================================
    section("SECTION 12 -- Amplitude")
    print(f"Minimum amplitude : {min_amp:.5f}")
    print(f"Maximum amplitude : {max_amp:.5f}")
    print(f"Mean amplitude    : {mean_amp:.6f}")
    print(f"RMS amplitude     : {rms_amp:.5f}")
    short_len = min(sr, len(wave_for_plot))
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(wave_for_plot[:short_len], sr=sr)
    plt.title("Amplitude Over a Short Section (first 1 second)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    show_and_save("12_amplitude")

    # =====================================================================
    # SECTION 13 -- RMS ENERGY
    # =====================================================================
    section("SECTION 13 -- RMS Energy")
    rms_mono = librosa.feature.rms(y=wave_for_plot)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms_mono)), sr=sr)
    plt.figure(figsize=(12, 4))
    plt.plot(rms_times, rms_mono)
    plt.title("RMS Energy Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("RMS Energy")
    show_and_save("13_rms_energy")

    # =====================================================================
    # SECTION 14 -- SILENCE DETECTION
    # =====================================================================
    section("SECTION 14 -- Silence Detection")
    intervals = librosa.effects.split(wave_for_plot, top_db=30)
    non_silent_samples = sum((end - start) for start, end in intervals)
    silent_samples = n_samples - non_silent_samples
    non_silent_pct = 100 * non_silent_samples / n_samples if n_samples else 0
    silent_pct = 100 - non_silent_pct
    print(f"Estimated non-silent time : {non_silent_samples/sr:.3f} s ({non_silent_pct:.1f}%)")
    print(f"Estimated silent time     : {silent_samples/sr:.3f} s ({silent_pct:.1f}%)")
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(wave_for_plot, sr=sr, alpha=0.5)
    for start, end in intervals:
        plt.axvspan(start / sr, end / sr, color="green", alpha=0.15)
    plt.title("Waveform with Estimated Non-Silent Regions Highlighted (green)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    show_and_save("14_silence_detection")

    # =====================================================================
    # SECTION 15 -- NOISE (SYNTHETIC DEMO)
    # =====================================================================
    section("SECTION 15 -- Noise (Synthetic Demonstration)")
    demo_sr2 = 8000
    t_demo = np.linspace(0, 1.0, demo_sr2, endpoint=False)
    clean_signal = 0.5 * np.sin(2 * np.pi * 220 * t_demo)
    noise = np.random.normal(0, 0.15, size=clean_signal.shape)
    noisy_signal = clean_signal + noise
    snr_db = 10 * np.log10(np.mean(clean_signal ** 2) / np.mean(noise ** 2))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    axes[0].plot(t_demo[:400], clean_signal[:400])
    axes[0].set_title("Synthetic CLEAN Signal (220 Hz tone)")
    axes[1].plot(t_demo[:400], noisy_signal[:400])
    axes[1].set_title("Synthetic NOISY Signal (clean + Gaussian noise)")
    axes[1].set_xlabel("Time (seconds)")
    show_and_save("15_noise_demo")
    print(f"Synthetic-demo SNR: {snr_db:.2f} dB")

    # =====================================================================
    # SECTION 16 -- FREQUENCY (SYNTHETIC)
    # =====================================================================
    section("SECTION 16 -- Frequency (Synthetic Demonstration)")
    demo_sr3 = 5000
    demo_freqs = [10, 100, 1000]
    window_lengths = [0.5, 0.05, 0.01]
    fig, axes = plt.subplots(len(demo_freqs), 1, figsize=(12, 8))
    for ax, f, win in zip(axes, demo_freqs, window_lengths):
        t_f = np.linspace(0, win, int(demo_sr3 * win))
        wave = np.sin(2 * np.pi * f * t_f)
        ax.plot(t_f, wave)
        ax.set_title(f"{f} Hz sine wave (first {win*1000:.0f} ms)")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
    show_and_save("16_frequency_demo")

    # =====================================================================
    # SECTION 17 -- FFT PREVIEW (UPLOADED AUDIO)
    # =====================================================================
    section("SECTION 17 -- Frequency Content of Uploaded Audio [Preview: Stage 6]")
    fft_window_sec = min(1.0, duration_sec)
    fft_window_samples = int(fft_window_sec * sr)
    fft_segment = wave_for_plot[:fft_window_samples]
    fft_magnitude = np.abs(np.fft.rfft(fft_segment))
    fft_freqs = np.fft.rfftfreq(len(fft_segment), d=1 / sr)
    plt.figure(figsize=(12, 4))
    plt.plot(fft_freqs, fft_magnitude)
    plt.title(f"FFT Magnitude Spectrum (first {fft_window_sec:.2f}s)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, nyquist)
    show_and_save("17_fft_preview")
    dominant_freq = fft_freqs[np.argmax(fft_magnitude)]
    print(f"Dominant frequency component in this segment: ~{dominant_freq:.1f} Hz")

    # =====================================================================
    # SECTION 18 -- F0 PITCH PREVIEW
    # =====================================================================
    section("SECTION 18 -- Fundamental Frequency (Pitch) Preview [Optional]")
    valid_f0 = np.array([])
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            wave_for_plot.astype(np.float32),
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            print(f"Estimated F0 range : {valid_f0.min():.1f} Hz - {valid_f0.max():.1f} Hz")
            print(f"Estimated F0 mean  : {valid_f0.mean():.1f} Hz")
            times_f0 = librosa.times_like(f0, sr=sr)
            plt.figure(figsize=(12, 4))
            plt.plot(times_f0, f0)
            plt.title("Estimated Pitch (F0) Over Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Frequency (Hz)")
            show_and_save("18_pitch_f0")
        else:
            print("No voiced pitch could be reliably detected (expected for music/noise/silence).")
    except Exception as e:
        print(f"Pitch estimation failed: {e}")

    # =====================================================================
    # SECTION 19 -- DURATION
    # =====================================================================
    section("SECTION 19 -- Audio Duration")
    computed_duration = n_samples / sr
    print(f"Duration = N / Fs = {n_samples} / {sr} = {computed_duration:.3f} seconds")

    # =====================================================================
    # SECTION 20 -- BIT DEPTH
    # =====================================================================
    section("SECTION 20 -- Bit Depth")
    for bits in [8, 16, 24]:
        print(f"{bits}-bit -> 2^{bits} = {2**bits:,} possible amplitude levels")
    print(f"This file's reported subtype/bit depth: {bit_depth_str}")

    # =====================================================================
    # SECTION 21 -- WAV vs MP3 (conceptual, no computation needed)
    # =====================================================================
    section("SECTION 21 -- WAV vs MP3 (Conceptual)")
    explain("""
    WAV: typically lossless PCM, larger files, no compression artifacts.
    MP3: lossy compression, much smaller files, can introduce artifacts.
    Research pitfall: mixing WAV (e.g. human) and MP3 (e.g. AI) recordings across
    classes can let a classifier learn compression artifacts instead of real
    speech differences -- a dataset confound.
    """)

    # =====================================================================
    # SECTION 22 -- CHANNEL ANALYSIS
    # =====================================================================
    section("SECTION 22 -- Channel Analysis")
    if is_stereo:
        left_rms = float(np.sqrt(np.mean(audio[0].astype(np.float64) ** 2)))
        right_rms = float(np.sqrt(np.mean(audio[1].astype(np.float64) ** 2))) if n_channels > 1 else left_rms
        print(f"Left channel RMS  : {left_rms:.5f}")
        print(f"Right channel RMS : {right_rms:.5f}")
    else:
        print("This audio is MONO -- no left/right comparison applies.")

    # =====================================================================
    # SECTION 23 -- MONO CONVERSION
    # =====================================================================
    section("SECTION 23 -- Mono Conversion Demonstration")
    if is_stereo:
        original_left = audio[0].copy()
        original_right = audio[1].copy() if n_channels > 1 else audio[0].copy()
        converted_mono = (original_left + original_right) / 2.0
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)
        librosa.display.waveshow(original_left, sr=sr, ax=axes[0])
        axes[0].set_title("Original Left")
        librosa.display.waveshow(original_right, sr=sr, ax=axes[1])
        axes[1].set_title("Original Right")
        librosa.display.waveshow(converted_mono, sr=sr, ax=axes[2])
        axes[2].set_title("Converted Mono (average of L and R)")
        axes[2].set_xlabel("Time (seconds)")
        show_and_save("23_mono_conversion")
    else:
        print("This audio is already mono -- no conversion needed.")

    # =====================================================================
    # SECTION 24 -- NORMALIZATION
    # =====================================================================
    section("SECTION 24 -- Audio Normalization")
    max_abs_amp = float(np.max(np.abs(wave_for_plot)))
    print(f"Maximum absolute amplitude before normalization: {max_abs_amp:.5f}")
    normalized_audio = wave_for_plot / max_abs_amp if max_abs_amp > 0 else wave_for_plot.copy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    librosa.display.waveshow(wave_for_plot, sr=sr, ax=axes[0])
    axes[0].set_title("Original Waveform")
    librosa.display.waveshow(normalized_audio, sr=sr, ax=axes[1])
    axes[1].set_title("Normalized Waveform (peak scaled to +/-1.0)")
    axes[1].set_xlabel("Time (seconds)")
    show_and_save("24_normalization")

    # =====================================================================
    # SECTION 25 -- SPEECH SIGNAL SUMMARY (text only)
    # =====================================================================
    section("SECTION 25 -- Speech Signal Analysis (Summary)")
    explain("""
    Pipeline: Speech -> Waveform -> Energy (RMS) -> Frequency (FFT) -> Pitch (F0)
    -> Features -> Machine Learning / Deep Learning.
    See Sections 8, 13, 14, 17, 18 above for each stage computed on your audio.
    """)

    # =====================================================================
    # SECTION 26 -- CONCEPT TABLE
    # =====================================================================
    section("SECTION 26 -- Important Audio Concept Table")
    concept_table = pd.DataFrame([
        ["Digital Audio", "Sound represented as a sequence of numbers", "-", "A .wav file's sample array"],
        ["Sampling", "Measuring a signal's value at regular intervals", "-", "16,000 measurements/second"],
        ["Sampling Rate", "Number of samples taken per second", "Hz", "16000 Hz, 44100 Hz"],
        ["Bit Depth", "Bits used to store each sample's amplitude", "bits", "16-bit -> 65,536 levels"],
        ["Amplitude", "Instantaneous value of a sample", "unitless/dB", "0.42, -0.15"],
        ["Frequency", "Cycles per second of oscillation", "Hz", "440 Hz"],
        ["Channel", "An independent audio stream", "-", "Left, Right"],
        ["Mono/Stereo", "1 vs 2 channel audio", "-", "(samples,) vs (2, samples)"],
        ["Waveform", "Amplitude plotted against time", "-", "Section 8 plot"],
        ["Noise", "Unwanted signal mixed with the desired signal", "-", "Hiss, hum, static"],
        ["Silence", "Low/no signal energy over a region", "-", "Pause between words"],
        ["Speech Signal", "Audio containing spoken language", "-", "A recorded sentence"],
        ["Duration", "Total length of the audio", "seconds", "3.250 s"],
    ], columns=["Concept", "Definition", "Unit", "Example"])
    print(concept_table.to_string(index=False))

    # =====================================================================
    # SECTION 27 -- RESEARCH CONNECTION (text)
    # =====================================================================
    section("SECTION 27 -- Research Connection: AI-Generated Speech Detection")
    explain("""
    Human and AI speech are both digital audio signals. Detection models may use
    differences in waveform, spectral, and temporal patterns, pitch, harmonic
    structure, high-frequency content, artifacts, and noise robustness.
    No single feature reliably detects AI speech alone.
    Pipeline: Waveform -> FFT -> STFT -> Spectrogram -> Mel-Spectrogram -> MFCC
    -> Self-Supervised Representations -> Classifier -> Detection.
    """)

    # =====================================================================
    # SECTION 28 -- AUTOMATIC REPORT
    # =====================================================================
    section("SECTION 28 -- Automatic Audio Report")
    f0_line = "Pitch: Not reliably estimated (see Section 18)."
    if len(valid_f0) > 0:
        f0_line = f"Pitch (F0): {valid_f0.min():.1f}-{valid_f0.max():.1f} Hz (mean {valid_f0.mean():.1f} Hz)"
    report_lines = [
        "================ AUDIO REPORT ================",
        f"File               : {audio_file}",
        f"Duration           : {duration_sec:.3f} s",
        f"Sampling Rate      : {sr} Hz",
        f"Channels           : {n_channels} ({'Stereo' if is_stereo else 'Mono'})",
        f"Number of Samples  : {n_samples}",
        f"Bit Depth          : {bit_depth_str}",
        f"Min Amplitude      : {min_amp:.5f}",
        f"Max Amplitude      : {max_amp:.5f}",
        f"RMS                : {rms_amp:.5f}",
        f"Nyquist Frequency  : {nyquist:.0f} Hz",
        "",
        f"Estimated Non-Silence : {non_silent_pct:.1f}%",
        f"Estimated Silence      : {silent_pct:.1f}%",
        "",
        f0_line,
        "================================================",
    ]
    report = "\n".join(report_lines)
    print(report)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "audio_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n[report saved -> {report_path}]")

    # =====================================================================
    # SECTION 30 -- MINI EXPERIMENT: RESAMPLING (implemented)
    # =====================================================================
    section("SECTION 30 -- Mini Experiment: Resampling (8kHz / 16kHz)")
    resample_targets = [8000, 16000]
    resampled_versions = {
        t_sr: librosa.resample(wave_for_plot.astype(np.float32), orig_sr=sr, target_sr=t_sr)
        for t_sr in resample_targets
    }
    fig, axes = plt.subplots(1 + len(resample_targets), 1, figsize=(12, 3 * (1 + len(resample_targets))), sharex=True)
    librosa.display.waveshow(wave_for_plot, sr=sr, ax=axes[0])
    axes[0].set_title(f"Original ({sr} Hz)")
    for ax, (t_sr, resampled) in zip(axes[1:], resampled_versions.items()):
        librosa.display.waveshow(resampled, sr=t_sr, ax=ax)
        ax.set_title(f"Resampled to {t_sr} Hz")
    axes[-1].set_xlabel("Time (seconds)")
    show_and_save("30_resampling")
    print("(Original audio variable was NOT modified.)")

    # =====================================================================
    # SECTION 31 -- CONCEPT MAP (text)
    # =====================================================================
    section("SECTION 31 -- Complete Concept Map")
    print("""
REAL WORLD SOUND -> MICROPHONE -> ANALOG SIGNAL -> SAMPLING -> SAMPLING RATE
-> QUANTIZATION -> BIT DEPTH -> DIGITAL AUDIO -> WAVEFORM -> FREQUENCY ANALYSIS
-> SPECTROGRAM -> AUDIO FEATURES -> DEEP LEARNING -> AUDIO RESEARCH
""")

    section("DONE")
    print(f"All figures and the report were saved to: {OUTPUT_DIR}")
    print("Next stage preview: FFT -> STFT -> Spectrogram -> Mel-Spectrogram.")


if __name__ == "__main__":
    main()
