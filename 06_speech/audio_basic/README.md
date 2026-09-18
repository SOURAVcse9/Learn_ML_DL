# 🎙️ Audio & Speech Processing Basics (audio_basic)

Welcome to the **Audio & Speech Processing Basics** module of **Learn_ML_DL**! This section covers foundational concepts, signal processing techniques, and feature extraction for audio, speech, and sound data using Python.

---

## 📌 Topics & Roadmap

- [ ] **Audio Fundamentals**:
  - Sound waves, Amplitude, Frequency ($Hz$), Period ($T$)
  - Sampling Theorem (Nyquist-Shannon), Sampling Rate (e.g., $16kHz, 22.05kHz, 44.1kHz$)
  - Bit Depth & Quantization (16-bit, 24-bit, 32-bit float)
  - Mono vs. Stereo Channels

- [ ] **Audio Signal Representations**:
  - **Time Domain**: Raw Waveform ($x(t)$)
  - **Frequency Domain**: Fast Fourier Transform (FFT) & Short-Time Fourier Transform (STFT)
  - **Time-Frequency Domain**: Spectrograms & Mel-Spectrograms
  - **Cepstral Domain**: Mel-Frequency Cepstral Coefficients (MFCCs), Chroma features, Spectral Centroid / Rolloff

- [ ] **Core Python Audio Libraries**:
  - `librosa`: Feature extraction and audio analysis
  - `torchaudio`: PyTorch audio processing pipeline
  - `soundfile` / `scipy.io.wavfile`: Reading & writing uncompressed audio formats

- [ ] **Hands-on Practice & Projects**:
  - Loading and visualizing raw audio waveforms
  - Generating and plotting Spectrograms and Mel-Spectrograms
  - Audio Data Augmentation (Time stretching, Pitch shifting, Noise injection)
  - Speech Emotion / Audio Classification baseline models

---

## 🚀 Quick Start Example (Librosa)

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Load an audio file (default sr=22050 Hz)
audio_path = "sample.wav"  # or librosa.example("trumpet")
y, sr = librosa.load(audio_path, sr=None)
print(f"Sample Rate: {sr} Hz, Audio Shape: {y.shape}, Duration: {len(y)/sr:.2f}s")

# 2. Plot Waveform (Time Domain)
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# 3. Compute and Plot Mel-Spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel-Spectrogram")
plt.tight_layout()
plt.show()
```

---
