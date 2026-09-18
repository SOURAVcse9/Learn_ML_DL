# 🎛️ Audio Processing & Feature Engineering (audio_processing)

This module focuses on advanced **Audio Preprocessing, Signal Filtering, Audio Data Augmentation**, and deep feature engineering for Speech and Sound Analysis pipelines.

---

## 📌 Topics & Curriculum

- [ ] **Audio Preprocessing & Cleaning**:
  - Silence Removal / Trimming (`librosa.effects.trim`, VAD - Voice Activity Detection)
  - Pre-emphasis Filtering (High-frequency boosting)
  - Normalization (Peak normalization, RMS / Loudness LUFS normalization)
  - Noise Reduction (Spectral Gating, Wiener filtering)

- [ ] **Audio Data Augmentation Techniques**:
  - Time Stretching (Speed up / Slow down without altering pitch)
  - Pitch Shifting (Changing semitones without changing duration)
  - Random Noise Injection (White noise, Gaussian noise, Environmental background)
  - Room Impulse Response (RIR) & Reverberation simulation
  - SpecAugment (Time masking & Frequency masking on Spectrograms)

- [ ] **Advanced Feature Extraction**:
  - Filterbank Energies (Log Mel-Filterbanks)
  - Delta and Delta-Delta (Velocity & Acceleration features)
  - Zero Crossing Rate (ZCR), Spectral Roll-off, Spectral Flatness
  - Chroma Short-Time Fourier Transform (Chroma STFT)

- [ ] **End-to-End Pipelines**:
  - Audio feature extraction pipeline for PyTorch / TensorFlow datasets
  - Audio classification baseline (Speech Commands / Environmental Sound Classification)

---

## 🚀 Starter Code: Audio Augmentation with Librosa

```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Audio
y, sr = librosa.load("sample.wav", sr=22050)

# 2. Time Stretch (Faster & Slower)
y_fast = librosa.effects.time_stretch(y, rate=1.25)
y_slow = librosa.effects.time_stretch(y, rate=0.8)

# 3. Pitch Shift (Shift up by 2 semitones)
y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)

# 4. Add Random Gaussian Noise
noise = np.random.randn(len(y))
y_noisy = y + 0.005 * noise

print(f"Original shape: {y.shape}, Fast shape: {y_fast.shape}, Noisy shape: {y_noisy.shape}")
```

---
