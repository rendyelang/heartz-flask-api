import os
import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, sosfilt
import tensorflow as tf

# === PARAMETER BAKU DARI TIM DATA SCIENCE ===
TARGET_SR = 16000
BANDPASS_LOWCUT = 80
BANDPASS_HIGHCUT = 3000
BANDPASS_ORDER = 4
DENOISE_PROP_DECREASE = 0.8
DENOISE_STATIONARY = True
TARGET_RMS = 0.1

# === PARAMETER BAKU DARI TIM AI ===
EXPECTED_SAMPLES = 16000 # Fix 1 detik

def butter_bandpass(lowcut, highcut, fs, order=BANDPASS_ORDER):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def apply_bandpass(y, sr, lowcut=BANDPASS_LOWCUT, highcut=BANDPASS_HIGHCUT, order=BANDPASS_ORDER):
    sos = butter_bandpass(lowcut, highcut, sr, order)
    return sosfilt(sos, y)

def denoise_audio(y, sr, stationary=DENOISE_STATIONARY, prop_decrease=DENOISE_PROP_DECREASE):
    if len(y) < sr * 0.5:
        stationary = False
    return nr.reduce_noise(y=y, sr=sr, stationary=stationary, prop_decrease=prop_decrease)

def normalize_rms(y, target_rms=TARGET_RMS):
    rms = np.sqrt(np.mean(y**2))
    if rms <= 0:
        return y
    return y * (target_rms / rms)

def clean_audio_for_inference(file_path):
    """
    Fungsi gabungan: Preprocessing ala DS + Formatting ala AI
    Dipakai SAAT INFERENCE (Testing dengan suara user baru)
    """
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    
    # 2. Denoising (DS)
    y_denoised = denoise_audio(y, sr)
    
    # 3. Bandpass Filter (DS)
    y_filtered = apply_bandpass(y_denoised, sr)
    
    # 4. RMS Normalization (DS)
    y_normalized = normalize_rms(y_filtered)
    
    # 5. Fix Dimension untuk CNN (AI) -> Pad/Truncate ke 16.000 sampel
    audio_length = len(y_normalized)
    if audio_length < EXPECTED_SAMPLES:
        # Tambah keheningan di akhir kalau kurang dari 1 detik
        padding = EXPECTED_SAMPLES - audio_length
        y_final = np.pad(y_normalized, (0, padding), 'constant')
    else:
        # Ambil bagian tengah kalau kepanjangan
        start_idx = (audio_length - EXPECTED_SAMPLES) // 2
        y_final = y_normalized[start_idx : start_idx + EXPECTED_SAMPLES]
        
    # 6. Convert ke Tensor & tambah dimensi Batch
    audio_tensor = tf.convert_to_tensor(y_final, dtype=tf.float32)
    audio_batch = tf.expand_dims(audio_tensor, 0)
    
    return audio_batch, y_final
