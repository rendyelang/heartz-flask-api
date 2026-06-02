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
EXPECTED_SAMPLES = 16000  # Fix 1 detik

# === PARAMETER SMART CROP ===
TRIM_TOP_DB = 20  # Threshold dB untuk mendeteksi suara vs keheningan


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
    """
    RMS Normalization dengan silence guard.
    Jika RMS terlalu rendah (< 1e-4), artinya audio hampir silence total,
    maka JANGAN amplifikasi agar noise tidak meledak 1000x lipat.
    """
    rms = np.sqrt(np.mean(y**2))
    if rms < 1e-4:
        # Audio hampir silence — skip normalization agar noise tidak ter-amplifikasi
        return y
    return y * (target_rms / rms)


def smart_crop(y, sr, target_samples, top_db=TRIM_TOP_DB):
    """
    Voice Activity Detection (VAD) based smart crop.
    
    Alih-alih blindly mengambil bagian tengah audio (center crop),
    fungsi ini mendeteksi di mana suara sebenarnya berada, lalu
    memotong window 1 detik di sekitar area suara tersebut.
    
    Strategi:
        1. Trim leading/trailing silence menggunakan librosa.effects.trim
        2. Jika trimmed speech <= target_samples: right-pad (konsisten dengan training)
        3. Jika trimmed speech > target_samples: cari window 1s dengan energi tertinggi
        4. Jika tidak ada suara terdeteksi (pure silence): return zeros
    
    Args:
        y: audio signal (numpy array)
        sr: sample rate
        target_samples: jumlah sampel yang diinginkan (16000 = 1 detik)
        top_db: threshold dB untuk mendeteksi suara vs keheningan
        
    Returns:
        numpy array dengan panjang tepat target_samples
    """
    # Step 1: Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    
    if len(y_trimmed) == 0:
        # Tidak ada suara terdeteksi — kembalikan silence
        print("[SmartCrop] Tidak ada suara terdeteksi, mengembalikan silence.")
        return np.zeros(target_samples)
    
    if len(y_trimmed) <= target_samples:
        # Suara cukup pendek, muat dalam 1 detik
        # Gunakan RIGHT-PAD (konsisten dengan training pipeline)
        padding = target_samples - len(y_trimmed)
        result = np.pad(y_trimmed, (0, padding), 'constant')
        print(f"[SmartCrop] Suara ditemukan ({len(y_trimmed)} sampel), right-padded ke {target_samples}.")
        return result
    else:
        # Suara lebih panjang dari 1 detik — cari window dengan energi tertinggi
        energy = y_trimmed ** 2
        kernel = np.ones(target_samples) / target_samples
        smoothed = np.convolve(energy, kernel, mode='valid')
        best_start = int(np.argmax(smoothed))
        result = y_trimmed[best_start : best_start + target_samples]
        print(f"[SmartCrop] Suara panjang ({len(y_trimmed)} sampel), crop energi tertinggi mulai idx {best_start}.")
        return result


def clean_audio_for_inference(file_path):
    """
    Fungsi gabungan: Preprocessing ala DS + Smart Crop ala AI
    Dipakai SAAT INFERENCE (Testing dengan suara user baru)
    
    Pipeline:
        1. Load Audio (librosa, mono, 16kHz)
        2. Denoising (noisereduce)
        3. Bandpass Filter (Butterworth SOS)
        4. RMS Normalization (dengan silence guard)
        5. Smart Crop (VAD-based, right-pad konsisten dengan training)
        6. Convert ke Tensor & tambah dimensi Batch
    """
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    
    # 2. Denoising (DS)
    y_denoised = denoise_audio(y, sr)
    
    # 3. Bandpass Filter (DS)
    y_filtered = apply_bandpass(y_denoised, sr)
    
    # 4. RMS Normalization (DS) — dengan silence guard
    y_normalized = normalize_rms(y_filtered)
    
    # 5. Smart Crop (AI) -> VAD-based crop ke 16.000 sampel
    y_final = smart_crop(y_normalized, sr, EXPECTED_SAMPLES)
        
    # 6. Convert ke Tensor & tambah dimensi Batch
    audio_tensor = tf.convert_to_tensor(y_final, dtype=tf.float32)
    audio_batch = tf.expand_dims(audio_tensor, 0)
    
    return audio_batch, y_final
