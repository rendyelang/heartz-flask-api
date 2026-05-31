import os
import asyncio
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv

# Import fungsi andalan tim DS dari preprocessing.py
from preprocessing import clean_audio_for_inference

# ==============================================================================
# BONGKAR RAHASIA COLAB: Mendefinisikan Custom Layer MelSpectrogram
# ==============================================================================
@tf.keras.utils.register_keras_serializable()
class MelSpectrogramLayer(tf.keras.layers.Layer):
    def __init__(self, sample_rate=16000, frame_length=256, frame_step=128, num_mel_bins=64, lower_freq=80.0, upper_freq=8000.0, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_freq = lower_freq
        self.upper_freq = upper_freq

    def call(self, audio):
        stfts = tf.signal.stft(audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.frame_length)
        spectrograms = tf.abs(stfts)
        
        num_spectrogram_bins = stfts.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, num_spectrogram_bins, self.sample_rate, self.lower_freq, self.upper_freq)
        
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        
        mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        return tf.expand_dims(mel_spectrograms, -1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_rate": self.sample_rate,
            "frame_length": self.frame_length,
            "frame_step": self.frame_step,
            "num_mel_bins": self.num_mel_bins,
            "lower_freq": self.lower_freq,
            "upper_freq": self.upper_freq,
        })
        return config

# ==============================================================================
# INISIALISASI FLASK & AI
# ==============================================================================
load_dotenv()
app = Flask(__name__)
CORS(app)

# Konfigurasi Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

llm_model = genai.GenerativeModel('gemini-3.5-flash')

# Konfigurasi Local VM (Ollama) — prioritas utama
LOCAL_VM_URL = os.getenv("LOCAL_VM_URL")

# Load Model AI dengan mengenalkan Custom Layer-nya
print("Memuat model Heartz...")
model = tf.keras.models.load_model(
    "Heartz_Model_Ready.keras",
    custom_objects={'MelSpectrogramLayer': MelSpectrogramLayer}
)
print("Model berhasil dimuat!")

# Load Daftar Label
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]


# ==============================================================================
# FUNGSI ASYNC: generate_motivation (System Rules — Fallback Logic)
# ==============================================================================
# Urutan: LOCAL_VM_URL (Ollama, timeout 5s) ──> GEMINI_API_KEY (cloud fallback)
# ==============================================================================
async def generate_motivation(hasil_latihan: dict) -> str:
    """
    Membangkitkan kata-kata semangat berdasarkan hasil latihan suara user.

    Fallback Logic (sesuai System Rules):
        1. Tembak LOCAL_VM_URL (Ollama di GCE VM) dengan timeout 30 detik.
        2. Jika timeout / error apapun  →  fallback ke Gemini API (cloud).

    Args:
        hasil_latihan: dict berisi minimal:
            - "target_label" (str): suku kata yang ingin dilatih
            - "predicted_label" (str): suku kata yang diprediksi, misal "ba"
            - "confidence" (float): tingkat akurasi 0.0 – 1.0

    Returns:
        str: satu kalimat motivasi yang ramah untuk user.
    """

    target_label = hasil_latihan.get("target_label", hasil_latihan["predicted_label"])
    predicted_label = hasil_latihan["predicted_label"]
    confidence = hasil_latihan["confidence"]

    if target_label == predicted_label:
        prompt = (
            f"Seorang teman Tuli baru saja berhasil melafalkan suku kata '{target_label}' "
            f"dengan tingkat akurasi {confidence * 100:.1f}%. Berikan satu kalimat "
            f"singkat, ramah, dan memotivasi untuk memujinya."
        )
    else:
        prompt = (
            f"Seorang teman Tuli sedang berlatih melafalkan suku kata '{target_label}'. "
            f"Namun, pelafalannya terdengar seperti '{predicted_label}' dengan tingkat kemiripan {confidence * 100:.1f}%. "
            f"Berikan satu kalimat singkat, ramah, dan memotivasi untuk mengoreksinya dan menyemangatinya agar mencoba lagi."
        )

    # ------------------------------------------------------------------
    # LANGKAH 1 — Coba LOCAL_VM_URL (Ollama) dengan timeout 30 detik
    # ------------------------------------------------------------------
    if LOCAL_VM_URL:
        try:
            # requests.post bersifat blocking, jalankan di thread pool
            # agar tidak memblokir event loop asyncio.
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,  # default ThreadPoolExecutor
                lambda: requests.post(
                    LOCAL_VM_URL,
                    json={
                        "model": "gemma3:4b",
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=30,  # diperpanjang agar lebih sabar menunggu VM
                ),
            )
            response.raise_for_status()
            data = response.json()

            # Ollama mengembalikan field "response" untuk non-streaming
            motivation_text = data.get("response", "").strip()
            if motivation_text:
                print("[Motivation] Sumber: LOCAL_VM (Ollama)")
                return motivation_text

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as net_err:
            print(f"[Motivation] Local VM timeout/koneksi gagal: {net_err}")
        except Exception as e:
            print(f"[Motivation] Local VM error tak terduga: {e}")

    # ------------------------------------------------------------------
    # LANGKAH 2 — Fallback ke Gemini API (cloud)
    # ------------------------------------------------------------------
    try:
        # genai bersifat blocking; jalankan di thread pool juga
        loop = asyncio.get_event_loop()
        gemini_response = await loop.run_in_executor(
            None,
            lambda: llm_model.generate_content(prompt),
        )
        motivation_text = gemini_response.text.strip()
        print("[Motivation] Sumber: GEMINI API (cloud fallback)")
        return motivation_text

    except Exception as e:
        print(f"[Motivation] Gemini API juga gagal: {e}")
        # Fallback statis terakhir agar user tidak mendapat respons kosong
        if target_label == predicted_label:
            return (
                f"Hebat! Kamu sudah berhasil mengucapkan '{target_label}' dengan sangat baik. "
                f"Terus semangat, ya!"
            )
        else:
            return (
                f"Kamu sudah berani berlatih mengucapkan '{target_label}'. "
                f"Masih terdengar seperti '{predicted_label}', tapi jangan menyerah ya! Ayo coba lagi!"
            )


# ==============================================================================
# ENDPOINT API
# ==============================================================================

# 1. Endpoint Base (Welcome Message)
@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to Heartz ML API"}), 200

# 2. Endpoint Prediksi
@app.route('/predict', methods=['POST'])
async def predict_audio(): 
    if 'audio' not in request.files:
        return jsonify({"error": "File audio tidak ditemukan"}), 400
        
    if 'target_label' not in request.form:
        return jsonify({"error": "Parameter target_label tidak ditemukan"}), 400
        
    target_label = request.form['target_label']
    file = request.files['audio']
    temp_path = "temp_inference.wav"
    
    try:
        file.save(temp_path)
        audio_batch, _ = clean_audio_for_inference(temp_path)
        
        predictions = model.predict(audio_batch)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_label = class_names[predicted_index]
        confidence = float(predictions[predicted_index])

        hasil_latihan = {
            "target_label": target_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
        }
        
        motivation_text = await generate_motivation(hasil_latihan) 
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({
            "status": "success",
            "prediction": predicted_label,
            "confidence": confidence,
            "motivation_message": motivation_text
        }), 200
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)