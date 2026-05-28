import os
import numpy as np
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
# ENDPOINT API
# ==============================================================================

# 1. Endpoint Base (Welcome Message)
@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to Heartz ML API"}), 200

# 2. Endpoint Prediksi
@app.route('/predict', methods=['POST'])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "File audio tidak ditemukan"}), 400
        
    file = request.files['audio']
    temp_path = "temp_inference.wav"
    
    try:
        file.save(temp_path)
        audio_batch, _ = clean_audio_for_inference(temp_path)
        
        predictions = model.predict(audio_batch)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_label = class_names[predicted_index]
        confidence = float(predictions[predicted_index])
        
        prompt = f"Seorang teman Tuli baru saja berlatih melafalkan suku kata '{predicted_label}' dengan tingkat akurasi {confidence * 100:.1f}%. Berikan satu kalimat singkat, ramah, dan memotivasi untuk menyemangatinya."
        response = llm_model.generate_content(prompt)
        motivation_text = response.text
        
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