# Gunakan base image Python resmi yang ringan
FROM python:3.11-slim

# Set working directory di dalam kontainer
WORKDIR /app

# Install system dependencies yang dibutuhkan oleh librosa & soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code ke dalam kontainer
COPY . .

# Set environment variable untuk Flask port (Cloud Run mewajibkan port dinamis via env)
ENV PORT 8080

# Jalankan server menggunakan Gunicorn (Standar Production agar Flask lu stabil)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app