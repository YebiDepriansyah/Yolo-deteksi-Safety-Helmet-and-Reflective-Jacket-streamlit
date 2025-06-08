import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import requests
from pathlib import Path
import gdown

# Download model dari Google Drive jika belum ada
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

model_path = Path("best.pt")
if not model_path.exists():
    download_file_from_google_drive("1JtYh2YZ1Lc-2UShqSiLFi5CaOntE1kAh", model_path)

# Load model sekali saja agar efisien
model = YOLO(str(model_path))

# Sidebar sebagai Navbar
st.sidebar.title("🔍 Menu Deteksi")
option = st.sidebar.radio("Pilih Jenis Input:", ["📷 Gambar", "🎞️ Video", "📹 Kamera (Real-Time)"])

# Judul Halaman
st.title("🦺 Deteksi Helm & Rompi dengan YOLOv11")
st.markdown("Aplikasi deteksi otomatis untuk helm dan rompi menggunakan model YOLOv11.")
st.markdown("---")

# --- Deteksi Gambar ---
if option == "📷 Gambar":
    st.header("📷 Deteksi pada Gambar")
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        results = model.predict(img_array, conf=0.5)
        result_img = results[0].plot()

        st.image(result_img, caption="🟢 Hasil Deteksi", use_column_width=True)

# --- Deteksi Video ---
elif option == "🎞️ Video":
    st.header("🎞️ Deteksi pada Video")
    uploaded_file = st.file_uploader("Unggah video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = open("temp_video.mp4", 'wb')
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture("temp_video.mp4")
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=0.5)
            result_frame = results[0].plot()
            stframe.image(result_frame, channels="BGR", use_column_width=True)

        cap.release()

# --- Deteksi Kamera Real-Time dengan st.camera_input ---
elif option == "📹 Kamera (Real-Time)":
    st.header("📹 Deteksi Kamera (Real-Time)")
    st.markdown("Tekan tombol di bawah untuk mengambil gambar dari kamera Anda.")
    
    img_file_buffer = st.camera_input("📸 Ambil Foto")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert("RGB")
        img_array = np.array(image)

        results = model.predict(img_array, conf=0.5)
        result_img = results[0].plot()

        st.image(result_img, caption="🟢 Hasil Deteksi dari Kamera", use_column_width=True)

# --- Footer Informasi ---
st.markdown("---")
st.markdown("#### 👨‍💻 Dibuat oleh:")
st.markdown("""
- ZONI ARYANTONI ALBAB (G1A022043)  
- YEBI DEPRIANSYAH (G1A022063)  
- AHMAD ZUL ZHAFRAN (G1A022088)  
""")
st.markdown("**Mata Kuliah:** Artificial Neural Network (ANN)  \n**Dosen Pengampu:** Ir. Arie Vatresia, S.T., M.T.I., Ph.D")
