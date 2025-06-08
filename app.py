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
@st.cache_resource
def download_model():
    model_path = Path("best.pt")
    if not model_path.exists():
        # Gunakan ID dari Google Drive link
        url = "https://drive.google.com/uc?id=1ZIzU9H-dbyaHdyILA2XoWpxxNxEZ1qOw"
        gdown.download(url, str(model_path), quiet=False)
    return YOLO(str(model_path))


# Load model
model = download_model()

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

# --- Deteksi Kamera Real-Time ---
elif option == "📹 Kamera (Real-Time)":
    st.header("📹 Deteksi Kamera (Real-Time)")
    run = st.checkbox("✅ Mulai Kamera")
    stframe = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Tidak dapat mengakses kamera.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Gagal membaca frame dari kamera.")
                    break

                results = model.predict(frame, conf=0.5)
                result_frame = results[0].plot()

                stframe.image(result_frame, channels="BGR", use_column_width=True)

            cap.release()

# --- Footer Informasi ---
st.markdown("---")
st.markdown("#### 👨‍💻 Dibuat oleh:")
st.markdown("""
- ZONI ARYANTONI ALBAB (G1A022043)  
- YEBI DEPRIANSYAH (G1A022063)  
- AHMAD ZUL ZHAFRAN (G1A022088)  
""")
st.markdown("**Mata Kuliah:** Artificial Neural Network (ANN)  \n**Dosen Pengampu:** Ir. Arie Vatresia, S.T., M.T.I., Ph.D")
