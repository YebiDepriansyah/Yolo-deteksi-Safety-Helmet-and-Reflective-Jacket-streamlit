import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import requests
from pathlib import Path
import gdown
import io # Added for st.camera_input handling

# Download model from Google Drive if not present
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
    st.info("Mengunduh model dari Google Drive...")
    try:
        # Using gdown for potentially more robust download
        gdown.download(id="1JtYh2YZ1Lc-2UShqSiLFi5CaOntE1kAh", output=str(model_path), quiet=False)
        st.success("Model berhasil diunduh!")
    except Exception as e:
        st.error(f"❌ Gagal mengunduh model: {e}. Pastikan model tersedia di Google Drive ID tersebut.")
        st.stop() # Stop execution if model download fails

# Load the YOLO model
try:
    model = YOLO(model_path)
    st.sidebar.success("✅ Model YOLO berhasil dimuat!")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model YOLO: {e}")
    st.stop() # Stop the app if model can't be loaded

# Sidebar as Navbar
st.sidebar.title("🔍 Menu Deteksi")
option = st.sidebar.radio("Pilih Jenis Input:", ["📷 Gambar", "🎞️ Video", "📹 Kamera (Real-Time)"])

# Page Title
st.title("🦺 Deteksi Helm & Rompi dengan YOLOv11")
st.markdown("Aplikasi deteksi otomatis untuk helm dan rompi menggunakan model YOLOv11.")
st.markdown("---")

# --- Image Detection ---
if option == "📷 Gambar":
    st.header("📷 Deteksi pada Gambar")
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Open image using PIL for consistency
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        st.image(img_array, caption="Gambar Asli", use_column_width=True)
        
        with st.spinner("Mendeteksi objek di gambar..."):
            results = model.predict(img_array, conf=0.5)
            # results[0].plot() returns a numpy array (BGR format for OpenCV)
            result_img = results[0].plot() 

        st.image(result_img, caption="🟢 Hasil Deteksi", use_column_width=True, channels="BGR")

# --- Video Detection ---
elif option == "🎞️ Video":
    st.header("🎞️ Deteksi pada Video")
    uploaded_file = st.file_uploader("Unggah video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = Path("temp_video.mp4")
        with open(tfile, 'wb') as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(str(tfile)) # Ensure path is string
        stframe = st.empty()
        
        st.info("Memproses video... Ini mungkin membutuhkan waktu tergantung panjang video.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Selesai memproses video atau gagal membaca frame.")
                break

            results = model.predict(frame, conf=0.5)
            result_frame = results[0].plot()
            stframe.image(result_frame, channels="BGR", use_column_width=True)

        cap.release()
        tfile.unlink(missing_ok=True) # Clean up temp file
        st.success("✅ Deteksi video selesai!")

# --- Real-Time Camera Detection ---
elif option == "📹 Kamera (Real-Time)":
    st.header("📹 Deteksi Kamera (Real-Time)")
    st.warning("Fitur ini menggunakan kamera browser dan cocok untuk mengambil gambar sesaat. Untuk live stream berkelanjutan, ada keterbatasan di lingkungan deployment.")
    
    img_file_buffer = st.camera_input("Ambil Foto dari Kamera")

    if img_file_buffer is not None:
        # Read image file buffer as bytes
        bytes_data = img_file_buffer.getvalue()
        # Convert to PIL Image and then numpy array
        image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        img_array = np.array(image)

        st.image(img_array, caption="Gambar yang Diambil", use_column_width=True)

        with st.spinner("Mendeteksi objek di gambar dari kamera..."):
            results = model.predict(img_array, conf=0.5)
            result_img = results[0].plot()
        
        st.image(result_img, caption="🟢 Hasil Deteksi dari Kamera", use_column_width=True, channels="BGR")

# --- Footer Information ---
st.markdown("---")
st.markdown("#### 👨‍💻 Dibuat oleh:")
st.markdown("""
- ZONI ARYANTONI ALBAB (G1A022043)  
- YEBI DEPRIANSYAH (G1A022063)  
- AHMAD ZUL ZHAFRAN (G1A022088)  
""")
st.markdown("**Mata Kuliah:** Artificial Neural Network (ANN)  \n**Dosen Pengampu:** Ir. Arie Vatresia, S.T., M.T.I., Ph.D")
