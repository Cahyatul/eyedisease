import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load Model (pastikan file lenet_model.h5 sudah ada di repo)
@st.cache_resource
def load_lenet():
    model = load_model("lenet_model.h5")
    return model

model = load_lenet()

# Kelas yang diprediksi
classes = ["Normal", "Diabetic Retinopathy", "Cataract", "Glaucoma"]

# Judul aplikasi
st.title("🔍 Prediksi Penyakit Mata dari Citra Fundus")
st.write("Upload citra fundus (512x512) untuk memprediksi kondisi mata.")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar fundus", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing
    img = img.resize((128, 128))  # samakan dengan input LeNet modifikasi
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediksi
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Output hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.success(f"✅ {classes[class_idx]} ({confidence:.2f}%)")

    # Tampilkan probabilitas semua kelas
    st.subheader("Probabilitas Tiap Kelas:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"- {classes[i]} : {prob*100:.2f}%")
