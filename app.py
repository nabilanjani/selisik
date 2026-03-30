import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Model Deteksi Pesan Rekayasa Sosial",
    page_icon="🛡️",
    layout="wide"   # WAJIB WIDE BIAR 2 KOLOM KELIHATAN
)

# =========================================================
# 2. NLTK SETUP
# =========================================================
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return stopwords.words('indonesian')

list_stopwords = download_nltk_data()

# =========================================================
# 3. MODEL CONFIG
# =========================================================
MAX_LEN = 100
CLASS_NAMES = ['Normal', 'Penipuan', 'Promosi']

# =========================================================
# 4. PREPROCESSING
# =========================================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = " ".join(text.split())

    tokens = text.split()
    tokens = [word for word in tokens if word not in list_stopwords]
    return " ".join(tokens)

# =========================================================
# 5. LOAD MODEL & TOKENIZER
# =========================================================
@st.cache_resource
def load_resources():
    model = load_model('best_model_fixed.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

# =========================================================
# 6. HEADER
# =========================================================
st.title("🛡️ Model Deteksi Pesan Rekayasa Sosial")
st.caption("Deteksi otomatis pesan Normal, Penipuan, dan Promosi menggunakan CNN + FastText")

# =========================================================
# 7. LOAD MODEL
# =========================================================
try:
    model, tokenizer = load_resources()
    st.success("Model berhasil dimuat dan siap digunakan.")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =========================================================
# 8. 2 KOLOM UTAMA
# =========================================================
col_input, col_result = st.columns(2)

# =========================================================
# 9. KOLOM KIRI - INPUT
# =========================================================
with col_input:
    st.subheader("📨 Analisis Pesan")

    input_text = st.text_area(
        "Masukkan teks pesan:",
        height=180,
        placeholder="Contoh: Selamat! Anda memenangkan hadiah. Klik link berikut..."
    )

    detect_btn = st.button("🔍 Deteksi Pesan", use_container_width=True)

# =========================================================
# 10. KOLOM KANAN - HASIL
# =========================================================
with col_result:
    st.subheader("📌 Hasil Analisis")

    if detect_btn:
        if not input_text.strip():
            st.warning("Silakan masukkan pesan terlebih dahulu.")
        else:
            cleaned_text = preprocess_text(input_text)

            sequences = tokenizer.texts_to_sequences([cleaned_text])
            processed_data = pad_sequences(
                sequences,
                maxlen=MAX_LEN,
                padding='post'
            )

            prediction_probs = model.predict(processed_data)
            predicted_class_index = np.argmax(prediction_probs, axis=1)[0]
            confidence = np.max(prediction_probs)

            result_label = CLASS_NAMES[predicted_class_index]

            if result_label == 'Penipuan':
                st.error("🚨 **PENIPUAN TERDETEKSI**")
                st.write("Pesan ini memiliki karakteristik kuat sebagai pesan rekayasa sosial.")
            elif result_label == 'Promosi':
                st.warning("⚠️ **PESAN PROMOSI**")
                st.write("Pesan terindikasi sebagai promosi atau iklan.")
            else:
                st.success("✅ **PESAN AMAN (NORMAL)**")
                st.write("Tidak ditemukan indikasi rekayasa sosial yang signifikan.")

            st.metric(
                label="Tingkat Keyakinan Model",
                value=f"{confidence * 100:.2f}%"
            )

# =========================================================
# 11. DETAIL TEKNIS (TETAP DI BAWAH)
# =========================================================
if detect_btn and input_text.strip():
    with st.expander("🔎 Detail Teknis (Opsional)"):
        st.markdown("**Teks setelah preprocessing:**")
        st.code(cleaned_text)

        probs_df = pd.DataFrame(prediction_probs, columns=CLASS_NAMES)
        st.markdown("**Distribusi Probabilitas Kelas:**")
        st.bar_chart(probs_df.T)

# =========================================================
# 12. MODEL PERFORMANCE
# =========================================================
st.markdown("---")
with st.expander("📊 Performa Model (Training History)"):
    try:
        df_history = pd.read_csv('best_model_fixed.csv')

        tab1, tab2 = st.tabs(["Akurasi", "Loss"])

        with tab1:
            st.line_chart(df_history[['accuracy', 'val_accuracy']])

        with tab2:
            st.line_chart(df_history[['loss', 'val_loss']])

    except FileNotFoundError:
        st.info("File history training belum tersedia.")
