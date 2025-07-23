import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Untuk load model
from sklearn.linear_model import LinearRegression

# Load model & fitur (pastikan simpan sebelumnya)
model = joblib.load('model_rumah.pkl')  # model LinearRegression
feature_names = joblib.load('fitur_model.pkl')  # List nama kolom input (21 kolom)

# Judul
st.title("Prediksi Harga Rumah di Kota Depok")

# Input user
k_tidur = st.number_input('Jumlah Kamar Tidur', min_value=0)
l_tanah = st.number_input('Luas Tanah (m2)', min_value=0)
l_bangunan = st.number_input('Luas Bangunan (m2)', min_value=0)
k_mandi = st.number_input('Jumlah Kamar Mandi', min_value=0)

daerah = st.selectbox(
    'Pilih Daerah', 
    ['Beji', 'Bojong Sari', 'Cibubur', 'Cilodong', 'Cimanggis', 'Cinangka', 
    'Cinere', 'Citayam', 'Cipayung', 'Harjamukti', 'Kukusan', 'Limo', 
    'Pancoran Mas', 'Rangkapan Jaya', 'Sawangan', 'Sukmajaya', 'Sukatani', 
    'Tanah Baru', 'Tapos', 'Tirtajaya'],
    index=None, placeholder="Masukkan Daerah"
)

# Siapkan input dengan semua kolom fitur
input_dict = {col: 0 for col in feature_names}
input_dict['k_tidur'] = k_tidur
input_dict['l_tanah'] = l_tanah
input_dict['l_bangunan'] = l_bangunan
input_dict['k_mandi'] = k_mandi

# One-hot daerah
daerah_col = f'daerah_{daerah}'
if daerah_col in input_dict:
    input_dict[daerah_col] = 1 

input_df = pd.DataFrame([input_dict])

# Prediksi
if st.button("Prediksi Harga"):
    prediksi = model.predict(input_df)[0]
    st.success(f"Perkiraan Harga Rumah: Rp {int(prediksi):,}")