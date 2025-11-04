import streamlit as st
import requests
import numpy as np
import json
from typing import List

#1. Configurasi Pages
st.set_page_config(
    page_title="F1 GP Winner Predictor 2025",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#2. Custom F1 Styles
st.markdown("""
<style>
/* Dark background dan warna teks */
.stApp {
    background-color: #1a1a1a;
    color: white;
}
/* Title and Header */
h1, h2, h3, h4, .st-emotion-cache-10trblm {
    color: #ff0000; /* Merah khas F1 */
    font-family: 'Arial Black', sans-serif;
    text-shadow: 2px 2px 4px #000000;
}
/* Tombol START PREDICTION */
div.stButton > button:first-child {
    background-color: #ff0000;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    border: 2px solid #ff0000;
    padding: 10px 20px;
}
div.stButton > button:first-child:hover {
    background-color: #e60000;
    border-color: #e60000;
}
/* Kotak Sidebar */
.st-emotion-cache-16niy5c {
    background-color: #2b2b2b; 
}
</style>
""", unsafe_allow_html=True)

#3. Load API URL dan Konfigurasi
try:
    API_URL = st.secrets["api"]["FASTAPI_URL"]
except :
    st.error(" API URL tidak ditemukan di secrets.toml. Gagal memuat model.")
    API_URL  = None

#4. Fungsi format Data
def format_input_data(input: dict) -> List[float]:
    feature_order = [
        'Year', 'GridPosition', 'LapTime (s)', 'BestQuali (s)', 'RacePace (s)',
        'Sector1Time (s)', 'Sector2Time (s)', 'Sector3Time (s)', 'SectorTimeConsistency',
        'QualiAdvantage', 'PositionImprovement', 'RacePaceEfficiency', 'Sector1Ratio',
        'Sector2Ratio', 'Sector3Ratio', 'TimeDiffFromFastest', 'DriverEncoded', 
        'AvgPrevPositions', 'AvgPrevPoints'
    ]
    return [inputs[key] for key in feature_keys]

#5. Tampilan Utama
st.markdown("<h1><span style='color:white'>PREDIKSI PEMENANG </span>F1 MEXICO CITY GP 2025</h1>", unsafe_allow_html=True)
st.markdown("### 🚦 Simulasi Kinerja Driver (Input Formula)")

# Gunakan Columns untuk Layout
col1, col2, col3 = st.columns(3)

# Input blok 1 : Posisi & Waktu utama
with col1:
    st.markdown("### Posisi Awal 🏁", unsafe_allow_html=True)
    GridPosition = st.slider("Grid Position (Kualifikasi)", 1, 20, 5)
    
    st.markdown("### Kecepatan 🔥", unsafe_allow_html=True)
    LapTime = st.number_input("Best Lap Time (s)", value=80.500, step=0.001, format="%.3f")
    BestQuali = st.number_input("Best Quali Time (s)", value=79.900, step=0.001, format="%.3f")

# --- Input Blok 2: Kecepatan Balapan & Sektor ---
with col2:
    st.markdown("### Pace Balapan 🏎️", unsafe_allow_html=True)
    RacePace = st.number_input("Race Pace (s)", value=81.200, step=0.001, format="%.3f")
    
    st.markdown("### Sektor Waktu ⏱️", unsafe_allow_html=True)
    Sector1Time = st.number_input("Sector 1 Time (s)", value=25.000, step=0.001, format="%.3f")
    Sector2Time = st.number_input("Sector 2 Time (s)", value=28.000, step=0.001, format="%.3f")
    Sector3Time = st.number_input("Sector 3 Time (s)", value=27.500, step=0.001, format="%.3f")

# --- Input Blok 3: Performa Historis & Driver ID ---
with col3:
    st.markdown("### Performa Driver 🏆", unsafe_allow_html=True)
    DriverEncoded = st.slider("Driver ID (Encoded)", 0, 19, 10, help="Representasi numerik driver (0=ALO, 1=VER, dst.)")
    AvgPrevPositions = st.number_input("Avg Prev Positions", value=5.0, step=0.1, format="%.1f")
    AvgPrevPoints = st.number_input("Avg Prev Points", value=15.0, step=0.1, format="%.1f")

    # Nilai Tetap (Diambil dari rata-rata atau asumsi di notebook Anda)
    Year = 2025.0
    TimeDiffFromFastest = st.number_input("Time Diff From Fastest (s)", value=0.5, step=0.1, format="%.1f", help="Asumsi perbedaan waktu dari lap tercepat rata-rata di tahun 2025.")

#6. Logika Feature Enginnering
total_sector_time = Sector1Time + Sector2Time + Sector3Time
SectorTimeConsistency = np.std([Sector1Time, Sector2Time, Sector3Time])
QualiAdvantage = BestQuali - LapTime
PositionImprovement = GridPosition - 5  # Asumsi posisi finish rata-rata adalah 10
RacePaceEfficiency = RacePace / LapTime
Sector1Ratio = Sector1Time / total_sector_time
Sector2Ratio = Sector2Time / total_sector_time
Sector3Ratio = Sector3Time / total_sector_time

# Merge semua input ke dalam satu dictionary
user_inputs = {
    'Year': Year,
    'GridPosition': GridPosition,
    'LapTime (s)': LapTime,
    'BestQuali (s)': BestQuali,
    'RacePace (s)': RacePace,
    'Sector1Time (s)': Sector1Time,     
    'Sector2Time (s)': Sector2Time,
    'Sector3Time (s)': Sector3Time,
    'SectorTimeConsistency': SectorTimeConsistency,
    'QualiAdvantage': QualiAdvantage,
    'PositionImprovement': float(PositionImprovement),
    'RacePaceEfficiency': RacePaceEfficiency,
    'Sector1Ratio': Sector1Ratio,
    'Sector2Ratio': Sector2Ratio,
    'Sector3Ratio': Sector3Ratio,
    'TimeDiffFromFastest': TimeDiffFromFastest,
    'DriverEncoded': float(DriverEncoded),
    'AvgPrevPositions': AvgPrevPositions,
    'AvgPrevPoints': AvgPrevPoints
}

#7. Button buat Predict
st.markdown("---")

if st.button ("🔴 START PREDICTION (GO! GO! GO!)"):
    if API_URL is None:
        st.warning("⚠️ Koneksi API Gagal. Cek URL di secrets.toml.")
    else:
        input_data_list = format_input_data(user_inputs)
        payload = {"features": input_data_list}
        
        with st.spinner("🔧 Mengolah data di backend...") :
            try:
                #Kirim request ke API
                response = request.post(API_URL, json=payload)
                response.raise_for_status()
                
                result = response.json()
                win_prob = result.get("winner_probability", 0)
                
                # Tampilan Hasil dengan Gaya F1
                st.markdown("## 🏆 HASIL PREDIKSI PEMENANG  🏆", unsafe_allow_html=True)
                
                col_res1, col_res2 = st.columns([1, 3])
                
                with col_res1:
                    st.metric(label = "Probabilitas Menang",
                            value=f"{win_prob*100:.2f} %",
                            delta = "Target : 100%", delta_color = "off")

                with col_res2:
                    if win_prob > 0.35:
                        st.success(f"**CHEQUERED FLAG!** Peluang kemenangan sangat tinggi ({win_prob*100:.1f}%). Driver ini diprediksi menjadi pemenang.")
                    elif win_prob > 0.15:
                        st.info(f"**TOP 3 POTENTIAL.** Peluang sedang ({win_prob*100:.1f}%). Driver ini berpotensi kuat untuk podium.")
                    else:
                        st.warning(f"**MID-PACK.** Peluang rendah ({win_prob*100:.1f}%). Driver ini kemungkinan finis di luar 5 besar.")

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Kesalahan Koneksi API: Gagal terhubung ke {API_URL}. Detail: {e}")
                st.info("Pastikan layanan Cloud Run (FastAPI) Anda aktif dan dapat diakses publik.")