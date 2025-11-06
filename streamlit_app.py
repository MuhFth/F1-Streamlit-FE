import streamlit as st
import requests
import numpy as np
import json
from typing import List

# 1. Configurasi Pages
st.set_page_config(
    page_title="F1 GP Winner Predictor 2025",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom F1 Styles
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
/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 2.5rem;
    color: #ff0000;
}
</style>
""", unsafe_allow_html=True)

# 3. Load API URL dan Konfigurasi
# Prioritas: secrets.toml > hardcoded URL
try:
    API_URL = st.secrets["api"]["FASTAPI_URL"]
    st.sidebar.success("✅ API URL loaded from secrets")
except:
    # Fallback ke Railway URL
    API_URL = "https://web-production-24d62.up.railway.app/predict"
    st.sidebar.info("ℹ️ Using default Railway API URL")

# Display API URL di sidebar untuk debugging
with st.sidebar:
    st.markdown("### 🔧 API Configuration")
    st.text(f"Backend: {API_URL}")
    st.info("📊 Sending 20 features (19 real + 1 dummy)")
    
    # Test connection button
    if st.button("🔌 Test API Connection"):
        with st.spinner("Testing..."):
            try:
                health_url = API_URL.replace("/predict", "/health")
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    st.success("✅ API is online!")
                    st.json(response.json())
                else:
                    st.error(f"❌ API returned {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")

# 4. Fungsi format Data
def format_input_data(inputs: dict) -> List[float]:
    """Convert dictionary to ordered list of 20 features for backend"""
    # Backend expects exactly 20 features in this order
    feature_order = [
        'Year',                    # 1
        'GridPosition',            # 2
        'LapTime (s)',            # 3
        'BestQuali (s)',          # 4
        'RacePace (s)',           # 5
        'Sector1Time (s)',        # 6
        'Sector2Time (s)',        # 7
        'Sector3Time (s)',        # 8
        'SectorTimeConsistency',  # 9
        'QualiAdvantage',         # 10
        'PositionImprovement',    # 11
        'RacePaceEfficiency',     # 12
        'Sector1Ratio',           # 13
        'Sector2Ratio',           # 14
        'Sector3Ratio',           # 15
        'TimeDiffFromFastest',    # 16
        'DriverEncoded',          # 17
        'AvgPrevPositions',       # 18
        'AvgPrevPoints',          # 19
        'Dummy'                   # 20 - Placeholder untuk backend
    ]
    
    result = []
    for key in feature_order:
        if key == 'Dummy':
            result.append(0.0)  # Add dummy 20th feature
        else:
            result.append(float(inputs[key]))
    
    # Validate feature count
    if len(result) != 20:
        raise ValueError(f"Expected 20 features, got {len(result)}")
    
    return result

# 5. Tampilan Utama
st.markdown("<h1><span style='color:white'>PREDIKSI PEMENANG </span>F1 MEXICO CITY GP 2025</h1>", unsafe_allow_html=True)
st.markdown("### 🚦 Simulasi Kinerja Driver (Input Formula)")

# Gunakan Columns untuk Layout
col1, col2, col3 = st.columns(3)

# Input blok 1 : Posisi & Waktu utama
with col1:
    st.markdown("### Posisi Awal 🏁")
    GridPosition = st.slider("Grid Position (Kualifikasi)", 1, 20, 5)
    
    st.markdown("### Kecepatan 🔥")
    LapTime = st.number_input("Best Lap Time (s)", value=80.500, step=0.001, format="%.3f")
    BestQuali = st.number_input("Best Quali Time (s)", value=79.900, step=0.001, format="%.3f")

# Input Blok 2: Kecepatan Balapan & Sektor
with col2:
    st.markdown("### Pace Balapan 🏎️")
    RacePace = st.number_input("Race Pace (s)", value=81.200, step=0.001, format="%.3f")
    
    st.markdown("### Sektor Waktu ⏱️")
    Sector1Time = st.number_input("Sector 1 Time (s)", value=25.000, step=0.001, format="%.3f")
    Sector2Time = st.number_input("Sector 2 Time (s)", value=28.000, step=0.001, format="%.3f")
    Sector3Time = st.number_input("Sector 3 Time (s)", value=27.500, step=0.001, format="%.3f")

# Input Blok 3: Performa Historis & Driver ID
with col3:
    st.markdown("### Performa Driver 🏆")
    DriverEncoded = st.slider("Driver ID (Encoded)", 0, 19, 10, 
                              help="Representasi numerik driver (0=ALO, 1=VER, dst.)")
    AvgPrevPositions = st.number_input("Avg Prev Positions", value=5.0, step=0.1, format="%.1f")
    AvgPrevPoints = st.number_input("Avg Prev Points", value=15.0, step=0.1, format="%.1f")

    # Nilai Tetap
    Year = 2025.0
    TimeDiffFromFastest = st.number_input("Time Diff From Fastest (s)", value=0.5, step=0.1, 
                                          format="%.1f", 
                                          help="Perbedaan waktu dari lap tercepat")

# 6. Logika Feature Engineering
total_sector_time = Sector1Time + Sector2Time + Sector3Time
SectorTimeConsistency = np.std([Sector1Time, Sector2Time, Sector3Time])
QualiAdvantage = BestQuali - LapTime
PositionImprovement = GridPosition - 5
RacePaceEfficiency = RacePace / LapTime
Sector1Ratio = Sector1Time / total_sector_time
Sector2Ratio = Sector2Time / total_sector_time
Sector3Ratio = Sector3Time / total_sector_time

# Merge semua input ke dalam satu dictionary
user_inputs = {
    'Year': Year,
    'GridPosition': float(GridPosition),
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

# 7. Button untuk Predict
st.markdown("---")

if st.button("🔴 START PREDICTION (GO! GO! GO!)"):
    if API_URL is None:
        st.warning("⚠️ Koneksi API Gagal. Cek URL di secrets.toml.")
    else:
        # Format data
        input_data_list = format_input_data(user_inputs)
        payload = {"features": input_data_list}
        
        # Debug info (bisa di-comment untuk production)
        with st.expander("🔍 Debug Info - Request Payload"):
            st.json(payload)
            st.text(f"Total features: {len(input_data_list)}")
        
        with st.spinner("🔧 Mengolah data di backend..."):
            try:
                # Kirim request ke API dengan timeout
                response = requests.post(
                    API_URL, 
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10  # 10 second timeout
                )
                
                # Check status code
                if response.status_code == 200:
                    result = response.json()
                    win_prob = result.get("winner_probability", 0)
                    
                    # Tampilan Hasil dengan Gaya F1
                    st.markdown("## 🏆 HASIL PREDIKSI PEMENANG 🏆")
                    
                    col_res1, col_res2 = st.columns([1, 3])
                    
                    with col_res1:
                        st.metric(
                            label="Probabilitas Menang",
                            value=f"{win_prob*100:.2f}%",
                            delta="Target: 100%",
                            delta_color="off"
                        )
                    
                    with col_res2:
                        if win_prob > 0.35:
                            st.success(
                                f"**CHEQUERED FLAG!** 🏁 Peluang kemenangan sangat tinggi "
                                f"({win_prob*100:.1f}%). Driver ini diprediksi menjadi pemenang."
                            )
                        elif win_prob > 0.15:
                            st.info(
                                f"**TOP 3 POTENTIAL.** 🥉 Peluang sedang ({win_prob*100:.1f}%). "
                                f"Driver ini berpotensi kuat untuk podium."
                            )
                        else:
                            st.warning(
                                f"**MID-PACK.** ⚠️ Peluang rendah ({win_prob*100:.1f}%). "
                                f"Driver ini kemungkinan finis di luar 5 besar."
                            )
                    
                    # Show full response
                    with st.expander("📊 Full API Response"):
                        st.json(result)
                
                else:
                    # Handle error responses
                    st.error(f"❌ API Error: Status Code {response.status_code}")
                    try:
                        error_detail = response.json()
                        st.json(error_detail)
                    except:
                        st.text(response.text)
                
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timeout! API tidak merespons dalam 10 detik.")
                st.info("Cek apakah backend Railway masih aktif.")
                
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Connection Error: Tidak bisa connect ke {API_URL}")
                st.info("Pastikan:\n- URL benar\n- Backend Railway sudah di-deploy\n- Backend tidak sleep")
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Request Error: {str(e)}")
                
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON response from API")
                st.text(response.text)
                
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
                st.exception(e)

# 8. Footer Info dengan Watermark
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px 0;'>
    <p style='font-size: 14px; margin-bottom: 8px;'>🏎️ F1 Winner Predictor 2025 | Powered by Machine Learning</p>
    <p style='font-size: 12px; margin-bottom: 15px;'>Backend: Flask + Waitress on Railway | Frontend: Streamlit</p>
    <div style='border-top: 1px solid #444; padding-top: 15px; margin-top: 10px;'>
        <p style='font-size: 13px; color: #ff0000; font-weight: bold; letter-spacing: 1px;'>
            © DGX 2025 • All Rights Reserved
        </p>
    </div>
</div>
""", unsafe_allow_html=True)