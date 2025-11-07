import streamlit as st
import requests
import numpy as np
import json
from typing import List
import plotly.graph_objects as go
import plotly.express as px
import base64
from pathlib import Path

# 1. Configurasi Pages
st.set_page_config(
    page_title="F1 GP Winner Predictor 2025",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Function to load background image
def get_base64_image(image_path):
    """Convert image to base64 for CSS background"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Load background image (letakkan gambar dengan nama 'f1_background.jpg' di folder yang sama)
bg_image_path = Path("Ferrari.jpg")  # Ganti dengan nama file gambar Anda
bg_base64 = get_base64_image(bg_image_path)

# 3. Custom F1 Styles with Background
if bg_base64:
    background_style = f"""
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)), url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    """
else:
    background_style = """
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d0a0a 50%, #1a1a1a 100%);
    }
    """

st.markdown(f"""
<style>
{background_style}

/* Global text color */
.stApp {{
    color: white;
}}

/* Title and Header */
h1, h2, h3, h4, h5, h6 {{
    color: #ff0000 !important;
    font-family: 'Arial Black', sans-serif;
    text-shadow: 2px 2px 8px #000000;
}}

/* Input labels */
label {{
    color: white !important;
    font-weight: bold;
}}

/* Tombol START PREDICTION */
div.stButton > button:first-child {{
    background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
    color: white;
    font-weight: bold;
    font-size: 1.2rem;
    border-radius: 10px;
    border: 3px solid #ff0000;
    padding: 15px 40px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 0, 0, 0.5);
}}

div.stButton > button:first-child:hover {{
    background: linear-gradient(135deg, #cc0000 0%, #ff0000 100%);
    border-color: #ff3333;
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(255, 0, 0, 0.8);
}}

/* Metric styling */
[data-testid="stMetricValue"] {{
    font-size: 2.8rem;
    color: #ff0000;
    font-weight: bold;
    text-shadow: 2px 2px 4px #000000;
}}

[data-testid="stMetricLabel"] {{
    color: white !important;
    font-size: 1.1rem;
}}

/* Visualization container */
.viz-container {{
    background: rgba(43, 43, 43, 0.8);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    border: 2px solid rgba(255, 0, 0, 0.3);
    backdrop-filter: blur(10px);
}}

/* Section divider */
.section-divider {{
    height: 4px;
    background: linear-gradient(90deg, transparent 0%, #ff0000 20%, #ffffff 50%, #ff0000 80%, transparent 100%);
    margin: 50px 0;
    box-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
}}

/* About section card */
.about-card {{
    background: linear-gradient(135deg, rgba(43, 43, 43, 0.95) 0%, rgba(26, 26, 26, 0.95) 100%);
    padding: 25px;
    border-radius: 15px;
    border: 2px solid #ff0000;
    margin: 15px 0;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(255, 0, 0, 0.3);
}}

/* Input containers */
.stNumberInput, .stSlider {{
    background: rgba(43, 43, 43, 0.6);
    border-radius: 10px;
    padding: 10px;
}}

/* Header banner */
.header-banner {{
    background: linear-gradient(135deg, rgba(255, 0, 0, 0.9) 0%, rgba(0, 0, 0, 0.9) 100%);
    padding: 40px 20px;
    border-radius: 20px;
    margin-bottom: 30px;
    border: 3px solid #ff0000;
    box-shadow: 0 8px 30px rgba(255, 0, 0, 0.5);
}}

/* Result card */
.result-card {{
    background: linear-gradient(135deg, rgba(255, 0, 0, 0.15) 0%, rgba(0, 0, 0, 0.8) 100%);
    border: 3px solid #ff0000;
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 30px rgba(255, 0, 0, 0.6);
}}
</style>
""", unsafe_allow_html=True)

# 4. Load API URL
try:
    API_URL = st.secrets["api"]["FASTAPI_URL"]
except:
    API_URL = "https://web-production-24d62.up.railway.app/predict"

# 5. Fungsi format Data
def format_input_data(inputs: dict) -> List[float]:
    """Convert dictionary to ordered list of 20 features for backend"""
    feature_order = [
        'Year', 'GridPosition', 'LapTime (s)', 'BestQuali (s)', 'RacePace (s)',
        'Sector1Time (s)', 'Sector2Time (s)', 'Sector3Time (s)', 
        'SectorTimeConsistency', 'QualiAdvantage', 'PositionImprovement',
        'RacePaceEfficiency', 'Sector1Ratio', 'Sector2Ratio', 'Sector3Ratio',
        'TimeDiffFromFastest', 'DriverEncoded', 'AvgPrevPositions', 'AvgPrevPoints', 'Dummy'
    ]
    
    result = []
    for key in feature_order:
        if key == 'Dummy':
            result.append(0.0)
        else:
            result.append(float(inputs[key]))
    
    if len(result) != 20:
        raise ValueError(f"Expected 20 features, got {len(result)}")
    
    return result

# 6. Fungsi Visualisasi
def create_speedometer(probability):
    """Create a speedometer gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Win Probability", 'font': {'size': 28, 'color': 'white', 'family': 'Arial Black'}},
        delta = {'reference': 50, 'increasing': {'color': "#00ff00"}, 'font': {'size': 20}},
        number = {'font': {'size': 50, 'color': '#ff0000', 'family': 'Arial Black'}, 'suffix': '%'},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white", 'tickfont': {'size': 14}},
            'bar': {'color': "#ff0000", 'thickness': 0.8},
            'bgcolor': "rgba(255,255,255,0.2)",
            'borderwidth': 3,
            'bordercolor': "#ff0000",
            'steps': [
                {'range': [0, 15], 'color': 'rgba(150, 150, 150, 0.3)'},
                {'range': [15, 35], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [35, 100], 'color': 'rgba(0, 255, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"},
        height=350,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

def create_sector_comparison(s1, s2, s3):
    """Create sector time comparison bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Sector 1', 'Sector 2', 'Sector 3'],
            y=[s1, s2, s3],
            marker=dict(
                color=['#ff0000', '#ffffff', '#ff0000'],
                line=dict(color='#ff0000', width=3),
                pattern_shape=["", "/", ""]
            ),
            text=[f'{s1:.3f}s', f'{s2:.3f}s', f'{s3:.3f}s'],
            textposition='outside',
            textfont=dict(size=16, color='white', family='Arial Black')
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Sector Time Analysis",
            'font': {'size': 24, 'color': 'white', 'family': 'Arial Black'},
            'x': 0.5,
            'xanchor': 'center'
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(43,43,43,0.6)',
        font={'color': "white", 'size': 14},
        yaxis_title="Time (seconds)",
        yaxis_title_font={'size': 16, 'color': 'white'},
        xaxis_tickfont={'size': 14, 'color': 'white'},
        height=350,
        showlegend=False,
        margin=dict(l=50, r=30, t=80, b=50)
    )
    
    return fig

def create_performance_radar(inputs):
    """Create radar chart for driver performance"""
    categories = ['Qualifying', 'Race Pace', 'Consistency', 'Experience', 'Grid Position']
    
    # Normalize values to 0-100 scale
    quali_score = max(0, min(100, 100 - (inputs['BestQuali (s)'] - 75) * 10))
    pace_score = max(0, min(100, 100 - (inputs['RacePace (s)'] - 75) * 10))
    consistency_score = max(0, min(100, 100 - inputs['SectorTimeConsistency'] * 50))
    experience_score = min(100, (inputs['AvgPrevPoints'] / 25) * 100)
    grid_score = (21 - inputs['GridPosition']) * 5
    
    values = [quali_score, pace_score, consistency_score, experience_score, grid_score]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.4)',
        line=dict(color='#ff0000', width=3),
        marker=dict(size=8, color='#ff0000')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(255, 255, 255, 0.3)',
                tickfont={'size': 12, 'color': 'white'}
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.3)',
                tickfont={'size': 14, 'color': 'white', 'family': 'Arial Black'}
            ),
            bgcolor='rgba(43,43,43,0.6)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        title={
            'text': "Driver Performance Profile",
            'font': {'size': 24, 'color': 'white', 'family': 'Arial Black'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=450,
        margin=dict(l=80, r=80, t=100, b=50)
    )
    
    return fig

def create_comparison_metrics(inputs):
    """Create comparison bar chart for key metrics"""
    metrics = ['Quali Advantage', 'Race Efficiency', 'Position Improvement']
    values = [
        inputs['QualiAdvantage'] * 10,  # Scaled for visibility
        inputs['RacePaceEfficiency'] * 100,
        abs(inputs['PositionImprovement']) * 5
    ]
    colors = ['#ff0000' if v > 0 else '#666666' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            y=metrics,
            x=values,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f'{v:.1f}' for v in values],
            textposition='outside',
            textfont=dict(size=16, color='white', family='Arial Black')
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Key Performance Metrics",
            'font': {'size': 24, 'color': 'white', 'family': 'Arial Black'},
            'x': 0.5,
            'xanchor': 'center'
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(43,43,43,0.6)',
        font={'color': "white"},
        xaxis_title="Score",
        height=350,
        showlegend=False,
        margin=dict(l=150, r=50, t=80, b=50)
    )
    
    return fig

# ============ HEADER SECTION ============
st.markdown("""
<div class='header-banner'>
    <div style='text-align: center;'>
        <h1 style='font-size: 3.5rem; margin-bottom: 10px; color: white !important; text-shadow: 3px 3px 10px #000000;'>
            🏎️ F1 WINNER PREDICTOR 🏎️
        </h1>
        <h2 style='color: #ff0000 !important; font-weight: bold; margin-top: 5px; font-size: 2.2rem; text-shadow: 2px 2px 8px #000000;'>
            MEXICO CITY GRAND PRIX 2025
        </h2>
        <p style='color: white; font-size: 1.2rem; margin-top: 15px; font-weight: bold;'>
            ⚡ Predict • Analyze • Win ⚡
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============ INPUT SECTION ============
st.markdown("<h3 style='text-align: center; color: white !important; margin-bottom: 30px;'>🚦 DRIVER PERFORMANCE INPUT</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("#### 🏁 Starting Position")
    GridPosition = st.slider("Grid Position", 1, 20, 5, help="Posisi start pada kualifikasi")
    
    st.markdown("#### 🔥 Speed Performance")
    LapTime = st.number_input("Best Lap Time (s)", value=80.500, step=0.001, format="%.3f")
    BestQuali = st.number_input("Best Quali Time (s)", value=79.900, step=0.001, format="%.3f")

with col2:
    st.markdown("#### 🏎️ Race Performance")
    RacePace = st.number_input("Race Pace (s)", value=81.200, step=0.001, format="%.3f")
    
    st.markdown("#### ⏱️ Sector Times")
    Sector1Time = st.number_input("Sector 1 Time (s)", value=25.000, step=0.001, format="%.3f")
    Sector2Time = st.number_input("Sector 2 Time (s)", value=28.000, step=0.001, format="%.3f")
    Sector3Time = st.number_input("Sector 3 Time (s)", value=27.500, step=0.001, format="%.3f")

with col3:
    st.markdown("#### 🏆 Driver Statistics")
    
    # Driver ID dengan informasi nama
    driver_names = {
        0: "Max Verstappen (Red Bull)",
        1: "Sergio Pérez (Red Bull)",
        2: "Lewis Hamilton (Mercedes)",
        3: "George Russell (Mercedes)",
        4: "Charles Leclerc (Ferrari)",
        5: "Carlos Sainz (Ferrari)",
        6: "Lando Norris (McLaren)",
        7: "Oscar Piastri (McLaren)",
        8: "Fernando Alonso (Aston Martin)",
        9: "Lance Stroll (Aston Martin)",
        10: "Pierre Gasly (Alpine)",
        11: "Esteban Ocon (Alpine)",
        12: "Valtteri Bottas (Alfa Romeo)",
        13: "Zhou Guanyu (Alfa Romeo)",
        14: "Kevin Magnussen (Haas)",
        15: "Nico Hülkenberg (Haas)",
        16: "Yuki Tsunoda (AlphaTauri)",
        17: "Daniel Ricciardo (AlphaTauri)",
        18: "Alexander Albon (Williams)",
        19: "Logan Sargeant (Williams)"
    }
    
    DriverEncoded = st.slider("Driver ID", 0, 19, 10, help="Pilih ID driver (lihat daftar di bawah)")
    
    # Tampilkan nama driver yang dipilih
    st.markdown(f"""
    <div style='background: rgba(255, 0, 0, 0.2); padding: 10px; border-radius: 8px; border: 1px solid #ff0000; margin: 10px 0;'>
        <p style='color: white; font-size: 1rem; margin: 0; text-align: center; font-weight: bold;'>
            🏎️ <span style='color: #ff0000;'>{driver_names.get(DriverEncoded, "Unknown Driver")}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Expander untuk daftar lengkap driver
    with st.expander("📋 Lihat Daftar Lengkap Driver ID"):
        st.markdown("""
        <div style='background: rgba(43, 43, 43, 0.8); padding: 15px; border-radius: 10px;'>
        """, unsafe_allow_html=True)
        
        # Bagi menjadi 2 kolom
        driver_col1, driver_col2 = st.columns(2)
        
        with driver_col1:
            for i in range(0, 10):
                st.markdown(f"**ID {i}:** {driver_names[i]}")
        
        with driver_col2:
            for i in range(10, 20):
                st.markdown(f"**ID {i}:** {driver_names[i]}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Historical performance data berdasarkan Driver ID (2024 season average)
    driver_historical_data = {
        0: {"avg_pos": 1.5, "avg_points": 22.0},   # Verstappen
        1: {"avg_pos": 5.0, "avg_points": 10.0},   # Pérez
        2: {"avg_pos": 4.0, "avg_points": 12.0},   # Hamilton
        3: {"avg_pos": 5.5, "avg_points": 9.0},    # Russell
        4: {"avg_pos": 3.0, "avg_points": 15.0},   # Leclerc
        5: {"avg_pos": 4.5, "avg_points": 11.0},   # Sainz
        6: {"avg_pos": 3.5, "avg_points": 14.0},   # Norris
        7: {"avg_pos": 6.0, "avg_points": 8.0},    # Piastri
        8: {"avg_pos": 7.0, "avg_points": 6.0},    # Alonso
        9: {"avg_pos": 12.0, "avg_points": 2.0},   # Stroll
        10: {"avg_pos": 10.0, "avg_points": 4.0},  # Gasly
        11: {"avg_pos": 11.0, "avg_points": 3.0},  # Ocon
        12: {"avg_pos": 14.0, "avg_points": 1.0},  # Bottas
        13: {"avg_pos": 16.0, "avg_points": 0.5},  # Zhou
        14: {"avg_pos": 13.0, "avg_points": 1.5},  # Magnussen
        15: {"avg_pos": 11.0, "avg_points": 3.0},  # Hülkenberg
        16: {"avg_pos": 12.0, "avg_points": 2.0},  # Tsunoda
        17: {"avg_pos": 10.0, "avg_points": 4.0},  # Ricciardo
        18: {"avg_pos": 15.0, "avg_points": 0.8},  # Albon
        19: {"avg_pos": 18.0, "avg_points": 0.2},  # Sargeant
    }
    
    # Auto-calculate berdasarkan driver yang dipilih
    AvgPrevPositions = driver_historical_data[DriverEncoded]["avg_pos"]
    AvgPrevPoints = driver_historical_data[DriverEncoded]["avg_points"]
    Year = 2025.0

# Feature Engineering
total_sector_time = Sector1Time + Sector2Time + Sector3Time
SectorTimeConsistency = np.std([Sector1Time, Sector2Time, Sector3Time])
QualiAdvantage = BestQuali - LapTime
PositionImprovement = GridPosition - 5
RacePaceEfficiency = RacePace / LapTime
Sector1Ratio = Sector1Time / total_sector_time
Sector2Ratio = Sector2Time / total_sector_time
Sector3Ratio = Sector3Time / total_sector_time

# Auto-calculate TimeDiffFromFastest based on BestQuali
# Assuming fastest quali time is around 78.0s for Mexico City GP
fastest_quali_time = 78.0
TimeDiffFromFastest = max(0, BestQuali - fastest_quali_time)

user_inputs = {
    'Year': Year, 'GridPosition': float(GridPosition), 'LapTime (s)': LapTime,
    'BestQuali (s)': BestQuali, 'RacePace (s)': RacePace,
    'Sector1Time (s)': Sector1Time, 'Sector2Time (s)': Sector2Time, 'Sector3Time (s)': Sector3Time,
    'SectorTimeConsistency': SectorTimeConsistency, 'QualiAdvantage': QualiAdvantage,
    'PositionImprovement': float(PositionImprovement), 'RacePaceEfficiency': RacePaceEfficiency,
    'Sector1Ratio': Sector1Ratio, 'Sector2Ratio': Sector2Ratio, 'Sector3Ratio': Sector3Ratio,
    'TimeDiffFromFastest': TimeDiffFromFastest, 'DriverEncoded': float(DriverEncoded),
    'AvgPrevPositions': AvgPrevPositions, 'AvgPrevPoints': AvgPrevPoints
}

# ============ PREDICTION BUTTON ============
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🏁 START PREDICTION • LIGHTS OUT! 🏁", use_container_width=True)

# ============ PREDICTION RESULTS ============
if predict_button:
    input_data_list = format_input_data(user_inputs)
    payload = {"features": input_data_list}
    
    with st.spinner("🔧 Analyzing driver performance..."):
        try:
            response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                win_prob = result.get("winner_probability", 0)
                
                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                
                # Result Header
                st.markdown("""
                <div class='result-card'>
                    <h2 style='text-align: center; color: white !important; margin-bottom: 20px;'>
                        🏆 PREDICTION RESULTS 🏆
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Main Metrics
                metric_col1, metric_col2, metric_col3 = st.columns([1, 2, 1])
                
                with metric_col1:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.metric(
                        label="🎯 Win Probability",
                        value=f"{win_prob*100:.2f}%",
                        delta=f"{(win_prob*100 - 50):.1f}% vs avg"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with metric_col2:
                    if win_prob > 0.35:
                        st.success(
                            f"**🏁 CHEQUERED FLAG!** Peluang kemenangan sangat tinggi "
                            f"({win_prob*100:.1f}%). Driver ini diprediksi menjadi PEMENANG!"
                        )
                    elif win_prob > 0.15:
                        st.info(
                            f"**🥉 PODIUM POTENTIAL!** Peluang sedang ({win_prob*100:.1f}%). "
                            f"Driver ini berpotensi finish di TOP 3."
                        )
                    else:
                        st.warning(
                            f"**⚠️ MID-PACK FINISH.** Peluang rendah ({win_prob*100:.1f}%). "
                            f"Driver ini kemungkinan finish posisi 6-10."
                        )
                
                with metric_col3:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    predicted_position = int(1 + (1 - win_prob) * 10)
                    st.metric(
                        label="📊 Est. Position",
                        value=f"P{predicted_position}",
                        delta="Predicted"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Visualization Section - Organized Layout
                st.markdown("<h3 style='text-align: center; color: white !important; margin: 30px 0;'>📊 DETAILED ANALYSIS</h3>", unsafe_allow_html=True)
                
                # Row 1: Main visualizations
                viz_col1, viz_col2 = st.columns(2, gap="large")
                
                with viz_col1:
                    st.markdown("<div class='viz-container'>", unsafe_allow_html=True)
                    fig_speedometer = create_speedometer(win_prob)
                    st.plotly_chart(fig_speedometer, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with viz_col2:
                    st.markdown("<div class='viz-container'>", unsafe_allow_html=True)
                    fig_sectors = create_sector_comparison(Sector1Time, Sector2Time, Sector3Time)
                    st.plotly_chart(fig_sectors, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Row 2: Performance radar and metrics
                viz_col3, viz_col4 = st.columns(2, gap="large")
                
                with viz_col3:
                    st.markdown("<div class='viz-container'>", unsafe_allow_html=True)
                    fig_radar = create_performance_radar(user_inputs)
                    st.plotly_chart(fig_radar, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with viz_col4:
                    st.markdown("<div class='viz-container'>", unsafe_allow_html=True)
                    fig_comparison = create_comparison_metrics(user_inputs)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            else:
                st.error(f"❌ API Error: Status Code {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============ ABOUT F1 SECTION ============
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white !important; margin-bottom: 40px;'>🏎️ ABOUT FORMULA 1</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='about-card'>
    <h3 style='color: #ff0000; margin-top: 0;'>What is Formula 1?</h3>
    <p style='font-size: 1.15rem; line-height: 1.9; color: #ddd;'>
    Formula 1 (F1) adalah kelas tertinggi dari balap mobil roda terbuka internasional yang diatur oleh 
    <strong>Fédération Internationale de l'Automobile (FIA)</strong>. F1 dikenal sebagai puncak teknologi 
    otomotif dan keterampilan mengemudi, dengan mobil-mobil yang mampu mencapai kecepatan lebih dari 
    <strong style='color: #ff0000;'>350 km/jam</strong> dan menghasilkan <strong>5G force</strong> di tikungan cepat.
    </p>
</div>
""", unsafe_allow_html=True)

about_col1, about_col2, about_col3 = st.columns(3, gap="large")

with about_col1:
    st.markdown("""
    <div class='about-card'>
        <h4 style='color: #ff0000; text-align: center;'>⚡ EXTREME SPEED</h4>
        <p style='color: #ddd; text-align: center; font-size: 1rem; line-height: 1.6;'>
        Mobil F1 berakselerasi <strong style='color: #ff0000;'>0-100 km/jam</strong> dalam 
        <strong>2.6 detik</strong> dan mencapai top speed <strong style='color: #ff0000;'>350+ km/jam</strong>.
        <br><br>
        Sistem DRS menambah <strong>10-12 km/jam</strong> ekstra di straight!
        </p>
    </div>
    """, unsafe_allow_html=True)

with about_col2:
    st.markdown("""
    <div class='about-card'>
        <h4 style='color: #ff0000; text-align: center;'>🌍 GLOBAL REACH</h4>
        <p style='color: #ddd; text-align: center; font-size: 1rem; line-height: 1.6;'>
        F1 mengadakan <strong style='color: #ff0000;'>23+ races</strong> di 5 benua setiap musim, 
        dari Monaco hingga Singapore.
        <br><br>
        Ditonton oleh <strong style='color: #ff0000;'>500+ juta</strong> fans di seluruh dunia!
        </p>
    </div>
    """, unsafe_allow_html=True)

with about_col3:
    st.markdown("""
    <div class='about-card'>
        <h4 style='color: #ff0000; text-align: center;'>🔧 CUTTING-EDGE TECH</h4>
        <p style='color: #ddd; text-align: center; font-size: 1rem; line-height: 1.6;'>
        Setiap mobil memiliki <strong style='color: #ff0000;'>300+ sensors</strong> yang 
        menghasilkan <strong>1.5 TB data</strong> per race.
        <br><br>
        Power unit hybrid menghasilkan <strong style='color: #ff0000;'>1000+ HP</strong>!
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class='about-card'>
    <h3 style='color: #ff0000; text-align: center;'>🏆 THE ELITE COMPETITION</h3>
    <p style='font-size: 1.1rem; line-height: 1.8; color: #ddd; text-align: center;'>
    Hanya <strong style='color: #ff0000;'>20 pembalap terbaik dunia</strong> yang berkompetisi di F1, 
    mewakili 10 tim konstruktor. Mereka bersaing untuk dua kejuaraan: 
    <strong>Kejuaraan Dunia Pembalap</strong> dan <strong>Kejuaraan Dunia Konstruktor</strong>.
    </p>
    <br>
    <h4 style='color: #ff0000; text-align: center;'>🇲🇽 MEXICO CITY GRAND PRIX</h4>
    <p style='color: #ddd; font-size: 1rem; line-height: 1.7; text-align: center;'>
    <strong>Autódromo Hermanos Rodríguez</strong> di Mexico City adalah salah satu sirkuit paling ikonik di F1, 
    terkenal dengan <strong style='color: #ff0000;'>atmosfer elektrik</strong> dan elevasi tinggi 
    <strong>(2,200m di atas permukaan laut)</strong> yang menantang performa mesin dan aero.
    <br><br>
    Sirkuit ini memiliki <strong style='color: #ff0000;'>Foro Sol</strong>, sebuah stadion baseball 
    yang diubah menjadi tribun dengan kapasitas <strong>30,000+ penonton</strong> yang menciptakan 
    suasana luar biasa dengan Mexican wave dan chants yang legendaris!
    </p>
</div>
""", unsafe_allow_html=True)

# ============ FOOTER ============
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #888; padding: 40px 0 20px 0;'>
    <p style='font-size: 1rem; margin-bottom: 10px; color: #ccc;'>
        🏎️ <strong style='color: #ff0000;'>F1 Winner Predictor 2025</strong> | Powered by Machine Learning & AI
    </p>
    <p style='font-size: 0.9rem; margin-bottom: 20px; color: #999;'>
        Backend: Flask + Waitress on Railway | Frontend: Streamlit Cloud
    </p>
    <div style='border-top: 2px solid #444; padding-top: 20px; margin: 20px auto; max-width: 600px;'>
        <p style='font-size: 1.1rem; color: #ff0000; font-weight: bold; letter-spacing: 2px; text-shadow: 2px 2px 4px #000;'>
            © DGX 2025 • ALL RIGHTS RESERVED
        </p>
        <p style='font-size: 0.85rem; color: #666; margin-top: 10px;'>
            Racing Analytics • Performance Prediction • Data Science
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
