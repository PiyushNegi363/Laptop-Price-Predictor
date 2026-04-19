import streamlit as st
import pickle
import pandas as pd
import numpy as np
import logging
import sys
import os

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Laptop Price Predictor Pro",
    page_icon="💻",
    layout="wide"
)

# --- UTILS ---
def calculate_ppi(resolution_str, screen_size_inches):
    """Calculates Pixels Per Inch from resolution string (WxH) and screen size."""
    try:
        x_res, y_res = map(int, resolution_str.split('x'))
        return ((x_res**2) + (y_res**2))**0.5 / screen_size_inches
    except Exception as e:
        logger.error(f"PPI Calculation Error: {e}")
        return 0

# --- PREMIUM CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* Global Overrides */
    * { font-family: 'Inter', sans-serif; }
    
    .main { 
        background: radial-gradient(circle at top right, #1e293b, #0f172a); 
        color: #f8fafc; 
    }

    /* Glassmorphism Containers */
    div[data-testid="stVerticalBlock"] > div:has(div.stSelectbox) {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    div[data-testid="stVerticalBlock"] > div:has(div.stSelectbox):hover {
        border-color: rgba(16, 185, 129, 0.3);
        transform: translateY(-2px);
    }

    /* Subheaders */
    h3 {
        color: #10b981 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 1.5rem !important;
        border-bottom: 2px solid rgba(16, 185, 129, 0.1);
        padding-bottom: 8px;
    }

    /* Custom Button */
    div.stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 0.75rem 2.5rem !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.2) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div.stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 20px 25px -5px rgba(16, 185, 129, 0.3) !important;
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
    }

    /* Prediction Card */
    .prediction-card {
        background: rgba(16, 185, 129, 0.1);
        backdrop-filter: blur(20px);
        padding: 40px;
        border-radius: 24px;
        text-align: center;
        border: 1px solid rgba(16, 185, 129, 0.3);
        margin-top: 30px;
        animation: slideUp 0.6s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .price-text {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(to right, #34d399, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }

    /* Inputs Focus */
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- ARTIFACT LOADING ---
@st.cache_resource
def load_assets():
    """Loads the model and reference dataframe from the models/ directory."""
    try:
        # Use absolute paths relative to this file
        base_path = os.path.dirname(__file__)
        pipe_path = os.path.join(base_path, 'models', 'pipe.pkl')
        df_path = os.path.join(base_path, 'models', 'df.pkl')
        
        logger.info(f"Loading artifacts. Path: {base_path}")
        
        if not os.path.exists(pipe_path) or not os.path.exists(df_path):
            logger.error(f"Missing artifacts at {pipe_path} or {df_path}")
            return None, None
            
        with open(pipe_path, 'rb') as f:
            pipe_loaded = pickle.load(f)
        with open(df_path, 'rb') as f:
            df_loaded = pickle.load(f)
            
        logger.info("Artifacts loaded successfully.")
        return pipe_loaded, df_loaded
    except Exception as e:
        logger.error(f"Asset loading failed: {e}")
        return None, None

pipe, df = load_assets()

def main():
    if df is None or pipe is None:
        st.error("⚠️ Critical Error: Model artifacts not found.")
        st.info(f"**Application Root**: `{os.path.dirname(__file__)}`")
        try:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            st.info(f"**Models Dir Contents**: `{os.listdir(models_dir)}`")
        except Exception as e:
            st.error(f"Cannot access 'models' folder: {e}")
        st.stop()

    # --- HEADER ---
    col_header, _ = st.columns([2, 1])
    with col_header:
        st.title("💻 Laptop Price Predictor Pro")
        st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-top: -15px;'>Precision valuation powered by Stacking Ensemble AI</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- INPUT FORM ---
    try:
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.subheader("🏢 Build & Brand")
            company = st.selectbox('Brand Name', sorted(df['Company'].unique()), help="Select the manufacturer")
            laptop_type = st.selectbox("Device Category", sorted(df['TypeName'].unique()))
            
            ram_options = [2, 4, 6, 8, 12, 16, 24, 32, 64]
            weight = st.number_input('Weight (kg)', 0.5, 5.0, 2.0, 0.1, help="Expected weight of the laptop")

        with col2:
            st.subheader("⚙️ Performance Core")
            cpu_series = st.selectbox('CPU Series', sorted(df['Cpu_Series'].unique()))
            
            # --- REAL WORLD CONSTRAINT: Filter RAM based on CPU ---
            if any(x in cpu_series for x in ['Celeron', 'Pentium']) or laptop_type == 'Netbook':
                ram_options = [2, 4, 8]
                st.caption("ℹ️ Low-power configurations are limited to 8GB RAM max.")
            
            ram = st.selectbox('RAM (GB)', ram_options, index=min(3, len(ram_options)-1))
            cpu_ghz = st.number_input('CPU Clock Speed (GHz)', 0.5, 4.5, 2.5, 0.1)
            gpu_model = st.selectbox('Graphics Model', sorted(df['Gpu_Model'].unique()))
            
            # --- REAL WORLD CONSTRAINT: OS Lock for Brands ---
            os_options = ['Mac'] if company == 'Apple' else [opt for opt in df['OS'].unique() if opt != 'Mac']
            operating_system = st.selectbox('Operating System', sorted(os_options))

        with col3:
            st.subheader("🖥️ Display & Storage")
            screen_size = st.number_input('Screen Size (in)', 10.0, 20.0, 15.6, 0.1)
            resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2560x1600','2560x1440'])
            
            s1, s2 = st.columns(2)
            with s1: touchscreen = st.selectbox('Touch', ['No', 'Yes'])
            with s2: ips = st.selectbox('IPS Panel', ['No', 'Yes'])
            
            # --- REAL WORLD CONSTRAINT: No HDD for Ultrabooks ---
            hdd_options = [0, 128, 256, 512, 1024, 2048]
            if laptop_type == 'Ultrabook':
                hdd_options = [0]
                st.caption("ℹ️ Ultrabooks exclusively use SSD storage.")
            
            st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
            st1, st2 = st.columns(2)
            with st1: hdd = st.selectbox('HDD (GB)', hdd_options)
            with st2: ssd = st.selectbox('SSD (GB)', [0, 128, 256, 512, 1024])

        # --- PREDICTION ACTION ---
        st.markdown("<br>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1, 1, 1])
        with btn_col:
            predict_btn = st.button('🎯 GENERATE MARKET VALUATION')

        if predict_btn:
            if screen_size <= 0:
                st.warning("⚠️ Invalid screen size. Please enter a positive value.")
                return

            try:
                # Mapping & PPI Calculation
                touch_val = 1 if touchscreen == 'Yes' else 0
                ips_val = 1 if ips == 'Yes' else 0
                ppi = calculate_ppi(resolution, screen_size)
                
                # Log inputs for auditing
                logger.info(f"Prediction requested: {company}, {laptop_type}, RAM:{ram}, CPU:{cpu_series}@{cpu_ghz}, GPU:{gpu_model}, OS:{operating_system}")
                
                query = pd.DataFrame([[company, laptop_type, ram, weight, cpu_ghz, cpu_series, gpu_model, touch_val, ips_val, ppi, hdd, ssd, operating_system]], 
                                     columns=['Company', 'TypeName', 'Ram', 'Weight', 'Cpu_GHz', 'Cpu_Series', 'Gpu_Model', 'Touchscreen', 'IPS', 'ppi', 'HDD', 'SSD', 'OS'])
                
                with st.spinner('🚀 Analyzing market trends and calculating valuation...'):
                    log_price = pipe.predict(query)[0]
                    result = np.exp(log_price)
                    
                    st.markdown(f"""
                        <div class="prediction-card">
                            <p style="color: #a7f3d0; letter-spacing: 3px; font-weight: 600; font-size: 0.85rem; text-transform: uppercase;">Estimated Market Value</p>
                            <p class="price-text">₹ {int(result):,}</p>
                            <p style="color: #6ee7b7; font-size: 0.9rem; margin-top: 15px;">
                                <span style="background: rgba(16, 185, 129, 0.2); padding: 4px 12px; border-radius: 20px;">
                                    Confidence: High (Stacking Ensemble)
                                </span>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                    logger.info(f"Prediction successful: ₹ {int(result)}")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                logger.error(f"Prediction Error: {e}")

    except Exception as e:
        st.error(f"❌ An unexpected UI error occurred: {e}")
        logger.error(f"UI Error: {e}")

if __name__ == "__main__":
    main()