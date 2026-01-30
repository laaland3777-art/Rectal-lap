import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Laparoscopic Surgery Difficulty Prediction",
    page_icon="🏥",
    layout="wide"
)

# Title and Introduction
st.title("🏥 Laparoscopic Surgery Difficulty Prediction Model")
st.markdown("""
**Introduction**:  
This system is designed to predict the difficulty of laparoscopic rectal surgery.
Please input the patient's pelvic measurements and clinical characteristics in the sidebar to assess the surgical difficulty risk.
""")

st.divider() 

# ---------------------------------------------------------
# 2. Load Model and Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Ensure these three files are in the same directory
    model = joblib.load("final_ensemble_model.pkl")
    scaler = joblib.load("final_scaler.pkl")
    # Note: This is essential for column alignment
    model_columns = joblib.load("final_columns.pkl") 
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(f"❌ System startup failed: Necessary files not found. Error details: {e}")
    st.warning("Please ensure 'final_ensemble_model.pkl', 'final_scaler.pkl', and 'final_columns.pkl' are in the current directory.")
    st.stop()

# ---------------------------------------------------------
# 3. Sidebar: Patient Features Input
# ---------------------------------------------------------
st.sidebar.header("📋 Clinical Parameters")

def user_input_features():
    # --- 1. History of abdominal surgery (0/1 variable) ---
    # UI displays No/Yes, logic converts to 0/1
    history_display = st.sidebar.radio(
        "History of abdominal surgery",
        options=["No", "Yes"],
        index=0,
        horizontal=True
    )
    history_value = 1 if history_display == "Yes" else 0

    st.sidebar.markdown("---") 
    st.sidebar.subheader("📏 Pelvic Measurements")

    # --- 2. Numerical Variables (with units) ---
    # Ranges (min, max) and default values are estimates; adjust based on actual data distribution
    
    dist_anal = st.sidebar.number_input(
        "Distance from anal verge",
        min_value=0.0, max_value=20.0, value=5.0, step=0.5,
        format="%.1f", help="Unit: cm"
    )
    
    inter_dist = st.sidebar.number_input(
        "Intertuberous distance",
        min_value=5.0, max_value=20.0, value=10.0, step=0.1,
        format="%.1f", help="Unit: cm"
    )
    
    ap_diameter = st.sidebar.number_input(
        "Anteroposterior diameter of the pelvic inlet",
        min_value=5.0, max_value=20.0, value=11.0, step=0.1,
        format="%.1f", help="Unit: cm"
    )
    
    sacro_dist = st.sidebar.number_input(
        "Sacrococcygeal distance",
        min_value=5.0, max_value=20.0, value=10.0, step=0.1,
        format="%.1f", help="Unit: cm"
    )
    
    fat_area = st.sidebar.number_input(
        "Mesorectal fat area",
        min_value=0.0, max_value=100.0, value=20.0, step=0.1,
        format="%.1f", help="Unit: cm²"
    )

    # --- Construct DataFrame ---
    # Keys must match the column names in your training CSV exactly!
    data = {
        'History of abdominal surgery': history_value,
        'Distance from anal verge': dist_anal,
        'Intertuberous distance': inter_dist,
        'Anteroposterior diameter of the pelvic inlet': ap_diameter,
        'Sacrococcygeal distance': sacro_dist,
        'Mesorectal fat area': fat_area
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# ---------------------------------------------------------
# 4. Main Interface: Display and Prediction
# ---------------------------------------------------------

# Display overview of input data
with st.expander("Review Input Patient Data", expanded=True):
    st.dataframe(input_df)

# Prediction Button
if st.button("🚀 Predict Difficulty", type="primary", use_container_width=True):
    
    # --- Data Preprocessing ---
    # 1. One-Hot Encoding (Maintains consistency with training flow)
    input_df_encoded = pd.get_dummies(input_df)
    
    # 2. Column Alignment (Critical: Ensures column order/count matches training model)
    input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # 3. Standardization (Use the pre-trained scaler)
    input_scaled = scaler.transform(input_df_encoded)
    input_scaled_df = pd.DataFrame(input_scaled, columns=model_columns)

    # --- Model Inference ---
    prediction = model.predict(input_scaled_df)
    prediction_proba = model.predict_proba(input_scaled_df)
    
    # Get probability for Class 1 (High Difficulty)
    prob_high = prediction_proba[0][1]

    # --- Result Display ---
    st.subheader("📊 Prediction Results")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Gauge-style metric
        st.metric(
            label="Probability of High Difficulty", 
            value=f"{prob_high * 100:.1f}%",
            delta="Risk Score"
        )
    
    with col2:
        if prediction[0] == 1:
            st.error("⚠️ **High Difficulty**")
            st.write("The model predicts high surgical difficulty. Adequate preoperative preparation is recommended.")
        else:
            st.success("✅ **Low Difficulty**")
            st.write("The model predicts relatively low surgical difficulty.")

    # Visualize probability bar
    st.write("Risk Probability Bar:")
    st.progress(int(prob_high * 100))
