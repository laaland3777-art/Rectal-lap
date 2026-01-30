import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# 1. Page Configuration (Centered Layout)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Laparoscopic Surgery Difficulty Prediction",
    layout="centered"
)

# ---------------------------------------------------------
# 2. Load Model and Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # Ensure these three files are in the same directory as this script
        model = joblib.load("final_ensemble_model.pkl")
        scaler = joblib.load("final_scaler.pkl")
        model_columns = joblib.load("final_columns.pkl") # Critical for column alignment
        return model, scaler, model_columns
    except FileNotFoundError as e:
        st.error(f"Error: Necessary files not found. Details: {e}")
        st.warning("Please ensure 'final_ensemble_model.pkl', 'final_scaler.pkl', and 'final_columns.pkl' are in the same directory.")
        return None, None, None

model, scaler, model_columns = load_artifacts()

# ---------------------------------------------------------
# 3. Title and Introduction
# ---------------------------------------------------------
st.title("Laparoscopic Surgery Difficulty Prediction Model")
st.markdown("""
This application predicts the difficulty probability of **laparoscopic rectal surgery** based on preoperative clinical features and pelvic measurements.
Please input the patient's parameters below.
""")

st.markdown("---")

# ---------------------------------------------------------
# 4. Patient Features Input (Two-Column Layout)
# ---------------------------------------------------------
st.subheader("Patient Features Input")

col1, col2 = st.columns(2)

with col1:
    # 1. History of abdominal surgery (Selectbox)
    # Chinese: 腹部手术史
    f_history_display = st.selectbox(
        "History of abdominal surgery",
        options=["No", "Yes"],
        index=0,
        help="History of previous abdominal surgery."
    )
    f_history = 1 if f_history_display == "Yes" else 0
    
    # 2. Distance from anal verge
    # Chinese: 肿瘤与肛缘距离
    f_dist_anal = st.number_input(
        "Distance from anal verge (cm)", 
        min_value=0.0, max_value=20.0, value=5.0, step=0.5,
        format="%.1f",
        help="Distance from the tumor to the anal verge."
    )
    
    # 3. Intertuberous distance
    # Chinese: 坐骨结节最低点间距离
    f_inter_dist = st.number_input(
        "Intertuberous distance (cm)", 
        min_value=5.0, max_value=20.0, value=10.0, step=0.1,
        format="%.1f",
        help="Distance between the lowest points of the ischial tuberosities."
    )

with col2:
    # 4. Anteroposterior diameter of the pelvic inlet
    # Chinese: 骨盆入口长度（耻骨联合上端至骶骨岬距离）
    f_ap_diameter = st.number_input(
        "AP diameter of pelvic inlet (cm)", 
        min_value=5.0, max_value=20.0, value=11.0, step=0.1,
        format="%.1f",
        help="Length of the pelvic inlet (distance from the superior border of the pubic symphysis to the sacral promontory)."
    )
    
    # 5. Sacrococcygeal distance
    # Chinese: 骶骨岬至尾骨尖距离
    f_sacro_dist = st.number_input(
        "Sacrococcygeal distance (cm)", 
        min_value=5.0, max_value=20.0, value=10.0, step=0.1,
        format="%.1f",
        help="Distance from the sacral promontory to the tip of the coccyx."
    )
    
    # 6. Mesorectal fat area
    # Chinese: 直肠系膜脂肪区面积
    f_fat_area = st.number_input(
        "Mesorectal fat area (cm²)", 
        min_value=0.0, max_value=100.0, value=20.0, step=0.1,
        format="%.1f",
        help="Area of the mesorectal fat."
    )

# ---------------------------------------------------------
# 5. Prediction Logic
# ---------------------------------------------------------
if st.button("Predict Difficulty", type="primary", use_container_width=True):
    if model is not None and scaler is not None:
        
        # --- A. Construct Input DataFrame ---
        # Keys must match the training data columns exactly
        input_data = pd.DataFrame([{
            'History of abdominal surgery': f_history,
            'Distance from anal verge': f_dist_anal,
            'Intertuberous distance': f_inter_dist,
            'Anteroposterior diameter of the pelvic inlet': f_ap_diameter,
            'Sacrococcygeal distance': f_sacro_dist,
            'Mesorectal fat area': f_fat_area
        }])
        
        # --- B. Data Preprocessing ---
        # 1. One-Hot Encoding
        input_df_encoded = pd.get_dummies(input_data)
        
        # 2. Column Alignment (Critical Step)
        input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        
        # 3. Standardization
        input_scaled = scaler.transform(input_df_encoded)
        input_scaled_df = pd.DataFrame(input_scaled, columns=model_columns)
        
        # --- C. Model Inference ---
        # Get probability for Class 1 (High Difficulty)
        probability = model.predict_proba(input_scaled_df)[0][1]
        prediction_class = 1 if probability >= 0.5 else 0
        
        # ---------------------------------------------------------
        # 6. Result Display
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("Prediction Result")
        
        # Progress bar
        st.progress(probability)
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.metric(label="Difficulty Probability", value=f"{probability:.1%}")
            
        with result_col2:
            if prediction_class == 1:
                st.error("⚠️ High Difficulty Predicted")
            else:
                st.success("✅ Low Difficulty Predicted")
                
        st.info(f"The model predicts a **{probability:.1%}** chance of the surgery being difficult based on the provided parameters.")

# --- Footer ---
st.markdown("---")
st.caption("Model based on Ensemble Learning (GaussianNB + SVM + XGBoost).")

