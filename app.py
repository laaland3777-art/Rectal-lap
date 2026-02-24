import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Laparoscopic Surgery Difficulty Prediction",
    page_icon="🏥",
    layout="centered"
)

# ---------------------------------------------------------
# 2. Load Model and Artifacts (Modified for Flat Structure)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # Directly load files from the current directory
        # Make sure these files are uploaded to the SAME folder as app.py
        model = joblib.load("final_ensemble_model.pkl")
        scaler = joblib.load("final_scaler.pkl")
        model_columns = joblib.load("final_feature_columns.pkl")
        return model, scaler, model_columns
    except FileNotFoundError as e:
        st.error(f"❌ Error: Model files not found. Details: {e}")
        st.warning("Please ensure 'final_ensemble_model.pkl', 'final_scaler.pkl', and 'final_feature_columns.pkl' are in the SAME directory as app.py.")
        return None, None, None

model, scaler, model_columns = load_artifacts()

# ---------------------------------------------------------
# 3. Title and Introduction
# ---------------------------------------------------------
st.title("🏥 Laparoscopic Surgery Difficulty Prediction")
st.markdown("""
This application predicts the difficulty probability of **laparoscopic rectal surgery** based on preoperative clinical features and pelvic measurements.
Please input the patient's parameters below.
""")

st.markdown("---")

# ---------------------------------------------------------
# 4. Patient Features Input
# ---------------------------------------------------------
st.subheader("Patient Features Input")

if model is not None:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            # 1. History of abdominal surgery
            f_history_display = st.selectbox(
                "History of abdominal surgery",
                options=["No", "Yes"],
                index=0
            )
            f_history = 1 if f_history_display == "Yes" else 0
            
            # 2. Distance from anal verge
            f_dist_anal = st.number_input(
                "Distance from anal verge (cm)", 
                min_value=0.0, max_value=20.0, value=5.0, step=0.5, format="%.1f"
            )
            
            # 3. Intertuberous distance
            f_inter_dist = st.number_input(
                "Intertuberous distance (cm)", 
                min_value=5.0, max_value=20.0, value=10.0, step=0.1, format="%.1f"
            )
            
            # 4. True conjugate
            f_true_conjugate = st.number_input(
                "True conjugate (cm)", 
                min_value=5.0, max_value=20.0, value=11.0, step=0.1, format="%.1f"
            )

        with col2:
            # 5. Anteroposterior diameter of the pelvic outlet
            f_outlet = st.number_input(
                "AP diameter of pelvic outlet (cm)", 
                min_value=5.0, max_value=20.0, value=9.0, step=0.1, format="%.1f"
            )
            
            # 6. Sacral chord length
            f_sacral = st.number_input(
                "Sacral chord length (cm)", 
                min_value=5.0, max_value=20.0, value=10.0, step=0.1, format="%.1f"
            )
            
            # 7. Mesorectal fat area
            f_fat_area = st.number_input(
                "Mesorectal fat area (cm²)", 
                min_value=0.0, max_value=100.0, value=20.0, step=0.1, format="%.1f"
            )

        submit_btn = st.form_submit_button("🚀 Predict Difficulty", use_container_width=True)

    # ---------------------------------------------------------
    # 5. Prediction Logic
    # ---------------------------------------------------------
    if submit_btn:
        input_data = pd.DataFrame([{
            'History of abdominal surgery': f_history,
            'Distance from anal verge': f_dist_anal,
            'Intertuberous distance': f_inter_dist,
            'True conjugate': f_true_conjugate,
            'Anteroposterior diameter of the pelvic outlet': f_outlet,
            'Sacral chord length': f_sacral,
            'Mesorectal fat area': f_fat_area
        }])
        
        # Preprocessing
        input_df_encoded = pd.get_dummies(input_data)
        input_df_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        input_scaled = scaler.transform(input_df_aligned)
        input_scaled_df = pd.DataFrame(input_scaled, columns=model_columns)
        
        # Prediction
        probability = model.predict_proba(input_scaled_df)[0][1]
        prediction_class = 1 if probability >= 0.5 else 0
        
        # Display
        st.markdown("---")
        st.subheader("Prediction Result")
        st.progress(float(probability))
        
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric(label="Difficulty Probability", value=f"{probability:.1%}")
        with result_col2:
            if prediction_class == 1:
                st.error("⚠️ High Difficulty Predicted")
            else:
                st.success("✅ Low Difficulty Predicted")
                
        if probability >= 0.5:
            st.warning(f"The model predicts a **{probability:.1%}** chance of high surgical difficulty.")
        else:
            st.info(f"The model predicts a **{probability:.1%}** chance of high surgical difficulty (Low Risk).")

st.markdown("---")
st.caption("Model based on Ensemble Learning (GaussianNB + SVM + Random Forest).")
