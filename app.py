import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# 1. 页面配置 (Page Configuration) - 采用附件的居中布局
# ---------------------------------------------------------
st.set_page_config(
    page_title="Laparoscopic Surgery Difficulty Prediction",
    page_icon="🏥",
    layout="centered"
)

# ---------------------------------------------------------
# 2. 加载模型和工具 (Load Model and Artifacts)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # 请确保这三个文件在 GitHub 或本地文件夹中
        model = joblib.load("final_ensemble_model.pkl")
        scaler = joblib.load("final_scaler.pkl")
        model_columns = joblib.load("final_columns.pkl") # 关键文件：用于列对齐
        return model, scaler, model_columns
    except FileNotFoundError as e:
        st.error(f"Error: Necessary files not found. Details: {e}")
        st.warning("Please ensure 'final_ensemble_model.pkl', 'final_scaler.pkl', and 'final_columns.pkl' are in the same directory.")
        return None, None, None

model, scaler, model_columns = load_artifacts()

# ---------------------------------------------------------
# 3. 标题和介绍 (Title and Introduction)
# ---------------------------------------------------------
st.title("🏥 Laparoscopic Surgery Difficulty Prediction Model")
st.markdown("""
This application predicts the difficulty probability of laparoscopic rectal surgery based on preoperative clinical features and pelvic measurements.
Please input the patient's parameters below.
""")

st.markdown("---")

# ---------------------------------------------------------
# 4. 输入表单 (Patient Features Input) - 双列布局
# ---------------------------------------------------------
st.subheader("Patient Features Input")

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    # 1. History of abdominal surgery (0/1)
    history_display = st.radio(
        "History of abdominal surgery",
        options=["No", "Yes"],
        index=0,
        horizontal=True,
        help="Does the patient have a history of previous abdominal surgeries?"
    )
    # 转换逻辑：Yes -> 1, No -> 0
    f_history = 1 if history_display == "Yes" else 0
    
    # 2. Distance from anal verge
    f_dist_anal = st.number_input(
        "Distance from anal verge (cm)", 
        min_value=0.0, max_value=20.0, value=5.0, step=0.5,
        format="%.1f",
        help="Distance from the anal verge to the tumor."
    )
    
    # 3. Intertuberous distance
    f_inter_dist = st.number_input(
        "Intertuberous distance (cm)", 
        min_value=5.0, max_value=20.0, value=10.0, step=0.1,
        format="%.1f",
        help="Distance between the ischial tuberosities."
    )

with col2:
    # 4. Anteroposterior diameter of the pelvic inlet
    f_ap_diameter = st.number_input(
        "AP diameter of pelvic inlet (cm)", 
        min_value=5.0, max_value=20.0, value=11.0, step=0.1,
        format="%.1f",
        help="Anteroposterior diameter of the pelvic inlet."
    )
    
    # 5. Sacrococcygeal distance
    f_sacro_dist = st.number_input(
        "Sacrococcygeal distance (cm)", 
        min_value=5.0, max_value=20.0, value=10.0, step=0.1,
        format="%.1f",
        help="Distance between the sacrum and coccyx."
    )
    
    # 6. Mesorectal fat area
    f_fat_area = st.number_input(
        "Mesorectal fat area (cm²)", 
        min_value=0.0, max_value=100.0, value=20.0, step=0.1,
        format="%.1f",
        help="Cross-sectional area of the mesorectal fat."
    )

# ---------------------------------------------------------
# 5. 预测逻辑 (Prediction Logic)
# ---------------------------------------------------------
if st.button("Predict Difficulty", type="primary", use_container_width=True):
    if model is not None and scaler is not None:
        
        # --- A. 构造输入 DataFrame ---
        # 这里的 Key 必须与您训练数据 CSV 中的列名完全一致！
        input_data = pd.DataFrame([{
            'History of abdominal surgery': f_history,
            'Distance from anal verge': f_dist_anal,
            'Intertuberous distance': f_inter_dist,
            'Anteroposterior diameter of the pelvic inlet': f_ap_diameter,
            'Sacrococcygeal distance': f_sacro_dist,
            'Mesorectal fat area': f_fat_area
        }])
        
        # --- B. 数据预处理 (关键步骤) ---
        # 1. 独热编码 (保持流程一致)
        input_df_encoded = pd.get_dummies(input_data)
        
        # 2. 列对齐 (Critical Step: 确保列顺序和数量与训练模型时完全一致)
        input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        
        # 3. 标准化
        input_scaled = scaler.transform(input_df_encoded)
        input_scaled_df = pd.DataFrame(input_scaled, columns=model_columns)
        
        # --- C. 模型预测 ---
        # 获取属于类别 1 (High Difficulty) 的概率
        probability = model.predict_proba(input_scaled_df)[0][1]
        prediction_class = 1 if probability >= 0.5 else 0
        
        # ---------------------------------------------------------
        # 6. 结果展示 (Result Display) - 仿照附件风格
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("Prediction Result")
        
        # 进度条显示风险概率
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

# --- 页脚 ---
st.markdown("---")
st.caption("Model based on Ensemble Learning (GaussianNB + SVM + XGBoost).")
