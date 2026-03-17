import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- Load CSV ----------------
CSV_PATH = r"C:\Users\Ruchir\Downloads\Wildfire_classification\data\best_models_summary.csv"
df = pd.read_csv(CSV_PATH)

# ---------------- Streamlit Page Config ----------------
st.set_page_config(
    page_title="🔥 Wildfire Classification",
    layout="wide",
    page_icon="🔥"
)

# ---------------- Custom CSS for Colors ----------------
st.markdown("""
<style>
h1 {
    color: #ff6f61;
    text-align: center;
}
.stButton>button {
    background-color: #ff6f61;
    color:white;
    font-weight:bold;
}
.stSelectbox>div>div>div>span {
    color: #ff6f61;
}
.stFileUploader>div>div>div>label {
    font-weight:bold;
}
.stMarkdown p {
    font-size:16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("<h1>🔥 Wildfire Image Classification 🔥</h1>", unsafe_allow_html=True)
st.markdown("Upload an image, select dataset type and metric priority. The app will automatically select the best model.", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Layout ----------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=['png','jpg','jpeg'])

with col2:
    dataset_type = st.selectbox("Select Dataset Type", ['satellite','uav'])
    metric_priority = st.selectbox("Select Metric Priority", ['accuracy','precision','recall','f1_score','best_overall'])

# ---------------- Process Uploaded Image ----------------
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # ---------------- Select Model ----------------
    subset = df[df['dataset']==dataset_type]
    if metric_priority != "best_overall":
        subset = subset[subset['metric']==metric_priority]
    else:
        subset = subset.loc[subset['f1_score'].idxmax():subset['f1_score'].idxmax()+1]

    model_row = subset.iloc[0]
    model_path = r"{}".format(model_row['model_path'])  # ensures backslashes handled
    model_type = model_row['model_type']

    st.markdown(f"### Using Model: **{model_row['model_name']}** ({model_type})")

    # ---------------- Load Model ----------------
    model = None
    try:
        if model_type.lower() in ["h5","keras"]:
            model = load_model(model_path)
        elif model_type.lower() == "joblib":
            model = joblib.load(model_path)
        else:
            st.error("Unsupported model type!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

    # ---------------- Preprocess Image ----------------
    img_array = np.array(img.resize((224,224)))/255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    # ---------------- Predict ----------------
    pred = None
    try:
        if model_type.lower() in ['h5','keras']:
            pred_probs = model.predict(img_array_exp)
            pred_class = np.argmax(pred_probs, axis=1)[0]
            pred = pred_class
        elif model_type.lower() == 'joblib':
            # Flatten image for ML models if needed
            pred = model.predict(img_array_exp.reshape(1,-1))[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")

    # ---------------- Display Results ----------------
    st.markdown("---")
    st.markdown("## Prediction Result")
    st.success(f"Predicted Class: **{pred}** (class index for now)")
    st.info(f"Metric Priority: **{metric_priority}**")
    st.write(f"Model Accuracy: {model_row['accuracy']:.4f} | Precision: {model_row['precision']:.4f} | Recall: {model_row['recall']:.4f} | F1 Score: {model_row['f1_score']:.4f}")
    
    # Optional style enhancements
    st.markdown("<hr style='border:2px solid #ff6f61'>", unsafe_allow_html=True)
    st.balloons()
