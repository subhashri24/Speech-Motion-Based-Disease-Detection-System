from pathlib import Path
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

from utils.plots import (
    CLASS_NAMES,
    plot_class_distribution,
    plot_histogram,
    plot_box,
    plot_heatmap,
    plot_confusion,
    plot_ovr_roc,
    plot_ovr_pr,
    plot_feature_importance,
    plot_metric_bars,
)

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "multiclass_dataset.csv"
MODEL_PATH = BASE / "models" / "multiclass_model.joblib"
METRICS_PATH = BASE / "models" / "metrics.joblib"
BUNDLE_PATH = BASE / "models" / "evaluation_bundle.joblib"

st.set_page_config(page_title="Neuro Multiclass Research Dashboard", layout="wide")

# -------------------------
# LOAD FUNCTIONS
# -------------------------
def ensure_assets():
    if not DATA_PATH.exists():
        raise FileNotFoundError("multiclass_dataset.csv not found in data/")
    if not MODEL_PATH.exists():
        import train  # auto-train

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metrics = joblib.load(METRICS_PATH)
    bundle = joblib.load(BUNDLE_PATH)
    return model, metrics, bundle

ensure_assets()
df = load_data()
model, metrics, bundle = load_artifacts()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("Research Dashboard")
page = st.sidebar.radio(
    "Select tab",
    [
        "Project Overview",
        "Dataset Explorer",
        "Speech EDA",
        "Motion EDA",
        "Correlation Analysis",
        "Model Metrics",
        "Feature Importance",
        "Speech Prediction (WAV)",   # ✅ NEW TAB
        "Prediction Lab",
        "Dataset Information",
    ],
)

st.title("Speech and Motion based Disease Detection system")

# -------------------------
# PROJECT OVERVIEW
# -------------------------
if page == "Project Overview":

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### Problem Statement
Neurodegenerative disease screening benefits from early detection using non-invasive speech and motion biomarkers.

### Disease Classes
- Healthy
- Parkinson's Disease
- Alzheimer's Disease
- Essential Tremor
        """)
    with c2:
        st.markdown("""
### Research Contributions
- Multiclass speech-motion classification
- Research-style exploratory analysis
- Publication-oriented metric visualization
- Interpretable feature importance analysis
        """)
    # st.info("This dashboard is for academic research and prototype experimentation only.")

# -------------------------
# DATASET EXPLORER
# -------------------------
elif page == "Dataset Explorer":
    st.header("Dataset Explorer")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("Shape:", df.shape)
    with c2:
        st.write("Missing values:", int(df.isna().sum().sum()))

    st.pyplot(plot_class_distribution(df))

# -------------------------
# SPEECH EDA
# -------------------------
elif page == "Speech EDA":
    st.header("Speech EDA")

    speech_cols = ["pitch_mean","pitch_std","energy_mean","energy_std","jitter","shimmer"] + [f"mfcc_{i}" for i in range(1,14)]

    col = st.selectbox("Select speech feature", speech_cols)

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_histogram(df, col))
    with c2:
        st.pyplot(plot_box(df, col))

    x_col = st.selectbox("Scatter X", speech_cols, index=0)
    y_col = st.selectbox("Scatter Y", speech_cols, index=1)

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=df["label"].map(dict(enumerate(CLASS_NAMES))),
        title=f"{x_col} vs {y_col}"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# MOTION EDA
# -------------------------
elif page == "Motion EDA":
    st.header("Motion EDA")

    motion_cols = ["motion_mean","motion_std","motion_peak","frame_diff_mean","frame_diff_std","symmetry_score","rhythm_score"]

    col = st.selectbox("Select motion feature", motion_cols)

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_histogram(df, col))
    with c2:
        st.pyplot(plot_box(df, col))

    x_col = st.selectbox("Motion Scatter X", motion_cols, index=0)
    y_col = st.selectbox("Motion Scatter Y", motion_cols, index=1)

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=df["label"].map(dict(enumerate(CLASS_NAMES))),
        title=f"{x_col} vs {y_col}"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# CORRELATION
# -------------------------
elif page == "Correlation Analysis":
    st.header("Correlation Analysis")
    st.pyplot(plot_heatmap(df.drop(columns=["label"])))
    st.caption("Correlation analysis across multimodal features.")

# -------------------------
# MODEL METRICS
# -------------------------
elif page == "Model Metrics":
    st.header("Evaluation Metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision (Macro)", f"{metrics['precision_macro']:.3f}")
    m3.metric("Recall (Macro)", f"{metrics['recall_macro']:.3f}")
    m4.metric("F1 (Macro)", f"{metrics['f1_macro']:.3f}")

    st.pyplot(plot_metric_bars(metrics))

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_confusion(bundle["y_test"], bundle["pred"]))
    with c2:
        st.pyplot(plot_ovr_roc(bundle["y_test"], bundle["proba"]))

    st.pyplot(plot_ovr_pr(bundle["y_test"], bundle["proba"]))

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
elif page == "Feature Importance":
    st.header("Feature Importance")
    st.pyplot(plot_feature_importance(bundle["feature_names"], bundle["importances"], top_n=20))

# -------------------------
# 🎤 SPEECH PREDICTION TAB
# -------------------------
elif page == "Speech Prediction (WAV)":
    st.header("🎤 Speech-Based Parkinson Detection")

    import librosa
    import numpy as np
    import tempfile

    MODEL_SPEECH_PATH = BASE / "models" / "parkinson_model.pkl"

    if not MODEL_SPEECH_PATH.exists():
        st.warning("⚠️ parkinson_model.pkl not found in models/")
        st.stop()

    speech_model = joblib.load(MODEL_SPEECH_PATH)


    def extract_features(file_path):
        y, sr = librosa.load(file_path, duration=5)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)

        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        return np.hstack([mfcc_mean, mfcc_std, chroma_mean, contrast_mean, zcr])

    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        with st.spinner("Analyzing voice..."):
            features = extract_features(temp_path).reshape(1, -1)
            pred = speech_model.predict(features)[0]
            probs = speech_model.predict_proba(features)[0]

        if pred == 1:
            st.error("⚠️ Parkinson's Detected")
        else:
            st.success("✅ Healthy")

        prob_df = pd.DataFrame({
            "Class": ["Healthy", "Parkinson's"],
            "Probability": probs
        })

        fig = px.bar(prob_df, x="Class", y="Probability", title="Prediction Confidence")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(prob_df, use_container_width=True)

        st.info("⚠️ This is not a medical diagnosis. For research use only.")

# -------------------------
# MULTICLASS PREDICTION
# -------------------------
elif page == "Prediction Lab":
    st.header("Prediction Lab")

    feature_names = [c for c in df.columns if c != "label"]
    inputs = {}

    cols = st.columns(3)
    for i, col in enumerate(feature_names):
        default_val = float(df[col].median())
        inputs[col] = cols[i % 3].number_input(col, value=default_val)

    if st.button("Predict Class"):
        input_df = pd.DataFrame([inputs])
        pred = int(model.predict(input_df)[0])
        probs = model.predict_proba(input_df)[0]

        st.success(f"Predicted Class: {CLASS_NAMES[pred]}")

        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probs
        })

        fig = px.bar(prob_df, x="Class", y="Probability", title="Predicted Class Probabilities")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(prob_df, use_container_width=True)

# -------------------------
# DATASET INFO
# -------------------------
elif page == "Dataset Information":
    st.header("Dataset Information")

    st.markdown("""
### Included file
- `data/multiclass_dataset.csv`

### Label mapping
- `0` = Healthy
- `1` = Parkinson's Disease
- `2` = Alzheimer's Disease
- `3` = Essential Tremor
    """)

    st.write(df["label"].value_counts().sort_index())