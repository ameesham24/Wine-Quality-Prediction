# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path

# ==== CONFIG ====
DATA_PATH = Path("./data/WineQT.csv")
MODEL_PATH = Path("./model/best_model.pkl")
METRICS_PATH = Path("./outputs/test_metrics.csv")

# ==== PAGE SETTINGS ====
st.set_page_config(page_title="🍷 Wine Quality Classifier", layout="wide")

# ==== CACHE FUNCTIONS ====
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["quality_label"] = (df["quality"] >= 7).astype(int)
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH).to_dict(orient="records")[0]
    return {}

# ==== LOAD ====
df = load_data()
model = load_model()
metrics = load_metrics()

# ==== SIDEBAR ====
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Exploration", "📈 Visualisations", "🤖 Prediction", "📋 Model Performance"])

# ==== PAGE 1: DATA EXPLORATION ====
if page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    st.markdown("Explore the Wine Quality dataset before training and prediction.")

    st.subheader("Dataset Overview")
    st.write(f"*Shape:* {df.shape}")
    st.write("*Columns:*", list(df.columns))
    st.dataframe(df.head(20))

    st.subheader("Data Types & Missing Values")
    col1, col2 = st.columns(2)
    with col1:
        st.write(df.dtypes)
    with col2:
        st.write("Missing values per column:")
        st.write(df.isnull().sum())

    st.subheader("Interactive Filtering")
    quality_filter = st.multiselect("Select Quality values", sorted(df["quality"].unique()), default=sorted(df["quality"].unique()))
    filtered_df = df[df["quality"].isin(quality_filter)]
    st.write(f"Filtered dataset: {filtered_df.shape[0]} rows")
    st.dataframe(filtered_df)

# ==== PAGE 2: VISUALISATIONS ====
elif page == "📈 Visualisations":
    st.title("📈 Visualisations")
    st.markdown("Visual insights into wine quality and its features.")

    # Chart 1 — Quality distribution
    st.subheader("Quality Distribution")
    fig1 = px.histogram(df, x="quality", nbins=7, title="Wine Quality Distribution", color="quality_label")
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 — Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.drop(columns=["quality_label"]).corr()
    fig2 = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap", color_continuous_scale="RdBu")
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 — Scatter: Alcohol vs Quality
    st.subheader("Alcohol vs Quality")
    fig3 = px.scatter(df, x="alcohol", y="quality", color="quality_label", title="Alcohol Content vs Quality")
    st.plotly_chart(fig3, use_container_width=True)

    # Interactive histogram
    st.subheader("Feature Distribution")
    feature_choice = st.selectbox("Select feature", df.drop(columns=["quality_label", "quality"]).columns)
    fig4 = px.histogram(df, x=feature_choice, nbins=30, title=f"Distribution of {feature_choice}")
    st.plotly_chart(fig4, use_container_width=True)

# ==== PAGE 3: PREDICTION ====
elif page == "🤖 Prediction":
    st.title("🤖 Wine Quality Prediction")
    st.markdown("Enter wine feature values to get a prediction.")

    # Create input form
    st.subheader("Input Wine Features")
    feature_cols = df.drop(columns=["quality", "quality_label"]).columns.tolist()
    inputs = {}

    with st.form("prediction_form"):
        cols = st.columns(2)
        for i, feat in enumerate(feature_cols):
            mean = float(df[feat].mean())
            min_v = float(df[feat].min())
            max_v = float(df[feat].max())
            step = (max_v - min_v) / 100
            with cols[i % 2]:
                inputs[feat] = st.number_input(
                    label=feat, min_value=min_v, max_value=max_v, value=mean, step=step, format="%.4f"
                )
        submitted = st.form_submit_button("Predict")

    if submitted:
        X_input = pd.DataFrame([inputs])
        with st.spinner("Predicting..."):
            try:
                pred = model.predict(X_input)[0]
                proba = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else None

                if pred == 1:
                    st.success("🍷 This wine is predicted to be *Good Quality* (>=7)")
                else:
                    st.warning("🍷 This wine is predicted to be *Not Good Quality* (<7)")

                if proba is not None:
                    st.info(f"Prediction confidence (good quality): {proba:.2%}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ==== PAGE 4: MODEL PERFORMANCE ====
elif page == "📋 Model Performance":
    st.title("📋 Model Performance")
    st.markdown("Performance of the trained model on the test set.")

    if metrics:
        st.subheader("Test Set Metrics")
        st.json(metrics)

        # Confusion matrix (if saved separately, or compute from test data)
        st.subheader("Confusion Matrix")
        try:
            from sklearn.metrics import confusion_matrix
            y_test = pd.read_csv("outputs/y_test.csv")
            X_test = pd.read_csv("outputs/X_test.csv")
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], colorscale="Blues"))
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)
        except FileNotFoundError:
            st.warning("Confusion matrix data not available.")
    else:
        st.warning("Metrics not found. Please run model training first.")