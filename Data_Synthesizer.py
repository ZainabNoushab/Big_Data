# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
import joblib, datetime, warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ===============================
# Helper functions
# ===============================
def detect_type(s, cardinality_threshold=50):
    if pd.api.types.is_numeric_dtype(s):
        return "numerical"
    elif pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    elif s.nunique() <= 2:
        return "binary"
    elif s.nunique() < cardinality_threshold:
        return "categorical"
    else:
        return "text"

def advanced_stats(real, synth, cardinality_threshold=50):
    rows = []
    for col in real.columns:
        try:
            t = detect_type(real[col], cardinality_threshold)
            if t == "numerical":
                real_vals = real[col].dropna().astype(float)
                synth_vals = synth[col].dropna().astype(float)
                rmse = np.sqrt(mean_squared_error(real_vals, synth_vals)) if len(real_vals) == len(synth_vals) else np.nan
                ks_stat, ks_p = ks_2samp(real_vals, synth_vals) if len(real_vals) and len(synth_vals) else (np.nan, np.nan)
                bins = np.histogram_bin_edges(
                    np.concatenate([real_vals, synth_vals]) if len(real_vals) and len(synth_vals) else real_vals,
                    bins="auto"
                )
                real_hist = np.histogram(real_vals, bins=bins, density=True)[0] if len(real_vals) else np.array([0.0])
                synth_hist = np.histogram(synth_vals, bins=bins, density=True)[0] if len(synth_vals) else np.array([0.0])
                js = float(jensenshannon(real_hist + 1e-10, synth_hist + 1e-10))
                rows.append({
                    "Column": col, "Type": "Numerical",
                    "Real_Mean": float(real_vals.mean()) if len(real_vals) else np.nan,
                    "Synth_Mean": float(synth_vals.mean()) if len(synth_vals) else np.nan,
                    "RMSE": float(rmse) if not np.isnan(rmse) else np.nan,
                    "KS_Stat": float(ks_stat) if not pd.isna(ks_stat) else np.nan,
                    "KS_p": float(ks_p) if not pd.isna(ks_p) else np.nan,
                    "JS": js
                })
            else:
                rows.append({"Column": col, "Type": t})
        except Exception as e:
            rows.append({"Column": col, "Error": str(e)})
    return pd.DataFrame(rows)

# ===============================
# Streamlit App
# ===============================
st.set_page_config(layout="wide", page_title="Data Synthesizer")
st.title("Data Synthesizer App")

# -------------------------------
# Upload dataset
# -------------------------------
st.header("Upload dataset")
uploaded_file = st.file_uploader("Upload CSV / Excel / JSON", type=["csv", "xls", "xlsx", "json"], key="main_upload")

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            st.error("❌ Unsupported file type.")
            st.stop()
        st.success(f"Loaded dataset: {uploaded_file.name} — shape: {df.shape}")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    st.info("Please upload a dataset to begin.")
    st.stop()

# -------------------------------
# Options
# -------------------------------
st.header("Options")
col1, col2 = st.columns([2, 1])
with col1:
    model_choice = st.selectbox("Model", ["CTGAN", "Statistical (Gaussian Copula)"])
    target_score = st.slider("Target quality score (stop early if reached)", 0.80, 0.99, 0.95, 0.01)
    epochs_list_input = st.text_input("CTGAN epochs sequence (comma separated)", "100,300,500")
    try:
        epochs_list = [int(x.strip()) for x in epochs_list_input.split(",") if x.strip()]
    except:
        epochs_list = [100, 300, 500]
with col2:
    pairplot_opt = st.checkbox("Generate pairplots (optional, can be slow)")
    pairplot_max_cols = st.number_input("Max numeric columns for pairplot", min_value=3, max_value=8, value=5)

# -------------------------------
# Train / Synthesize
# -------------------------------
st.header("Train / Synthesize")
if st.button("Start synthesis"):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    best_model = None
    best_synthetic = None
    best_score = -np.inf

    if model_choice == "CTGAN":
        st.info("Training CTGAN iteratively...")
        for epochs in epochs_list:
            with st.spinner(f"Training CTGAN for {epochs} epochs..."):
                try:
                    synth = CTGANSynthesizer(metadata, epochs=epochs, batch_size=500)
                    synth.fit(df)
                    synthetic = synth.sample(num_rows=len(df))
                except Exception as e:
                    st.error(f"Error training/sampling CTGAN at {epochs} epochs: {e}")
                    continue
            try:
                score = evaluate_quality(df, synthetic, metadata).get_score()
            except Exception:
                score = -np.inf
            st.write(f"Epochs: {epochs} → quality score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_model = synth
                best_synthetic = synthetic
            if best_score >= target_score:
                st.success(f"Target score {target_score} reached ({best_score:.4f}). Stopping early.")
                break
        if best_model is None:
            st.error("CTGAN training failed for all epoch settings.")
            st.stop()
    else:
        st.info("Training Gaussian Copula (statistical model)...")
        try:
            synth = GaussianCopulaSynthesizer(metadata)
            synth.fit(df)
            synthetic = synth.sample(num_rows=len(df))
            best_model = synth
            best_synthetic = synthetic
            best_score = evaluate_quality(df, best_synthetic, metadata).get_score()
            st.write(f"Gaussian Copula quality score: {best_score:.4f}")
        except Exception as e:
            st.error(f"Gaussian Copula training failed: {e}")
            st.stop()

    # Save model
    try:
        joblib.dump(best_model, "best_synthesizer.pkl")
        st.success("Saved best model to best_synthesizer.pkl")
    except Exception as e:
        st.warning(f"Couldn't save model: {e}")

    # -------------------------------
    # Validation + Advanced Stats
    # -------------------------------
    st.header("Validation & Advanced statistics")
    with st.spinner("Running SDV evaluation and diagnostics..."):
        quality_report = evaluate_quality(df, best_synthetic, metadata)
        diagnostic_report = run_diagnostic(df, best_synthetic, metadata)
    stats_df = advanced_stats(df, best_synthetic)
    st.subheader("Advanced statistics (sample)")
    st.dataframe(stats_df)

    # -------------------------------
    # Visualizations
    # -------------------------------
    st.header("Visualizations")

    # 📊 Statistical Summary
    st.subheader("Statistical Summary: Original vs Synthetic")
    real_summary = df.describe(include="all").T
    synth_summary = best_synthetic.describe(include="all").T
    summary = real_summary.join(synth_summary, lsuffix="_Real", rsuffix="_Synthetic")
    st.dataframe(summary)

    # 📈 Distribution Plots
    st.subheader("Distribution comparison (numeric)")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, color="blue", label="Real", stat="density", ax=ax, alpha=0.5)
        sns.histplot(best_synthetic[col].dropna(), kde=True, color="red", label="Synthetic", stat="density", ax=ax, alpha=0.5)
        ax.set_title(f"Distribution of {col}")
        ax.legend()
        st.pyplot(fig)

    # Heatmap & PCA
    if num_cols:
        with st.expander("Correlation heatmap & PCA"):
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0, ax=ax)
            st.pyplot(fig)
            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(df[num_cols].dropna())
            synth_scaled = scaler.transform(best_synthetic[num_cols].dropna())
            pca = PCA(n_components=2)
            real_pca = pca.fit_transform(real_scaled)
            synth_pca = pca.transform(synth_scaled)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label="Real")
            ax.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.5, label="Synthetic")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
            st.pyplot(fig)

    # -------------------------------
    # Report & downloads
    # -------------------------------
    st.header("Report & downloads")
    diagnostic_text = str(diagnostic_report) if diagnostic_report is not None else "N/A"
    report_text = f"""Data Synthesis Report
Model: {model_choice}
Input file: {uploaded_file.name}
Input shape: {df.shape}
Synthetic shape: {best_synthetic.shape}
Best quality score: {best_score:.4f}
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Diagnostics:
{diagnostic_text}

Advanced statistics:
{stats_df.to_string(index=False)}
"""
    synthetic_csv = best_synthetic.to_csv(index=False).encode("utf-8")
    stats_csv = stats_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download synthetic dataset (CSV)", synthetic_csv, "synthetic.csv", "text/csv")
    st.download_button("Download advanced statistics (CSV)", stats_csv, "advanced_stats.csv", "text/csv")
    st.download_button("Download synthesis report (TXT)", report_text.encode("utf-8"), "synthesis_report.txt", "text/plain")

    st.success("Done — downloads are ready above.")

