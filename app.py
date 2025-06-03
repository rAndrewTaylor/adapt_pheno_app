import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from itertools import product
from datetime import datetime
import base64
from io import BytesIO

st.set_page_config(page_title="Validation Simulation", layout="wide")

st.title("Stratified Validation Simulation")

# Sidebar controls
st.sidebar.header("Simulation Parameters")

# Population and sample size controls
pop_size = st.sidebar.number_input("Population Size", min_value=1000, max_value=100000, value=10000, step=1000)
sample_per_stratum = st.sidebar.number_input("Samples per Stratum", min_value=10, max_value=200, value=50, step=10)

# Prevalence controls
st.sidebar.subheader("Prevalence Range")
prev_min = st.sidebar.slider("Minimum Prevalence (%)", 0.1, 5.0, 0.5, 0.1)
prev_max = st.sidebar.slider("Maximum Prevalence (%)", 0.1, 5.0, 1.5, 0.1)
prev_step = st.sidebar.slider("Prevalence Step (%)", 0.1, 1.0, 0.5, 0.1)
prevalence_grid = np.arange(prev_min/100, prev_max/100 + prev_step/100, prev_step/100)

# Model performance controls
st.sidebar.subheader("Model Performance")
model_sens_min = st.sidebar.slider("Minimum Model Sensitivity", 0.5, 1.0, 0.75, 0.05)
model_sens_max = st.sidebar.slider("Maximum Model Sensitivity", 0.5, 1.0, 0.95, 0.05)
model_sens_step = st.sidebar.slider("Model Sensitivity Step", 0.05, 0.2, 0.05, 0.05)
model_sens_grid = np.arange(model_sens_min, model_sens_max + model_sens_step, model_sens_step)

model_spec_min = st.sidebar.slider("Minimum Model Specificity", 0.5, 1.0, 0.85, 0.05)
model_spec_max = st.sidebar.slider("Maximum Model Specificity", 0.5, 1.0, 0.95, 0.05)
model_spec_step = st.sidebar.slider("Model Specificity Step", 0.05, 0.2, 0.05, 0.05)
model_spec_grid = np.arange(model_spec_min, model_spec_max + model_spec_step, model_spec_step)

# Rule performance controls
st.sidebar.subheader("Rule Performance")
rule_sens = st.sidebar.slider("Rule Sensitivity", 0.5, 1.0, 0.70, 0.05)
rule_spec = st.sidebar.slider("Rule Specificity", 0.5, 1.0, 0.92, 0.05)

# Number of simulations
n_reps = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=5000, value=1000, step=100)

def simulate_once(prev, model_sens, model_spec, rng):
    """One Monte‑Carlo replicate with design weights."""
    true = rng.random(pop_size) < prev
    model_pred = np.where(
        true,
        rng.random(pop_size) < model_sens,
        rng.random(pop_size) < (1 - model_spec),
    )
    rule_pred = np.where(
        true,
        rng.random(pop_size) < rule_sens,
        rng.random(pop_size) < (1 - rule_spec),
    )
    strata = model_pred.astype(int) * 2 + rule_pred.astype(int)
    pop_counts = np.bincount(strata, minlength=4)

    sample_idx = []
    for s in range(4):
        idx = np.where(strata == s)[0]
        k = min(sample_per_stratum, len(idx))
        sample_idx.extend(rng.choice(idx, size=k, replace=False))
    sample_idx = np.array(sample_idx)

    weights = pop_counts[strata[sample_idx]] / sample_per_stratum

    tp = ((model_pred[sample_idx] == 1) & (true[sample_idx] == 1)) * weights
    fp = ((model_pred[sample_idx] == 1) & (true[sample_idx] == 0)) * weights
    fn = ((model_pred[sample_idx] == 0) & (true[sample_idx] == 1)) * weights
    tn = ((model_pred[sample_idx] == 0) & (true[sample_idx] == 0)) * weights

    TP, FP, FN, TN = tp.sum(), fp.sum(), fn.sum(), tn.sum()
    sens = TP / (TP + FN) if (TP + FN) else np.nan
    spec = TN / (TN + FP) if (TN + FP) else np.nan
    ppv = TP / (TP + FP) if (TP + FP) else np.nan
    npv = TN / (TN + FN) if (TN + FN) else np.nan
    return sens, spec, ppv, npv

def generate_html_report(summary_df, fig_ppv, fig_npv):
    """Generate HTML report with simulation results."""
    equations_section = r"""
    <p>
    Weights: \(w_i = \frac{N_i}{n_i}\)<br>
    Metrics: \(\text{Se}=\frac{\tilde{TP}}{\tilde{TP}+\tilde{FN}}\),
    \(\text{Sp}=\frac{\tilde{TN}}{\tilde{TN}+\tilde{FP}}\),
    \(\text{PPV}=\frac{\tilde{TP}}{\tilde{TP}+\tilde{FP}}\),
    \(\text{NPV}=\frac{\tilde{TN}}{\tilde{TN}+\tilde{FN}}\).
    </p>
    """

    # Convert Plotly figures to static images
    ppv_img = fig_ppv.to_image(format="png")
    npv_img = fig_npv.to_image(format="png")
    
    # Convert images to base64 for embedding in HTML
    ppv_b64 = base64.b64encode(ppv_img).decode()
    npv_b64 = base64.b64encode(npv_img).decode()

    html = f"""<!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>Validation Simulation Report</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1, h2 {{ color:#003366; }}
    table {{ border-collapse:collapse; width:100%; }}
    th,td {{ border:1px solid #ccc; padding:8px; text-align:center; }}
    th {{ background:#f2f2f2; }}
    img {{ max-width:800px; margin:20px 0; }}
    .container {{ max-width:1200px; margin:0 auto; }}
    </style>
    </head>
    <body>
    <div class="container">
    <h1>Design‑Weighted Validation Report<br>(Generated {datetime.utcnow().date()})</h1>
    
    <h2>Equations</h2>
    {equations_section}
    
    <h2>Parameter Grids</h2>
    <p>Prevalence levels: {prevalence_grid}<br>
    Model sensitivity grid: {model_sens_grid}<br>
    Model specificity grid: {model_spec_grid}</p>
    
    <h2>Summary Metrics</h2>
    {summary_df.to_html(index=False)}
    
    <h2>PPV vs Prevalence</h2>
    <img src="data:image/png;base64,{ppv_b64}" alt="PPV curve">
    
    <h2>NPV vs Prevalence</h2>
    <img src="data:image/png;base64,{npv_b64}" alt="NPV curve">
    </div>
    </body></html>
    """
    return html

# Run simulation
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        rng = np.random.default_rng(seed=2025)
        rows = []
        
        for prev, m_sens, m_spec in product(prevalence_grid, model_sens_grid, model_spec_grid):
            sims = np.array([simulate_once(prev, m_sens, m_spec, rng) for _ in range(n_reps)])
            df_tmp = pd.DataFrame(sims, columns=["Sensitivity", "Specificity", "PPV", "NPV"])
            rows.append({
                "Prevalence": prev,
                "Model_Sens": m_sens,
                "Model_Spec": m_spec,
                "Sens_mean": df_tmp["Sensitivity"].mean(),
                "Sens_LCL": df_tmp["Sensitivity"].quantile(0.025),
                "Sens_UCL": df_tmp["Sensitivity"].quantile(0.975),
                "Spec_mean": df_tmp["Specificity"].mean(),
                "Spec_LCL": df_tmp["Specificity"].quantile(0.025),
                "Spec_UCL": df_tmp["Specificity"].quantile(0.975),
                "PPV_mean": df_tmp["PPV"].mean(),
                "PPV_LCL": df_tmp["PPV"].quantile(0.025),
                "PPV_UCL": df_tmp["PPV"].quantile(0.975),
                "NPV_mean": df_tmp["NPV"].mean(),
                "NPV_LCL": df_tmp["NPV"].quantile(0.025),
                "NPV_UCL": df_tmp["NPV"].quantile(0.975),
            })

        summary_df = pd.DataFrame(rows).round(4)
        
        # Display summary metrics
        st.subheader("Summary Metrics")
        st.dataframe(summary_df)
        
        # Create PPV plot
        fig_ppv = go.Figure()
        for m_sens, m_spec in product(model_sens_grid, model_sens_grid):
            subset = summary_df[(summary_df["Model_Sens"] == m_sens) & (summary_df["Model_Spec"] == m_spec)]
            fig_ppv.add_trace(go.Scatter(
                x=subset["Prevalence"]*100,
                y=subset["PPV_mean"],
                mode='lines+markers',
                name=f"Se {m_sens:.2f}, Sp {m_spec:.2f}"
            ))
        
        fig_ppv.update_layout(
            title="PPV vs Prevalence",
            xaxis_title="Prevalence (%)",
            yaxis_title="Positive Predictive Value",
            showlegend=True
        )
        st.plotly_chart(fig_ppv, use_container_width=True)
        
        # Create NPV plot
        fig_npv = go.Figure()
        for m_sens, m_spec in product(model_sens_grid, model_sens_grid):
            subset = summary_df[(summary_df["Model_Sens"] == m_sens) & (summary_df["Model_Spec"] == m_spec)]
            fig_npv.add_trace(go.Scatter(
                x=subset["Prevalence"]*100,
                y=subset["NPV_mean"],
                mode='lines+markers',
                name=f"Se {m_sens:.2f}, Sp {m_spec:.2f}"
            ))
        
        fig_npv.update_layout(
            title="NPV vs Prevalence",
            xaxis_title="Prevalence (%)",
            yaxis_title="Negative Predictive Value",
            showlegend=True
        )
        st.plotly_chart(fig_npv, use_container_width=True)
        
        # Generate HTML report
        html_report = generate_html_report(summary_df, fig_ppv, fig_npv)
        
        # Add download buttons
        col1, col2 = st.columns(2)
        with col1:
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary Metrics CSV",
                data=csv,
                file_name="summary_metrics.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="Download HTML Report",
                data=html_report,
                file_name="validation_report.html",
                mime="text/html"
            ) 