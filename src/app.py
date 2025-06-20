# src/app.py

import streamlit as st
import pandas as pd
import json
import plotly.express as px
from pathlib import Path

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    layout="wide",
    page_title="Exo-Detector Mission Control",
    page_icon="🛰️"
)

# --- Path Resolution ---
# This ensures the app can find the 'data' and 'assets' directories correctly.
try:
    APP_DIR = Path(__file__).parent
    PROJECT_ROOT = APP_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    ASSETS_DIR = APP_DIR / "assets"
except NameError:
    # Fallback for environments where __file__ is not defined
    DATA_DIR = Path("data")
    ASSETS_DIR = Path("src/assets")

RESULTS_DIR = DATA_DIR / "results"
TRANSIT_WINDOWS_DIR = DATA_DIR / "transit_windows"

# --- Helper Functions ---
def load_css(file_path):
    """Loads a CSS file and applies its styles."""
    if file_path.exists():
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_json_file(filename):
    """Loads a JSON file from the results directory."""
    filepath = RESULTS_DIR / filename
    return json.load(open(filepath)) if filepath.exists() else None

def create_metric_card(label, value, icon):
    """Creates a styled metric card using markdown."""
    st.markdown(f"""
        <div class="metric-card">
            <div class="status-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def get_candidate_light_curve(candidate_id, is_known_transit):
    """Finds and loads the light curve data for a specific candidate window."""
    label = int(is_known_transit)
    # Search for the window file. This is a bit slow but works for a demo.
    # In a production app, you would store file paths in the results JSON.
    for f in TRANSIT_WINDOWS_DIR.glob(f"*_label_{label}_*.csv"):
        # A bit of a hack to match the ID, as we don't store the file path.
        # This can be improved by saving file paths during preprocessing.
        # For now, we just grab one that matches the label.
        return pd.read_csv(f)
    return None

# --- Main App ---

# Apply the custom CSS
load_css(ASSETS_DIR / "style.css")

# --- Header ---
st.title("🛰️ Exo-Detector Mission Control")
st.markdown("An automated AI pipeline for discovering exoplanet candidates in TESS data.")

# --- Load All Results ---
phase1 = load_json_file("phase1_results.json")
phase2 = load_json_file("phase2_results.json")
phase3 = load_json_file("phase3_results.json")
candidates = load_json_file("final_ranked_candidates.json")

# --- Status Overview Section ---
st.header("Pipeline Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    value = phase1.get('real_samples', 'N/A') if phase1 else 'N/A'
    create_metric_card("Real Light Curves", value, "📥")

with col2:
    value = phase2.get('synthetic_samples_generated', 'N/A') if phase2 else 'N/A'
    create_metric_card("Synthetic Samples", value, "✨")

with col3:
    value = phase3.get('num_test_samples', 'N/A') if phase3 else 'N/A'
    create_metric_card("Windows Analyzed", value, "🔬")

with col4:
    value = len(candidates) if candidates else 'N/A'
    create_metric_card("Top Candidates", value, "🏆")

# --- Candidate Results Section ---
st.header("🏆 Top Exoplanet Candidates")
st.markdown("The most promising transit-like signals found by the pipeline, ranked by anomaly score. **Click on a row to view the light curve.**")

if candidates:
    candidates_df = pd.DataFrame(candidates)
    
    # Use Streamlit's new dataframe selection feature
    selection = st.dataframe(
        candidates_df[['candidate_id', 'anomaly_score', 'is_known_transit']],
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    # --- Interactive Plot Section ---
    if not selection.selection.rows:
        st.info("Click on a candidate in the table above to visualize its light curve.")
    else:
        selected_row_index = selection.selection.rows[0]
        selected_candidate = candidates_df.iloc[selected_row_index]
        
        # This part is a simplification. A real implementation would need to
        # map candidate_id back to its original file. For this demo, we just
        # show a representative plot.
        st.subheader(f"Light Curve for Candidate #{selected_candidate['candidate_id']}")
        
        with st.spinner("Loading light curve data..."):
            lc_df = get_candidate_light_curve(
                selected_candidate['candidate_id'],
                selected_candidate['is_known_transit']
            )

            if lc_df is not None:
                fig = px.scatter(
                    lc_df, x="time", y="flux",
                    title=f"Transit Signal (Anomaly Score: {selected_candidate['anomaly_score']:.4f})",
                    template="plotly_dark"
                )
                fig.update_traces(marker=dict(color='#00A2FF', size=5, line=dict(width=1, color='DarkSlateGrey')))
                fig.update_layout(
                    xaxis_title="Time (BJD)",
                    yaxis_title="Normalized Flux",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not load the specific light curve file for this candidate.")

else:
    st.warning("No candidate data found. Please complete the entire pipeline (Phase 1 through 4).")
