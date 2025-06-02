# src/app.py

import streamlit as st
import pandas as pd
import json
from pathlib import Path

# --- Definitive Path Resolution ---
try:
    APP_DIR = Path(__file__).parent
    PROJECT_ROOT = APP_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = DATA_DIR / "results"
except NameError:
    RESULTS_DIR = Path("data") / "results"

# --- Helper Functions ---
def load_json_file(filename):
    """Loads a JSON file from the results directory, returns None if not found."""
    filepath = RESULTS_DIR / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def display_phase_status(phase_name, results_data, metrics_map):
    """Displays a formatted status box for a pipeline phase."""
    if results_data:
        st.success(f"**{phase_name}:** Completed Successfully")
        for key, label in metrics_map.items():
            if key in results_data:
                st.metric(label, results_data[key])
    else:
        st.error(f"**{phase_name}:** Results Not Found")

# --- Dashboard Layout ---
st.set_page_config(layout="wide", page_title="Exo-Detector Dashboard")

st.title("🛰️ Exo-Detector Pipeline Dashboard")
st.markdown("Final results and status for the exoplanet detection pipeline.")

# --- Load All Results ---
phase1_results = load_json_file("phase1_results.json")
phase2_results = load_json_file("phase2_results.json")
phase3_results = load_json_file("phase3_results.json")
phase4_candidates_data = load_json_file("final_ranked_candidates.json")

# --- Status Overview Section ---
st.header("Pipeline Status Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    display_phase_status("Phase 1: Ingestion", phase1_results, {
        "real_samples": "Real Light Curves Downloaded"
    })
with col2:
    display_phase_status("Phase 2: Augmentation", phase2_results, {
        "synthetic_samples_generated": "Synthetic Samples Created"
    })
with col3:
    display_phase_status("Phase 3: Detection", phase3_results, {
        "num_test_samples": "Windows Analyzed"
    })
with col4:
    if phase4_candidates_data is not None:
         st.success(f"**Phase 4: Ranking:** Completed Successfully")
         st.metric("Top Candidates Found", len(phase4_candidates_data))
    else:
        st.error(f"**Phase 4: Ranking:** Results Not Found")

# --- Candidate Results Section ---
st.header("🏆 Top Exoplanet Candidates")
st.markdown("These are the most promising transit-like signals found by the pipeline, ranked by their anomaly score.")

if phase4_candidates_data is not None:
    candidates_df = pd.DataFrame(phase4_candidates_data)
    candidates_df['anomaly_score'] = candidates_df['anomaly_score'].map('{:.4f}'.format)
    display_cols = ['candidate_id', 'anomaly_score', 'is_known_transit']
    
    # *** DEFINITIVE FIX ***
    # The 'hide_index' argument is removed to ensure compatibility with your version of Streamlit.
    st.dataframe(candidates_df[display_cols], use_container_width=True)
    
    st.info(f"Displaying the top {len(candidates_df)} candidates found and ranked by the pipeline.")
else:
    st.warning("No candidate data found. Please complete the entire pipeline (Phase 1 through 4) and ensure results are in the `data/results` directory.")

