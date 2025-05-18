import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import json
from PIL import Image
import lightkurve as lk
import glob
from datetime import datetime

# Add parent directory to path for importing project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="Exo-Detector Dashboard",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #6B7280;
        font-size: 0.875rem;
    }
    /* Improve table styling */
    .dataframe {
        font-size: 0.9rem !important;
    }
    /* Custom styling for plotly charts */
    .js-plotly-plot {
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading functions
@st.cache_data(ttl=3600)
def load_candidates(file_path="../data/candidates/top_candidates.csv"):
    """Load candidate data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading candidate data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_all_candidates(file_path="../data/candidates/all_candidates.csv"):
    """Load all candidate data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading all candidate data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_light_curve(tic_id, sector=None):
    """Load light curve data for a specific TIC ID and sector."""
    try:
        # Define path pattern
        if sector is not None:
            path_pattern = f"../data/processed/TIC_{tic_id}/sector_{sector}_lc_processed.fits"
        else:
            path_pattern = f"../data/processed/TIC_{tic_id}/*_lc_processed.fits"
        
        # Find matching files
        lc_files = glob.glob(path_pattern)
        
        if not lc_files:
            return None
        
        # Load light curves
        lcs = []
        for lc_file in lc_files:
            lc = lk.read(lc_file)
            lcs.append(lc)
        
        # Combine light curves if multiple
        if len(lcs) > 1:
            combined_lc = lk.LightCurveCollection(lcs).stitch()
            return combined_lc
        else:
            return lcs[0]
    except Exception as e:
        st.error(f"Error loading light curve: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_folded_light_curve(tic_id, period):
    """Load and fold light curve data for a specific TIC ID with given period."""
    try:
        lc = load_light_curve(tic_id)
        if lc is None:
            return None
        
        # Fold light curve
        folded_lc = lc.fold(period=period)
        return folded_lc
    except Exception as e:
        st.error(f"Error folding light curve: {str(e)}")
        return None

def display_header():
    """Display dashboard header with logo and title."""
    col1, col2 = st.columns([1, 5])
    
    # Try to load logo if it exists
    try:
        logo = Image.open("../dashboard/assets/logo.png")
        col1.image(logo, width=100)
    except:
        col1.markdown("🔭")
    
    col2.markdown("<div class='main-header'>Exo-Detector Dashboard</div>", unsafe_allow_html=True)
    col2.markdown("Interactive dashboard for exoplanet transit candidate vetting")

def display_metrics(df):
    """Display key metrics as cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    # Total candidates
    col1.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    col1.markdown(f"<div class='metric-value'>{len(df)}</div>", unsafe_allow_html=True)
    col1.markdown("<div class='metric-label'>Total Candidates</div>", unsafe_allow_html=True)
    col1.markdown("</div>", unsafe_allow_html=True)
    
    # Average planet likelihood
    avg_likelihood = df['planet_likelihood'].mean()
    col2.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-value'>{avg_likelihood:.2f}</div>", unsafe_allow_html=True)
    col2.markdown("<div class='metric-label'>Avg. Planet Likelihood</div>", unsafe_allow_html=True)
    col2.markdown("</div>", unsafe_allow_html=True)
    
    # Average period
    avg_period = df['best_period'].mean()
    col3.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-value'>{avg_period:.2f}</div>", unsafe_allow_html=True)
    col3.markdown("<div class='metric-label'>Avg. Period (days)</div>", unsafe_allow_html=True)
    col3.markdown("</div>", unsafe_allow_html=True)
    
    # Unique stars
    unique_stars = df['tic_id'].nunique()
    col4.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-value'>{unique_stars}</div>", unsafe_allow_html=True)
    col4.markdown("<div class='metric-label'>Unique Stars</div>", unsafe_allow_html=True)
    col4.markdown("</div>", unsafe_allow_html=True)

def plot_candidate_distribution(df):
    """Plot distribution of candidate scores and periods."""
    col1, col2 = st.columns(2)
    
    # Plot planet likelihood distribution
    with col1:
        st.markdown("<div class='sub-header'>Planet Likelihood Distribution</div>", unsafe_allow_html=True)
        fig = px.histogram(
            df, 
            x='planet_likelihood',
            nbins=20,
            color_discrete_sequence=['#3B82F6'],
            opacity=0.7,
            marginal='box'
        )
        fig.update_layout(
            xaxis_title='Planet Likelihood Score',
            yaxis_title='Count',
            plot_bgcolor='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Plot period distribution
    with col2:
        st.markdown("<div class='sub-header'>Period Distribution</div>", unsafe_allow_html=True)
        fig = px.histogram(
            df, 
            x='best_period',
            nbins=20,
            color_discrete_sequence=['#10B981'],
            opacity=0.7,
            marginal='box'
        )
        fig.update_layout(
            xaxis_title='Period (days)',
            yaxis_title='Count',
            plot_bgcolor='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_candidate_scatter(df):
    """Plot scatter plot of candidates by period and likelihood."""
    st.markdown("<div class='sub-header'>Candidates by Period and Likelihood</div>", unsafe_allow_html=True)
    
    fig = px.scatter(
        df,
        x='best_period',
        y='planet_likelihood',
        color='max_score',
        size='num_candidates',
        hover_name='tic_id',
        hover_data=['num_sectors', 'mean_score'],
        color_continuous_scale='Viridis',
        opacity=0.7
    )
    fig.update_layout(
        xaxis_title='Period (days)',
        yaxis_title='Planet Likelihood Score',
        coloraxis_colorbar_title='Max Score',
        plot_bgcolor='white',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def display_candidate_table(df):
    """Display interactive table of candidates."""
    st.markdown("<div class='sub-header'>Candidate Table</div>", unsafe_allow_html=True)
    
    # Create a copy of the dataframe for display
    display_df = df.copy()
    
    # Format columns for display
    display_df['planet_likelihood'] = display_df['planet_likelihood'].round(2)
    display_df['max_score'] = display_df['max_score'].round(2)
    display_df['mean_score'] = display_df['mean_score'].round(2)
    display_df['best_period'] = display_df['best_period'].round(2)
    
    # Select columns to display
    display_cols = ['tic_id', 'num_candidates', 'num_sectors', 'max_score', 'mean_score', 'planet_likelihood', 'best_period']
    
    # Display table
    st.dataframe(
        display_df[display_cols],
        use_container_width=True,
        hide_index=True
    )

def display_candidate_details(df):
    """Display detailed information for a selected candidate."""
    st.markdown("<div class='sub-header'>Candidate Details</div>", unsafe_allow_html=True)
    
    # Select candidate
    selected_tic = st.selectbox("Select TIC ID", df['tic_id'].unique())
    
    # Get candidate data
    candidate = df[df['tic_id'] == selected_tic].iloc[0]
    
    # Display candidate details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Candidate Information")
        st.markdown(f"**TIC ID:** {candidate['tic_id']}")
        st.markdown(f"**Planet Likelihood:** {candidate['planet_likelihood']:.2f}")
        st.markdown(f"**Best Period:** {candidate['best_period']:.2f} days")
        st.markdown(f"**Number of Candidates:** {candidate['num_candidates']}")
        st.markdown(f"**Number of Sectors:** {candidate['num_sectors']}")
        st.markdown(f"**Max Score:** {candidate['max_score']:.2f}")
        st.markdown(f"**Mean Score:** {candidate['mean_score']:.2f}")
    
    with col2:
        # Load and display folded light curve
        st.markdown("### Folded Light Curve")
        folded_lc = load_folded_light_curve(selected_tic, candidate['best_period'])
        
        if folded_lc is not None:
            # Create binned version
            binned_lc = folded_lc.bin(time_bin_size=0.01)
            
            # Create figure
            fig = go.Figure()
            
            # Add scatter plot for raw data
            fig.add_trace(go.Scatter(
                x=folded_lc.time.value,
                y=folded_lc.flux.value,
                mode='markers',
                marker=dict(color='rgba(59, 130, 246, 0.3)', size=3),
                name='Raw Data'
            ))
            
            # Add line plot for binned data
            fig.add_trace(go.Scatter(
                x=binned_lc.time.value,
                y=binned_lc.flux.value,
                mode='lines+markers',
                marker=dict(color='rgba(220, 38, 38, 0.8)', size=6),
                line=dict(color='rgba(220, 38, 38, 0.8)', width=2),
                name='Binned Data'
            ))
            
            # Update layout
            fig.update_layout(
                xaxis_title='Phase',
                yaxis_title='Normalized Flux',
                plot_bgcolor='white',
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Light curve data not available for this candidate.")
    
    # Display transit times
    st.markdown("### Transit Times")
    
    # Parse transit times from string if needed
    transit_times = candidate['transit_times']
    if isinstance(transit_times, str):
        try:
            transit_times = [float(t) for t in transit_times.split(',')]
        except:
            transit_times = []
    
    if transit_times:
        # Create a dataframe for display
        times_df = pd.DataFrame({
            'Transit Number': range(1, len(transit_times) + 1),
            'Time (BTJD)': transit_times
        })
        
        # Display table
        st.dataframe(times_df, use_container_width=True, hide_index=True)
        
        # Display raw light curve with transit times
        st.markdown("### Raw Light Curve with Transit Times")
        
        # Load light curve
        lc = load_light_curve(selected_tic)
        
        if lc is not None:
            # Create figure
            fig = go.Figure()
            
            # Add scatter plot for light curve
            fig.add_trace(go.Scatter(
                x=lc.time.value,
                y=lc.flux.value,
                mode='markers',
                marker=dict(color='rgba(59, 130, 246, 0.3)', size=3),
                name='Light Curve'
            ))
            
            # Add vertical lines for transit times
            for i, time in enumerate(transit_times):
                fig.add_vline(
                    x=time,
                    line=dict(color='rgba(220, 38, 38, 0.8)', width=1, dash='dash'),
                    annotation_text=f"Transit {i+1}",
                    annotation_position="top right"
                )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Time (BTJD)',
                yaxis_title='Normalized Flux',
                plot_bgcolor='white',
                height=400
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Light curve data not available for this candidate.")
    else:
        st.warning("No transit times available for this candidate.")

def display_vetting_form(df):
    """Display form for vetting candidates."""
    st.markdown("<div class='sub-header'>Candidate Vetting</div>", unsafe_allow_html=True)
    
    # Select candidate
    selected_tic = st.selectbox("Select TIC ID for Vetting", df['tic_id'].unique(), key="vetting_tic_id")
    
    # Get candidate data
    candidate = df[df['tic_id'] == selected_tic].iloc[0]
    
    # Display candidate summary
    st.markdown(f"**TIC ID:** {candidate['tic_id']} | **Planet Likelihood:** {candidate['planet_likelihood']:.2f} | **Period:** {candidate['best_period']:.2f} days")
    
    # Create vetting form
    with st.form("vetting_form"):
        # Vetting criteria
        st.markdown("### Vetting Criteria")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transit_shape = st.radio(
                "Transit Shape",
                ["Good", "Acceptable", "Poor"],
                help="Assess the shape of the transit (U or V shaped, consistent depth)"
            )
            
            secondary_eclipse = st.radio(
                "Secondary Eclipse",
                ["None", "Possible", "Present"],
                help="Check for presence of secondary eclipse which may indicate an eclipsing binary"
            )
            
            odd_even_diff = st.radio(
                "Odd-Even Difference",
                ["None", "Slight", "Significant"],
                help="Check for differences between odd and even transits which may indicate an eclipsing binary"
            )
        
        with col2:
            out_of_transit_var = st.radio(
                "Out-of-Transit Variability",
                ["Low", "Medium", "High"],
                help="Assess the level of variability outside of transit events"
            )
            
            snr_quality = st.radio(
                "Signal-to-Noise Ratio",
                ["High", "Medium", "Low"],
                help="Assess the signal-to-noise ratio of the transit signal"
            )
            
            data_quality = st.radio(
                "Data Quality",
                ["Good", "Acceptable", "Poor"],
                help="Assess the overall quality of the data (gaps, outliers, etc.)"
            )
        
        # Overall assessment
        st.markdown("### Overall Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            planet_candidate = st.radio(
                "Planet Candidate",
                ["Yes", "Maybe", "No"],
                help="Is this a good planet candidate?"
            )
        
        with col2:
            priority = st.slider(
                "Priority for Follow-up",
                min_value=1,
                max_value=5,
                value=3,
                help="Priority for follow-up observations (1 = lowest, 5 = highest)"
            )
        
        # Notes
        notes = st.text_area(
            "Notes",
            help="Additional notes or comments about this candidate"
        )
        
        # Submit button
        submitted = st.form_submit_button("Submit Vetting Results")
        
        if submitted:
            # Create vetting results dictionary
            vetting_results = {
                "tic_id": selected_tic,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "transit_shape": transit_shape,
                "secondary_eclipse": secondary_eclipse,
                "odd_even_diff": odd_even_diff,
                "out_of_transit_var": out_of_transit_var,
                "snr_quality": snr_quality,
                "data_quality": data_quality,
                "planet_candidate": planet_candidate,
                "priority": priority,
                "notes": notes
            }
            
            # Save vetting results
            vetting_dir = "../data/vetting"
            os.makedirs(vetting_dir, exist_ok=True)
            
            vetting_file = os.path.join(vetting_dir, f"vetting_TIC_{selected_tic}.json")
            
            with open(vetting_file, "w") as f:
                json.dump(vetting_results, f, indent=4)
            
            st.success(f"Vetting results for TIC {selected_tic} saved successfully!")

def display_vetting_results():
    """Display summary of vetting results."""
    st.markdown("<div class='sub-header'>Vetting Results Summary</div>", unsafe_allow_html=True)
    
    # Load vetting results
    vetting_dir = "../data/vetting"
    vetting_files = glob.glob(os.path.join(vetting_dir, "vetting_TIC_*.json"))
    
    if not vetting_files:
        st.info("No vetting results available yet.")
        return
    
    # Load all vetting results
    vetting_results = []
    for file in vetting_files:
        try:
            with open(file, "r") as f:
                result = json.load(f)
                vetting_results.append(result)
        except Exception as e:
            st.warning(f"Error loading vetting file {file}: {str(e)}")
    
    # Create dataframe
    vetting_df = pd.DataFrame(vetting_results)
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Vetted", len(vetting_df))
    
    with col2:
        planet_counts = vetting_df['planet_candidate'].value_counts()
        yes_count = planet_counts.get('Yes', 0)
        st.metric("Planet Candidates", yes_count)
    
    with col3:
        avg_priority = vetting_df['priority'].mean()
        st.metric("Avg. Priority", f"{avg_priority:.1f}")
    
    # Display vetting results table
    st.dataframe(
        vetting_df[['tic_id', 'timestamp', 'planet_candidate', 'priority', 'transit_shape', 'data_quality']],
        use_container_width=True,
        hide_index=True
    )
    
    # Plot planet candidate distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            vetting_df, 
            names='planet_candidate',
            color='planet_candidate',
            color_discrete_map={
                'Yes': '#10B981',
                'Maybe': '#F59E0B',
                'No': '#EF4444'
            },
            title='Planet Candidate Distribution'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            vetting_df,
            x='priority',
            nbins=5,
            color='planet_candidate',
            color_discrete_map={
                'Yes': '#10B981',
                'Maybe': '#F59E0B',
                'No': '#EF4444'
            },
            title='Priority Distribution'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the dashboard."""
    # Display header
    display_header()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Candidate Explorer", "Candidate Vetting", "Vetting Results"]
    )
    
    # Load data
    top_candidates = load_candidates()
    all_candidates = load_all_candidates()
    
    if len(top_candidates) == 0:
        st.error("No candidate data available. Please run the candidate ranking pipeline first.")
        return
    
    # Display selected page
    if page == "Overview":
        # Display metrics
        display_metrics(top_candidates)
        
        # Display candidate distribution
        plot_candidate_distribution(top_candidates)
        
        # Display candidate scatter plot
        plot_candidate_scatter(top_candidates)
        
        # Display candidate table
        display_candidate_table(top_candidates)
    
    elif page == "Candidate Explorer":
        # Display candidate details
        display_candidate_details(top_candidates)
    
    elif page == "Candidate Vetting":
        # Display vetting form
        display_vetting_form(top_candidates)
    
    elif page == "Vetting Results":
        # Display vetting results
        display_vetting_results()
    
    # Footer
    st.markdown("<div class='footer'>Exo-Detector Dashboard | Developed by Manus AI | © 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
