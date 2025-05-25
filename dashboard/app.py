#!/usr/bin/env python3
"""
Exo-Detector: Streamlit Dashboard for Candidate Vetting

This dashboard provides an interactive interface for reviewing and vetting
exoplanet transit candidates identified by the Exo-Detector pipeline.

Author: Manus AI
Date: May 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import json
import glob
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Set page configuration
st.set_page_config(
    page_title="Exo-Detector Dashboard",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Dashboard title
st.markdown(
    """
    <div class="dashboard-title">
        <h1>🔭 Exo-Detector Dashboard</h1>
        <p class="subtitle">Exoplanet Transit Candidate Vetting System</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Select Page",
        ["Overview", "Candidate Explorer", "Vetting Tool", "Pipeline Status", "Documentation"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("## Data Settings")
    data_dir = st.text_input("Data Directory", value="data", help="Directory containing Exo-Detector data")
    
    # Convert to absolute path
    data_dir = os.path.abspath(data_dir)
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        st.warning(f"Directory not found: {data_dir}")
    
    st.markdown("---")
    
    st.markdown("## Filters")
    min_score = st.slider("Minimum Score", 0.0, 1.0, 0.3, 0.05, 
                         help="Minimum candidate score to display")
    
    min_snr = st.slider("Minimum SNR", 0.0, 20.0, 5.0, 0.5,
                       help="Minimum signal-to-noise ratio")
    
    min_transits = st.slider("Minimum Transits", 1, 10, 2,
                            help="Minimum number of detected transits")
    
    st.markdown("---")
    
    st.markdown("## About")
    st.markdown("""
    **Exo-Detector** is a machine learning pipeline for detecting exoplanet transits in TESS light curves.
    
    This dashboard provides tools for reviewing and vetting candidates identified by the pipeline.
    
    Version: 1.0.0
    """)

# Function to load candidate data
@st.cache_data(ttl=300)
def load_candidate_data(data_dir):
    """
    Load candidate data from files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data
        
    Returns:
    --------
    tuple
        (catalog_df, results_dict, light_curves)
    """
    try:
        # Define paths
        candidates_dir = os.path.join(data_dir, "candidates")
        catalog_file = os.path.join(candidates_dir, "candidate_catalog.csv")
        results_file = os.path.join(candidates_dir, "candidate_results.json")
        
        # Check if files exist
        if not os.path.exists(catalog_file):
            return None, None, {}
        
        if not os.path.exists(results_file):
            return None, None, {}
        
        # Load catalog
        catalog_df = pd.read_csv(catalog_file)
        
        # Load results
        with open(results_file, 'r') as f:
            results_dict = json.load(f)
        
        # Load light curves
        light_curves = {}
        processed_dir = os.path.join(data_dir, "processed")
        
        for _, row in catalog_df.iterrows():
            tic_id = row['tic_id']
            sector = row['sector']
            
            # Find light curve file
            lc_file = os.path.join(processed_dir, f"TIC_{tic_id}", f"sector_{sector}_lc.csv")
            
            if os.path.exists(lc_file):
                # Load light curve
                lc_df = pd.read_csv(lc_file)
                
                # Store in dictionary
                key = f"TIC_{tic_id}_sector_{sector}"
                light_curves[key] = lc_df
        
        return catalog_df, results_dict, light_curves
    
    except Exception as e:
        st.error(f"Error loading candidate data: {str(e)}")
        return None, None, {}

# Function to filter candidates
def filter_candidates(catalog_df, min_score=0.0, min_snr=0.0, min_transits=1):
    """
    Filter candidates based on criteria.
    
    Parameters:
    -----------
    catalog_df : pandas.DataFrame
        Candidate catalog
    min_score : float
        Minimum candidate score
    min_snr : float
        Minimum signal-to-noise ratio
    min_transits : int
        Minimum number of detected transits
        
    Returns:
    --------
    pandas.DataFrame
        Filtered catalog
    """
    if catalog_df is None:
        return None
    
    # Apply filters
    filtered_df = catalog_df[
        (catalog_df['score'] >= min_score) &
        (catalog_df['snr'] >= min_snr) &
        (catalog_df['num_transits'] >= min_transits)
    ]
    
    return filtered_df

# Function to create light curve plot
def create_light_curve_plot(time, flux, candidates=None, period=None):
    """
    Create a light curve plot.
    
    Parameters:
    -----------
    time : numpy.ndarray
        Array of time values
    flux : numpy.ndarray
        Array of flux values
    candidates : list or None
        List of candidate dictionaries
    period : float or None
        Orbital period in days
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Full Light Curve", "Phase-folded Light Curve" if period else ""),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Add full light curve
    fig.add_trace(
        go.Scattergl(
            x=time,
            y=flux,
            mode='markers',
            marker=dict(
                size=2,
                color='rgba(0, 0, 0, 0.5)'
            ),
            name='Flux'
        ),
        row=1, col=1
    )
    
    # Add transit times if available
    if candidates:
        transit_times = [c['mid_time'] for c in candidates]
        
        for t in transit_times:
            fig.add_vline(
                x=t,
                line=dict(
                    color="red",
                    width=1,
                    dash="dash"
                ),
                row=1, col=1
            )
    
    # Add phase-folded light curve if period is available
    if period:
        # Calculate phase
        phase = ((time % period) / period + 0.5) % 1.0 - 0.5
        
        # Sort by phase
        sort_idx = np.argsort(phase)
        phase = phase[sort_idx]
        folded_flux = flux[sort_idx]
        
        # Add trace
        fig.add_trace(
            go.Scattergl(
                x=phase,
                y=folded_flux,
                mode='markers',
                marker=dict(
                    size=2,
                    color='rgba(0, 0, 0, 0.5)'
                ),
                name='Folded Flux'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Time (BTJD)", row=1, col=1)
    
    if period:
        fig.update_xaxes(title_text="Phase", row=2, col=1)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Normalized Flux", row=1, col=1)
    
    if period:
        fig.update_yaxes(title_text="Normalized Flux", row=2, col=1)
    
    return fig

# Function to create transit parameter plot
def create_transit_parameter_plot(depth, duration, period, snr):
    """
    Create a transit parameter plot.
    
    Parameters:
    -----------
    depth : float
        Transit depth
    duration : float
        Transit duration in hours
    period : float
        Orbital period in days
    snr : float
        Signal-to-noise ratio
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add bar for each parameter
    parameters = ['Depth', 'Duration', 'Period', 'SNR']
    values = [depth, duration, period, snr]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Normalize values for display
    norm_values = [
        depth * 1000,  # Convert to parts per thousand
        duration,      # Hours
        period,        # Days
        snr            # As is
    ]
    
    # Add bars
    for i, (param, value, norm_value, color) in enumerate(zip(parameters, values, norm_values, colors)):
        fig.add_trace(
            go.Bar(
                x=[param],
                y=[norm_value],
                text=[f"{value:.4f}" if i == 0 else f"{value:.2f}"],
                textposition='auto',
                marker_color=color,
                name=param
            )
        )
    
    # Update layout
    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(
            title="Value",
            showgrid=True
        ),
        xaxis=dict(
            title="Parameter",
            showgrid=False
        )
    )
    
    return fig

# Function to create pipeline status plot
@st.cache_data(ttl=300)
def create_pipeline_status_plot(data_dir):
    """
    Create a pipeline status plot.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    # Define phases
    phases = [
        "Phase 1: Data Ingestion & Preprocessing",
        "Phase 2: GAN-based Transit Augmentation",
        "Phase 3: Semi-Supervised Anomaly Detection",
        "Phase 4: Candidate Scoring & Ranking"
    ]
    
    # Check if result files exist
    results_dir = os.path.join(data_dir, "results")
    
    phase1_file = os.path.join(results_dir, "phase1_results.json")
    phase2_file = os.path.join(results_dir, "phase2_results.json")
    phase3_file = os.path.join(results_dir, "phase3_results.json")
    phase4_file = os.path.join(results_dir, "phase4_results.json")
    
    # Check status of each phase
    status = []
    
    if os.path.exists(phase1_file):
        with open(phase1_file, 'r') as f:
            phase1_results = json.load(f)
            if phase1_results.get('steps_executed', {}).get('data_ingestion', False):
                status.append("Complete")
            else:
                status.append("Partial")
    else:
        status.append("Not Run")
    
    if os.path.exists(phase2_file):
        with open(phase2_file, 'r') as f:
            phase2_results = json.load(f)
            if phase2_results.get('steps_executed', {}).get('gan_training', False):
                status.append("Complete")
            else:
                status.append("Partial")
    else:
        status.append("Not Run")
    
    if os.path.exists(phase3_file):
        with open(phase3_file, 'r') as f:
            phase3_results = json.load(f)
            if phase3_results.get('steps_executed', {}).get('anomaly_detection', False):
                status.append("Complete")
            else:
                status.append("Partial")
    else:
        status.append("Not Run")
    
    if os.path.exists(phase4_file):
        with open(phase4_file, 'r') as f:
            phase4_results = json.load(f)
            if phase4_results.get('steps_executed', {}).get('candidate_ranking', False):
                status.append("Complete")
            else:
                status.append("Partial")
    else:
        status.append("Not Run")
    
    # Define colors for status
    colors = {
        "Complete": "#2ca02c",
        "Partial": "#ff7f0e",
        "Not Run": "#d62728"
    }
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    for i, (phase, stat) in enumerate(zip(phases, status)):
        fig.add_trace(
            go.Bar(
                x=[phase],
                y=[1],
                text=[stat],
                textposition='auto',
                marker_color=colors[stat],
                name=stat
            )
        )
    
    # Update layout
    fig.update_layout(
        height=300,
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(
            title="Status",
            showticklabels=False,
            showgrid=False
        ),
        xaxis=dict(
            title="Pipeline Phase",
            showgrid=False
        )
    )
    
    return fig

# Function to create candidate distribution plot
def create_candidate_distribution_plot(catalog_df):
    """
    Create a candidate distribution plot.
    
    Parameters:
    -----------
    catalog_df : pandas.DataFrame
        Candidate catalog
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    if catalog_df is None or len(catalog_df) == 0:
        # Create empty figure
        fig = go.Figure()
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[dict(
                text="No candidates found",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return fig
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Period Distribution", "Depth Distribution", "SNR Distribution"),
        horizontal_spacing=0.1
    )
    
    # Add period histogram
    fig.add_trace(
        go.Histogram(
            x=catalog_df['period'],
            marker_color='#1f77b4',
            nbinsx=20,
            name='Period'
        ),
        row=1, col=1
    )
    
    # Add depth histogram
    fig.add_trace(
        go.Histogram(
            x=catalog_df['depth'] * 1000,  # Convert to parts per thousand
            marker_color='#ff7f0e',
            nbinsx=20,
            name='Depth'
        ),
        row=1, col=2
    )
    
    # Add SNR histogram
    fig.add_trace(
        go.Histogram(
            x=catalog_df['snr'],
            marker_color='#2ca02c',
            nbinsx=20,
            name='SNR'
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Period (days)", row=1, col=1)
    fig.update_xaxes(title_text="Depth (ppt)", row=1, col=2)
    fig.update_xaxes(title_text="SNR", row=1, col=3)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
    return fig

# Function to create candidate scatter plot
def create_candidate_scatter_plot(catalog_df):
    """
    Create a candidate scatter plot.
    
    Parameters:
    -----------
    catalog_df : pandas.DataFrame
        Candidate catalog
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    if catalog_df is None or len(catalog_df) == 0:
        # Create empty figure
        fig = go.Figure()
        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[dict(
                text="No candidates found",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=catalog_df['period'],
            y=catalog_df['depth'] * 1000,  # Convert to parts per thousand
            mode='markers',
            marker=dict(
                size=10,
                color=catalog_df['score'],
                colorscale='Viridis',
                colorbar=dict(
                    title="Score"
                ),
                line=dict(
                    width=1,
                    color='rgba(0, 0, 0, 0.5)'
                )
            ),
            text=catalog_df.apply(
                lambda row: f"TIC {row['tic_id']}<br>Score: {row['score']:.3f}<br>SNR: {row['snr']:.1f}",
                axis=1
            ),
            hoverinfo='text'
        )
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            title="Period (days)",
            type="log",
            showgrid=True
        ),
        yaxis=dict(
            title="Depth (ppt)",
            type="log",
            showgrid=True
        )
    )
    
    return fig

# Function to save vetting results
def save_vetting_results(data_dir, tic_id, sector, vetting_status, vetting_notes):
    """
    Save vetting results to a file.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data
    tic_id : int
        TIC ID
    sector : int
        Sector
    vetting_status : str
        Vetting status
    vetting_notes : str
        Vetting notes
        
    Returns:
    --------
    bool
        Whether results were successfully saved
    """
    try:
        # Define paths
        vetting_dir = os.path.join(data_dir, "vetting")
        os.makedirs(vetting_dir, exist_ok=True)
        
        vetting_file = os.path.join(vetting_dir, "vetting_results.csv")
        
        # Create DataFrame
        vetting_data = {
            'tic_id': [tic_id],
            'sector': [sector],
            'status': [vetting_status],
            'notes': [vetting_notes],
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        
        vetting_df = pd.DataFrame(vetting_data)
        
        # Check if file exists
        if os.path.exists(vetting_file):
            # Load existing file
            existing_df = pd.read_csv(vetting_file)
            
            # Check if entry already exists
            mask = (existing_df['tic_id'] == tic_id) & (existing_df['sector'] == sector)
            
            if mask.any():
                # Update existing entry
                existing_df.loc[mask, 'status'] = vetting_status
                existing_df.loc[mask, 'notes'] = vetting_notes
                existing_df.loc[mask, 'timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Save to file
                existing_df.to_csv(vetting_file, index=False)
            else:
                # Append new entry
                combined_df = pd.concat([existing_df, vetting_df], ignore_index=True)
                
                # Save to file
                combined_df.to_csv(vetting_file, index=False)
        else:
            # Create new file
            vetting_df.to_csv(vetting_file, index=False)
        
        return True
    
    except Exception as e:
        st.error(f"Error saving vetting results: {str(e)}")
        return False

# Function to load vetting results
@st.cache_data(ttl=300)
def load_vetting_results(data_dir):
    """
    Load vetting results from a file.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data
        
    Returns:
    --------
    pandas.DataFrame
        Vetting results
    """
    try:
        # Define paths
        vetting_dir = os.path.join(data_dir, "vetting")
        vetting_file = os.path.join(vetting_dir, "vetting_results.csv")
        
        # Check if file exists
        if not os.path.exists(vetting_file):
            return None
        
        # Load file
        vetting_df = pd.read_csv(vetting_file)
        
        return vetting_df
    
    except Exception as e:
        st.error(f"Error loading vetting results: {str(e)}")
        return None

# Load data
catalog_df, results_dict, light_curves = load_candidate_data(data_dir)

# Filter candidates
filtered_df = filter_candidates(catalog_df, min_score, min_snr, min_transits)

# Overview page
if page == "Overview":
    st.markdown("## Pipeline Overview")
    
    # Create columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Pipeline Status")
        
        # Create pipeline status plot
        status_plot = create_pipeline_status_plot(data_dir)
        st.plotly_chart(status_plot, use_container_width=True)
    
    with col2:
        st.markdown("### Candidate Summary")
        
        # Check if data is available
        if filtered_df is not None and len(filtered_df) > 0:
            # Create metrics
            total_candidates = len(filtered_df)
            avg_score = filtered_df['score'].mean()
            max_score = filtered_df['score'].max()
            
            # Create columns for metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Total Candidates", total_candidates)
            
            with metric_col2:
                st.metric("Average Score", f"{avg_score:.3f}")
            
            with metric_col3:
                st.metric("Max Score", f"{max_score:.3f}")
        else:
            st.warning("No candidates found matching the current filters.")
    
    st.markdown("## Candidate Distribution")
    
    # Create candidate distribution plot
    dist_plot = create_candidate_distribution_plot(filtered_df)
    st.plotly_chart(dist_plot, use_container_width=True)
    
    st.markdown("## Candidate Scatter Plot")
    
    # Create candidate scatter plot
    scatter_plot = create_candidate_scatter_plot(filtered_df)
    st.plotly_chart(scatter_plot, use_container_width=True)
    
    st.markdown("## Top Candidates")
    
    # Check if data is available
    if filtered_df is not None and len(filtered_df) > 0:
        # Sort by score
        top_df = filtered_df.sort_values('score', ascending=False).head(10)
        
        # Display table
        st.dataframe(
            top_df[['tic_id', 'sector', 'score', 'period', 'depth', 'duration', 'snr', 'num_transits']],
            use_container_width=True,
            column_config={
                'tic_id': st.column_config.NumberColumn("TIC ID"),
                'sector': st.column_config.NumberColumn("Sector"),
                'score': st.column_config.NumberColumn("Score", format="%.3f"),
                'period': st.column_config.NumberColumn("Period (days)", format="%.2f"),
                'depth': st.column_config.NumberColumn("Depth", format="%.6f"),
                'duration': st.column_config.NumberColumn("Duration (hours)", format="%.2f"),
                'snr': st.column_config.NumberColumn("SNR", format="%.1f"),
                'num_transits': st.column_config.NumberColumn("Transits")
            }
        )
    else:
        st.warning("No candidates found matching the current filters.")

# Candidate Explorer page
elif page == "Candidate Explorer":
    st.markdown("## Candidate Explorer")
    
    # Check if data is available
    if filtered_df is not None and len(filtered_df) > 0:
        # Create columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### Candidate List")
            
            # Sort by score
            sorted_df = filtered_df.sort_values('score', ascending=False)
            
            # Create selection
            selected_index = st.selectbox(
                "Select Candidate",
                range(len(sorted_df)),
                format_func=lambda i: f"TIC {sorted_df.iloc[i]['tic_id']} - Score: {sorted_df.iloc[i]['score']:.3f}"
            )
            
            # Get selected candidate
            selected_candidate = sorted_df.iloc[selected_index]
            
            # Display candidate details
            st.markdown("### Candidate Details")
            
            st.markdown(f"**TIC ID:** {selected_candidate['tic_id']}")
            st.markdown(f"**Sector:** {selected_candidate['sector']}")
            st.markdown(f"**Score:** {selected_candidate['score']:.3f}")
            st.markdown(f"**Period:** {selected_candidate['period']:.2f} days")
            st.markdown(f"**Depth:** {selected_candidate['depth']:.6f}")
            st.markdown(f"**Duration:** {selected_candidate['duration']:.2f} hours")
            st.markdown(f"**SNR:** {selected_candidate['snr']:.1f}")
            st.markdown(f"**Transits:** {selected_candidate['num_transits']}")
            
            # Create transit parameter plot
            param_plot = create_transit_parameter_plot(
                selected_candidate['depth'],
                selected_candidate['duration'],
                selected_candidate['period'],
                selected_candidate['snr']
            )
            st.plotly_chart(param_plot, use_container_width=True)
        
        with col2:
            st.markdown("### Light Curve")
            
            # Get light curve
            tic_id = selected_candidate['tic_id']
            sector = selected_candidate['sector']
            key = f"TIC_{tic_id}_sector_{sector}"
            
            if key in light_curves:
                lc_df = light_curves[key]
                
                # Get time and flux
                time = lc_df['time'].values
                flux = lc_df['flux'].values
                
                # Get candidates
                candidates = None
                for result in results_dict:
                    if result['tic_id'] == tic_id and result['sector'] == sector:
                        candidates = result['candidates']
                        break
                
                # Create light curve plot
                lc_plot = create_light_curve_plot(
                    time,
                    flux,
                    candidates,
                    selected_candidate['period']
                )
                st.plotly_chart(lc_plot, use_container_width=True)
            else:
                st.warning(f"Light curve not found for TIC {tic_id}, Sector {sector}")
    else:
        st.warning("No candidates found matching the current filters.")

# Vetting Tool page
elif page == "Vetting Tool":
    st.markdown("## Candidate Vetting Tool")
    
    # Check if data is available
    if filtered_df is not None and len(filtered_df) > 0:
        # Load vetting results
        vetting_df = load_vetting_results(data_dir)
        
        # Create columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### Candidate List")
            
            # Sort by score
            sorted_df = filtered_df.sort_values('score', ascending=False)
            
            # Create selection
            selected_index = st.selectbox(
                "Select Candidate",
                range(len(sorted_df)),
                format_func=lambda i: f"TIC {sorted_df.iloc[i]['tic_id']} - Score: {sorted_df.iloc[i]['score']:.3f}",
                key="vetting_candidate_select"
            )
            
            # Get selected candidate
            selected_candidate = sorted_df.iloc[selected_index]
            
            # Display candidate details
            st.markdown("### Candidate Details")
            
            st.markdown(f"**TIC ID:** {selected_candidate['tic_id']}")
            st.markdown(f"**Sector:** {selected_candidate['sector']}")
            st.markdown(f"**Score:** {selected_candidate['score']:.3f}")
            st.markdown(f"**Period:** {selected_candidate['period']:.2f} days")
            st.markdown(f"**Depth:** {selected_candidate['depth']:.6f}")
            st.markdown(f"**Duration:** {selected_candidate['duration']:.2f} hours")
            st.markdown(f"**SNR:** {selected_candidate['snr']:.1f}")
            st.markdown(f"**Transits:** {selected_candidate['num_transits']}")
            
            # Vetting form
            st.markdown("### Vetting Form")
            
            # Check if candidate has been vetted
            vetting_status = "Unvetted"
            vetting_notes = ""
            
            if vetting_df is not None:
                mask = (vetting_df['tic_id'] == selected_candidate['tic_id']) & (vetting_df['sector'] == selected_candidate['sector'])
                
                if mask.any():
                    vetting_status = vetting_df.loc[mask, 'status'].iloc[0]
                    vetting_notes = vetting_df.loc[mask, 'notes'].iloc[0]
            
            # Create form
            with st.form("vetting_form"):
                status = st.radio(
                    "Vetting Status",
                    ["Unvetted", "Confirmed Planet", "Likely Planet", "Uncertain", "Likely False Positive", "Confirmed False Positive"],
                    index=["Unvetted", "Confirmed Planet", "Likely Planet", "Uncertain", "Likely False Positive", "Confirmed False Positive"].index(vetting_status)
                )
                
                notes = st.text_area("Vetting Notes", value=vetting_notes)
                
                submitted = st.form_submit_button("Save Vetting Results")
                
                if submitted:
                    # Save vetting results
                    success = save_vetting_results(
                        data_dir,
                        selected_candidate['tic_id'],
                        selected_candidate['sector'],
                        status,
                        notes
                    )
                    
                    if success:
                        st.success("Vetting results saved successfully!")
                        
                        # Refresh vetting results
                        load_vetting_results.clear()
                    else:
                        st.error("Error saving vetting results.")
        
        with col2:
            st.markdown("### Light Curve")
            
            # Get light curve
            tic_id = selected_candidate['tic_id']
            sector = selected_candidate['sector']
            key = f"TIC_{tic_id}_sector_{sector}"
            
            if key in light_curves:
                lc_df = light_curves[key]
                
                # Get time and flux
                time = lc_df['time'].values
                flux = lc_df['flux'].values
                
                # Get candidates
                candidates = None
                for result in results_dict:
                    if result['tic_id'] == tic_id and result['sector'] == sector:
                        candidates = result['candidates']
                        break
                
                # Create light curve plot
                lc_plot = create_light_curve_plot(
                    time,
                    flux,
                    candidates,
                    selected_candidate['period']
                )
                st.plotly_chart(lc_plot, use_container_width=True)
                
                # Create transit parameter plot
                param_plot = create_transit_parameter_plot(
                    selected_candidate['depth'],
                    selected_candidate['duration'],
                    selected_candidate['period'],
                    selected_candidate['snr']
                )
                st.plotly_chart(param_plot, use_container_width=True)
            else:
                st.warning(f"Light curve not found for TIC {tic_id}, Sector {sector}")
        
        # Display vetting statistics
        st.markdown("## Vetting Statistics")
        
        if vetting_df is not None and len(vetting_df) > 0:
            # Count by status
            status_counts = vetting_df['status'].value_counts()
            
            # Create columns
            stat_cols = st.columns(len(status_counts))
            
            for i, (status, count) in enumerate(status_counts.items()):
                with stat_cols[i]:
                    st.metric(status, count)
            
            # Create pie chart
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Vetting Status Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.dataframe(
                vetting_df,
                use_container_width=True,
                column_config={
                    'tic_id': st.column_config.NumberColumn("TIC ID"),
                    'sector': st.column_config.NumberColumn("Sector"),
                    'status': st.column_config.TextColumn("Status"),
                    'notes': st.column_config.TextColumn("Notes"),
                    'timestamp': st.column_config.DatetimeColumn("Timestamp")
                }
            )
        else:
            st.warning("No vetting results found.")
    else:
        st.warning("No candidates found matching the current filters.")

# Pipeline Status page
elif page == "Pipeline Status":
    st.markdown("## Pipeline Status")
    
    # Create pipeline status plot
    status_plot = create_pipeline_status_plot(data_dir)
    st.plotly_chart(status_plot, use_container_width=True)
    
    # Check data directories
    st.markdown("### Data Directory Status")
    
    # Create columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Raw Data")
        
        raw_dir = os.path.join(data_dir, "raw")
        if os.path.exists(raw_dir):
            # Count files
            raw_files = glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True)
            raw_files = [f for f in raw_files if os.path.isfile(f)]
            
            st.success(f"Raw data directory exists with {len(raw_files)} files.")
        else:
            st.error("Raw data directory not found.")
        
        st.markdown("#### Processed Data")
        
        processed_dir = os.path.join(data_dir, "processed")
        if os.path.exists(processed_dir):
            # Count files
            processed_files = glob.glob(os.path.join(processed_dir, "**", "*"), recursive=True)
            processed_files = [f for f in processed_files if os.path.isfile(f)]
            
            st.success(f"Processed data directory exists with {len(processed_files)} files.")
        else:
            st.error("Processed data directory not found.")
        
        st.markdown("#### Transit Windows")
        
        transit_dir = os.path.join(data_dir, "transit_windows")
        if os.path.exists(transit_dir):
            # Count files
            transit_files = glob.glob(os.path.join(transit_dir, "**", "*"), recursive=True)
            transit_files = [f for f in transit_files if os.path.isfile(f)]
            
            st.success(f"Transit windows directory exists with {len(transit_files)} files.")
        else:
            st.error("Transit windows directory not found.")
        
        st.markdown("#### Non-Transit Windows")
        
        non_transit_dir = os.path.join(data_dir, "non_transit_windows")
        if os.path.exists(non_transit_dir):
            # Count files
            non_transit_files = glob.glob(os.path.join(non_transit_dir, "**", "*"), recursive=True)
            non_transit_files = [f for f in non_transit_files if os.path.isfile(f)]
            
            st.success(f"Non-transit windows directory exists with {len(non_transit_files)} files.")
        else:
            st.error("Non-transit windows directory not found.")
    
    with col2:
        st.markdown("#### Synthetic Transits")
        
        synthetic_dir = os.path.join(data_dir, "synthetic_transits")
        if os.path.exists(synthetic_dir):
            # Count files
            synthetic_files = glob.glob(os.path.join(synthetic_dir, "**", "*"), recursive=True)
            synthetic_files = [f for f in synthetic_files if os.path.isfile(f)]
            
            st.success(f"Synthetic transits directory exists with {len(synthetic_files)} files.")
        else:
            st.error("Synthetic transits directory not found.")
        
        st.markdown("#### Models")
        
        models_dir = os.path.join(data_dir, "models")
        if os.path.exists(models_dir):
            # Count files
            model_files = glob.glob(os.path.join(models_dir, "**", "*"), recursive=True)
            model_files = [f for f in model_files if os.path.isfile(f)]
            
            st.success(f"Models directory exists with {len(model_files)} files.")
        else:
            st.error("Models directory not found.")
        
        st.markdown("#### Candidates")
        
        candidates_dir = os.path.join(data_dir, "candidates")
        if os.path.exists(candidates_dir):
            # Count files
            candidate_files = glob.glob(os.path.join(candidates_dir, "**", "*"), recursive=True)
            candidate_files = [f for f in candidate_files if os.path.isfile(f)]
            
            st.success(f"Candidates directory exists with {len(candidate_files)} files.")
        else:
            st.error("Candidates directory not found.")
        
        st.markdown("#### Validation")
        
        validation_dir = os.path.join(data_dir, "validation")
        if os.path.exists(validation_dir):
            # Count files
            validation_files = glob.glob(os.path.join(validation_dir, "**", "*"), recursive=True)
            validation_files = [f for f in validation_files if os.path.isfile(f)]
            
            st.success(f"Validation directory exists with {len(validation_files)} files.")
        else:
            st.error("Validation directory not found.")
    
    # Pipeline results
    st.markdown("### Pipeline Results")
    
    # Create tabs
    tabs = st.tabs(["Phase 1", "Phase 2", "Phase 3", "Phase 4"])
    
    # Define results directory
    results_dir = os.path.join(data_dir, "results")
    
    # Phase 1 tab
    with tabs[0]:
        phase1_file = os.path.join(results_dir, "phase1_results.json")
        
        if os.path.exists(phase1_file):
            with open(phase1_file, 'r') as f:
                phase1_results = json.load(f)
            
            st.json(phase1_results)
        else:
            st.warning("Phase 1 results not found.")
    
    # Phase 2 tab
    with tabs[1]:
        phase2_file = os.path.join(results_dir, "phase2_results.json")
        
        if os.path.exists(phase2_file):
            with open(phase2_file, 'r') as f:
                phase2_results = json.load(f)
            
            st.json(phase2_results)
        else:
            st.warning("Phase 2 results not found.")
    
    # Phase 3 tab
    with tabs[2]:
        phase3_file = os.path.join(results_dir, "phase3_results.json")
        
        if os.path.exists(phase3_file):
            with open(phase3_file, 'r') as f:
                phase3_results = json.load(f)
            
            st.json(phase3_results)
        else:
            st.warning("Phase 3 results not found.")
    
    # Phase 4 tab
    with tabs[3]:
        phase4_file = os.path.join(results_dir, "phase4_results.json")
        
        if os.path.exists(phase4_file):
            with open(phase4_file, 'r') as f:
                phase4_results = json.load(f)
            
            st.json(phase4_results)
        else:
            st.warning("Phase 4 results not found.")

# Documentation page
elif page == "Documentation":
    st.markdown("## Exo-Detector Documentation")
    
    # Create tabs
    tabs = st.tabs(["Overview", "Pipeline", "Dashboard", "Vetting Guide"])
    
    # Overview tab
    with tabs[0]:
        st.markdown("""
        ### Exo-Detector: Exoplanet Transit Detection Pipeline
        
        Exo-Detector is a machine learning pipeline for detecting exoplanet transits in TESS light curves. The pipeline uses a combination of techniques:
        
        1. **Data Ingestion & Preprocessing**: Downloads and processes TESS light curves
        2. **GAN-based Transit Augmentation**: Generates synthetic transit signals to improve detection
        3. **Semi-Supervised Anomaly Detection**: Uses a 1D-Convolutional Autoencoder + One-Class SVM to identify unusual patterns
        4. **Candidate Scoring & Ranking**: Ranks potential transit signals by likelihood of being real planets
        
        This dashboard provides tools for reviewing and vetting candidates identified by the pipeline.
        """)
    
    # Pipeline tab
    with tabs[1]:
        st.markdown("""
        ### Pipeline Components
        
        #### Phase 1: Data Ingestion & Preprocessing
        
        - Downloads TESS light curves from MAST
        - Performs detrending and normalization
        - Extracts transit and non-transit windows
        
        #### Phase 2: GAN-based Transit Augmentation
        
        - Trains a Generative Adversarial Network (GAN) on known transit signals
        - Generates synthetic transit signals with varying parameters
        - Augments the training data for the anomaly detection system
        
        #### Phase 3: Semi-Supervised Anomaly Detection
        
        - Trains a 1D-Convolutional Autoencoder on non-transit data
        - Uses reconstruction error to identify unusual patterns
        - Trains a One-Class SVM to classify anomalies
        
        #### Phase 4: Candidate Scoring & Ranking
        
        - Scans light curves using a sliding window approach
        - Identifies potential transit signals using the anomaly detection system
        - Estimates orbital periods and transit parameters
        - Ranks candidates by likelihood of being real planets
        """)
    
    # Dashboard tab
    with tabs[2]:
        st.markdown("""
        ### Dashboard Components
        
        #### Overview
        
        - Displays pipeline status and candidate summary
        - Shows distribution of candidate parameters
        - Provides a scatter plot of candidates by period and depth
        - Lists top candidates by score
        
        #### Candidate Explorer
        
        - Allows detailed exploration of individual candidates
        - Displays light curves and phase-folded plots
        - Shows transit parameters and candidate scores
        
        #### Vetting Tool
        
        - Provides a form for vetting candidates
        - Allows users to classify candidates as planets or false positives
        - Tracks vetting progress and statistics
        
        #### Pipeline Status
        
        - Shows detailed status of each pipeline phase
        - Displays data directory status and file counts
        - Provides access to pipeline results
        """)
    
    # Vetting Guide tab
    with tabs[3]:
        st.markdown("""
        ### Candidate Vetting Guide
        
        #### Vetting Process
        
        1. Review the candidate's light curve and phase-folded plot
        2. Check the transit parameters (depth, duration, period)
        3. Look for multiple transit events at the expected period
        4. Consider the signal-to-noise ratio and overall score
        5. Classify the candidate using the vetting form
        
        #### Vetting Classifications
        
        - **Confirmed Planet**: Clear transit signal with multiple events and consistent parameters
        - **Likely Planet**: Good transit signal but with some uncertainty
        - **Uncertain**: Ambiguous signal that could be a planet or false positive
        - **Likely False Positive**: Probably not a planet, but some uncertainty
        - **Confirmed False Positive**: Clearly not a planet (e.g., stellar variability, instrumental artifact)
        
        #### Common False Positives
        
        - Stellar variability (e.g., spots, flares)
        - Eclipsing binaries
        - Instrumental artifacts
        - Cosmic ray hits
        - Data gaps or discontinuities
        """)

# Footer
st.markdown("""
<div class="footer">
    <p>Exo-Detector Dashboard v1.0.0 | &copy; 2025</p>
</div>
""", unsafe_allow_html=True)
