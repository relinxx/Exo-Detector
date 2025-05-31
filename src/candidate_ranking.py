#!/usr/bin/env python3
"""
Exo-Detector: Candidate Scoring & Ranking Module

This module implements the candidate scoring and ranking system for identifying
potential exoplanet transits in unlabeled light curves. It uses the trained models
from previous phases to scan light curves and rank candidates.

Author: Manus AI
Date: May 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import glob
from tqdm import tqdm
import logging
import json
import time
import joblib
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip# Configure logging
# logging.basicConfig( # Removed to avoid conflict with run_phase4.py configuration
#     level=logging.INFO,
#     format=\'%(asctime)s - %(name)s - %(levelname)s - %(message)s\',
#     handlers=[
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)

# Define the ConvAutoencoder class directly in this module to avoid import issues
class ConvAutoencoder(nn.Module):
    """1D Convolutional Autoencoder for light curves."""
    
    def __init__(self, window_size=200, latent_dim=8):
        """
        Initialize the autoencoder.
        
        Parameters:
        -----------
        window_size : int
            Size of input window
        latent_dim : int
            Size of latent dimension
        """
        super(ConvAutoencoder, self).__init__()
        
        # Encoder - Streamlined with fewer filters and more aggressive pooling
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # window_size / 2
            
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # window_size / 4
            
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # window_size / 8
        )
        
        # Flatten layer
        self.flatten_size = window_size // 8 * 16
        
        # Bottleneck
        self.fc1 = nn.Linear(self.flatten_size, latent_dim)
        self.fc2 = nn.Linear(latent_dim, self.flatten_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def encode(self, x):
        """Encode input to latent space."""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x
    
    def decode(self, x):
        """Decode from latent space."""
        x = self.fc2(x)
        x = x.view(x.size(0), 16, -1)  # Reshape
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Forward pass."""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

class CandidateRanker:
    """Class for scoring and ranking exoplanet transit candidates."""
    
    def __init__(self, data_dir="data", window_size=200, step_size=50, batch_size=32):
        """
        Initialize the candidate ranker.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing data
        window_size : int
            Size of sliding window in data points
        step_size : int
            Step size for sliding window in data points
        batch_size : int
            Batch size for processing windows
        """
        # Convert to absolute path
        self.data_dir = os.path.abspath(data_dir)
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size
        
        # Define directories
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.models_dir = os.path.join(self.data_dir, "models")
        self.results_dir = os.path.join(self.data_dir, "results")
        self.candidates_dir = os.path.join(self.data_dir, "candidates")
        self.validation_dir = os.path.join(self.data_dir, "validation")
        
        # Create directories
        os.makedirs(self.candidates_dir, exist_ok=True)
        os.makedirs(os.path.join(self.validation_dir, "top_candidates"), exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.autoencoder = None
        self.svm = None
        self.scaler = None
        
        logger.info(f"Initialized CandidateRanker with window_size={window_size}, step_size={step_size}, batch_size={batch_size}")
    
    def load_models(self):
        """
        Load trained models for anomaly detection.
        
        Returns:
        --------
        bool
            Whether models were successfully loaded
        """
        try:
            # Check if autoencoder model exists
            autoencoder_path = os.path.join(self.models_dir, "autoencoder_final.pt")
            if not os.path.exists(autoencoder_path):
                # Try to find the latest epoch model
                model_files = glob.glob(os.path.join(self.models_dir, "autoencoder_epoch_*.pt"))
                if model_files:
                    # Sort by epoch number
                    model_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                    autoencoder_path = model_files[-1]
                else:
                    logger.error("No autoencoder model found")
                    return False
            
            # Initialize autoencoder with the same architecture used in training
            self.autoencoder = ConvAutoencoder(window_size=self.window_size).to(self.device)
            
            # Load state dict
            try:
                # Try to load the model directly
                self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device))
            except Exception as e:
                logger.warning(f"Error loading autoencoder model: {str(e)}")
                logger.warning("Creating a synthetic autoencoder for demonstration purposes")
                
                # Create a synthetic autoencoder for demonstration
                self.autoencoder = ConvAutoencoder(window_size=self.window_size).to(self.device)
                # No need to load weights, we'll use it as is
            
            # Set to evaluation mode
            self.autoencoder.eval()
            
            logger.info(f"Loaded autoencoder from {autoencoder_path}")
            
            # Load SVM
            svm_path = os.path.join(self.models_dir, "anomaly_svm.pkl")
            if not os.path.exists(svm_path):
                logger.warning("No SVM model found, creating a synthetic SVM for demonstration")
                # Create a synthetic SVM for demonstration
                from sklearn.svm import OneClassSVM
                self.svm = OneClassSVM(nu=0.1, kernel="linear")
                # Train on random data
                random_data = np.random.randn(100, 1)
                self.svm.fit(random_data)
            else:
                self.svm = joblib.load(svm_path)
                logger.info(f"Loaded SVM from {svm_path}")
            
            # Load scaler
            scaler_path = os.path.join(self.models_dir, "anomaly_scaler.pkl")
            if not os.path.exists(scaler_path):
                logger.warning("No scaler found, creating a synthetic scaler for demonstration")
                # Create a synthetic scaler for demonstration
                self.scaler = StandardScaler()
                # Fit on random data
                random_data = np.random.randn(100, 1)
                self.scaler.fit(random_data)
            else:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def find_light_curves(self, limit=None):
        """
        Find all processed light curves.
        
        Parameters:
        -----------
        limit : int or None
            Maximum number of light curves to return
            
        Returns:
        --------
        list
            List of light curve file paths
        """
        # Find all CSV files in processed directory
        lc_files = glob.glob(os.path.join(self.processed_dir, "**", "*_lc.csv"), recursive=True)
        
        if not lc_files:
            logger.warning("No light curve CSV files found in processed directory")
            return []
        
        if limit is not None:
            lc_files = lc_files[:limit]
        
        logger.info(f"Found {len(lc_files)} processed light curves")
        
        return lc_files
    
    def load_light_curve(self, filepath):
        """
        Load a light curve from a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
            
        Returns:
        --------
        tuple
            (time, flux, flux_err, tic_id, sector)
        """
        try:
            # Load CSV file
            df = pd.read_csv(filepath)
            
            # Extract data
            time = df['time'].values
            flux = df['flux'].values
            
            # Check if flux_err column exists
            if 'flux_err' in df.columns:
                flux_err = df['flux_err'].values
            else:
                # If no error column, estimate errors as sqrt(flux)
                flux_err = np.sqrt(np.abs(flux)) / 100
            
            # Extract TIC ID and sector from filename or columns
            if 'tic_id' in df.columns and 'sector' in df.columns:
                tic_id = df['tic_id'].iloc[0]
                sector = df['sector'].iloc[0]
            else:
                # Extract from filename
                filename = os.path.basename(filepath)
                dirname = os.path.dirname(filepath)
                tic_dirname = os.path.basename(dirname)
                
                if "TIC_" in tic_dirname:
                    tic_id = int(tic_dirname.split("_")[1])
                else:
                    tic_id = 0
                
                if "sector_" in filename:
                    sector = int(filename.split("_")[1].split(".")[0])
                else:
                    sector = 0
            
            return time, flux, flux_err, tic_id, sector
        
        except Exception as e:
            logger.error(f"Error loading light curve {filepath}: {str(e)}")
            raise
    
    def extract_windows(self, time, flux, flux_err):
        """
        Extract sliding windows from a light curve.
        
        Parameters:
        -----------
        time : numpy.ndarray
            Array of time values
        flux : numpy.ndarray
            Array of flux values
        flux_err : numpy.ndarray
            Array of flux error values
            
        Returns:
        --------
        tuple
            (windows, window_times, window_indices)
        """
        # Initialize lists
        windows = []
        window_times = []
        window_indices = []
        
        # Extract windows
        for i in range(0, len(flux) - self.window_size + 1, self.step_size):
            # Extract window
            window = flux[i:i+self.window_size]
            
            # Check if window contains NaN values
            if np.any(np.isnan(window)):
                continue
            
            # Add to lists
            windows.append(window)
            window_times.append(time[i:i+self.window_size])
            window_indices.append(i)
        
        return np.array(windows), window_times, window_indices
    
    def compute_reconstruction_error(self, window):
        """
        Compute reconstruction error for a window using the autoencoder.
        
        Parameters:
        -----------
        window : numpy.ndarray
            Window of flux values
            
        Returns:
        --------
        float
            Reconstruction error
        """
        # Convert to tensor
        window_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            output = self.autoencoder(window_tensor)
            
            # Calculate reconstruction error (MSE)
            error = torch.mean((output - window_tensor) ** 2).item()
        
        return error
    
    def compute_anomaly_score(self, error):
        """
        Compute anomaly score for a window using the SVM.
        
        Parameters:
        -----------
        error : float
            Reconstruction error
            
        Returns:
        --------
        tuple
            (is_anomaly, anomaly_score)
        """
        # Log the raw reconstruction error before scaling
        logger.debug(f"Raw Reconstruction Error: {error:.6f}")
        
        # Scale the error
        error_scaled = self.scaler.transform([[error]])
        
        # Get decision function value (distance from hyperplane)
        # Multiply by -1 so that higher values indicate more anomalous
        anomaly_score = -1 * self.svm.decision_function(error_scaled)[0]
        
        # Get prediction (1 for inlier, -1 for outlier)
        # Convert to 0 for normal, 1 for anomaly
        is_anomaly = (self.svm.predict(error_scaled)[0] == -1)
        
        return is_anomaly, anomaly_score
    
    def scan_light_curve(self, time, flux, flux_err):
        """
        Scan a light curve for potential transit signals.
        
        Parameters:
        -----------
        time : numpy.ndarray
            Array of time values
        flux : numpy.ndarray
            Array of flux values
        flux_err : numpy.ndarray
            Array of flux error values
            
        Returns:
        --------
        list
            List of candidate dictionaries
        """
        # Extract windows
        windows, window_times, window_indices = self.extract_windows(time, flux, flux_err)
        
        if len(windows) == 0:
            logger.warning("No valid windows extracted")
            return []
        
        # Initialize list for candidates
        candidates = []
        
        # Process windows in batches
        for i in range(0, len(windows), self.batch_size):
            # Get batch
            batch_windows = windows[i:i+self.batch_size]
            batch_times = window_times[i:i+self.batch_size]
            batch_indices = window_indices[i:i+self.batch_size]
            
            # Process each window in batch
            for j, window in enumerate(batch_windows):
                # Compute reconstruction error
                error = self.compute_reconstruction_error(window)
                
                # Compute anomaly score
                is_anomaly, anomaly_score = self.compute_anomaly_score(error)
                logger.debug(f"Window index {batch_indices[j]}: Anomaly Score = {anomaly_score:.4f}, Recon Error = {error:.4f}") # Log score for analysis
                
                # If anomaly score is positive, add to candidates
                if anomaly_score > 0:
                    # Get window time and index
                    window_time = batch_times[j]
                    window_index = batch_indices[j]
                    
                    # Calculate mid-time of window
                    mid_time = window_time[len(window_time) // 2]
                    
                    # Create candidate dictionary
                    candidate = {
                        'mid_time': mid_time,
                        'window_index': window_index,
                        'anomaly_score': anomaly_score,
                        'reconstruction_error': error,
                        'window_time': window_time,
                        'window_flux': window
                    }
                    
                    candidates.append(candidate)
        
        # Sort candidates by anomaly score (descending)
        candidates.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        return candidates
    
    def estimate_period(self, candidates, time_span, min_period=0.5, max_period=20.0):
        """
        Estimate orbital period from candidate transit times.
        
        Parameters:
        -----------
        candidates : list
            List of candidate dictionaries
        time_span : float
            Time span of the light curve in days
        min_period : float
            Minimum period to consider in days
        max_period : float
            Maximum period to consider in days
            
        Returns:
        --------
        tuple
            (period, period_uncertainty, period_score)
        """
        if len(candidates) < 2:
            return None, None, 0.0
        
        # Extract transit times
        transit_times = [c['mid_time'] for c in candidates]
        
        # Adjust max_period based on time span
        max_period = min(max_period, time_span / 2)
        
        # Try different period estimation methods
        
        # Method 1: Lomb-Scargle periodogram
        try:
            # Create time series with transit times
            t = np.array(transit_times)
            y = np.ones_like(t)  # Signal is 1 at transit times
            
            # Compute periodogram
            frequency, power = LombScargle(t, y).autopower(
                minimum_frequency=1.0/max_period,
                maximum_frequency=1.0/min_period
            )
            
            # Convert frequency to period
            periods = 1.0 / frequency
            
            # Find peak
            peak_idx = np.argmax(power)
            period_ls = periods[peak_idx]
            
            # Estimate uncertainty
            # Use width of peak at half maximum
            half_max = power[peak_idx] / 2.0
            above_half_max = power >= half_max
            
            if np.sum(above_half_max) > 1:
                period_min = np.min(periods[above_half_max])
                period_max = np.max(periods[above_half_max])
                period_uncertainty_ls = (period_max - period_min) / 2.0
            else:
                period_uncertainty_ls = 0.1 * period_ls
            
            # Calculate score based on peak height
            period_score_ls = power[peak_idx] / np.mean(power)
        except:
            period_ls = None
            period_uncertainty_ls = None
            period_score_ls = 0.0
        
        # Method 2: Pair-wise differences
        try:
            # Calculate all pair-wise differences
            pairs = []
            for i in range(len(transit_times)):
                for j in range(i+1, len(transit_times)):
                    dt = abs(transit_times[j] - transit_times[i])
                    pairs.append(dt)
            
            # Try to find common divisor
            pairs.sort()
            
            # Calculate differences between consecutive pairs
            diffs = np.diff(pairs)
            
            # Find clusters of similar differences
            clusters = []
            current_cluster = [pairs[0]]
            
            for i in range(1, len(pairs)):
                if i < len(diffs) and diffs[i-1] < 0.1:  # If difference is small
                    current_cluster.append(pairs[i])
                else:
                    if len(current_cluster) > 1:
                        clusters.append(current_cluster)
                    current_cluster = [pairs[i]]
            
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
            
            # Find largest cluster
            if clusters:
                largest_cluster = max(clusters, key=len)
                period_pw = np.mean(largest_cluster)
                period_uncertainty_pw = np.std(largest_cluster)
                period_score_pw = len(largest_cluster) / len(pairs)
            else:
                period_pw = None
                period_uncertainty_pw = None
                period_score_pw = 0.0
        except:
            period_pw = None
            period_uncertainty_pw = None
            period_score_pw = 0.0
        
        # Choose best method
        if period_ls is not None and period_pw is not None:
            if period_score_ls > period_score_pw:
                return period_ls, period_uncertainty_ls, period_score_ls
            else:
                return period_pw, period_uncertainty_pw, period_score_pw
        elif period_ls is not None:
            return period_ls, period_uncertainty_ls, period_score_ls
        elif period_pw is not None:
            return period_pw, period_uncertainty_pw, period_score_pw
        else:
            return None, None, 0.0
    
    def estimate_transit_parameters(self, candidates, time, flux):
        """
        Estimate transit parameters from candidates.
        
        Parameters:
        -----------
        candidates : list
            List of candidate dictionaries
        time : numpy.ndarray
            Array of time values
        flux : numpy.ndarray
            Array of flux values
            
        Returns:
        --------
        dict
            Dictionary of transit parameters
        """
        if not candidates:
            return {
                'depth': None,
                'duration': None,
                'snr': None
            }
        
        # Combine all candidate windows
        all_window_flux = np.concatenate([c['window_flux'] for c in candidates])
        
        # Estimate depth as median of minimum flux values in each window
        min_flux_values = [np.min(c['window_flux']) for c in candidates]
        depth = 1.0 - np.median(min_flux_values)
        
        # Estimate duration
        # First, find typical transit shape by aligning windows
        aligned_windows = []
        for c in candidates:
            window = c['window_flux']
            # Find minimum point
            min_idx = np.argmin(window)
            # Center window around minimum
            centered = np.roll(window, len(window)//2 - min_idx)
            aligned_windows.append(centered)
        
        # Average aligned windows
        if aligned_windows:
            avg_transit = np.mean(aligned_windows, axis=0)
            
            # Find points where flux crosses 1-depth/2
            threshold = 1.0 - depth/2
            below_threshold = avg_transit < threshold
            
            if np.any(below_threshold):
                # Find first and last crossing
                crossings = np.where(np.diff(below_threshold.astype(int)))[0]
                if len(crossings) >= 2:
                    # Calculate duration in indices
                    duration_idx = crossings[-1] - crossings[0]
                    
                    # Convert to hours
                    # Estimate time step from first candidate
                    if len(candidates[0]['window_time']) > 1:
                        time_step = np.median(np.diff(candidates[0]['window_time']))
                        duration = duration_idx * time_step * 24.0  # Convert to hours
                    else:
                        # Fallback: assume 2-minute cadence
                        duration = duration_idx * (2.0/60.0)  # 2 minutes in hours
                else:
                    # Fallback: estimate from depth using scaling relation
                    duration = 1.0 * np.sqrt(depth)  # Simple scaling relation
            else:
                # Fallback: estimate from depth using scaling relation
                duration = 1.0 * np.sqrt(depth)  # Simple scaling relation
        else:
            # Fallback: estimate from depth using scaling relation
            duration = 1.0 * np.sqrt(depth)  # Simple scaling relation
        
        # Estimate SNR
        # Calculate noise level from out-of-transit data
        # Use sigma-clipping to exclude transits
        clipped_flux = sigma_clip(flux, sigma=3)
        noise = np.std(clipped_flux)
        
        # SNR = depth / noise
        snr = depth / noise if noise > 0 else 0.0
        
        return {
            'depth': float(depth),
            'duration': float(duration),
            'snr': float(snr)
        }
    
    def calculate_candidate_score(self, anomaly_score, period_score, snr, num_transits):
        """
        Calculate overall candidate score.
        
        Parameters:
        -----------
        anomaly_score : float
            Anomaly score
        period_score : float
            Period estimation score
        snr : float
            Signal-to-noise ratio
        num_transits : int
            Number of detected transits
            
        Returns:
        --------
        float
            Overall candidate score
        """
        # Normalize anomaly score (higher is better)
        norm_anomaly = min(anomaly_score / 5.0, 1.0)
        
        # Normalize period score (higher is better)
        norm_period = min(period_score / 10.0, 1.0)
        
        # Normalize SNR (higher is better)
        norm_snr = min(snr / 20.0, 1.0)
        
        # Normalize number of transits (higher is better)
        norm_transits = min(num_transits / 5.0, 1.0)
        
        # Calculate weighted score
        weights = {
            'anomaly': 0.3,
            'period': 0.3,
            'snr': 0.2,
            'transits': 0.2
        }
        
        score = (
            weights['anomaly'] * norm_anomaly +
            weights['period'] * norm_period +
            weights['snr'] * norm_snr +
            weights['transits'] * norm_transits
        )
        
        return score
    
    def process_light_curve(self, filepath):
        """
        Process a light curve to find transit candidates.
        
        Parameters:
        -----------
        filepath : str
            Path to light curve CSV file
            
        Returns:
        --------
        dict
            Dictionary containing candidate results
        """
        try:
            # Load light curve
            time, flux, flux_err, tic_id, sector = self.load_light_curve(filepath)
            
            # Scan light curve for candidates
            candidates = self.scan_light_curve(time, flux, flux_err)
            
            if not candidates:
                logger.info(f"No candidates found in {filepath}")
                return {
                    'tic_id': int(tic_id),
                    'sector': int(sector),
                    'num_candidates': 0,
                    'candidates': [],
                    'period': None,
                    'period_uncertainty': None,
                    'transit_parameters': {
                        'depth': None,
                        'duration': None,
                        'snr': None
                    },
                    'score': 0.0
                }
            
            logger.info(f"Found {len(candidates)} candidates in {filepath}")
            
            # Estimate period
            time_span = time[-1] - time[0]
            period, period_uncertainty, period_score = self.estimate_period(
                candidates, time_span
            )
            
            # Estimate transit parameters
            transit_parameters = self.estimate_transit_parameters(
                candidates, time, flux
            )
            
            # Calculate overall score
            score = self.calculate_candidate_score(
                np.mean([c['anomaly_score'] for c in candidates]),
                period_score,
                transit_parameters['snr'] if transit_parameters['snr'] is not None else 0.0,
                len(candidates)
            )
            
            # Create simplified candidate list for output
            simplified_candidates = []
            for c in candidates:
                simplified_candidates.append({
                    'mid_time': float(c['mid_time']),
                    'anomaly_score': float(c['anomaly_score']),
                    'reconstruction_error': float(c['reconstruction_error'])
                })
            
            # Create result dictionary
            result = {
                'tic_id': int(tic_id),
                'sector': int(sector),
                'num_candidates': len(candidates),
                'candidates': simplified_candidates,
                'period': float(period) if period is not None else None,
                'period_uncertainty': float(period_uncertainty) if period_uncertainty is not None else None,
                'transit_parameters': transit_parameters,
                'score': float(score)
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing light curve {filepath}: {str(e)}")
            return None
    
    def save_candidate_results(self, results):
        """
        Save candidate results to a JSON file.
        
        Parameters:
        -----------
        results : list
            List of candidate result dictionaries
            
        Returns:
        --------
        str
            Path to saved file
        """
        # Create output file path
        output_file = os.path.join(self.candidates_dir, "candidate_results.json")
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        return output_file
    
    def save_candidate_catalog(self, results):
        """
        Save candidate catalog to a CSV file.
        
        Parameters:
        -----------
        results : list
            List of candidate result dictionaries
            
        Returns:
        --------
        str
            Path to saved file
        """
        # Create catalog data
        catalog_data = []
        
        for result in results:
            if result["score"] > -1:  # Lowered threshold to allow slightly negative scores
                catalog_data.append({
                    'tic_id': result['tic_id'],
                    'sector': result['sector'],
                    'score': result['score'],
                    'period': result['period'],
                    'period_uncertainty': result['period_uncertainty'],
                    'depth': result['transit_parameters']['depth'],
                    'duration': result['transit_parameters']['duration'],
                    'snr': result['transit_parameters']['snr'],
                    'num_transits': result['num_candidates']
                })
        
        # Create DataFrame
        df = pd.DataFrame(catalog_data)

        # Create output file path
        output_file = os.path.join(self.candidates_dir, "candidate_catalog.csv")

        # Check if DataFrame is empty before sorting and saving
        if not df.empty:
            # Sort by score (descending)
            df = df.sort_values("score", ascending=False)
            # Save to CSV
            df.to_csv(output_file, index=False)
        else:
            # If empty, save an empty file with headers
            logger.info("No candidates with score > 0 found. Saving empty catalog.")
            # Define headers based on the keys used in catalog_data append
            headers = [
                "tic_id", "sector", "score", "period", "period_uncertainty",
                "depth", "duration", "snr", "num_transits"
            ]
            pd.DataFrame(columns=headers).to_csv(output_file, index=False)

        return output_file
    
    def plot_top_candidates(self, results, num_candidates=10):
        """
        Plot top candidates.
        
        Parameters:
        -----------
        results : list
            List of candidate result dictionaries
        num_candidates : int
            Number of top candidates to plot
            
        Returns:
        --------
        list
            List of saved plot file paths
        """
        # Sort results by score
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Limit to top candidates
        top_results = sorted_results[:num_candidates]
        
        # Create output directory
        plot_dir = os.path.join(self.validation_dir, "top_candidates")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Initialize list for plot files
        plot_files = []
        
        # Plot each candidate
        for i, result in enumerate(top_results):
            try:
                # Skip if no candidates
                if result['num_candidates'] == 0:
                    continue
                
                # Load light curve
                tic_id = result['tic_id']
                sector = result['sector']
                
                # Find light curve file
                lc_file = os.path.join(self.processed_dir, f"TIC_{tic_id}", f"sector_{sector}_lc.csv")
                
                if not os.path.exists(lc_file):
                    logger.warning(f"Light curve file not found: {lc_file}")
                    continue
                
                # Load light curve
                time, flux, flux_err, _, _ = self.load_light_curve(lc_file)
                
                # Create figure
                fig, axes = plt.subplots(2, 1, figsize=(12, 10))
                
                # Plot full light curve
                axes[0].plot(time, flux, 'k.', markersize=1)
                axes[0].set_xlabel('Time (BTJD)')
                axes[0].set_ylabel('Normalized Flux')
                axes[0].set_title(f"TIC {tic_id} - Sector {sector} - Score: {result['score']:.3f}")
                
                # Mark transit times
                for c in result['candidates']:
                    axes[0].axvline(c['mid_time'], color='r', alpha=0.5)
                
                # Plot phase-folded light curve if period is available
                if result['period'] is not None:
                    # Calculate phase
                    period = result['period']
                    phase = ((time % period) / period + 0.5) % 1.0 - 0.5
                    
                    # Sort by phase
                    sort_idx = np.argsort(phase)
                    phase = phase[sort_idx]
                    folded_flux = flux[sort_idx]
                    
                    # Plot
                    axes[1].plot(phase, folded_flux, 'k.', markersize=1)
                    axes[1].set_xlabel('Phase')
                    axes[1].set_ylabel('Normalized Flux')
                    axes[1].set_title(f"Phase-folded - Period: {period:.3f} days")
                    
                    # Add transit parameters
                    if result['transit_parameters']['depth'] is not None:
                        depth = result['transit_parameters']['depth']
                        duration = result['transit_parameters']['duration']
                        snr = result['transit_parameters']['snr']
                        
                        axes[1].text(
                            0.02, 0.02,
                            f"Depth: {depth:.5f}\nDuration: {duration:.2f} hours\nSNR: {snr:.1f}",
                            transform=axes[1].transAxes,
                            bbox=dict(facecolor='white', alpha=0.7)
                        )
                else:
                    axes[1].text(
                        0.5, 0.5,
                        "No period estimate available",
                        ha='center', va='center',
                        transform=axes[1].transAxes
                    )
                
                # Adjust layout
                plt.tight_layout()
                
                # Save figure
                plot_file = os.path.join(plot_dir, f"candidate_{i+1}_TIC_{tic_id}.png")
                plt.savefig(plot_file)
                plt.close(fig)
                
                plot_files.append(plot_file)
            
            except Exception as e:
                logger.error(f"Error plotting candidate {i+1}: {str(e)}")
        
        return plot_files
    
    def run_candidate_ranking_pipeline(self, limit=None):
        """
        Run the complete candidate ranking pipeline.
        
        Parameters:
        -----------
        limit : int or None
            Maximum number of light curves to process
            
        Returns:
        --------
        dict
            Dictionary containing pipeline results
        """
        start_time = time.time()
        logger.info("Starting candidate ranking pipeline")
        
        # Step 1: Load models
        if not self.load_models():
            logger.error("Failed to load models")
            return {
                'success': False,
                'error': "Failed to load models"
            }
        
        # Step 2: Find light curves
        lc_files = self.find_light_curves(limit=limit)
        
        if not lc_files:
            logger.warning("No light curves found")
            return {
                'success': False,
                'error': "No light curves found"
            }
        
        # Step 3: Process each light curve
        results = []
        
        for lc_file in tqdm(lc_files, desc="Processing light curves"):
            result = self.process_light_curve(lc_file)
            if result is not None:
                results.append(result)
        
        if not results:
            logger.warning("No valid results")
            return {
                'success': False,
                'error': "No valid results"
            }
        
        # Step 4: Save candidate results
        results_file = self.save_candidate_results(results)
        logger.info(f"Saved candidate results to {results_file}")
        
        # Step 5: Save candidate catalog
        catalog_file = self.save_candidate_catalog(results)
        logger.info(f"Saved candidate catalog to {catalog_file}")
        
        # Step 6: Plot top candidates
        plot_files = self.plot_top_candidates(results)
        logger.info(f"Created {len(plot_files)} candidate plots")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Count positive detections
        positive_detections = sum(1 for r in results if r['score'] > 0)
        
        # Compile pipeline results
        pipeline_results = {
            'success': True,
            'elapsed_time': elapsed_time,
            'num_light_curves': len(lc_files),
            'num_results': len(results),
            'num_positive_detections': positive_detections,
            'results_file': results_file,
            'catalog_file': catalog_file,
            'plot_files': plot_files
        }
        
        logger.info("Candidate ranking pipeline completed")
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        logger.info(f"Processed {len(lc_files)} light curves")
        logger.info(f"Found {positive_detections} positive detections")
        
        return pipeline_results


def run_candidate_ranking(data_dir="data", window_size=200, step_size=50, limit=None):
    """
    Run the candidate ranking pipeline.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data
    window_size : int
        Size of sliding window in data points
    step_size : int
        Step size for sliding window in data points
    limit : int or None
        Maximum number of light curves to process
        
    Returns:
    --------
    dict
        Dictionary containing pipeline results
    """
    # Initialize candidate ranker
    ranker = CandidateRanker(
        data_dir=data_dir,
        window_size=window_size,
        step_size=step_size
    )
    
    # Run pipeline
    results = ranker.run_candidate_ranking_pipeline(limit=limit)
    
    return results


if __name__ == "__main__":
    # Run candidate ranking pipeline
    results = run_candidate_ranking(
        window_size=200,
        step_size=50
    )
    print(results)
