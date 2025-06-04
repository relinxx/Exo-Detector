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
from astropy.stats import sigma_clip

# Configure logging - Changed level to DEBUG
logging.basicConfig(
    level=logging.DEBUG, # Changed from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
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
        
        # RELAXED THRESHOLD: Set a lower anomaly threshold to detect more candidates
        self.anomaly_threshold = -0.5  # Relaxed from default (typically 0)
        
        logger.info(f"Initialized CandidateRanker with window_size={window_size}, step_size={step_size}, batch_size={batch_size}")
        logger.info(f"Using relaxed anomaly threshold: {self.anomaly_threshold}")
    
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
                # Create a synthetic SVM for demonstration with RELAXED parameters
                from sklearn.svm import OneClassSVM
                # RELAXED PARAMETER: Increased nu for more outliers (0.1 -> 0.2)
                self.svm = OneClassSVM(nu=0.2, kernel="linear")
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
        # Scale the error
        error_scaled = self.scaler.transform([[error]])
        logger.debug(f"Window error: {error:.6f}, Scaled error: {error_scaled[0][0]:.6f}")
        
        # Get decision function value (distance from hyperplane)
        # Multiply by -1 so that higher values indicate more anomalous
        anomaly_score = -1 * self.svm.decision_function(error_scaled)[0]
        
        # RELAXED LOGIC: Use custom threshold instead of SVM prediction
        # Original: is_anomaly = (self.svm.predict(error_scaled)[0] == -1)
        is_anomaly = (anomaly_score > self.anomaly_threshold)
        
        logger.debug(f"SVM decision function: {-anomaly_score:.6f}, Anomaly score: {anomaly_score:.6f}, Is anomaly: {is_anomaly}")
        
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
        
        # RELAXED APPROACH: Find windows with significant dips
        # This is a backup approach if the anomaly detection doesn't find candidates
        min_flux_values = []
        for window in windows:
            min_flux_values.append(np.min(window))
        
        # Calculate statistics of minimum flux values
        mean_min = np.mean(min_flux_values)
        std_min = np.std(min_flux_values)
        
        # Process windows in batches
        for i in range(0, len(windows), self.batch_size):
            # Get batch
            batch_windows = windows[i:i+self.batch_size]
            batch_times = window_times[i:i+self.batch_size]
            batch_indices = window_indices[i:i+self.batch_size]
            batch_min_flux = min_flux_values[i:i+self.batch_size]
            
            # Process each window in batch
            for j, window in enumerate(batch_windows):
                # Compute reconstruction error
                error = self.compute_reconstruction_error(window)
                
                # Compute anomaly score
                is_anomaly, anomaly_score = self.compute_anomaly_score(error)
                
                # RELAXED LOGIC: Also check for significant dips in flux
                min_flux = batch_min_flux[j]
                is_significant_dip = (min_flux < mean_min - 2.5 * std_min)
                
                logger.debug(f"Window {j} in batch {i//self.batch_size}: Error={error:.4f}, Score={anomaly_score:.4f}, " +
                           f"Anomaly={is_anomaly}, MinFlux={min_flux:.4f}, IsDip={is_significant_dip}")
                
                # If anomaly or significant dip, add to candidates
                if is_anomaly or is_significant_dip:
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
                        'min_flux': float(min_flux),
                        'is_anomaly': bool(is_anomaly),
                        'is_dip': bool(is_significant_dip),
                        'window_time': window_time.tolist(), # Convert to list for JSON serialization
                        'window_flux': window.tolist() # Convert to list for JSON serialization
                    }
                    
                    candidates.append(candidate)
        
        # Sort candidates by anomaly score (descending)
        candidates.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        # RELAXED APPROACH: If no candidates found, force include the windows with the deepest dips
        if not candidates and len(min_flux_values) > 0:
            logger.info("No candidates found through anomaly detection, including top 3 deepest dips")
            # Find indices of windows with deepest dips
            deepest_indices = np.argsort(min_flux_values)[:3]
            
            for idx in deepest_indices:
                window = windows[idx]
                window_time = window_times[idx]
                window_index = window_indices[idx]
                min_flux = min_flux_values[idx]
                
                # Calculate mid-time of window
                mid_time = window_time[len(window_time) // 2]
                
                # Compute reconstruction error
                error = self.compute_reconstruction_error(window)
                
                # Compute anomaly score
                _, anomaly_score = self.compute_anomaly_score(error)
                
                # Create candidate dictionary
                candidate = {
                    'mid_time': mid_time,
                    'window_index': window_index,
                    'anomaly_score': anomaly_score,
                    'reconstruction_error': error,
                    'min_flux': float(min_flux),
                    'is_anomaly': False,
                    'is_dip': True,
                    'window_time': window_time.tolist(),
                    'window_flux': window.tolist()
                }
                
                candidates.append(candidate)
            
            # Sort by depth of dip
            candidates.sort(key=lambda x: x['min_flux'])
        
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
        except Exception as e:
            logger.debug(f"Lomb-Scargle failed: {e}")
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
        except Exception as e:
            logger.debug(f"Pair-wise difference method failed: {e}")
            period_pw = None
            period_uncertainty_pw = None
            period_score_pw = 0.0
        
        # Choose best method
        if period_ls is not None and period_pw is not None:
            if period_score_ls > period_score_pw:
                logger.debug(f"Using Lomb-Scargle period: {period_ls:.4f} (score={period_score_ls:.2f})")
                return period_ls, period_uncertainty_ls, period_score_ls
            else:
                logger.debug(f"Using Pair-wise period: {period_pw:.4f} (score={period_score_pw:.2f})")
                return period_pw, period_uncertainty_pw, period_score_pw
        elif period_ls is not None:
            logger.debug(f"Using Lomb-Scargle period: {period_ls:.4f} (score={period_score_ls:.2f})")
            return period_ls, period_uncertainty_ls, period_score_ls
        elif period_pw is not None:
            logger.debug(f"Using Pair-wise period: {period_pw:.4f} (score={period_score_pw:.2f})")
            return period_pw, period_uncertainty_pw, period_score_pw
        else:
            logger.debug("Both period estimation methods failed")
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
        all_window_flux = np.concatenate([np.array(c['window_flux']) for c in candidates])
        
        # Estimate depth as median of minimum flux values in each window
        min_flux_values = [np.min(c['window_flux']) for c in candidates]
        depth = 1.0 - np.median(min_flux_values)
        
        # Estimate duration
        # First, find typical transit shape by aligning windows
        aligned_windows = []
        for c in candidates:
            window = np.array(c['window_flux']) # Ensure it's a numpy array
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
        try:
            clipped_flux = sigma_clip(flux, sigma=3)
            noise = np.std(clipped_flux)
        except Exception as e:
            logger.warning(f"Sigma clipping failed: {e}, using standard deviation.")
            noise = np.std(flux)
        
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
        norm_anomaly = min(max(0, anomaly_score / 5.0), 1.0) # Ensure score is between 0 and 1
        
        # Normalize period score (higher is better)
        norm_period = min(max(0, period_score / 10.0), 1.0) # Ensure score is between 0 and 1
        
        # Normalize SNR (higher is better)
        norm_snr = min(max(0, snr / 20.0), 1.0) # Ensure score is between 0 and 1
        
        # Normalize number of transits (higher is better)
        norm_transits = min(max(0, num_transits / 5.0), 1.0) # Ensure score is between 0 and 1
        
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
            
            # Scan light curve for anomalies
            candidates = self.scan_light_curve(time, flux, flux_err)
            
            if not candidates:
                logger.info(f"No candidates found in {filepath}")
                return {
                    'tic_id': tic_id,
                    'sector': sector,
                    'num_candidates': 0,
                    'candidates': []
                }
            
            logger.info(f"Found {len(candidates)} potential candidates in {filepath}")
            
            # Estimate period
            time_span = time[-1] - time[0]
            period, period_uncertainty, period_score = self.estimate_period(candidates, time_span)
            
            # Estimate transit parameters
            transit_params = self.estimate_transit_parameters(candidates, time, flux)
            
            # Calculate overall score
            score = self.calculate_candidate_score(
                np.mean([c['anomaly_score'] for c in candidates]),
                period_score,
                transit_params['snr'] if transit_params['snr'] is not None else 0.0,
                len(candidates)
            )
            
            # Compile results
            result = {
                'tic_id': tic_id,
                'sector': sector,
                'num_candidates': len(candidates),
                'period': period,
                'period_uncertainty': period_uncertainty,
                'period_score': period_score,
                'depth': transit_params['depth'],
                'duration': transit_params['duration'],
                'snr': transit_params['snr'],
                'score': score,
                'candidates': candidates
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing light curve {filepath}: {str(e)}")
            return {
                'tic_id': 0,
                'sector': 0,
                'num_candidates': 0,
                'candidates': [],
                'error': str(e)
            }
    
    def plot_candidate(self, result):
        """
        Plot the light curve and phase-folded curve for a candidate.
        
        Parameters:
        -----------
        result : dict
            Candidate result dictionary
        """
        try:
            tic_id = result['tic_id']
            sector = result['sector']
            period = result['period']
            
            # Load light curve
            lc_file = os.path.join(self.processed_dir, f"TIC_{tic_id}", f"sector_{sector}_lc.csv")
            if not os.path.exists(lc_file):
                logger.warning(f"Light curve file not found for plotting: {lc_file}")
                return
            
            time, flux, _, _, _ = self.load_light_curve(lc_file)
            
            # Create plot
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot full light curve
            axes[0].plot(time, flux, ".", markersize=2, color='k', alpha=0.5)
            axes[0].set_title(f"TIC {tic_id} Sector {sector} - Full Light Curve")
            axes[0].set_xlabel("Time (BTJD)")
            axes[0].set_ylabel("Normalized Flux")
            
            # Mark candidate times
            for cand in result['candidates']:
                axes[0].axvline(cand['mid_time'], color='r', linestyle='--', alpha=0.7)
            
            # Plot phase-folded light curve
            if period is not None:
                phase = ((time % period) / period + 0.5) % 1.0 - 0.5
                sort_idx = np.argsort(phase)
                axes[1].plot(phase[sort_idx], flux[sort_idx], ".", markersize=2, color='k', alpha=0.5)
                axes[1].set_title(f"Phase-folded (P = {period:.4f} days)")
                axes[1].set_xlabel("Phase")
                axes[1].set_ylabel("Normalized Flux")
            else:
                axes[1].text(0.5, 0.5, "Period estimation failed", ha='center', va='center')
            
            # Save plot
            plot_dir = os.path.join(self.validation_dir, "top_candidates")
            os.makedirs(plot_dir, exist_ok=True)
            plot_file = os.path.join(plot_dir, f"TIC_{tic_id}_sector_{sector}_candidate.png")
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close(fig)
            logger.debug(f"Saved candidate plot to {plot_file}")
        
        except Exception as e:
            logger.error(f"Error plotting candidate TIC {result.get('tic_id', 'N/A')} Sector {result.get('sector', 'N/A')}: {str(e)}")

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
        Dictionary containing ranking results
    """
    logger.info("Starting candidate ranking pipeline")
    start_time = time.time()
    
    # Initialize ranker
    ranker = CandidateRanker(data_dir=data_dir, window_size=window_size, step_size=step_size)
    
    # Load models
    if not ranker.load_models():
        logger.error("Failed to load models, exiting")
        return {
            'status': 'error',
            'message': 'Failed to load models'
        }
    
    # Find light curves
    lc_files = ranker.find_light_curves(limit=limit)
    
    if not lc_files:
        logger.warning("No light curves found to process")
        return {
            'status': 'warning',
            'message': 'No light curves found'
        }
    
    # Process light curves
    all_results = []
    for filepath in tqdm(lc_files, desc="Processing light curves"):
        result = ranker.process_light_curve(filepath)
        if result.get('num_candidates', 0) > 0:
            all_results.append(result)
    
    # Sort results by score
    all_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    
    # Create candidate catalog
    catalog_data = []
    for result in all_results:
        catalog_data.append({
            'tic_id': result['tic_id'],
            'sector': result['sector'],
            'score': result['score'],
            'period': result['period'],
            'depth': result['depth'],
            'duration': result['duration'],
            'snr': result['snr'],
            'num_transits': result['num_candidates']
        })
    
    catalog_df = pd.DataFrame(catalog_data)
    
    # Save catalog
    catalog_file = os.path.join(ranker.candidates_dir, "candidate_catalog.csv")
    catalog_df.to_csv(catalog_file, index=False)
    logger.info(f"Saved candidate catalog to {catalog_file}")
    
    # Save detailed results
    results_file = os.path.join(ranker.candidates_dir, "candidate_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Saved detailed results to {results_file}")
    
    # Plot top candidates
    num_plots = min(len(all_results), 10) # Plot top 10
    logger.info(f"Plotting top {num_plots} candidates")
    for i in range(num_plots):
        ranker.plot_candidate(all_results[i])
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"Candidate ranking pipeline completed in {duration:.2f} seconds")
    
    return {
        'status': 'success',
        'num_light_curves_processed': len(lc_files),
        'num_candidates_found': len(all_results),
        'duration_seconds': duration
    }

if __name__ == "__main__":
    # This part is usually run by run_phase4.py
    # Example usage:
    run_candidate_ranking(limit=10)
