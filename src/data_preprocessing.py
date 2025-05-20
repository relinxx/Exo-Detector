#!/usr/bin/env python3
"""
Exo-Detector: Data Preprocessing Module

This module handles the preprocessing of TESS light curves, including detrending,
normalization, and extraction of transit windows.

Author: Manus AI
Date: May 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import glob
from tqdm import tqdm
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TESSDataPreprocessing:
    """Class for preprocessing TESS light curves."""
    
    def __init__(self, data_dir="data", window_size_hours=5):
        """
        Initialize the data preprocessing class.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing data
        window_size_hours : float
            Size of transit windows in hours
        """
        # Convert to absolute path
        self.data_dir = os.path.abspath(data_dir)
        self.window_size_hours = window_size_hours
        
        # Define directories
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.catalog_dir = os.path.join(self.data_dir, "catalogs")
        self.transit_windows_dir = os.path.join(self.data_dir, "transit_windows")
        self.non_transit_windows_dir = os.path.join(self.data_dir, "non_transit_windows")
        
        # Create directories
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.transit_windows_dir, exist_ok=True)
        os.makedirs(self.non_transit_windows_dir, exist_ok=True)
        
        logger.info(f"Initialized TESSDataPreprocessing with window_size_hours={window_size_hours}")
        logger.info(f"Using data directory: {self.data_dir}")
    
    def load_csv_light_curve(self, filepath):
        """
        Load a CSV light curve file.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
            
        Returns:
        --------
        tuple
            (time_array, flux_array, flux_err_array)
        """
        try:
            # Read CSV file, skipping comment lines
            df = pd.read_csv(filepath, comment='#')
            
            # Extract data
            time = df['time'].values
            flux = df['flux'].values
            
            # Check if flux_err column exists
            if 'flux_err' in df.columns:
                flux_err = df['flux_err'].values
            else:
                # If no error column, estimate errors as sqrt(flux)
                flux_err = np.sqrt(np.abs(flux)) / 100
            
            return time, flux, flux_err
        
        except Exception as e:
            logger.error(f"Error reading CSV file {filepath}: {str(e)}")
            raise
    
    def detrend_light_curve(self, time, flux, flux_err, window_length=101):
        """
        Detrend a light curve using a Savitzky-Golay filter.
        
        Parameters:
        -----------
        time : numpy.ndarray
            Array of time values
        flux : numpy.ndarray
            Array of flux values
        flux_err : numpy.ndarray
            Array of flux error values
        window_length : int
            Window length for Savitzky-Golay filter
            
        Returns:
        --------
        tuple
            (time, detrended_flux, flux_err)
        """
        # Handle NaN values
        mask = np.isfinite(flux)
        if not np.any(mask):
            raise ValueError("All flux values are NaN")
        
        # Apply Savitzky-Golay filter to the valid data
        trend = np.ones_like(flux) * np.nan
        if np.sum(mask) > window_length:
            trend[mask] = signal.savgol_filter(flux[mask], window_length, 2)
        else:
            # If too few points, use a simple median
            trend[mask] = np.median(flux[mask])
        
        # Detrend
        detrended_flux = flux / trend
        
        # Adjust errors
        detrended_flux_err = flux_err / trend
        
        return time, detrended_flux, detrended_flux_err
    
    def normalize_light_curve(self, time, flux, flux_err):
        """
        Normalize a light curve to have a median of 1.0.
        
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
            (time, normalized_flux, normalized_flux_err)
        """
        # Handle NaN values
        mask = np.isfinite(flux)
        if not np.any(mask):
            raise ValueError("All flux values are NaN")
        
        # Calculate median
        median = np.median(flux[mask])
        
        # Normalize
        normalized_flux = flux / median
        normalized_flux_err = flux_err / median
        
        return time, normalized_flux, normalized_flux_err
    
    def preprocess_light_curve(self, filepath):
        """
        Preprocess a light curve: load, detrend, and normalize.
        
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
            # Extract TIC ID and sector from filename
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
            
            # Load light curve
            time, flux, flux_err = self.load_csv_light_curve(filepath)
            
            # Detrend
            time, flux, flux_err = self.detrend_light_curve(time, flux, flux_err)
            
            # Normalize
            time, flux, flux_err = self.normalize_light_curve(time, flux, flux_err)
            
            return time, flux, flux_err, tic_id, sector
        
        except Exception as e:
            logger.error(f"Error preprocessing light curve {filepath}: {str(e)}")
            raise
    
    def save_processed_light_curve(self, time, flux, flux_err, tic_id, sector):
        """
        Save a processed light curve to a CSV file.
        
        Parameters:
        -----------
        time : numpy.ndarray
            Array of time values
        flux : numpy.ndarray
            Array of flux values
        flux_err : numpy.ndarray
            Array of flux error values
        tic_id : int
            TIC ID
        sector : int
            TESS sector number
            
        Returns:
        --------
        str
            Path to saved CSV file
        """
        # Create directory for this target
        target_dir = os.path.join(self.processed_dir, f"TIC_{tic_id}")
        os.makedirs(target_dir, exist_ok=True)
        
        # Define output file path
        csv_file = os.path.join(target_dir, f"sector_{sector}_lc.csv")
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time,
            'flux': flux,
            'flux_err': flux_err
        })
        
        # Save to CSV
        df.to_csv(csv_file, index=False)
        
        return csv_file
    
    def load_transit_parameters(self, filepath=None):
        """
        Load transit parameters from a CSV file.
        
        Parameters:
        -----------
        filepath : str or None
            Path to CSV file (if None, use default)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing transit parameters
        """
        if filepath is None:
            filepath = os.path.join(self.catalog_dir, "transit_parameters.csv")
        
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Error loading transit parameters from {filepath}: {str(e)}")
            return pd.DataFrame(columns=['tic_id', 'planet_name', 'period', 'epoch', 'duration', 'depth', 'source'])
    
    def extract_transit_windows(self, time, flux, flux_err, tic_id, sector, transit_params):
        """
        Extract transit windows from a light curve.
        
        Parameters:
        -----------
        time : numpy.ndarray
            Array of time values
        flux : numpy.ndarray
            Array of flux values
        flux_err : numpy.ndarray
            Array of flux error values
        tic_id : int
            TIC ID
        sector : int
            TESS sector number
        transit_params : dict
            Transit parameters
            
        Returns:
        --------
        tuple
            (transit_windows, non_transit_windows)
        """
        # Extract transit parameters
        period = transit_params['period']
        epoch = transit_params['epoch']
        duration = transit_params['duration']  # hours
        
        # Convert duration from hours to days
        duration_days = duration / 24.0
        
        # Calculate window size in days
        window_size_days = self.window_size_hours / 24.0
        
        # Calculate transit times within the light curve time range
        transit_times = []
        t = epoch
        while t < time[-1]:
            if t > time[0]:
                transit_times.append(t)
            t += period
        
        t = epoch - period
        while t > time[0]:
            transit_times.append(t)
            t -= period
        
        transit_times = sorted(transit_times)
        
        # Extract transit windows
        transit_windows = []
        non_transit_windows = []
        
        # For each transit time
        for t_mid in transit_times:
            # Define window boundaries
            t_start = t_mid - window_size_days / 2
            t_end = t_mid + window_size_days / 2
            
            # Find indices within window
            idx = (time >= t_start) & (time <= t_end)
            
            if np.sum(idx) > 10:  # Require at least 10 points
                # Extract window
                window_time = time[idx]
                window_flux = flux[idx]
                window_flux_err = flux_err[idx]
                
                # Center time around transit
                window_time = window_time - t_mid
                
                # Create window dictionary
                window = {
                    'time': window_time,
                    'flux': window_flux,
                    'flux_err': window_flux_err,
                    'tic_id': tic_id,
                    'sector': sector,
                    'transit_time': t_mid,
                    'period': period,
                    'duration': duration
                }
                
                transit_windows.append(window)
        
        # Extract non-transit windows
        # For simplicity, we'll extract windows midway between transits
        for i in range(len(transit_times) - 1):
            t_mid = (transit_times[i] + transit_times[i+1]) / 2
            
            # Define window boundaries
            t_start = t_mid - window_size_days / 2
            t_end = t_mid + window_size_days / 2
            
            # Find indices within window
            idx = (time >= t_start) & (time <= t_end)
            
            if np.sum(idx) > 10:  # Require at least 10 points
                # Extract window
                window_time = time[idx]
                window_flux = flux[idx]
                window_flux_err = flux_err[idx]
                
                # Center time around midpoint
                window_time = window_time - t_mid
                
                # Create window dictionary
                window = {
                    'time': window_time,
                    'flux': window_flux,
                    'flux_err': window_flux_err,
                    'tic_id': tic_id,
                    'sector': sector,
                    'transit_time': t_mid,
                    'period': period,
                    'duration': duration
                }
                
                non_transit_windows.append(window)
        
        return transit_windows, non_transit_windows
    
    def save_window(self, window, is_transit=True):
        """
        Save a window to a CSV file.
        
        Parameters:
        -----------
        window : dict
            Window dictionary
        is_transit : bool
            Whether this is a transit window
            
        Returns:
        --------
        str
            Path to saved CSV file
        """
        # Determine output directory
        if is_transit:
            output_dir = self.transit_windows_dir
        else:
            output_dir = self.non_transit_windows_dir
        
        # Create filename
        filename = f"TIC_{window['tic_id']}_sector_{window['sector']}_{window['transit_time']:.1f}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': window['time'],
            'flux': window['flux'],
            'flux_err': window['flux_err']
        })
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def run_preprocessing_pipeline(self, limit=None):
        """
        Run the complete preprocessing pipeline.
        
        Parameters:
        -----------
        limit : int or None
            Maximum number of light curves to process
            
        Returns:
        --------
        dict
            Dictionary containing pipeline results
        """
        logger.info("Starting preprocessing pipeline")
        
        # Find all light curve files (CSV format)
        lc_files = glob.glob(os.path.join(self.raw_dir, "TIC_*", "sector_*_lc.csv"))
        
        if not lc_files:
            logger.warning("No CSV light curve files found in raw directory")
        
        if limit is not None:
            lc_files = lc_files[:limit]
        
        logger.info(f"Preprocessing {len(lc_files)} light curves")
        
        # Process each light curve
        processed_files = []
        
        for lc_file in tqdm(lc_files, desc="Preprocessing light curves"):
            try:
                # Preprocess light curve
                time, flux, flux_err, tic_id, sector = self.preprocess_light_curve(lc_file)
                
                # Save processed light curve
                processed_file = self.save_processed_light_curve(time, flux, flux_err, tic_id, sector)
                processed_files.append(processed_file)
            except Exception as e:
                logger.error(f"Error preprocessing light curve {lc_file}: {str(e)}")
        
        logger.info(f"Preprocessed {len(processed_files)} light curves")
        
        # Load transit parameters
        transit_params_df = self.load_transit_parameters()
        
        # Cross-match TIC IDs
        processed_tic_ids = set()
        for filepath in processed_files:
            dirname = os.path.dirname(filepath)
            tic_dirname = os.path.basename(dirname)
            if "TIC_" in tic_dirname:
                tic_id = int(tic_dirname.split("_")[1])
                processed_tic_ids.add(tic_id)
        
        logger.info(f"Found {len(processed_tic_ids)} unique TIC IDs in processed data")
        
        # Find TIC IDs with transit parameters
        transit_tic_ids = set(transit_params_df['tic_id'].unique())
        matched_tic_ids = processed_tic_ids.intersection(transit_tic_ids)
        
        logger.info(f"Found transit parameters for {len(matched_tic_ids)} TIC IDs")
        
        # Extract transit windows
        transit_windows = []
        non_transit_windows = []
        
        for tic_id in tqdm(matched_tic_ids, desc="Extracting transit windows"):
            # Get transit parameters for this TIC ID
            tic_params = transit_params_df[transit_params_df['tic_id'] == tic_id].iloc[0]
            transit_params = {
                'period': tic_params['period'],
                'epoch': tic_params['epoch'],
                'duration': tic_params['duration'],
                'depth': tic_params['depth']
            }
            
            # Find processed light curves for this TIC ID
            tic_files = glob.glob(os.path.join(self.processed_dir, f"TIC_{tic_id}", "sector_*_lc.csv"))
            
            for filepath in tic_files:
                try:
                    # Load processed light curve
                    df = pd.read_csv(filepath)
                    time = df['time'].values
                    flux = df['flux'].values
                    flux_err = df['flux_err'].values
                    
                    # Extract sector from filename
                    filename = os.path.basename(filepath)
                    sector = int(filename.split("_")[1].split(".")[0])
                    
                    # Extract transit windows
                    tic_transit_windows, tic_non_transit_windows = self.extract_transit_windows(
                        time, flux, flux_err, tic_id, sector, transit_params
                    )
                    
                    # Save transit windows
                    for window in tic_transit_windows:
                        self.save_window(window, is_transit=True)
                    
                    # Save non-transit windows
                    for window in tic_non_transit_windows:
                        self.save_window(window, is_transit=False)
                    
                    transit_windows.extend(tic_transit_windows)
                    non_transit_windows.extend(tic_non_transit_windows)
                
                except Exception as e:
                    logger.error(f"Error extracting windows from {filepath}: {str(e)}")
        
        logger.info(f"Extracted {len(transit_windows)} transit windows and {len(non_transit_windows)} non-transit windows")
        
        # Check if any windows were extracted
        if len(transit_windows) == 0 and len(non_transit_windows) == 0:
            logger.warning("No window files found")
        
        # Compile pipeline results
        pipeline_results = {
            'num_light_curves_preprocessed': len(processed_files),
            'num_stars_with_transits': len(matched_tic_ids),
            'num_transit_windows': len(transit_windows),
            'num_non_transit_windows': len(non_transit_windows)
        }
        
        logger.info("Preprocessing pipeline completed")
        logger.info(f"Summary: {pipeline_results}")
        
        return pipeline_results


if __name__ == "__main__":
    # Run preprocessing pipeline
    preprocessing = TESSDataPreprocessing()
    results = preprocessing.run_preprocessing_pipeline()
    print(results)
