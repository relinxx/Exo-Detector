#!/usr/bin/env python3
"""
Exo-Detector: Synthetic Light Curve Generation Module

This module generates synthetic TESS-like light curves with injected transit signals.
It creates realistic light curves that mimic TESS data without requiring API access.

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
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SyntheticLightCurveGenerator:
    """Class for generating synthetic TESS-like light curves."""
    
    def __init__(self, data_dir="data", sectors=[1, 2, 3, 4, 5]):
        """
        Initialize the synthetic light curve generator.
        
        Parameters:
        -----------
        data_dir : str
            Directory to store data
        sectors : list
            List of TESS sectors to simulate
        """
        # Convert to absolute path
        self.data_dir = os.path.abspath(data_dir)
        self.sectors = sectors
        
        # Create directories
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.catalog_dir = os.path.join(self.data_dir, "catalogs")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.catalog_dir, exist_ok=True)
        
        # TESS-specific parameters
        self.cadence = 2.0 / (24.0 * 60.0)  # 2-minute cadence in days
        self.sector_duration = 27.0  # days
        self.sector_gap = 3.0  # days
        
        # Noise parameters
        self.photon_noise_level = 0.0001  # Base photon noise level
        self.systematic_noise_level = 0.0002  # Base systematic noise level
        
        logger.info(f"Initialized SyntheticLightCurveGenerator with sectors={sectors}")
        logger.info(f"Using data directory: {self.data_dir}")
    
    def generate_time_array(self, sector):
        """
        Generate time array for a TESS sector.
        
        Parameters:
        -----------
        sector : int
            TESS sector number
            
        Returns:
        --------
        numpy.ndarray
            Array of time values in BTJD (TESS BJD - 2457000.0)
        """
        # Calculate start time for sector (approximate)
        sector_start = 1325.0 + (sector - 1) * (self.sector_duration + self.sector_gap)
        
        # Generate time array with 2-minute cadence
        n_points = int(self.sector_duration / self.cadence)
        time_array = sector_start + np.arange(n_points) * self.cadence
        
        # Add small gaps to simulate data downlink
        gap_start = sector_start + self.sector_duration / 2.0 - 0.5
        gap_end = gap_start + 1.0
        gap_mask = (time_array < gap_start) | (time_array > gap_end)
        time_array = time_array[gap_mask]
        
        return time_array
    
    def generate_stellar_variability(self, time_array, amplitude=0.001, period=5.0, phase=0.0):
        """
        Generate stellar variability signal.
        
        Parameters:
        -----------
        time_array : numpy.ndarray
            Array of time values
        amplitude : float
            Amplitude of variability
        period : float
            Period of variability in days
        phase : float
            Phase offset in radians
            
        Returns:
        --------
        numpy.ndarray
            Stellar variability signal
        """
        return amplitude * np.sin(2.0 * np.pi * time_array / period + phase)
    
    def generate_systematic_noise(self, time_array, level=None):
        """
        Generate systematic noise.
        
        Parameters:
        -----------
        time_array : numpy.ndarray
            Array of time values
        level : float or None
            Noise level (if None, use default)
            
        Returns:
        --------
        numpy.ndarray
            Systematic noise signal
        """
        if level is None:
            level = self.systematic_noise_level
        
        # Generate low-frequency noise using a sum of sines
        noise = np.zeros_like(time_array)
        for i in range(5):
            period = np.random.uniform(0.5, 5.0) * self.sector_duration
            amplitude = level * np.random.uniform(0.5, 1.5) / (i + 1)
            phase = np.random.uniform(0, 2.0 * np.pi)
            noise += amplitude * np.sin(2.0 * np.pi * time_array / period + phase)
        
        # Add a long-term trend
        trend = level * 2.0 * (time_array - time_array[0]) / (time_array[-1] - time_array[0])
        trend = trend - np.mean(trend)
        
        return noise + trend
    
    def generate_transit_signal(self, time_array, period, epoch, duration, depth):
        """
        Generate transit signal.
        
        Parameters:
        -----------
        time_array : numpy.ndarray
            Array of time values
        period : float
            Orbital period in days
        epoch : float
            Transit epoch in BTJD
        duration : float
            Transit duration in hours
        depth : float
            Transit depth as fraction of flux
            
        Returns:
        --------
        numpy.ndarray
            Transit signal (1.0 - depth during transit, 1.0 otherwise)
        """
        # Convert duration from hours to days
        duration_days = duration / 24.0
        
        # Calculate phase
        phase = ((time_array - epoch) % period) / period
        
        # Calculate transit width in phase
        width = duration_days / period
        
        # Generate transit signal
        transit = np.ones_like(time_array)
        in_transit = (phase < width / 2.0) | (phase > 1.0 - width / 2.0)
        transit[in_transit] = 1.0 - depth
        
        return transit
    
    def generate_light_curve(self, tic_id, sector, has_transit=False, transit_params=None):
        """
        Generate synthetic light curve for a given TIC ID and sector.
        
        Parameters:
        -----------
        tic_id : int
            TIC ID
        sector : int
            TESS sector number
        has_transit : bool
            Whether to include a transit signal
        transit_params : dict or None
            Transit parameters (if None, generate random parameters)
            
        Returns:
        --------
        tuple
            (time_array, flux_array, flux_err_array, transit_params)
        """
        # Generate time array
        time_array = self.generate_time_array(sector)
        
        # Generate base flux (normalized to 1.0)
        flux = np.ones_like(time_array)
        
        # Add stellar variability
        variability_amplitude = np.random.uniform(0.0001, 0.002)
        variability_period = np.random.uniform(0.5, 10.0)
        variability_phase = np.random.uniform(0, 2.0 * np.pi)
        variability = self.generate_stellar_variability(
            time_array, 
            amplitude=variability_amplitude,
            period=variability_period,
            phase=variability_phase
        )
        flux += variability
        
        # Add systematic noise
        systematic_level = self.systematic_noise_level * np.random.uniform(0.5, 1.5)
        systematics = self.generate_systematic_noise(time_array, level=systematic_level)
        flux += systematics
        
        # Add transit signal if requested
        if has_transit:
            if transit_params is None:
                # Generate random transit parameters
                period = np.random.uniform(1.0, 10.0)  # 1-10 days
                epoch = time_array[0] + np.random.uniform(0, period)
                duration = np.random.uniform(1.0, 6.0)  # 1-6 hours
                depth = np.random.uniform(0.0005, 0.02)  # 0.05% to 2%
                transit_params = {
                    'period': period,
                    'epoch': epoch,
                    'duration': duration,
                    'depth': depth
                }
            
            transit = self.generate_transit_signal(
                time_array,
                period=transit_params['period'],
                epoch=transit_params['epoch'],
                duration=transit_params['duration'],
                depth=transit_params['depth']
            )
            flux *= transit
        
        # Add photon noise
        photon_noise_level = self.photon_noise_level * np.random.uniform(0.8, 1.2)
        photon_noise = np.random.normal(0, photon_noise_level, size=len(time_array))
        flux += photon_noise
        
        # Generate flux errors
        flux_err = np.ones_like(flux) * photon_noise_level
        
        return time_array, flux, flux_err, transit_params
    
    def save_light_curve_as_csv(self, tic_id, sector, time_array, flux_array, flux_err_array):
        """
        Save synthetic light curve as CSV file.
        
        Parameters:
        -----------
        tic_id : int
            TIC ID
        sector : int
            TESS sector number
        time_array : numpy.ndarray
            Array of time values
        flux_array : numpy.ndarray
            Array of flux values
        flux_err_array : numpy.ndarray
            Array of flux error values
            
        Returns:
        --------
        str
            Path to saved CSV file
        """
        # Create directory for this target
        target_dir = os.path.join(self.raw_dir, f"TIC_{tic_id}")
        os.makedirs(target_dir, exist_ok=True)
        
        # Define output file path
        csv_file = os.path.join(target_dir, f"sector_{sector}_lc.csv")
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time_array,
            'flux': flux_array,
            'flux_err': flux_err_array,
            'quality': np.zeros_like(time_array, dtype=int)  # Add quality flags (all zeros)
        })
        
        # Add metadata as the first few rows with # comments
        with open(csv_file, 'w') as f:
            f.write(f"# SYNTHETIC TESS LIGHT CURVE\n")
            f.write(f"# OBJECT: TIC {tic_id}\n")
            f.write(f"# SECTOR: {sector}\n")
            f.write(f"# CREATED: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}\n")
            f.write(f"# CADENCE: 2-minute\n")
            f.write(f"# COLUMNS: time,flux,flux_err,quality\n")
        
        # Append data
        df.to_csv(csv_file, index=False, mode='a')
        
        return csv_file
    
    def create_transit_parameters_catalog(self, tic_ids, planet_fraction=0.3):
        """
        Create a catalog of transit parameters for synthetic light curves.
        
        Parameters:
        -----------
        tic_ids : list
            List of TIC IDs
        planet_fraction : float
            Fraction of stars with planets
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing transit parameters
        """
        logger.info(f"Creating synthetic transit parameters catalog for {len(tic_ids)} TIC IDs")
        
        # Initialize empty DataFrame
        transit_params = pd.DataFrame(columns=[
            'tic_id', 'planet_name', 'period', 'epoch', 'duration', 'depth', 'source'
        ])
        
        # Determine which stars have planets
        n_planets = int(len(tic_ids) * planet_fraction)
        if n_planets == 0 and len(tic_ids) > 0:
            n_planets = 1  # Ensure at least one planet if we have stars
            
        planet_indices = np.random.choice(len(tic_ids), size=min(n_planets, len(tic_ids)), replace=False)
        planet_tic_ids = [tic_ids[i] for i in planet_indices]
        
        # Generate transit parameters for each planet
        params_list = []
        for tic_id in planet_tic_ids:
            # Generate random transit parameters
            period = np.random.uniform(1.0, 10.0)  # 1-10 days
            epoch = 1325.0 + np.random.uniform(0, period)  # Start of sector 1 + random offset
            duration = np.random.uniform(1.0, 6.0)  # 1-6 hours
            depth = np.random.uniform(0.0005, 0.02)  # 0.05% to 2%
            
            params_list.append({
                'tic_id': tic_id,
                'planet_name': f"SYN-{tic_id}b",
                'period': period,
                'epoch': epoch,
                'duration': duration,
                'depth': depth,
                'source': 'synthetic'
            })
        
        # Create DataFrame
        if params_list:
            transit_params = pd.DataFrame(params_list)
        
        # Save transit parameters catalog
        transit_params_file = os.path.join(self.catalog_dir, "transit_parameters.csv")
        transit_params.to_csv(transit_params_file, index=False)
        
        logger.info(f"Created transit parameters catalog with {len(transit_params)} entries")
        return transit_params
    
    def generate_synthetic_data(self, num_stars=100, max_sectors_per_star=None):
        """
        Generate synthetic TESS data for multiple stars.
        
        Parameters:
        -----------
        num_stars : int
            Number of stars to generate
        max_sectors_per_star : int or None
            Maximum number of sectors per star (if None, use all sectors)
            
        Returns:
        --------
        dict
            Dictionary containing generation results
        """
        logger.info(f"Generating synthetic TESS data for {num_stars} stars")
        
        # Generate TIC IDs (random 5-digit numbers)
        tic_ids = np.random.randint(10000, 99999, size=num_stars)
        tic_ids = np.unique(tic_ids)  # Ensure uniqueness
        
        # Create transit parameters catalog with higher planet fraction
        transit_params_df = self.create_transit_parameters_catalog(tic_ids, planet_fraction=0.5)
        
        # Create dictionary mapping TIC ID to transit parameters
        transit_params_dict = {}
        for _, row in transit_params_df.iterrows():
            transit_params_dict[row['tic_id']] = {
                'period': row['period'],
                'epoch': row['epoch'],
                'duration': row['duration'],
                'depth': row['depth']
            }
        
        # Generate light curves for each star
        num_light_curves = 0
        transit_windows = []
        non_transit_windows = []
        
        for tic_id in tqdm(tic_ids, desc="Generating light curves"):
            # Determine which sectors to generate for this star
            if max_sectors_per_star is None or max_sectors_per_star >= len(self.sectors):
                star_sectors = self.sectors
            else:
                num_sectors = np.random.randint(1, max_sectors_per_star + 1)
                star_sectors = np.random.choice(self.sectors, size=num_sectors, replace=False)
            
            # Check if star has a planet
            has_transit = tic_id in transit_params_dict
            transit_params = transit_params_dict.get(tic_id, None)
            
            # Generate light curve for each sector
            for sector in star_sectors:
                time_array, flux_array, flux_err_array, _ = self.generate_light_curve(
                    tic_id, sector, has_transit, transit_params
                )
                
                # Save light curve as CSV file
                csv_file = self.save_light_curve_as_csv(
                    tic_id, sector, time_array, flux_array, flux_err_array
                )
                
                # Extract windows for GAN training
                if has_transit:
                    # Extract transit windows
                    transit_indices = self._find_transit_indices(time_array, transit_params)
                    for idx in transit_indices:
                        window = self._extract_window(time_array, flux_array, idx)
                        if window is not None:
                            transit_windows.append(window)
                else:
                    # Extract non-transit windows
                    for _ in range(3):  # Extract 3 non-transit windows per light curve
                        idx = np.random.randint(0, len(time_array) - 100)
                        window = self._extract_window(time_array, flux_array, idx)
                        if window is not None:
                            non_transit_windows.append(window)
                
                logger.info(f"Generated light curve for TIC {tic_id}, sector {sector}: {csv_file}")
                num_light_curves += 1
        
        # Save windows for GAN training
        self._save_windows(transit_windows, non_transit_windows)
        
        # Create a simple catalog of confirmed planets (empty but with correct structure)
        confirmed_planets_file = os.path.join(self.catalog_dir, "confirmed_planets.csv")
        if not os.path.exists(confirmed_planets_file):
            confirmed_df = pd.DataFrame(columns=['pl_name', 'pl_orbper', 'pl_tranmid', 'pl_trandur', 'pl_trandep'])
            confirmed_df.to_csv(confirmed_planets_file, index=False)
            logger.info(f"Created empty confirmed planets catalog")
        
        # Create a simple TOI catalog (empty but with correct structure)
        toi_file = os.path.join(self.catalog_dir, "toi_catalog.csv")
        if not os.path.exists(toi_file):
            toi_df = pd.DataFrame(columns=['TOI', 'TIC ID', 'Period (days)', 'Epoch (BJD)', 'Duration (hours)', 'Depth (ppm)'])
            toi_df.to_csv(toi_file, index=False)
            logger.info(f"Created empty TOI catalog")
        
        # Compile generation results
        generation_results = {
            'num_stars': len(tic_ids),
            'num_light_curves': num_light_curves,
            'num_planets': len(transit_params_df),
            'sectors': self.sectors,
            'confirmed_planets_catalog_size': 0,
            'toi_catalog_size': 0,
            'num_transit_windows': len(transit_windows),
            'num_non_transit_windows': len(non_transit_windows)
        }
        
        logger.info("Synthetic data generation completed")
        logger.info(f"Summary: {generation_results}")
        
        return generation_results

    def _find_transit_indices(self, time_array, transit_params):
        """Find indices where transits occur in the time array."""
        period = transit_params['period']
        epoch = transit_params['epoch']
        duration = transit_params['duration'] / 24.0  # Convert hours to days
        
        # Calculate phases
        phases = ((time_array - epoch) % period) / period
        width = duration / period
        
        # Find indices where transit occurs
        transit_indices = np.where((phases < width/2) | (phases > 1 - width/2))[0]
        
        # Group consecutive indices
        groups = []
        current_group = []
        for i in range(len(transit_indices)):
            if i == 0 or transit_indices[i] - transit_indices[i-1] == 1:
                current_group.append(transit_indices[i])
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [transit_indices[i]]
        if current_group:
            groups.append(current_group)
        
        # Return middle index of each group
        return [group[len(group)//2] for group in groups]

    def _extract_window(self, time_array, flux_array, center_idx, window_size=100):
        """Extract a window of data centered at the given index."""
        half_window = window_size // 2
        start_idx = max(0, center_idx - half_window)
        end_idx = min(len(time_array), center_idx + half_window)
        
        if end_idx - start_idx < window_size:
            return None
        
        return {
            'time': time_array[start_idx:end_idx],
            'flux': flux_array[start_idx:end_idx]
        }

    def _save_windows(self, transit_windows, non_transit_windows):
        """Save transit and non-transit windows for GAN training."""
        # Create directories if they don't exist
        transit_dir = os.path.join(self.data_dir, "transit_windows")
        non_transit_dir = os.path.join(self.data_dir, "non_transit_windows")
        os.makedirs(transit_dir, exist_ok=True)
        os.makedirs(non_transit_dir, exist_ok=True)
        
        # Save transit windows
        for i, window in enumerate(transit_windows):
            df = pd.DataFrame({
                'time': window['time'],
                'flux': window['flux']
            })
            df.to_csv(os.path.join(transit_dir, f"transit_{i:04d}.csv"), index=False)
        
        # Save non-transit windows
        for i, window in enumerate(non_transit_windows):
            df = pd.DataFrame({
                'time': window['time'],
                'flux': window['flux']
            })
            df.to_csv(os.path.join(non_transit_dir, f"non_transit_{i:04d}.csv"), index=False)
        
        logger.info(f"Saved {len(transit_windows)} transit windows and {len(non_transit_windows)} non-transit windows")


def run_data_ingestion(data_dir="data", sectors=[1, 2, 3, 4, 5], num_stars=100, max_sectors_per_star=3):
    """
    Run the synthetic data generation pipeline.
    
    Parameters:
    -----------
    data_dir : str
        Directory to store data
    sectors : list
        List of TESS sectors to simulate
    num_stars : int
        Number of stars to generate
    max_sectors_per_star : int
        Maximum number of sectors per star
        
    Returns:
    --------
    dict
        Dictionary containing pipeline results
    """
    # Create synthetic data generator
    generator = SyntheticLightCurveGenerator(data_dir=data_dir, sectors=sectors)
    
    # Generate synthetic data
    results = generator.generate_synthetic_data(num_stars=num_stars, max_sectors_per_star=max_sectors_per_star)
    
    return results


if __name__ == "__main__":
    # Run synthetic data generation
    results = run_data_ingestion(num_stars=20, max_sectors_per_star=2)  # Small test run
    print(results)
