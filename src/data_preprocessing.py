# src/data_preprocessing.py

import os
import logging
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import savgol_filter
import lightkurve as lk

# Configure logging
logger = logging.getLogger(__name__)

class TESSDataPreprocessor:
    """Handles the preprocessing of TESS light curves."""
    def __init__(self, data_dir="data", window_size=256):
        self.data_dir = os.path.abspath(data_dir)
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.catalog_dir = os.path.join(self.data_dir, "catalogs")
        self.transit_windows_dir = os.path.join(self.data_dir, "transit_windows")
        self.non_transit_windows_dir = os.path.join(self.data_dir, "non_transit_windows")
        
        self.window_size = window_size
        
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.transit_windows_dir, exist_ok=True)
        os.makedirs(self.non_transit_windows_dir, exist_ok=True)
        logger.info(f"Initialized TESSDataPreprocessor with window_size={window_size}.")

    def _detrend_and_normalize(self, time, flux):
        """Detrends and normalizes a light curve using a Savitzky-Golay filter."""
        # Ensure there are enough points for the filter window
        if len(flux[np.isfinite(flux)]) < 51:
            return flux / np.median(flux[np.isfinite(flux)])
        
        trend = savgol_filter(flux[np.isfinite(flux)], window_length=51, polyorder=2)
        detrended_flux = flux[np.isfinite(flux)] / trend
        
        final_flux = np.full_like(flux, np.nan)
        final_flux[np.isfinite(flux)] = detrended_flux
        return final_flux

    def run_preprocessing_pipeline(self):
        """Processes all raw light curves and extracts labeled windows."""
        logger.info("--- Starting Data Preprocessing Pipeline ---")
        
        toi_catalog_path = os.path.join(self.catalog_dir, "toi_catalog.csv")
        if not os.path.exists(toi_catalog_path):
            logger.error("TOI Catalog not found. Cannot extract labeled windows.")
            return {}
            
        # The TOI catalog does not have period/duration info, so we query the main PS table.
        planet_params_path = os.path.join(self.catalog_dir, "confirmed_planets.csv")
        if not os.path.exists(planet_params_path):
             logger.error("Confirmed Planets Catalog not found. Cannot get transit parameters.")
             return {}

        params_df = pd.read_csv(planet_params_path)
        raw_files = glob.glob(os.path.join(self.raw_dir, "*.csv"))
        
        if not raw_files:
            logger.warning("No raw light curve files found to process.")
            return {}
            
        logger.info(f"Found {len(raw_files)} raw light curves to process.")
        
        all_transit_windows, all_non_transit_windows = 0, 0

        for filepath in tqdm(raw_files, desc="Preprocessing Light Curves"):
            try:
                df = pd.read_csv(filepath)
                tic_id = int(df['tic_id'].iloc[0])
                
                df['flux'] = self._detrend_and_normalize(df['time'].values, df['flux'].values)
                df.to_csv(os.path.join(self.processed_dir, os.path.basename(filepath)), index=False)
                
                star_params = params_df[params_df['tic_id'] == f"TIC {tic_id}"]
                if star_params.empty:
                    logger.warning(f"No transit parameters found for TIC {tic_id}. Skipping window extraction.")
                    continue

                planet_period = star_params.iloc[0].get('pl_orbper')
                planet_t0 = star_params.iloc[0].get('pl_tranmid')
                
                if pd.isna(planet_period) or pd.isna(planet_t0):
                    logger.warning(f"Incomplete parameters for TIC {tic_id}. Skipping.")
                    continue
                
                # *** DEFINITIVE FIX ***
                # The 'get_transit_times' method does not exist. The correct approach is to
                # manually calculate the transit midpoints using the known period and epoch.
                t_start = df['time'].min()
                t_end = df['time'].max()
                
                # Calculate the first transit number visible in the data
                n_start = np.floor((t_start - planet_t0) / planet_period)
                transit_midpoints = []
                n = n_start
                while True:
                    t_mid = planet_t0 + n * planet_period
                    if t_mid > t_end:
                        break
                    if t_mid >= t_start:
                        transit_midpoints.append(t_mid)
                    n += 1
                
                if not transit_midpoints:
                    logger.warning(f"No transits found within the observation window for {os.path.basename(filepath)}")
                    continue
                
                transit_windows, non_transit_windows = self._extract_windows(df, transit_midpoints)
                all_transit_windows += len(transit_windows)
                all_non_transit_windows += len(non_transit_windows)
                
                self._save_windows(transit_windows, self.transit_windows_dir, tic_id, is_transit=True)
                self._save_windows(non_transit_windows, self.non_transit_windows_dir, tic_id, is_transit=False)
            
            except Exception as e:
                logger.error(f"Failed to process {os.path.basename(filepath)}: {e}", exc_info=True)

        summary = {
            'light_curves_processed': len(raw_files),
            'transit_windows_created': all_transit_windows,
            'non_transit_windows_created': all_non_transit_windows
        }
        logger.info(f"Preprocessing complete. Summary: {summary}")
        return summary

    def _extract_windows(self, df, transit_midpoints):
        """Extracts fixed-size windows around and between transits."""
        transit_windows, non_transit_windows = [], []
        
        is_transit_time = np.zeros(len(df), dtype=bool)
        # Approximate duration in days for masking (e.g., 4 hours)
        approx_duration_days = 4.0 / 24.0
        
        for t_mid in transit_midpoints:
            is_transit_time |= np.abs(df['time'] - t_mid) < approx_duration_days

        for t_mid in transit_midpoints:
            center_idx = np.argmin(np.abs(df['time'] - t_mid))
            start_idx = max(0, center_idx - self.window_size // 2)
            end_idx = start_idx + self.window_size
            if end_idx <= len(df):
                transit_windows.append(df.iloc[start_idx:end_idx])
        
        non_transit_indices = np.where(~is_transit_time)[0]
        # Generate more non-transit windows by shuffling and stepping
        np.random.shuffle(non_transit_indices)
        for i in range(0, len(non_transit_indices) - self.window_size, self.window_size):
             window_indices = non_transit_indices[i:i+self.window_size]
             if len(window_indices) == self.window_size:
                 non_transit_windows.append(df.iloc[window_indices])
                 
        return transit_windows, non_transit_windows
        
    def _save_windows(self, windows, directory, tic_id, is_transit):
        """Saves a list of window dataframes to CSV."""
        label = 1 if is_transit else 0
        for i, window_df in enumerate(windows):
            window_time = window_df['time'].median()
            filename = f"TIC_{tic_id}_label_{label}_time_{window_time:.2f}_{i}.csv"
            filepath = os.path.join(directory, filename)
            window_df.to_csv(filepath, index=False)

