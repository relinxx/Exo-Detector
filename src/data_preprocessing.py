import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import lightkurve as lk
from scipy.signal import savgol_filter
from tqdm import tqdm
import logging
import glob
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../data/preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress lightkurve warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightkurve")

class TESSDataPreprocessing:
    """Class for handling TESS data preprocessing operations."""
    
    def __init__(self, data_dir="../data", window_size_hours=5):
        """
        Initialize the data preprocessing module.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the downloaded data
        window_size_hours : float
            Size of the window to extract around transits (in hours)
        """
        self.data_dir = data_dir
        self.window_size_hours = window_size_hours
        
        # Define directories
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.catalog_dir = os.path.join(data_dir, "catalogs")
        self.transit_dir = os.path.join(data_dir, "transit_windows")
        self.non_transit_dir = os.path.join(data_dir, "non_transit_windows")
        
        # Create directories if they don't exist
        for directory in [self.processed_dir, self.transit_dir, self.non_transit_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Load catalogs if they exist
        self.confirmed_planets = self._load_catalog("confirmed_planets.csv")
        self.toi_catalog = self._load_catalog("toi_catalog.csv")
        
        logger.info(f"Initialized TESSDataPreprocessing with window_size_hours={window_size_hours}")
    
    def _load_catalog(self, filename):
        """
        Load a catalog from file.
        
        Parameters:
        -----------
        filename : str
            Name of the catalog file
            
        Returns:
        --------
        pandas.DataFrame or None
            Loaded catalog or None if file doesn't exist
        """
        filepath = os.path.join(self.catalog_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            logger.warning(f"Catalog file not found: {filepath}")
            return None
    
    def detrend_light_curve(self, lc, window_length=101, polyorder=2):
        """
        Detrend a light curve using Savitzky-Golay filter.
        
        Parameters:
        -----------
        lc : lightkurve.LightCurve
            Light curve to detrend
        window_length : int
            Window length for Savitzky-Golay filter
        polyorder : int
            Polynomial order for Savitzky-Golay filter
            
        Returns:
        --------
        lightkurve.LightCurve
            Detrended light curve
        """
        # Make sure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Remove NaNs
        mask = ~np.isnan(lc.flux.value)
        time = lc.time.value[mask]
        flux = lc.flux.value[mask]
        
        # Apply Savitzky-Golay filter
        try:
            trend = savgol_filter(flux, window_length, polyorder)
            detrended_flux = flux / trend
            
            # Create a new light curve with detrended flux
            detrended_lc = lk.LightCurve(time=time, flux=detrended_flux)
            return detrended_lc
        except Exception as e:
            logger.error(f"Error detrending light curve: {str(e)}")
            return None
    
    def normalize_light_curve(self, lc):
        """
        Normalize a light curve to have a median of 1.0.
        
        Parameters:
        -----------
        lc : lightkurve.LightCurve
            Light curve to normalize
            
        Returns:
        --------
        lightkurve.LightCurve
            Normalized light curve
        """
        if lc is None:
            return None
        
        try:
            # Remove NaNs
            mask = ~np.isnan(lc.flux.value)
            time = lc.time.value[mask]
            flux = lc.flux.value[mask]
            
            # Normalize flux to have median of 1.0
            median_flux = np.median(flux)
            normalized_flux = flux / median_flux
            
            # Create a new light curve with normalized flux
            normalized_lc = lk.LightCurve(time=time, flux=normalized_flux)
            return normalized_lc
        except Exception as e:
            logger.error(f"Error normalizing light curve: {str(e)}")
            return None
    
    def preprocess_light_curve(self, lc_file):
        """
        Preprocess a light curve file.
        
        Parameters:
        -----------
        lc_file : str
            Path to the light curve file
            
        Returns:
        --------
        tuple
            (tic_id, sector, preprocessed_lc) - TIC ID, sector, and preprocessed light curve
        """
        try:
            # Extract TIC ID and sector from filename
            tic_id = int(os.path.basename(os.path.dirname(lc_file)).split('_')[1])
            sector = int(os.path.basename(lc_file).split('_')[1])
            
            # Load the light curve
            lc = lk.read(lc_file)
            
            # Remove flagged data points
            if hasattr(lc, 'quality'):
                lc = lc[lc.quality == 0]
            
            # Detrend the light curve
            detrended_lc = self.detrend_light_curve(lc)
            
            # Normalize the light curve
            normalized_lc = self.normalize_light_curve(detrended_lc)
            
            if normalized_lc is None:
                logger.warning(f"Failed to preprocess light curve: {lc_file}")
                return tic_id, sector, None
            
            # Save the preprocessed light curve
            output_dir = os.path.join(self.processed_dir, f"TIC_{tic_id}")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"sector_{sector}_lc_processed.fits")
            normalized_lc.to_fits(output_file, overwrite=True)
            
            logger.info(f"Preprocessed light curve for TIC {tic_id}, sector {sector}")
            return tic_id, sector, normalized_lc
        
        except Exception as e:
            logger.error(f"Error preprocessing light curve {lc_file}: {str(e)}")
            return None, None, None
    
    def preprocess_all_light_curves(self, limit=None):
        """
        Preprocess all downloaded light curves.
        
        Parameters:
        -----------
        limit : int, optional
            Limit the number of light curves to preprocess (for testing)
            
        Returns:
        --------
        list
            List of preprocessed light curve files
        """
        # Get all light curve files
        lc_files = []
        for root, dirs, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith("_lc.fits"):
                    lc_files.append(os.path.join(root, file))
        
        if limit is not None:
            lc_files = lc_files[:limit]
        
        logger.info(f"Preprocessing {len(lc_files)} light curves")
        
        # Preprocess each light curve
        preprocessed_files = []
        for lc_file in tqdm(lc_files, desc="Preprocessing light curves"):
            tic_id, sector, _ = self.preprocess_light_curve(lc_file)
            
            if tic_id is not None and sector is not None:
                output_file = os.path.join(self.processed_dir, f"TIC_{tic_id}", f"sector_{sector}_lc_processed.fits")
                preprocessed_files.append(output_file)
        
        logger.info(f"Preprocessed {len(preprocessed_files)} light curves")
        
        # Save the list of preprocessed files
        with open(os.path.join(self.catalog_dir, 'preprocessed_files.txt'), 'w') as f:
            for file_path in preprocessed_files:
                f.write(f"{file_path}\n")
        
        return preprocessed_files
    
    def cross_match_with_catalogs(self):
        """
        Cross-match TIC IDs with exoplanet catalogs.
        
        Returns:
        --------
        dict
            Dictionary mapping TIC IDs to transit parameters
        """
        if self.confirmed_planets is None or self.toi_catalog is None:
            logger.error("Exoplanet catalogs not loaded")
            return {}
        
        logger.info("Cross-matching TIC IDs with exoplanet catalogs")
        
        # Get all TIC IDs from processed light curves
        tic_ids = set()
        for root, dirs, files in os.walk(self.processed_dir):
            for dir_name in dirs:
                if dir_name.startswith("TIC_"):
                    try:
                        tic_id = int(dir_name.split("_")[1])
                        tic_ids.add(tic_id)
                    except (IndexError, ValueError):
                        pass
        
        logger.info(f"Found {len(tic_ids)} unique TIC IDs in processed data")
        
        # Cross-match with TOI catalog
        transit_params = {}
        
        # Check if 'TIC ID' column exists in TOI catalog
        if 'TIC ID' in self.toi_catalog.columns:
            for tic_id in tic_ids:
                matches = self.toi_catalog[self.toi_catalog['TIC ID'] == tic_id]
                
                if len(matches) > 0:
                    for _, row in matches.iterrows():
                        # Extract transit parameters
                        try:
                            period = row.get('Period (days)', np.nan)
                            epoch = row.get('Epoch (BJD)', np.nan)
                            duration = row.get('Duration (hours)', np.nan)
                            depth = row.get('Depth (ppm)', np.nan)
                            
                            if np.isnan(period) or np.isnan(epoch):
                                continue
                            
                            if tic_id not in transit_params:
                                transit_params[tic_id] = []
                            
                            transit_params[tic_id].append({
                                'period': period,
                                'epoch': epoch,
                                'duration': duration if not np.isnan(duration) else 2.0,  # Default duration
                                'depth': depth if not np.isnan(depth) else 1000.0,  # Default depth
                                'source': 'TOI'
                            })
                        except Exception as e:
                            logger.error(f"Error extracting transit parameters for TIC {tic_id}: {str(e)}")
        
        # Save transit parameters
        transit_params_df = []
        for tic_id, params_list in transit_params.items():
            for params in params_list:
                transit_params_df.append({
                    'tic_id': tic_id,
                    'period': params['period'],
                    'epoch': params['epoch'],
                    'duration': params['duration'],
                    'depth': params['depth'],
                    'source': params['source']
                })
        
        transit_params_df = pd.DataFrame(transit_params_df)
        transit_params_df.to_csv(os.path.join(self.catalog_dir, 'transit_parameters.csv'), index=False)
        
        logger.info(f"Found transit parameters for {len(transit_params)} TIC IDs")
        return transit_params
    
    def extract_transit_windows(self, transit_params):
        """
        Extract windows around transit events.
        
        Parameters:
        -----------
        transit_params : dict
            Dictionary mapping TIC IDs to transit parameters
            
        Returns:
        --------
        tuple
            (transit_windows, non_transit_windows) - Lists of transit and non-transit windows
        """
        logger.info("Extracting transit windows")
        
        transit_windows = []
        non_transit_windows = []
        
        # Process each TIC ID with known transits
        for tic_id, params_list in tqdm(transit_params.items(), desc="Extracting transit windows"):
            # Get all processed light curves for this TIC ID
            lc_files = glob.glob(os.path.join(self.processed_dir, f"TIC_{tic_id}", "*_lc_processed.fits"))
            
            if len(lc_files) == 0:
                continue
            
            for lc_file in lc_files:
                try:
                    # Load the light curve
                    lc = lk.read(lc_file)
                    
                    # Extract sector from filename
                    sector = int(os.path.basename(lc_file).split('_')[1])
                    
                    # Process each set of transit parameters
                    for params in params_list:
                        period = params['period']
                        epoch = params['epoch']
                        duration = params['duration']
                        
                        # Calculate the window size in days
                        window_size_days = self.window_size_hours / 24.0
                        
                        # Calculate transit times within the light curve time range
                        time_range = lc.time.value.max() - lc.time.value.min()
                        num_transits = int(time_range / period) + 1
                        
                        transit_times = []
                        for i in range(-num_transits, num_transits + 1):
                            transit_time = epoch + i * period
                            if lc.time.value.min() <= transit_time <= lc.time.value.max():
                                transit_times.append(transit_time)
                        
                        # Extract windows around each transit
                        for transit_time in transit_times:
                            # Define window boundaries
                            window_start = transit_time - window_size_days / 2
                            window_end = transit_time + window_size_days / 2
                            
                            # Extract the window
                            window_mask = (lc.time.value >= window_start) & (lc.time.value <= window_end)
                            if np.sum(window_mask) < 10:  # Skip if too few points
                                continue
                            
                            window_time = lc.time.value[window_mask]
                            window_flux = lc.flux.value[window_mask]
                            
                            # Center the time around the transit
                            window_time = window_time - transit_time
                            
                            # Save the transit window
                            window_data = {
                                'tic_id': tic_id,
                                'sector': sector,
                                'transit_time': transit_time,
                                'period': period,
                                'duration': duration,
                                'time': window_time,
                                'flux': window_flux,
                                'label': 1  # 1 for transit
                            }
                            
                            transit_windows.append(window_data)
                            
                            # Save the window to a file
                            window_file = os.path.join(
                                self.transit_dir,
                                f"TIC_{tic_id}_sector_{sector}_transit_{len(transit_windows)}.npz"
                            )
                            np.savez(
                                window_file,
                                tic_id=tic_id,
                                sector=sector,
                                transit_time=transit_time,
                                period=period,
                                duration=duration,
                                time=window_time,
                                flux=window_flux,
                                label=1
                            )
                        
                        # Extract non-transit windows (3x the number of transit windows)
                        num_non_transit = 3 * len(transit_times)
                        
                        # Define transit masks (regions to avoid)
                        transit_masks = []
                        for transit_time in transit_times:
                            mask_start = transit_time - window_size_days
                            mask_end = transit_time + window_size_days
                            transit_masks.append((mask_start, mask_end))
                        
                        # Sample random non-transit windows
                        non_transit_count = 0
                        max_attempts = 100
                        attempts = 0
                        
                        while non_transit_count < num_non_transit and attempts < max_attempts:
                            # Random start time within the light curve
                            random_time = np.random.uniform(
                                lc.time.value.min() + window_size_days / 2,
                                lc.time.value.max() - window_size_days / 2
                            )
                            
                            # Check if this time overlaps with any transit
                            is_transit = False
                            for mask_start, mask_end in transit_masks:
                                if mask_start <= random_time <= mask_end:
                                    is_transit = True
                                    break
                            
                            if is_transit:
                                attempts += 1
                                continue
                            
                            # Define window boundaries
                            window_start = random_time - window_size_days / 2
                            window_end = random_time + window_size_days / 2
                            
                            # Extract the window
                            window_mask = (lc.time.value >= window_start) & (lc.time.value <= window_end)
                            if np.sum(window_mask) < 10:  # Skip if too few points
                                attempts += 1
                                continue
                            
                            window_time = lc.time.value[window_mask]
                            window_flux = lc.flux.value[window_mask]
                            
                            # Center the time around the random time
                            window_time = window_time - random_time
                            
                            # Save the non-transit window
                            window_data = {
                                'tic_id': tic_id,
                                'sector': sector,
                                'transit_time': random_time,  # Not a real transit time
                                'period': 0.0,  # No period
                                'duration': 0.0,  # No duration
                                'time': window_time,
                                'flux': window_flux,
                                'label': 0  # 0 for non-transit
                            }
                            
                            non_transit_windows.append(window_data)
                            
                            # Save the window to a file
                            window_file = os.path.join(
                                self.non_transit_dir,
                                f"TIC_{tic_id}_sector_{sector}_non_transit_{len(non_transit_windows)}.npz"
                            )
                            np.savez(
                                window_file,
                                tic_id=tic_id,
                                sector=sector,
                                transit_time=random_time,
                                period=0.0,
                                duration=0.0,
                                time=window_time,
                                flux=window_flux,
                                label=0
                            )
                            
                            non_transit_count += 1
                
                except Exception as e:
                    logger.error(f"Error extracting windows from {lc_file}: {str(e)}")
        
        logger.info(f"Extracted {len(transit_windows)} transit windows and {len(non_transit_windows)} non-transit windows")
        
        # Save summary to file
        with open(os.path.join(self.data_dir, "window_extraction_summary.txt"), "w") as f:
            f.write(f"Transit windows: {len(transit_windows)}\n")
            f.write(f"Non-transit windows: {len(non_transit_windows)}\n")
        
        return transit_windows, non_transit_windows
    
    def plot_example_windows(self, num_examples=5):
        """
        Plot example transit and non-transit windows.
        
        Parameters:
        -----------
        num_examples : int
            Number of examples to plot
            
        Returns:
        --------
        None
        """
        # Get transit window files
        transit_files = glob.glob(os.path.join(self.transit_dir, "*.npz"))
        non_transit_files = glob.glob(os.path.join(self.non_transit_dir, "*.npz"))
        
        if len(transit_files) == 0 or len(non_transit_files) == 0:
            logger.warning("No window files found")
            return
        
        # Sample random examples
        transit_samples = np.random.choice(transit_files, min(num_examples, len(transit_files)), replace=False)
        non_transit_samples = np.random.choice(non_transit_files, min(num_examples, len(non_transit_files)), replace=False)
        
        # Create output directory
        plots_dir = os.path.join(self.data_dir, "example_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot transit examples
        plt.figure(figsize=(15, 10))
        for i, file in enumerate(transit_samples):
            data = np.load(file)
            time = data['time']
            flux = data['flux']
            tic_id = data['tic_id']
            sector = data['sector']
            
            plt.subplot(2, num_examples, i + 1)
            plt.plot(time * 24, flux, 'b.')  # Convert time to hours
            plt.axvline(x=0, color='r', linestyle='--')  # Mark transit center
            plt.title(f"TIC {tic_id}, Sector {sector}")
            plt.xlabel("Time from transit (hours)")
            plt.ylabel("Normalized flux")
        
        # Plot non-transit examples
        for i, file in enumerate(non_transit_samples):
            data = np.load(file)
            time = data['time']
            flux = data['flux']
            tic_id = data['tic_id']
            sector = data['sector']
            
            plt.subplot(2, num_examples, i + 1 + num_examples)
            plt.plot(time * 24, flux, 'g.')  # Convert time to hours
            plt.title(f"TIC {tic_id}, Sector {sector} (Non-transit)")
            plt.xlabel("Time (hours)")
            plt.ylabel("Normalized flux")
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "example_windows.png"))
        plt.close()
        
        logger.info(f"Saved example window plots to {plots_dir}")
    
    def run_preprocessing_pipeline(self, limit=None):
        """
        Run the complete preprocessing pipeline.
        
        Parameters:
        -----------
        limit : int, optional
            Limit the number of light curves to process (for testing)
            
        Returns:
        --------
        dict
            Dictionary containing summary statistics
        """
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Preprocess all light curves
        preprocessed_files = self.preprocess_all_light_curves(limit=limit)
        
        # Step 2: Cross-match with exoplanet catalogs
        transit_params = self.cross_match_with_catalogs()
        
        # Step 3: Extract transit and non-transit windows
        transit_windows, non_transit_windows = self.extract_transit_windows(transit_params)
        
        # Step 4: Plot example windows
        self.plot_example_windows()
        
        # Generate summary statistics
        summary = {
            "num_light_curves_preprocessed": len(preprocessed_files),
            "num_stars_with_transits": len(transit_params),
            "num_transit_windows": len(transit_windows),
            "num_non_transit_windows": len(non_transit_windows)
        }
        
        # Save summary to file
        with open(os.path.join(self.data_dir, "preprocessing_summary.txt"), "w") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Preprocessing pipeline completed")
        logger.info(f"Summary: {summary}")
        
        return summary


if __name__ == "__main__":
    # Example usage
    preprocessing = TESSDataPreprocessing()
    
    # For testing, limit to a small number of light curves
    summary = preprocessing.run_preprocessing_pipeline(limit=10)
    print(summary)
