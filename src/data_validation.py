import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import lightkurve as lk
import glob
import logging
from tqdm import tqdm
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../data/validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataValidator:
    """Class for validating data outputs from ingestion and preprocessing phases."""
    
    def __init__(self, data_dir="../data"):
        """
        Initialize the data validator.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the project data
        """
        self.data_dir = data_dir
        
        # Define directories
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.catalog_dir = os.path.join(data_dir, "catalogs")
        self.transit_dir = os.path.join(data_dir, "transit_windows")
        self.non_transit_dir = os.path.join(data_dir, "non_transit_windows")
        self.validation_dir = os.path.join(data_dir, "validation")
        
        # Create validation directory if it doesn't exist
        os.makedirs(self.validation_dir, exist_ok=True)
        
        logger.info(f"Initialized DataValidator with data_dir={data_dir}")
    
    def validate_directory_structure(self):
        """
        Validate the directory structure.
        
        Returns:
        --------
        dict
            Dictionary containing validation results
        """
        logger.info("Validating directory structure")
        
        # Define expected directories
        expected_dirs = [
            self.raw_dir,
            self.processed_dir,
            self.catalog_dir,
            self.transit_dir,
            self.non_transit_dir
        ]
        
        # Check if directories exist
        dir_exists = {}
        for directory in expected_dirs:
            dir_exists[os.path.basename(directory)] = os.path.exists(directory)
        
        # Log results
        for dir_name, exists in dir_exists.items():
            if exists:
                logger.info(f"Directory {dir_name} exists")
            else:
                logger.warning(f"Directory {dir_name} does not exist")
        
        return dir_exists
    
    def validate_catalogs(self):
        """
        Validate the downloaded catalogs.
        
        Returns:
        --------
        dict
            Dictionary containing validation results
        """
        logger.info("Validating catalogs")
        
        # Define expected catalog files
        expected_files = [
            "confirmed_planets.csv",
            "toi_catalog.csv",
            "transit_parameters.csv"
        ]
        
        # Check if files exist and validate content
        catalog_validation = {}
        for filename in expected_files:
            filepath = os.path.join(self.catalog_dir, filename)
            exists = os.path.exists(filepath)
            
            if exists:
                try:
                    df = pd.read_csv(filepath)
                    catalog_validation[filename] = {
                        "exists": True,
                        "num_rows": len(df),
                        "num_columns": len(df.columns),
                        "columns": list(df.columns)
                    }
                    logger.info(f"Catalog {filename} exists with {len(df)} rows and {len(df.columns)} columns")
                except Exception as e:
                    catalog_validation[filename] = {
                        "exists": True,
                        "error": str(e)
                    }
                    logger.error(f"Error validating catalog {filename}: {str(e)}")
            else:
                catalog_validation[filename] = {
                    "exists": False
                }
                logger.warning(f"Catalog {filename} does not exist")
        
        return catalog_validation
    
    def validate_light_curves(self):
        """
        Validate the downloaded and preprocessed light curves.
        
        Returns:
        --------
        dict
            Dictionary containing validation results
        """
        logger.info("Validating light curves")
        
        # Count raw light curves
        raw_lc_files = []
        for root, dirs, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith("_lc.fits"):
                    raw_lc_files.append(os.path.join(root, file))
        
        # Count processed light curves
        processed_lc_files = []
        for root, dirs, files in os.walk(self.processed_dir):
            for file in files:
                if file.endswith("_lc_processed.fits"):
                    processed_lc_files.append(os.path.join(root, file))
        
        # Count unique TIC IDs
        raw_tic_ids = set()
        processed_tic_ids = set()
        
        for file in raw_lc_files:
            try:
                tic_id = int(os.path.basename(os.path.dirname(file)).split('_')[1])
                raw_tic_ids.add(tic_id)
            except (IndexError, ValueError):
                pass
        
        for file in processed_lc_files:
            try:
                tic_id = int(os.path.basename(os.path.dirname(file)).split('_')[1])
                processed_tic_ids.add(tic_id)
            except (IndexError, ValueError):
                pass
        
        # Validate a sample of processed light curves
        sample_size = min(10, len(processed_lc_files))
        if sample_size > 0:
            sample_files = np.random.choice(processed_lc_files, sample_size, replace=False)
            
            sample_validation = []
            for file in sample_files:
                try:
                    lc = lk.read(file)
                    sample_validation.append({
                        "file": os.path.basename(file),
                        "num_points": len(lc),
                        "time_range": [lc.time.value.min(), lc.time.value.max()],
                        "flux_median": np.median(lc.flux.value),
                        "valid": True
                    })
                except Exception as e:
                    sample_validation.append({
                        "file": os.path.basename(file),
                        "error": str(e),
                        "valid": False
                    })
        else:
            sample_validation = []
        
        # Compile results
        lc_validation = {
            "raw_light_curves": {
                "count": len(raw_lc_files),
                "unique_tic_ids": len(raw_tic_ids)
            },
            "processed_light_curves": {
                "count": len(processed_lc_files),
                "unique_tic_ids": len(processed_tic_ids)
            },
            "sample_validation": sample_validation
        }
        
        logger.info(f"Found {len(raw_lc_files)} raw light curves from {len(raw_tic_ids)} unique TIC IDs")
        logger.info(f"Found {len(processed_lc_files)} processed light curves from {len(processed_tic_ids)} unique TIC IDs")
        
        return lc_validation
    
    def validate_transit_windows(self):
        """
        Validate the extracted transit and non-transit windows.
        
        Returns:
        --------
        dict
            Dictionary containing validation results
        """
        logger.info("Validating transit windows")
        
        # Count transit windows
        transit_files = glob.glob(os.path.join(self.transit_dir, "*.npz"))
        non_transit_files = glob.glob(os.path.join(self.non_transit_dir, "*.npz"))
        
        # Validate a sample of transit windows
        transit_sample_size = min(10, len(transit_files))
        if transit_sample_size > 0:
            transit_sample_files = np.random.choice(transit_files, transit_sample_size, replace=False)
            
            transit_sample_validation = []
            for file in transit_sample_files:
                try:
                    data = np.load(file)
                    transit_sample_validation.append({
                        "file": os.path.basename(file),
                        "tic_id": int(data['tic_id']),
                        "sector": int(data['sector']),
                        "num_points": len(data['time']),
                        "label": int(data['label']),
                        "valid": True
                    })
                except Exception as e:
                    transit_sample_validation.append({
                        "file": os.path.basename(file),
                        "error": str(e),
                        "valid": False
                    })
        else:
            transit_sample_validation = []
        
        # Validate a sample of non-transit windows
        non_transit_sample_size = min(10, len(non_transit_files))
        if non_transit_sample_size > 0:
            non_transit_sample_files = np.random.choice(non_transit_files, non_transit_sample_size, replace=False)
            
            non_transit_sample_validation = []
            for file in non_transit_sample_files:
                try:
                    data = np.load(file)
                    non_transit_sample_validation.append({
                        "file": os.path.basename(file),
                        "tic_id": int(data['tic_id']),
                        "sector": int(data['sector']),
                        "num_points": len(data['time']),
                        "label": int(data['label']),
                        "valid": True
                    })
                except Exception as e:
                    non_transit_sample_validation.append({
                        "file": os.path.basename(file),
                        "error": str(e),
                        "valid": False
                    })
        else:
            non_transit_sample_validation = []
        
        # Compile results
        window_validation = {
            "transit_windows": {
                "count": len(transit_files),
                "sample_validation": transit_sample_validation
            },
            "non_transit_windows": {
                "count": len(non_transit_files),
                "sample_validation": non_transit_sample_validation
            }
        }
        
        logger.info(f"Found {len(transit_files)} transit windows and {len(non_transit_files)} non-transit windows")
        
        return window_validation
    
    def plot_data_distribution(self):
        """
        Plot the distribution of data across TIC IDs and sectors.
        
        Returns:
        --------
        None
        """
        logger.info("Plotting data distribution")
        
        # Count light curves per TIC ID and sector
        tic_counts = {}
        sector_counts = {}
        
        # Process raw light curves
        for root, dirs, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith("_lc.fits"):
                    try:
                        tic_id = int(os.path.basename(os.path.dirname(os.path.join(root, file))).split('_')[1])
                        sector = int(os.path.basename(file).split('_')[1])
                        
                        if tic_id not in tic_counts:
                            tic_counts[tic_id] = 0
                        tic_counts[tic_id] += 1
                        
                        if sector not in sector_counts:
                            sector_counts[sector] = 0
                        sector_counts[sector] += 1
                    except (IndexError, ValueError):
                        pass
        
        # Plot light curves per TIC ID
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(list(tic_counts.values()), bins=10)
        plt.xlabel("Number of light curves")
        plt.ylabel("Number of TIC IDs")
        plt.title("Light curves per TIC ID")
        
        # Plot light curves per sector
        plt.subplot(1, 2, 2)
        sectors = sorted(sector_counts.keys())
        counts = [sector_counts[sector] for sector in sectors]
        plt.bar(sectors, counts)
        plt.xlabel("Sector")
        plt.ylabel("Number of light curves")
        plt.title("Light curves per sector")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.validation_dir, "data_distribution.png"))
        plt.close()
        
        # Plot transit and non-transit window distribution
        transit_files = glob.glob(os.path.join(self.transit_dir, "*.npz"))
        non_transit_files = glob.glob(os.path.join(self.non_transit_dir, "*.npz"))
        
        transit_tic_counts = {}
        non_transit_tic_counts = {}
        
        for file in transit_files:
            try:
                tic_id = int(os.path.basename(file).split('_')[1])
                if tic_id not in transit_tic_counts:
                    transit_tic_counts[tic_id] = 0
                transit_tic_counts[tic_id] += 1
            except (IndexError, ValueError):
                pass
        
        for file in non_transit_files:
            try:
                tic_id = int(os.path.basename(file).split('_')[1])
                if tic_id not in non_transit_tic_counts:
                    non_transit_tic_counts[tic_id] = 0
                non_transit_tic_counts[tic_id] += 1
            except (IndexError, ValueError):
                pass
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(list(transit_tic_counts.values()), bins=10)
        plt.xlabel("Number of transit windows")
        plt.ylabel("Number of TIC IDs")
        plt.title("Transit windows per TIC ID")
        
        plt.subplot(1, 2, 2)
        plt.hist(list(non_transit_tic_counts.values()), bins=10)
        plt.xlabel("Number of non-transit windows")
        plt.ylabel("Number of TIC IDs")
        plt.title("Non-transit windows per TIC ID")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.validation_dir, "window_distribution.png"))
        plt.close()
        
        logger.info(f"Saved data distribution plots to {self.validation_dir}")
    
    def plot_example_light_curves(self, num_examples=3):
        """
        Plot example light curves from the raw and processed data.
        
        Parameters:
        -----------
        num_examples : int
            Number of examples to plot
            
        Returns:
        --------
        None
        """
        logger.info("Plotting example light curves")
        
        # Get processed light curve files
        processed_lc_files = []
        for root, dirs, files in os.walk(self.processed_dir):
            for file in files:
                if file.endswith("_lc_processed.fits"):
                    processed_lc_files.append(os.path.join(root, file))
        
        if len(processed_lc_files) == 0:
            logger.warning("No processed light curve files found")
            return
        
        # Sample random examples
        sample_files = np.random.choice(processed_lc_files, min(num_examples, len(processed_lc_files)), replace=False)
        
        # Plot each example
        for i, file in enumerate(sample_files):
            try:
                # Extract TIC ID and sector from filename
                tic_id = int(os.path.basename(os.path.dirname(file)).split('_')[1])
                sector = int(os.path.basename(file).split('_')[1])
                
                # Load the processed light curve
                processed_lc = lk.read(file)
                
                # Find the corresponding raw light curve
                raw_file = os.path.join(self.raw_dir, f"TIC_{tic_id}", f"sector_{sector}_lc.fits")
                if os.path.exists(raw_file):
                    raw_lc = lk.read(raw_file)
                    
                    # Plot raw and processed light curves
                    plt.figure(figsize=(12, 8))
                    
                    # Raw light curve
                    plt.subplot(2, 1, 1)
                    plt.plot(raw_lc.time.value, raw_lc.flux.value, 'b.', markersize=1)
                    plt.xlabel("Time (BTJD)")
                    plt.ylabel("Flux")
                    plt.title(f"TIC {tic_id}, Sector {sector} - Raw Light Curve")
                    
                    # Processed light curve
                    plt.subplot(2, 1, 2)
                    plt.plot(processed_lc.time.value, processed_lc.flux.value, 'r.', markersize=1)
                    plt.xlabel("Time (BTJD)")
                    plt.ylabel("Normalized Flux")
                    plt.title(f"TIC {tic_id}, Sector {sector} - Processed Light Curve")
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.validation_dir, f"example_lc_TIC_{tic_id}_sector_{sector}.png"))
                    plt.close()
                    
                    logger.info(f"Plotted example light curve for TIC {tic_id}, Sector {sector}")
                else:
                    logger.warning(f"Raw light curve not found for TIC {tic_id}, Sector {sector}")
            
            except Exception as e:
                logger.error(f"Error plotting example light curve {file}: {str(e)}")
    
    def run_validation(self):
        """
        Run the complete validation process.
        
        Returns:
        --------
        dict
            Dictionary containing validation results
        """
        logger.info("Starting validation process")
        
        # Step 1: Validate directory structure
        dir_validation = self.validate_directory_structure()
        
        # Step 2: Validate catalogs
        catalog_validation = self.validate_catalogs()
        
        # Step 3: Validate light curves
        lc_validation = self.validate_light_curves()
        
        # Step 4: Validate transit windows
        window_validation = self.validate_transit_windows()
        
        # Step 5: Plot data distribution
        self.plot_data_distribution()
        
        # Step 6: Plot example light curves
        self.plot_example_light_curves()
        
        # Compile validation results
        validation_results = {
            "directory_structure": dir_validation,
            "catalogs": catalog_validation,
            "light_curves": lc_validation,
            "transit_windows": window_validation
        }
        
        # Save validation results to file
        with open(os.path.join(self.validation_dir, "validation_results.json"), "w") as f:
            json.dump(validation_results, f, indent=4)
        
        # Generate validation summary
        validation_summary = {
            "directories_exist": all(dir_validation.values()),
            "catalogs_exist": all(cat["exists"] for cat in catalog_validation.values()),
            "raw_light_curves_count": lc_validation["raw_light_curves"]["count"],
            "processed_light_curves_count": lc_validation["processed_light_curves"]["count"],
            "transit_windows_count": window_validation["transit_windows"]["count"],
            "non_transit_windows_count": window_validation["non_transit_windows"]["count"]
        }
        
        # Save validation summary to file
        with open(os.path.join(self.validation_dir, "validation_summary.txt"), "w") as f:
            for key, value in validation_summary.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Validation process completed")
        logger.info(f"Summary: {validation_summary}")
        
        return validation_results


if __name__ == "__main__":
    # Run validation
    validator = DataValidator()
    validation_results = validator.run_validation()
    print(json.dumps(validation_results, indent=4))
