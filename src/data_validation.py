#!/usr/bin/env python3
"""
Exo-Detector: Data Validation Module

This module validates the data produced by the preprocessing pipeline,
ensuring that it meets the requirements for the subsequent phases.

Author: Manus AI
Date: May 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataValidator:
    """Class for validating processed data."""
    
    def __init__(self, data_dir="data"):
        """
        Initialize the data validator.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing data
        """
        # Convert to absolute path
        self.data_dir = os.path.abspath(data_dir)
        
        # Define directories
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.catalog_dir = os.path.join(self.data_dir, "catalogs")
        self.transit_windows_dir = os.path.join(self.data_dir, "transit_windows")
        self.non_transit_windows_dir = os.path.join(self.data_dir, "non_transit_windows")
        self.validation_dir = os.path.join(self.data_dir, "validation")
        
        # Create validation directory
        os.makedirs(self.validation_dir, exist_ok=True)
        
        logger.info(f"Initialized DataValidator with data_dir={data_dir}")
    
    def validate_directory_structure(self):
        """
        Validate the directory structure.
        
        Returns:
        --------
        bool
            True if all required directories exist
        """
        required_dirs = [
            self.raw_dir,
            self.processed_dir,
            self.catalog_dir,
            self.transit_windows_dir,
            self.non_transit_windows_dir
        ]
        
        all_exist = True
        
        for directory in required_dirs:
            if os.path.exists(directory):
                logger.info(f"Directory {os.path.basename(directory)} exists")
            else:
                logger.error(f"Directory {os.path.basename(directory)} does not exist")
                all_exist = False
        
        return all_exist
    
    def validate_catalogs(self):
        """
        Validate the catalog files.
        
        Returns:
        --------
        bool
            True if all required catalogs exist
        """
        required_catalogs = [
            "confirmed_planets.csv",
            "toi_catalog.csv",
            "transit_parameters.csv"
        ]
        
        all_exist = True
        
        for catalog in required_catalogs:
            filepath = os.path.join(self.catalog_dir, catalog)
            
            try:
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    logger.info(f"Catalog {catalog} exists with {len(df)} rows and {len(df.columns)} columns")
                else:
                    logger.error(f"Catalog {catalog} does not exist")
                    all_exist = False
            except Exception as e:
                logger.error(f"Error validating catalog {catalog}: {str(e)}")
                all_exist = False
        
        return all_exist
    
    def validate_light_curves(self):
        """
        Validate the light curve files.
        
        Returns:
        --------
        tuple
            (raw_count, processed_count, raw_tic_ids, processed_tic_ids)
        """
        # Find raw light curves (CSV format)
        raw_files = glob.glob(os.path.join(self.raw_dir, "TIC_*", "sector_*_lc.csv"))
        
        # Find processed light curves
        processed_files = glob.glob(os.path.join(self.processed_dir, "TIC_*", "sector_*_lc.csv"))
        
        # Extract TIC IDs
        raw_tic_ids = set()
        for filepath in raw_files:
            dirname = os.path.dirname(filepath)
            tic_dirname = os.path.basename(dirname)
            if "TIC_" in tic_dirname:
                tic_id = int(tic_dirname.split("_")[1])
                raw_tic_ids.add(tic_id)
        
        processed_tic_ids = set()
        for filepath in processed_files:
            dirname = os.path.dirname(filepath)
            tic_dirname = os.path.basename(dirname)
            if "TIC_" in tic_dirname:
                tic_id = int(tic_dirname.split("_")[1])
                processed_tic_ids.add(tic_id)
        
        logger.info(f"Found {len(raw_files)} raw light curves from {len(raw_tic_ids)} unique TIC IDs")
        logger.info(f"Found {len(processed_files)} processed light curves from {len(processed_tic_ids)} unique TIC IDs")
        
        return len(raw_files), len(processed_files), raw_tic_ids, processed_tic_ids
    
    def validate_transit_windows(self):
        """
        Validate the transit window files.
        
        Returns:
        --------
        tuple
            (transit_count, non_transit_count)
        """
        # Find transit windows
        transit_files = glob.glob(os.path.join(self.transit_windows_dir, "*.csv"))
        
        # Find non-transit windows
        non_transit_files = glob.glob(os.path.join(self.non_transit_windows_dir, "*.csv"))
        
        logger.info(f"Found {len(transit_files)} transit windows and {len(non_transit_files)} non-transit windows")
        
        return len(transit_files), len(non_transit_files)
    
    def plot_data_distribution(self):
        """
        Plot the distribution of data.
        
        Returns:
        --------
        list
            List of saved plot files
        """
        plot_files = []
        
        # Create figure for TIC ID distribution
        plt.figure(figsize=(10, 6))
        
        # Find all processed light curves
        processed_files = glob.glob(os.path.join(self.processed_dir, "TIC_*", "sector_*_lc.csv"))
        
        # Extract TIC IDs
        tic_ids = []
        for filepath in processed_files:
            dirname = os.path.dirname(filepath)
            tic_dirname = os.path.basename(dirname)
            if "TIC_" in tic_dirname:
                tic_id = int(tic_dirname.split("_")[1])
                tic_ids.append(tic_id)
        
        if tic_ids:
            plt.hist(tic_ids, bins=20)
            plt.xlabel('TIC ID')
            plt.ylabel('Count')
            plt.title('Distribution of TIC IDs')
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.validation_dir, "tic_id_distribution.png")
            plt.savefig(plot_file)
            plot_files.append(plot_file)
        
        plt.close()
        
        # Create figure for transit parameters
        plt.figure(figsize=(12, 8))
        
        # Load transit parameters
        transit_params_file = os.path.join(self.catalog_dir, "transit_parameters.csv")
        
        if os.path.exists(transit_params_file):
            try:
                transit_params = pd.read_csv(transit_params_file)
                
                if len(transit_params) > 0:
                    # Create subplots
                    plt.subplot(2, 2, 1)
                    plt.hist(transit_params['period'], bins=20)
                    plt.xlabel('Period (days)')
                    plt.ylabel('Count')
                    plt.title('Period Distribution')
                    
                    plt.subplot(2, 2, 2)
                    plt.hist(transit_params['duration'], bins=20)
                    plt.xlabel('Duration (hours)')
                    plt.ylabel('Count')
                    plt.title('Duration Distribution')
                    
                    plt.subplot(2, 2, 3)
                    plt.hist(transit_params['depth'], bins=20)
                    plt.xlabel('Depth')
                    plt.ylabel('Count')
                    plt.title('Depth Distribution')
                    
                    plt.subplot(2, 2, 4)
                    plt.scatter(transit_params['period'], transit_params['depth'], alpha=0.5)
                    plt.xlabel('Period (days)')
                    plt.ylabel('Depth')
                    plt.title('Period vs. Depth')
                    
                    plt.tight_layout()
                    
                    # Save plot
                    plot_file = os.path.join(self.validation_dir, "transit_parameters_distribution.png")
                    plt.savefig(plot_file)
                    plot_files.append(plot_file)
            
            except Exception as e:
                logger.error(f"Error plotting transit parameters: {str(e)}")
        
        plt.close()
        
        logger.info(f"Saved data distribution plots to {self.validation_dir}")
        
        return plot_files
    
    def plot_example_light_curves(self, num_examples=5):
        """
        Plot example light curves.
        
        Parameters:
        -----------
        num_examples : int
            Number of example light curves to plot
            
        Returns:
        --------
        list
            List of saved plot files
        """
        plot_files = []
        
        # Find all processed light curves
        processed_files = glob.glob(os.path.join(self.processed_dir, "TIC_*", "sector_*_lc.csv"))
        
        if not processed_files:
            logger.warning("No processed light curve files found")
            return plot_files
        
        # Select random examples
        if len(processed_files) > num_examples:
            example_files = np.random.choice(processed_files, num_examples, replace=False)
        else:
            example_files = processed_files
        
        # Plot each example
        for filepath in example_files:
            try:
                # Extract TIC ID and sector from filename
                dirname = os.path.dirname(filepath)
                tic_dirname = os.path.basename(dirname)
                filename = os.path.basename(filepath)
                
                if "TIC_" in tic_dirname:
                    tic_id = int(tic_dirname.split("_")[1])
                else:
                    tic_id = 0
                
                if "sector_" in filename:
                    sector = int(filename.split("_")[1].split(".")[0])
                else:
                    sector = 0
                
                # Load light curve
                df = pd.read_csv(filepath)
                
                # Create figure
                plt.figure(figsize=(12, 6))
                
                # Plot light curve
                plt.errorbar(df['time'], df['flux'], yerr=df['flux_err'], fmt='.', alpha=0.5)
                plt.xlabel('Time (BTJD)')
                plt.ylabel('Normalized Flux')
                plt.title(f'TIC {tic_id} - Sector {sector}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save plot
                plot_file = os.path.join(self.validation_dir, f"TIC_{tic_id}_sector_{sector}_lc.png")
                plt.savefig(plot_file)
                plot_files.append(plot_file)
                
                plt.close()
            
            except Exception as e:
                logger.error(f"Error plotting light curve {filepath}: {str(e)}")
        
        return plot_files
    
    def run_validation(self):
        """
        Run the complete validation process.
        
        Returns:
        --------
        dict
            Dictionary containing validation results
        """
        logger.info("Starting validation process")
        
        # Validate directory structure
        logger.info("Validating directory structure")
        directories_exist = self.validate_directory_structure()
        
        # Validate catalogs
        logger.info("Validating catalogs")
        catalogs_exist = self.validate_catalogs()
        
        # Validate light curves
        logger.info("Validating light curves")
        raw_count, processed_count, raw_tic_ids, processed_tic_ids = self.validate_light_curves()
        
        # Validate transit windows
        logger.info("Validating transit windows")
        transit_count, non_transit_count = self.validate_transit_windows()
        
        # Plot data distribution
        logger.info("Plotting data distribution")
        self.plot_data_distribution()
        
        # Plot example light curves
        logger.info("Plotting example light curves")
        self.plot_example_light_curves()
        
        # Compile validation results
        validation_results = {
            'directories_exist': directories_exist,
            'catalogs_exist': catalogs_exist,
            'raw_light_curves_count': raw_count,
            'processed_light_curves_count': processed_count,
            'transit_windows_count': transit_count,
            'non_transit_windows_count': non_transit_count
        }
        
        logger.info("Validation process completed")
        logger.info(f"Summary: {validation_results}")
        
        return validation_results


if __name__ == "__main__":
    # Run validation
    validator = DataValidator()
    results = validator.run_validation()
    print(results)
