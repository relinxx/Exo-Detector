import os
import numpy as np
import pandas as pd
from astroquery.mast import Observations
from astropy.table import Table
import lightkurve as lk
from tqdm import tqdm
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../data/data_ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress lightkurve warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightkurve")

class TESSDataIngestion:
    """Class for handling TESS data ingestion operations."""
    
    def __init__(self, data_dir="../data", max_tic_id=100000, sectors=[1, 2, 3, 4, 5]):
        """
        Initialize the data ingestion module.
        
        Parameters:
        -----------
        data_dir : str
            Directory to store downloaded data
        max_tic_id : int
            Maximum TIC ID to consider (focusing on brighter targets)
        sectors : list
            TESS sectors to download data from
        """
        self.data_dir = data_dir
        self.max_tic_id = max_tic_id
        self.sectors = sectors
        
        # Create data directories if they don't exist
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.catalog_dir = os.path.join(data_dir, "catalogs")
        
        for directory in [self.raw_dir, self.processed_dir, self.catalog_dir]:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"Initialized TESSDataIngestion with max_tic_id={max_tic_id}, sectors={sectors}")
    
    def query_tic_targets(self, limit=None):
        """
        Query MAST for TIC targets with IDs less than max_tic_id.
        
        Parameters:
        -----------
        limit : int, optional
            Limit the number of targets to query (for testing)
            
        Returns:
        --------
        astropy.table.Table
            Table of TIC targets
        """
        logger.info(f"Querying TIC targets with ID < {self.max_tic_id}")
        
        # Define the query criteria
        query_criteria = {
            'target_name': '*',
            'project': 'TESS',
            'obs_collection': 'TESS',
            'dataproduct_type': 'timeseries',
            'calib_level': 2  # We want calibrated light curves
        }
        
        # Execute the query
        obs_table = Observations.query_criteria(**query_criteria)
        
        # Filter for targets with TIC ID < max_tic_id
        tic_ids = []
        for target_name in obs_table['target_name']:
            try:
                # Extract TIC ID from target name (format: "TIC 12345678")
                tic_id = int(target_name.split(' ')[1])
                tic_ids.append(tic_id)
            except (IndexError, ValueError):
                tic_ids.append(np.nan)
        
        obs_table['tic_id'] = tic_ids
        filtered_table = obs_table[~np.isnan(obs_table['tic_id'])]
        filtered_table = filtered_table[filtered_table['tic_id'] < self.max_tic_id]
        
        # Filter for specified sectors
        sector_mask = np.zeros(len(filtered_table), dtype=bool)
        for idx, obs_id in enumerate(filtered_table['obs_id']):
            try:
                # Extract sector from obs_id (format: "tess-s0001-1-1")
                sector = int(obs_id.split('-')[1][1:])
                sector_mask[idx] = sector in self.sectors
            except (IndexError, ValueError):
                sector_mask[idx] = False
        
        filtered_table = filtered_table[sector_mask]
        
        # Apply limit if specified
        if limit is not None:
            filtered_table = filtered_table[:limit]
        
        # Save the query results
        filtered_table.write(os.path.join(self.catalog_dir, 'tic_targets.csv'), format='csv', overwrite=True)
        
        logger.info(f"Found {len(filtered_table)} TIC targets matching criteria")
        return filtered_table
    
    def download_light_curves(self, target_table=None, limit=None):
        """
        Download light curves for the specified targets.
        
        Parameters:
        -----------
        target_table : astropy.table.Table, optional
            Table of targets to download. If None, will use query_tic_targets()
        limit : int, optional
            Limit the number of light curves to download (for testing)
            
        Returns:
        --------
        list
            List of paths to downloaded light curve files
        """
        if target_table is None:
            target_table = self.query_tic_targets()
        
        if limit is not None:
            target_table = target_table[:limit]
        
        logger.info(f"Downloading light curves for {len(target_table)} targets")
        
        # Create a list to store the paths to downloaded files
        downloaded_files = []
        
        # Download light curves for each target
        for idx, row in enumerate(tqdm(target_table, desc="Downloading light curves")):
            try:
                target_name = row['target_name']
                tic_id = int(target_name.split(' ')[1])
                
                # Create a directory for this target
                target_dir = os.path.join(self.raw_dir, f"TIC_{tic_id}")
                os.makedirs(target_dir, exist_ok=True)
                
                # Use lightkurve to search for and download the light curve
                search_result = lk.search_lightcurve(target_name, mission='TESS')
                
                # Filter for 2-minute cadence data
                search_result = search_result[search_result.exptime.value == 120]
                
                # Filter for specified sectors
                sector_mask = np.zeros(len(search_result), dtype=bool)
                for i, s in enumerate(search_result.sector):
                    if s in self.sectors:
                        sector_mask[i] = True
                
                search_result = search_result[sector_mask]
                
                if len(search_result) == 0:
                    logger.warning(f"No 2-minute cadence data found for {target_name} in sectors {self.sectors}")
                    continue
                
                # Download each light curve
                for i, product in enumerate(search_result):
                    try:
                        sector = product.sector
                        lc_file = os.path.join(target_dir, f"sector_{sector}_lc.fits")
                        
                        # Skip if file already exists
                        if os.path.exists(lc_file):
                            logger.info(f"File already exists: {lc_file}")
                            downloaded_files.append(lc_file)
                            continue
                        
                        # Download the light curve
                        lc = product.download()
                        
                        # Save the light curve to a FITS file
                        lc.to_fits(lc_file, overwrite=True)
                        downloaded_files.append(lc_file)
                        
                        logger.info(f"Downloaded light curve for {target_name}, sector {sector}")
                    except Exception as e:
                        logger.error(f"Error downloading light curve for {target_name}, sector {sector}: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error processing target {row['target_name']}: {str(e)}")
        
        logger.info(f"Downloaded {len(downloaded_files)} light curves")
        
        # Save the list of downloaded files
        with open(os.path.join(self.catalog_dir, 'downloaded_files.txt'), 'w') as f:
            for file_path in downloaded_files:
                f.write(f"{file_path}\n")
        
        return downloaded_files
    
    def download_exoplanet_catalog(self):
        """
        Download the NASA Exoplanet Archive catalog of confirmed planets and TOIs.
        
        Returns:
        --------
        tuple
            (confirmed_planets_df, toi_df) - DataFrames containing catalog data
        """
        logger.info("Downloading exoplanet catalogs")
        
        # Download confirmed planets catalog
        confirmed_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv"
        confirmed_file = os.path.join(self.catalog_dir, "confirmed_planets.csv")
        
        try:
            confirmed_df = pd.read_csv(confirmed_url)
            confirmed_df.to_csv(confirmed_file, index=False)
            logger.info(f"Downloaded confirmed planets catalog with {len(confirmed_df)} entries")
        except Exception as e:
            logger.error(f"Error downloading confirmed planets catalog: {str(e)}")
            confirmed_df = None
        
        # Download TOI catalog
        toi_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv"
        toi_file = os.path.join(self.catalog_dir, "toi_catalog.csv")
        
        try:
            toi_df = pd.read_csv(toi_url)
            toi_df.to_csv(toi_file, index=False)
            logger.info(f"Downloaded TOI catalog with {len(toi_df)} entries")
        except Exception as e:
            logger.error(f"Error downloading TOI catalog: {str(e)}")
            toi_df = None
        
        return confirmed_df, toi_df
    
    def run_ingestion_pipeline(self, limit=None):
        """
        Run the complete data ingestion pipeline.
        
        Parameters:
        -----------
        limit : int, optional
            Limit the number of targets to process (for testing)
            
        Returns:
        --------
        dict
            Dictionary containing summary statistics
        """
        logger.info("Starting data ingestion pipeline")
        
        # Step 1: Download exoplanet catalogs
        confirmed_df, toi_df = self.download_exoplanet_catalog()
        
        # Step 2: Query TIC targets
        target_table = self.query_tic_targets(limit=limit)
        
        # Step 3: Download light curves
        downloaded_files = self.download_light_curves(target_table=target_table, limit=limit)
        
        # Generate summary statistics
        summary = {
            "num_targets_queried": len(target_table),
            "num_light_curves_downloaded": len(downloaded_files),
            "sectors_downloaded": self.sectors,
            "confirmed_planets_catalog_size": len(confirmed_df) if confirmed_df is not None else 0,
            "toi_catalog_size": len(toi_df) if toi_df is not None else 0
        }
        
        # Save summary to file
        with open(os.path.join(self.data_dir, "ingestion_summary.txt"), "w") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Data ingestion pipeline completed")
        logger.info(f"Summary: {summary}")
        
        return summary


if __name__ == "__main__":
    # Example usage
    ingestion = TESSDataIngestion()
    
    # For testing, limit to a small number of targets
    summary = ingestion.run_ingestion_pipeline(limit=10)
    print(summary)
