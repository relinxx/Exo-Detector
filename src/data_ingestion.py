import os
import re
import time
import logging
import json
from datetime import datetime

import lightkurve as lk
import numpy as np
import pandas as pd
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

# Configure logger for this module
logger = logging.getLogger(__name__)

class RealTransitDataIngestion:
    """
    Handles the download of real exoplanet data from the NASA Exoplanet Archive
    and TESS light curves from the MAST portal.
    """
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.toi_catalog = None

    def load_tess_objects_of_interest(self):
        """
        Downloads a catalog of TESS Objects of Interest (TOIs) that are
        confirmed planets, ensuring data availability.
        """
        try:
            logger.info("Downloading TESS Objects of Interest (TOI) catalog...")
            self.toi_catalog = NasaExoplanetArchive.query_criteria(
                table="toi",
                where="tfopwg_disp = 'CP' and tid is not null",
                select="toi,tid"
            )
            
            if self.toi_catalog is None or len(self.toi_catalog) == 0:
                logger.error("Failed to download TOI catalog. No targets to process.")
                return None

            catalog_dir = os.path.join(self.data_dir, "catalogs")
            os.makedirs(catalog_dir, exist_ok=True)
            self.toi_catalog.to_pandas().to_csv(os.path.join(catalog_dir, "toi_catalog.csv"), index=False)
            
            logger.info(f"Downloaded {len(self.toi_catalog)} confirmed planet TOIs.")
            return self.toi_catalog
        except Exception as e:
            logger.error(f"Failed to load TOI catalog: {e}")
            return None

    def download_tess_lightcurves(self, tic_ids, max_per_star=3):
        """Downloads TESS light curves for a list of TIC IDs."""
        downloaded_data = []
        raw_dir = os.path.join(self.data_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        logger.info(f"Attempting to download TESS data for {len(tic_ids)} unique stars...")
        logger.info(f"TIC IDs to be processed: {tic_ids[:5]}...")

        for tic_id in tic_ids:
            try:
                target_id_str = f"TIC {tic_id}"
                logger.info(f"Processing target: '{target_id_str}'")
                
                search_result = lk.search_lightcurve(target_id_str, mission="TESS", author="SPOC")
                
                if not search_result:
                    logger.warning(f"No SPOC-processed TESS data found for {target_id_str}. Trying other sources.")
                    search_result = lk.search_lightcurve(target_id_str, mission="TESS")
                    if not search_result:
                        logger.error(f"No data found for {target_id_str} from any source.")
                        continue

                downloaded_count = 0
                for item in search_result:
                    if downloaded_count >= max_per_star:
                        break
                    try:
                        # The download itself is working
                        lc = item.download()
                        if lc is None: continue
                        
                        lc = lc.remove_nans().remove_outliers(sigma=5).normalize()
                        
                        # *** DEFINITIVE FIX ***
                        # The error occurs because the code should use the 'lc' object (the
                        # downloaded LightCurve) to get the sector, not the 'item' from the search.
                        save_path = os.path.join(raw_dir, f"TIC_{tic_id}_sector_{lc.sector}.csv")
                        df = pd.DataFrame({
                            'time': lc.time.value, 'flux': lc.flux.value,
                            'flux_err': lc.flux_err.value if hasattr(lc, 'flux_err') and lc.flux_err is not None else np.nan,
                            'tic_id': tic_id, 'sector': lc.sector # Use lc.sector here
                        })
                        df.to_csv(save_path, index=False)
                        
                        downloaded_data.append({'tic_id': tic_id, 'sector': lc.sector, 'file_path': save_path})
                        downloaded_count += 1
                        logger.info(f"SUCCESS: Downloaded and saved {target_id_str} Sector {lc.sector}")
                        time.sleep(1)
                    except Exception as download_e:
                        logger.error(f"Failed during file-save for {target_id_str}: {download_e}")
            except Exception as process_e:
                logger.error(f"Failed to process {target_id_str}: {process_e}")
                
        logger.info(f"Successfully downloaded {len(downloaded_data)} real light curves.")
        return downloaded_data

def run_data_ingestion(data_dir="data", use_real_data=False, download_tess=False, **kwargs):
    """Orchestrates the data ingestion process."""
    results = {'timestamp': datetime.now().isoformat(), 'real_samples': 0}
    if not use_real_data:
        return results

    try:
        logger.info("--- Starting Real Data Ingestion ---")
        ingestion = RealTransitDataIngestion(data_dir)
        targets = ingestion.load_tess_objects_of_interest()

        if targets is not None and download_tess:
            num_stars = kwargs.get('num_stars', 5)
            targets_df = targets.to_pandas()
            tic_ids = targets_df['tid'].dropna().astype(int).unique()
            
            logger.info(f"Selected {num_stars} TIC IDs for download from {len(tic_ids)} available confirmed planets.")
            downloaded = ingestion.download_tess_lightcurves(
                tic_ids[:num_stars],
                max_per_star=kwargs.get('max_sectors_per_star', 3)
            )
            results['real_samples'] = len(downloaded)
    except Exception as e:
        logger.error(f"A critical error occurred during real data ingestion: {e}", exc_info=True)
    
    logger.info(f"Data ingestion summary: {results}")
    return results
