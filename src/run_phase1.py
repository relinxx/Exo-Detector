#!/usr/bin/env python3
"""
Exo-Detector: Phase 1 Runner Script
"""
import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def run_phase1(data_dir="data", num_stars=5):
    """Run the complete Phase 1 pipeline."""
    from data_ingestion import run_data_ingestion
    # These are placeholders for your other pipeline steps.
    # from data_preprocessing import TESSDataPreprocessing
    # from data_validation import DataValidator

    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Data Ingestion
    logger.info("Running data ingestion")
    ingestion_results = run_data_ingestion(
        data_dir=data_dir,
        num_stars=num_stars,
        use_real_data=True,
        download_tess=True
    )
    logger.info(f"Data ingestion completed: {ingestion_results}")

    # Subsequent steps would follow here.
    # logger.info("Running data preprocessing")
    # ...
    # logger.info("Running data validation")
    # ...

    logger.info("Phase 1 pipeline completed successfully.")
    return ingestion_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Exo-Detector Phase 1 pipeline')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to store data')
    parser.add_argument('--num-stars', type=int, default=5, help='Number of stars to download.')
    args = parser.parse_args()

    logger.info("Starting Phase 1 pipeline")
    summary = run_phase1(
        data_dir=args.data_dir,
        num_stars=args.num_stars
    )
    print(f"Pipeline Summary: {summary}")
