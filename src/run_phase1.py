#!/usr/bin/env python3
"""
Exo-Detector: Phase 1 Runner Script

This script runs the complete Phase 1 pipeline, including data ingestion,
preprocessing, and validation. It orchestrates the execution of all Phase 1
components and provides a unified interface.

This version works with CSV-based synthetic light curves for maximum compatibility.

Author: Manus AI
Date: May 2025
"""

import os
import sys
import argparse
import logging
import json
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

def run_phase1(data_dir="data", sectors=[1, 2, 3, 4, 5], limit=None, max_tic_id=100000):
    """
    Run the complete Phase 1 pipeline.
    
    Parameters:
    -----------
    data_dir : str
        Directory to store data
    sectors : list
        List of TESS sectors to process
    limit : int or None
        Maximum number of targets to process
    max_tic_id : int
        Maximum TIC ID to consider
        
    Returns:
    --------
    dict
        Dictionary containing pipeline results
    """
    # Import modules
    from data_ingestion import run_data_ingestion
    from data_preprocessing import TESSDataPreprocessing
    from data_validation import DataValidator
    
    # Create absolute path for data directory
    data_dir = os.path.abspath(data_dir)
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 1: Data Ingestion
    logger.info("Running data ingestion")
    
    # Determine number of stars based on limit
    num_stars = 20 if limit is None else min(limit, 20)
    
    # Run data ingestion
    ingestion_results = run_data_ingestion(
        data_dir=data_dir,
        sectors=sectors,
        num_stars=num_stars,
        max_sectors_per_star=3
    )
    
    logger.info(f"Data ingestion completed: {ingestion_results}")
    
    # Step 2: Data Preprocessing
    logger.info("Running data preprocessing")
    
    # Initialize preprocessing
    preprocessing = TESSDataPreprocessing(data_dir=data_dir)
    
    # Run preprocessing pipeline
    preprocessing_results = preprocessing.run_preprocessing_pipeline(limit=limit)
    
    logger.info(f"Data preprocessing completed: {preprocessing_results}")
    
    # Step 3: Data Validation
    logger.info("Running data validation")
    
    # Initialize validation
    validator = DataValidator(data_dir=data_dir)
    
    # Run validation
    validation_results = validator.run_validation()
    
    logger.info("Data validation completed")
    
    # Compile pipeline results
    pipeline_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'limit': limit,
        'steps_executed': {
            'data_ingestion': True,
            'preprocessing': True,
            'validation': True
        },
        'ingestion': ingestion_results,
        'preprocessing': preprocessing_results
    }
    
    # Save results for dashboard compatibility
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "phase1_results.json"), "w") as f:
        json.dump(pipeline_results, f, indent=4)
    
    logger.info("Phase 1 pipeline completed")
    
    return pipeline_results

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Exo-Detector Phase 1 pipeline')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to store data')
    parser.add_argument('--sectors', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='TESS sectors to process')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of targets to process')
    parser.add_argument('--max-tic-id', type=int, default=100000, help='Maximum TIC ID to consider')
    args = parser.parse_args()
    
    # Run Phase 1 pipeline
    logger.info("Starting Phase 1 pipeline")
    summary = run_phase1(
        data_dir=args.data_dir,
        sectors=args.sectors,
        limit=args.limit,
        max_tic_id=args.max_tic_id
    )
    
    print(f"Phase 1 pipeline completed: {summary}")
