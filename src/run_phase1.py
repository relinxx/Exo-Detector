#!/usr/bin/env python3
"""
Exo-Detector: Phase 1 Runner Script

This script runs the complete Phase 1 pipeline, including data ingestion,
preprocessing, and validation.

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

def run_phase1(data_dir="data", sectors=[1, 2, 3, 4, 5], limit=None):
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
    
    # Run data ingestion
    logger.info("Running data ingestion")
    
    # Determine number of stars based on limit
    num_stars = 20 if limit is None else min(limit, 20)
    
    ingestion_results = run_data_ingestion(
        data_dir=data_dir, 
        sectors=sectors, 
        num_stars=num_stars, 
        max_sectors_per_star=3
    )
    
    logger.info(f"Data ingestion completed: {ingestion_results}")
    
    # Run data preprocessing
    logger.info("Running data preprocessing")
    preprocessing = TESSDataPreprocessing(data_dir=data_dir)
    preprocessing_results = preprocessing.run_preprocessing_pipeline(limit=limit)
    
    logger.info(f"Data preprocessing completed: {preprocessing_results}")
    
    # Run data validation
    logger.info("Running data validation")
    validator = DataValidator(data_dir=data_dir)
    validation_results = validator.run_validation()
    
    logger.info("Data validation completed")
    
    # Compile pipeline results
    pipeline_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'limit': limit,
        'steps_executed': {
            'ingestion': True,
            'preprocessing': True,
            'validation': True
        },
        'ingestion': ingestion_results,
        'preprocessing': preprocessing_results
    }
    
    return pipeline_results

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Exo-Detector Phase 1 pipeline')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to store data')
    parser.add_argument('--sectors', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='TESS sectors to process')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of targets to process')
    
    args = parser.parse_args()
    
    # Run Phase 1 pipeline
    logger.info("Starting Phase 1 pipeline")
    summary = run_phase1(
        data_dir=args.data_dir,
        sectors=args.sectors,
        limit=args.limit
    )
    
    logger.info("Phase 1 pipeline completed")
    print(f"Phase 1 pipeline completed: {summary}")
