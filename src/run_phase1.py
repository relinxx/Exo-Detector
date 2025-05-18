import os
import sys
import logging
import argparse
from datetime import datetime

# Configure logging
log_dir = "../data/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"phase1_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_phase1(limit=None, skip_ingestion=False, skip_preprocessing=False, skip_validation=False):
    """
    Run the complete Phase 1 pipeline.
    
    Parameters:
    -----------
    limit : int, optional
        Limit the number of targets to process (for testing)
    skip_ingestion : bool
        Skip the data ingestion step
    skip_preprocessing : bool
        Skip the data preprocessing step
    skip_validation : bool
        Skip the data validation step
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    logger.info("Starting Phase 1 pipeline")
    
    # Import modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: Data ingestion
    if not skip_ingestion:
        logger.info("Running data ingestion")
        from data_ingestion import TESSDataIngestion
        
        ingestion = TESSDataIngestion()
        ingestion_summary = ingestion.run_ingestion_pipeline(limit=limit)
        logger.info(f"Data ingestion completed: {ingestion_summary}")
    else:
        logger.info("Skipping data ingestion")
    
    # Step 2: Data preprocessing
    if not skip_preprocessing:
        logger.info("Running data preprocessing")
        from data_preprocessing import TESSDataPreprocessing
        
        preprocessing = TESSDataPreprocessing()
        preprocessing_summary = preprocessing.run_preprocessing_pipeline(limit=limit)
        logger.info(f"Data preprocessing completed: {preprocessing_summary}")
    else:
        logger.info("Skipping data preprocessing")
    
    # Step 3: Data validation
    if not skip_validation:
        logger.info("Running data validation")
        from data_validation import DataValidator
        
        validator = DataValidator()
        validation_results = validator.run_validation()
        logger.info("Data validation completed")
    else:
        logger.info("Skipping data validation")
    
    logger.info("Phase 1 pipeline completed")
    
    # Compile summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "limit": limit,
        "steps_executed": {
            "ingestion": not skip_ingestion,
            "preprocessing": not skip_preprocessing,
            "validation": not skip_validation
        }
    }
    
    # Add step-specific summaries if available
    if not skip_ingestion and 'ingestion_summary' in locals():
        summary["ingestion"] = ingestion_summary
    
    if not skip_preprocessing and 'preprocessing_summary' in locals():
        summary["preprocessing"] = preprocessing_summary
    
    return summary

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Exo-Detector Phase 1 pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of targets to process (for testing)")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip the data ingestion step")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip the data preprocessing step")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the data validation step")
    
    args = parser.parse_args()
    
    # Run the pipeline
    summary = run_phase1(
        limit=args.limit,
        skip_ingestion=args.skip_ingestion,
        skip_preprocessing=args.skip_preprocessing,
        skip_validation=args.skip_validation
    )
    
    print(f"Phase 1 pipeline completed: {summary}")
