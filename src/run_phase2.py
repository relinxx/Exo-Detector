#!/usr/bin/env python3
"""
Exo-Detector: Phase 2 Runner Script

This script runs the complete Phase 2 pipeline:
1. GAN-based transit augmentation
2. Validation of synthetic data

Author: Manus AI
Date: May 2025
"""

import os
import sys
import logging
import argparse
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

def run_phase2(data_dir="data", num_epochs=100, num_synthetic_samples=1000):
    """
    Run the complete Phase 2 pipeline.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data
    num_epochs : int
        Number of epochs to train the GAN
    num_synthetic_samples : int
        Number of synthetic samples to generate
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    logger.info("Starting Phase 2 pipeline")
    
    # Create absolute path for data directory
    data_dir = os.path.abspath(data_dir)
    
    # Import modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: GAN-based transit augmentation
    logger.info("Running GAN-based transit augmentation")
    from gan_module import TransitGAN
    
    gan = TransitGAN(data_dir=data_dir)
    pipeline_results = gan.run_gan_pipeline(num_epochs=num_epochs, num_synthetic_samples=num_synthetic_samples)
    
    # Compile pipeline results
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_epochs": num_epochs,
        "num_synthetic_samples": num_synthetic_samples,
        "steps_executed": {
            "gan_training": True,
            "synthetic_generation": True
        },
        "gan_pipeline_results": pipeline_results
    }
    
    # Save results for dashboard compatibility
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "phase2_results.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    logger.info("Phase 2 pipeline completed")
    
    return summary

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Exo-Detector Phase 2 pipeline")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the GAN")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples to generate")
    
    args = parser.parse_args()
    
    # Run the pipeline
    summary = run_phase2(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        num_synthetic_samples=args.samples
    )
    
    print(f"Phase 2 pipeline completed: {summary}")
