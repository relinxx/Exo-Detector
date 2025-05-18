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
        logging.FileHandler(os.path.join(log_dir, f"phase2_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_phase2(num_epochs=100, num_synthetic_samples=1000):
    """
    Run the complete Phase 2 pipeline.
    
    Parameters:
    -----------
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
    
    # Import modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: GAN-based transit augmentation
    logger.info("Running GAN-based transit augmentation")
    from gan_module import TransitGAN
    
    gan = TransitGAN()
    pipeline_results = gan.run_gan_pipeline(num_epochs=num_epochs, num_synthetic_samples=num_synthetic_samples)
    
    logger.info("Phase 2 pipeline completed")
    
    # Compile summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_epochs": num_epochs,
        "num_synthetic_samples": num_synthetic_samples,
        "gan_pipeline_results": pipeline_results
    }
    
    return summary

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Exo-Detector Phase 2 pipeline")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the GAN")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples to generate")
    
    args = parser.parse_args()
    
    # Run the pipeline
    summary = run_phase2(
        num_epochs=args.epochs,
        num_synthetic_samples=args.samples
    )
    
    print(f"Phase 2 pipeline completed: {summary}")
