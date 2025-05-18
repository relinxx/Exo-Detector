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
        logging.FileHandler(os.path.join(log_dir, f"phase3_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_phase3(num_epochs=50, nu=0.1):
    """
    Run the complete Phase 3 pipeline.
    
    Parameters:
    -----------
    num_epochs : int
        Number of epochs to train the autoencoder
    nu : float
        Nu parameter for one-class SVM (controls the fraction of outliers)
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    logger.info("Starting Phase 3 pipeline")
    
    # Import modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: Anomaly detection
    logger.info("Running anomaly detection")
    from anomaly_detection import AnomalyDetector
    
    detector = AnomalyDetector()
    pipeline_results = detector.run_anomaly_detection_pipeline(num_epochs=num_epochs, nu=nu)
    
    logger.info("Phase 3 pipeline completed")
    
    # Compile summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_epochs": num_epochs,
        "nu": nu,
        "anomaly_detection_pipeline_results": pipeline_results
    }
    
    return summary

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Exo-Detector Phase 3 pipeline")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train the autoencoder")
    parser.add_argument("--nu", type=float, default=0.1, help="Nu parameter for one-class SVM")
    
    args = parser.parse_args()
    
    # Run the pipeline
    summary = run_phase3(
        num_epochs=args.epochs,
        nu=args.nu
    )
    
    print(f"Phase 3 pipeline completed: {summary}")
