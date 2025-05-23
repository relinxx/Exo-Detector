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

def run_phase3(data_dir="data", window_size=200, latent_dim=8, batch_size=128, epochs=50, limit=None):
    """
    Run the complete Phase 3 pipeline.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data
    window_size : int
        Size of input window
    latent_dim : int
        Size of latent dimension
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs to train the autoencoder
    limit : int or None
        Maximum number of files to load per class
        
    Returns:
    --------
    dict
        Dictionary containing pipeline results
    """
    # Import modules
    from anomaly_detection import run_anomaly_detection
    
    # Create absolute path for data directory
    data_dir = os.path.abspath(data_dir)
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 1: Anomaly Detection
    logger.info("Running anomaly detection")
    
    # Run anomaly detection
    anomaly_results = run_anomaly_detection(
        data_dir=data_dir,
        window_size=window_size,
        latent_dim=latent_dim,
        batch_size=batch_size,
        epochs=epochs,
        limit=limit
    )
    
    logger.info(f"Anomaly detection completed: {anomaly_results}")
    
    # Compile pipeline results
    pipeline_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'window_size': window_size,
        'latent_dim': latent_dim,
        'batch_size': batch_size,
        'epochs': epochs,
        'limit': limit,
        'steps_executed': {
            'anomaly_detection': True
        },
        'anomaly_detection': anomaly_results
    }
    
    # Save results
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "phase3_results.json"), "w") as f:
        json.dump(pipeline_results, f, indent=4)
    
    logger.info("Phase 3 pipeline completed")
    
    return pipeline_results

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Exo-Detector Phase 3 pipeline')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing data')
    parser.add_argument('--window-size', type=int, default=200, help='Size of input window')
    parser.add_argument('--latent-dim', type=int, default=8, help='Size of latent dimension')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the autoencoder')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of files to load per class')
    args = parser.parse_args()
    
    # Run Phase 3 pipeline
    logger.info("Starting Phase 3 pipeline")
    summary = run_phase3(
        data_dir=args.data_dir,
        window_size=args.window_size,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        limit=args.limit
    )
    
    print(f"Phase 3 pipeline completed: {summary}")
