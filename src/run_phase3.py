#!/usr/bin/env python3
"""Phase 3: Anomaly Detection Runner"""

import os
import argparse
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the corrected anomaly detector class
from transformer_anomaly_detection import EnhancedAnomalyDetector

def run_phase3(data_dir="data", epochs=50):
    """Run the complete Phase 3 anomaly detection pipeline."""
    logger.info("--- Starting Phase 3: Anomaly Detection ---")
    data_dir = os.path.abspath(data_dir)

    # Initialize the anomaly detector
    detector = EnhancedAnomalyDetector(data_dir=data_dir)

    # 1. Prepare data (this was the missing step)
    train_loader, test_loader = detector.prepare_data()

    # 2. Train the Transformer Autoencoder
    logger.info("Training the Transformer Autoencoder...")
    detector.train_transformer_ae(train_loader, epochs=epochs)

    # 3. Train the classical models
    logger.info("Training classical anomaly detectors...")
    normal_data_for_classical = detector.get_normal_data_array()
    detector.train_classical_detectors(normal_data_for_classical)

    # 4. Detect anomalies and get scores
    logger.info("Detecting anomalies on the test set...")
    anomaly_scores, true_labels = detector.detect_anomalies(test_loader)

    # --- Post-processing and results (ranking would go here) ---
    # For now, we save the scores.
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_used': 'Transformer AE + Isolation Forest',
        'num_test_samples': len(true_labels),
        'anomaly_scores': anomaly_scores.tolist(),
        'true_labels': true_labels.tolist()
    }

    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "phase3_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Phase 3 pipeline completed successfully.")
    logger.info(f"Saved results to {results_dir}/phase3_results.json")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Anomaly Detection (Phase 3)')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=20) # Lowered for faster testing
    args = parser.parse_args()

    run_phase3(data_dir=args.data_dir, epochs=args.epochs)
