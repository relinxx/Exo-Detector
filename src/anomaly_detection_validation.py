import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import glob
from tqdm import tqdm
import logging
import json
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import seaborn as sns

# Import anomaly detection module
from anomaly_detection import AnomalyDetector, LightCurveDataset, Autoencoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../data/anomaly_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnomalyDetectionValidator:
    """Class for validating anomaly detection outputs."""
    
    def __init__(self, data_dir="../data", window_size=200, batch_size=32):
        """
        Initialize the validator.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the data
        window_size : int
            Size of the window
        batch_size : int
            Batch size for evaluation
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.batch_size = batch_size
        
        # Define directories
        self.transit_dir = os.path.join(data_dir, "transit_windows")
        self.non_transit_dir = os.path.join(data_dir, "non_transit_windows")
        self.synthetic_dir = os.path.join(data_dir, "synthetic_transits")
        self.model_dir = os.path.join(data_dir, "models")
        self.plot_dir = os.path.join(data_dir, "plots")
        self.validation_dir = os.path.join(data_dir, "validation")
        
        # Create validation directory if it doesn't exist
        os.makedirs(self.validation_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create anomaly detector
        self.detector = AnomalyDetector(data_dir=data_dir, window_size=window_size, batch_size=batch_size)
        
        logger.info(f"Initialized AnomalyDetectionValidator with window_size={window_size}, batch_size={batch_size}")
    
    def load_models(self):
        """
        Load the trained autoencoder and SVM models.
        
        Returns:
        --------
        bool
            Whether the models were loaded successfully
        """
        logger.info("Loading trained models")
        
        # Find the latest autoencoder model
        autoencoder_files = glob.glob(os.path.join(self.model_dir, "autoencoder_epoch_*.pth"))
        if len(autoencoder_files) == 0:
            logger.error("No autoencoder models found")
            return False
        
        latest_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in autoencoder_files])
        autoencoder_loaded = self.detector.load_autoencoder(latest_epoch)
        
        # Load SVM model
        svm_loaded = self.detector.load_svm()
        
        if autoencoder_loaded and svm_loaded:
            logger.info("Models loaded successfully")
            return True
        else:
            logger.warning("Failed to load models")
            return False
    
    def create_test_datasets(self, test_ratio=0.2):
        """
        Create test datasets from transit and non-transit windows.
        
        Parameters:
        -----------
        test_ratio : float
            Ratio of data to use for testing
            
        Returns:
        --------
        tuple
            (transit_test_loader, non_transit_test_loader) - Test dataloaders
        """
        logger.info(f"Creating test datasets with test_ratio={test_ratio}")
        
        # Get all transit files
        transit_files = glob.glob(os.path.join(self.transit_dir, "*.npz"))
        
        # Get all non-transit files
        non_transit_files = glob.glob(os.path.join(self.non_transit_dir, "*.npz"))
        
        # Shuffle files
        np.random.shuffle(transit_files)
        np.random.shuffle(non_transit_files)
        
        # Split into train and test
        transit_test_size = int(len(transit_files) * test_ratio)
        non_transit_test_size = int(len(non_transit_files) * test_ratio)
        
        transit_test_files = transit_files[:transit_test_size]
        non_transit_test_files = non_transit_files[:non_transit_test_size]
        
        # Create test datasets
        transit_test_dataset = LightCurveDataset(self.transit_dir, window_size=self.window_size, label=1)
        transit_test_dataset.window_files = transit_test_files
        
        non_transit_test_dataset = LightCurveDataset(self.non_transit_dir, window_size=self.window_size, label=0)
        non_transit_test_dataset.window_files = non_transit_test_files
        
        # Create test dataloaders
        transit_test_loader = DataLoader(transit_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        non_transit_test_loader = DataLoader(non_transit_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        logger.info(f"Created test datasets with {len(transit_test_dataset)} transit and {len(non_transit_test_dataset)} non-transit windows")
        
        return transit_test_loader, non_transit_test_loader
    
    def evaluate_reconstruction_quality(self, dataloader, num_examples=5):
        """
        Evaluate the reconstruction quality of the autoencoder.
        
        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            Dataloader for the dataset
        num_examples : int
            Number of examples to visualize
            
        Returns:
        --------
        dict
            Dictionary containing reconstruction quality metrics
        """
        logger.info("Evaluating reconstruction quality")
        
        # Set autoencoder to eval mode
        self.detector.autoencoder.eval()
        
        # Lists to store errors
        errors = []
        
        # Get examples for visualization
        examples = []
        example_labels = []
        for inputs, labels in dataloader:
            if len(examples) < num_examples:
                examples.extend(inputs[:num_examples - len(examples)])
                example_labels.extend(labels[:num_examples - len(examples)])
            if len(examples) >= num_examples:
                break
        
        # Compute errors and visualize reconstructions
        with torch.no_grad():
            # Reconstruct examples
            example_inputs = torch.stack(examples).to(self.device)
            example_outputs, _ = self.detector.autoencoder(example_inputs)
            
            # Compute errors for all samples
            for inputs, _ in tqdm(dataloader, desc="Computing reconstruction errors"):
                # Move to device
                inputs = inputs.to(self.device)
                
                # Forward pass
                outputs, _ = self.detector.autoencoder(inputs)
                
                # Compute error
                error = torch.mean((outputs - inputs) ** 2, dim=(1, 2)).cpu().numpy()
                
                # Store error
                errors.extend(error)
        
        # Convert to numpy array
        errors = np.array(errors)
        
        # Calculate metrics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)
        
        # Plot error distribution
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=50)
        plt.axvline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.6f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Reconstruction Error Distribution')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.validation_dir, "reconstruction_error_distribution.png"))
        plt.close()
        
        # Plot example reconstructions
        plt.figure(figsize=(15, 10))
        for i in range(len(examples)):
            # Original
            plt.subplot(2, num_examples, i + 1)
            plt.plot(example_inputs[i, 0].cpu().numpy())
            plt.title(f"Original (Label: {example_labels[i]})")
            plt.grid(True)
            
            # Reconstruction
            plt.subplot(2, num_examples, i + 1 + num_examples)
            plt.plot(example_outputs[i, 0].cpu().numpy())
            plt.title(f"Reconstructed (Error: {torch.mean((example_outputs[i] - example_inputs[i]) ** 2).item():.6f})")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.validation_dir, "example_reconstructions.png"))
        plt.close()
        
        # Compile metrics
        metrics = {
            "mean_error": mean_error,
            "std_error": std_error,
            "min_error": min_error,
            "max_error": max_error
        }
        
        logger.info(f"Reconstruction quality metrics: {metrics}")
        
        return metrics
    
    def evaluate_anomaly_detection(self, transit_loader, non_transit_loader):
        """
        Evaluate the anomaly detection performance.
        
        Parameters:
        -----------
        transit_loader : torch.utils.data.DataLoader
            Dataloader for transit windows
        non_transit_loader : torch.utils.data.DataLoader
            Dataloader for non-transit windows
            
        Returns:
        --------
        dict
            Dictionary containing anomaly detection metrics
        """
        logger.info("Evaluating anomaly detection performance")
        
        # Detect anomalies in transit windows
        transit_anomalies, transit_scores, transit_labels = self.detector.detect_anomalies(transit_loader)
        
        # Detect anomalies in non-transit windows
        non_transit_anomalies, non_transit_scores, non_transit_labels = self.detector.detect_anomalies(non_transit_loader)
        
        if transit_anomalies is None or non_transit_anomalies is None:
            logger.warning("Anomaly detection failed")
            return {}
        
        # Combine results
        all_anomalies = np.concatenate([transit_anomalies, non_transit_anomalies])
        all_scores = np.concatenate([transit_scores, non_transit_scores])
        all_labels = np.concatenate([transit_labels, non_transit_labels])
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        
        # Compute precision-recall curve and average precision
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        avg_precision = average_precision_score(all_labels, all_scores)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_anomalies)
        
        # Compute classification report
        report = classification_report(all_labels, all_anomalies, output_dict=True)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Plot precision-recall curve
        plt.subplot(2, 1, 2)
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.validation_dir, "anomaly_detection_curves.png"))
        plt.close()
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Transit', 'Transit'], 
                    yticklabels=['Non-Transit', 'Transit'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.validation_dir, "confusion_matrix.png"))
        plt.close()
        
        # Plot score distributions
        plt.figure(figsize=(10, 5))
        plt.hist(non_transit_scores, bins=50, alpha=0.5, label='Non-Transit')
        plt.hist(transit_scores, bins=50, alpha=0.5, label='Transit')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distributions')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.validation_dir, "score_distributions.png"))
        plt.close()
        
        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Compile metrics
        metrics = {
            "roc_auc": roc_auc,
            "average_precision": avg_precision,
            "optimal_threshold": optimal_threshold,
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }
        
        logger.info(f"Anomaly detection metrics: {metrics}")
        
        return metrics
    
    def evaluate_latent_space(self, transit_loader, non_transit_loader):
        """
        Evaluate the latent space of the autoencoder.
        
        Parameters:
        -----------
        transit_loader : torch.utils.data.DataLoader
            Dataloader for transit windows
        non_transit_loader : torch.utils.data.DataLoader
            Dataloader for non-transit windows
            
        Returns:
        --------
        dict
            Dictionary containing latent space metrics
        """
        logger.info("Evaluating latent space")
        
        # Set autoencoder to eval mode
        self.detector.autoencoder.eval()
        
        # Lists to store latent representations
        transit_latents = []
        non_transit_latents = []
        
        # Extract latent representations
        with torch.no_grad():
            # Transit windows
            for inputs, _ in tqdm(transit_loader, desc="Extracting transit latents"):
                # Move to device
                inputs = inputs.to(self.device)
                
                # Forward pass
                _, latent = self.detector.autoencoder(inputs)
                
                # Store latent representation
                transit_latents.append(latent.cpu().numpy())
            
            # Non-transit windows
            for inputs, _ in tqdm(non_transit_loader, desc="Extracting non-transit latents"):
                # Move to device
                inputs = inputs.to(self.device)
                
                # Forward pass
                _, latent = self.detector.autoencoder(inputs)
                
                # Store latent representation
                non_transit_latents.append(latent.cpu().numpy())
        
        # Concatenate latent representations
        transit_latents = np.concatenate(transit_latents, axis=0)
        non_transit_latents = np.concatenate(non_transit_latents, axis=0)
        
        # Reshape latent representations
        transit_latents = transit_latents.reshape(transit_latents.shape[0], -1)
        non_transit_latents = non_transit_latents.reshape(non_transit_latents.shape[0], -1)
        
        # Calculate metrics
        transit_mean = np.mean(transit_latents, axis=0)
        non_transit_mean = np.mean(non_transit_latents, axis=0)
        
        transit_std = np.std(transit_latents, axis=0)
        non_transit_std = np.std(non_transit_latents, axis=0)
        
        # Calculate distance between means
        mean_distance = np.linalg.norm(transit_mean - non_transit_mean)
        
        # Plot latent space visualization
        plt.figure(figsize=(12, 8))
        
        # Plot mean latent representations
        plt.subplot(2, 1, 1)
        plt.plot(transit_mean, label='Transit')
        plt.plot(non_transit_mean, label='Non-Transit')
        plt.xlabel('Latent Dimension')
        plt.ylabel('Mean Value')
        plt.title('Mean Latent Representations')
        plt.legend()
        plt.grid(True)
        
        # Plot standard deviations
        plt.subplot(2, 1, 2)
        plt.plot(transit_std, label='Transit')
        plt.plot(non_transit_std, label='Non-Transit')
        plt.xlabel('Latent Dimension')
        plt.ylabel('Standard Deviation')
        plt.title('Latent Representation Variability')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.validation_dir, "latent_space_visualization.png"))
        plt.close()
        
        # Compile metrics
        metrics = {
            "mean_distance": mean_distance,
            "transit_latent_mean": transit_mean.tolist(),
            "non_transit_latent_mean": non_transit_mean.tolist(),
            "transit_latent_std": transit_std.tolist(),
            "non_transit_latent_std": non_transit_std.tolist()
        }
        
        logger.info(f"Latent space metrics: {metrics}")
        
        return metrics
    
    def run_validation(self):
        """
        Run the complete validation process.
        
        Returns:
        --------
        dict
            Dictionary containing validation results
        """
        logger.info("Starting validation process")
        
        # Step 1: Load models
        if not self.load_models():
            logger.error("Failed to load models, validation aborted")
            return {}
        
        # Step 2: Create test datasets
        transit_loader, non_transit_loader = self.create_test_datasets()
        
        # Step 3: Evaluate reconstruction quality
        reconstruction_metrics = self.evaluate_reconstruction_quality(transit_loader)
        
        # Step 4: Evaluate anomaly detection
        anomaly_metrics = self.evaluate_anomaly_detection(transit_loader, non_transit_loader)
        
        # Step 5: Evaluate latent space
        latent_metrics = self.evaluate_latent_space(transit_loader, non_transit_loader)
        
        # Compile validation results
        validation_results = {
            "reconstruction_quality": reconstruction_metrics,
            "anomaly_detection": anomaly_metrics,
            "latent_space": latent_metrics
        }
        
        # Save validation results to file
        with open(os.path.join(self.validation_dir, "anomaly_validation_results.json"), "w") as f:
            json.dump(validation_results, f, indent=4)
        
        # Generate validation summary
        validation_summary = {
            "mean_reconstruction_error": reconstruction_metrics.get("mean_error", "N/A"),
            "roc_auc": anomaly_metrics.get("roc_auc", "N/A"),
            "average_precision": anomaly_metrics.get("average_precision", "N/A"),
            "latent_mean_distance": latent_metrics.get("mean_distance", "N/A")
        }
        
        # Save validation summary to file
        with open(os.path.join(self.validation_dir, "anomaly_validation_summary.txt"), "w") as f:
            for key, value in validation_summary.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Validation process completed")
        logger.info(f"Summary: {validation_summary}")
        
        return validation_results


if __name__ == "__main__":
    # Run validation
    validator = AnomalyDetectionValidator()
    validation_results = validator.run_validation()
    print(json.dumps(validation_results, indent=4))
