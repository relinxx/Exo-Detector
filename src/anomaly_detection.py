import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import glob
from tqdm import tqdm
import logging
import json
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../data/anomaly_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class LightCurveDataset(Dataset):
    """Dataset class for light curve windows."""
    
    def __init__(self, data_dir, window_size=200, normalize=True, label=None):
        """
        Initialize the light curve dataset.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing window files
        window_size : int
            Size of the window to use (will pad or truncate)
        normalize : bool
            Whether to normalize the flux values
        label : int or None
            If specified, only load windows with this label (0 for non-transit, 1 for transit)
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.normalize = normalize
        self.label = label
        
        # Get all window files
        self.window_files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        # Filter by label if specified
        if label is not None:
            filtered_files = []
            for file in self.window_files:
                try:
                    data = np.load(file)
                    if 'label' in data and data['label'] == label:
                        filtered_files.append(file)
                except Exception as e:
                    logger.warning(f"Error loading {file}: {str(e)}")
            self.window_files = filtered_files
        
        logger.info(f"Found {len(self.window_files)} window files in {data_dir}" + 
                   (f" with label {label}" if label is not None else ""))
    
    def __len__(self):
        """Return the number of windows."""
        return len(self.window_files)
    
    def __getitem__(self, idx):
        """
        Get a light curve window.
        
        Parameters:
        -----------
        idx : int
            Index of the window
            
        Returns:
        --------
        tuple
            (flux_tensor, label)
        """
        # Load the window
        data = np.load(self.window_files[idx])
        flux = data['flux']
        
        # Get label (1 for transit, 0 for non-transit)
        label = data.get('label', 0)
        
        # Pad or truncate to window_size
        if len(flux) < self.window_size:
            # Pad with median value
            pad_value = np.median(flux)
            pad_width = self.window_size - len(flux)
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            flux = np.pad(flux, (pad_left, pad_right), mode='constant', constant_values=pad_value)
        elif len(flux) > self.window_size:
            # Truncate from center
            start = (len(flux) - self.window_size) // 2
            flux = flux[start:start + self.window_size]
        
        # Normalize if requested
        if self.normalize:
            flux = (flux - np.median(flux)) / np.std(flux)
        
        # Convert to tensor
        flux_tensor = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        return flux_tensor, label

class Autoencoder(nn.Module):
    """1D-Convolutional Autoencoder for anomaly detection."""
    
    def __init__(self, input_size=200, latent_dim=16):
        """
        Initialize the autoencoder.
        
        Parameters:
        -----------
        input_size : int
            Size of the input window
        latent_dim : int
            Dimension of the latent space
        """
        super(Autoencoder, self).__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Calculate sizes for each layer
        self.size1 = input_size // 2  # 100
        self.size2 = self.size1 // 2  # 50
        self.size3 = self.size2 // 2  # 25
        
        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: (1, input_size) -> (16, input_size/2)
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: (16, input_size/2) -> (32, input_size/4)
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: (32, input_size/4) -> (64, input_size/8)
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: (64, input_size/8) -> (latent_dim, input_size/8)
            nn.Conv1d(64, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1: (latent_dim, input_size/8) -> (64, input_size/8)
            nn.Conv1d(latent_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: (64, input_size/8) -> (32, input_size/4)
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: (32, input_size/4) -> (16, input_size/2)
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: (16, input_size/2) -> (1, input_size)
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, x):
        """
        Forward pass of the autoencoder.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input window
            
        Returns:
        --------
        tuple
            (reconstructed, latent) - Reconstructed window and latent representation
        """
        # Encoder
        latent = self.encoder(x)
        
        # Decoder
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent
    
    def encode(self, x):
        """
        Encode input to latent representation.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input window
            
        Returns:
        --------
        torch.Tensor
            Latent representation
        """
        return self.encoder(x)
    
    def decode(self, latent):
        """
        Decode latent representation to output.
        
        Parameters:
        -----------
        latent : torch.Tensor
            Latent representation
            
        Returns:
        --------
        torch.Tensor
            Reconstructed window
        """
        return self.decoder(latent)

class AnomalyDetector:
    """Class for anomaly detection using autoencoder and one-class SVM."""
    
    def __init__(self, data_dir="../data", window_size=200, latent_dim=16, batch_size=32, lr=0.001):
        """
        Initialize the anomaly detector.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the data
        window_size : int
            Size of the window
        latent_dim : int
            Dimension of the latent space
        batch_size : int
            Batch size for training
        lr : float
            Learning rate
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
        # Define directories
        self.transit_dir = os.path.join(data_dir, "transit_windows")
        self.non_transit_dir = os.path.join(data_dir, "non_transit_windows")
        self.synthetic_dir = os.path.join(data_dir, "synthetic_transits")
        self.model_dir = os.path.join(data_dir, "models")
        self.plot_dir = os.path.join(data_dir, "plots")
        
        # Create directories if they don't exist
        for directory in [self.model_dir, self.plot_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create datasets
        self.non_transit_dataset = LightCurveDataset(self.non_transit_dir, window_size=window_size, label=0)
        self.transit_dataset = LightCurveDataset(self.transit_dir, window_size=window_size, label=1)
        
        # Create dataloaders
        self.non_transit_loader = DataLoader(self.non_transit_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.transit_loader = DataLoader(self.transit_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Create autoencoder
        self.autoencoder = Autoencoder(input_size=window_size, latent_dim=latent_dim).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # One-class SVM
        self.svm = None
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized AnomalyDetector with window_size={window_size}, latent_dim={latent_dim}, batch_size={batch_size}")
    
    def train_autoencoder(self, num_epochs=50):
        """
        Train the autoencoder on non-transit windows.
        
        Parameters:
        -----------
        num_epochs : int
            Number of epochs to train
            
        Returns:
        --------
        list
            List of training losses
        """
        logger.info(f"Training autoencoder for {num_epochs} epochs")
        
        # Set autoencoder to train mode
        self.autoencoder.train()
        
        # List to store losses
        losses = []
        
        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for i, (inputs, _) in enumerate(tqdm(self.non_transit_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                # Move to device
                inputs = inputs.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs, _ = self.autoencoder(inputs)
                loss = self.criterion(outputs, inputs)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item()
            
            # Calculate epoch loss
            epoch_loss = running_loss / len(self.non_transit_loader)
            losses.append(epoch_loss)
            
            logger.info(f"[{epoch+1}/{num_epochs}] Loss: {epoch_loss:.6f}")
            
            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
                self.save_autoencoder(epoch + 1)
        
        logger.info("Autoencoder training completed")
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Autoencoder Training Loss")
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, "autoencoder_training_loss.png"))
        plt.close()
        
        return losses
    
    def save_autoencoder(self, epoch):
        """
        Save the autoencoder model.
        
        Parameters:
        -----------
        epoch : int
            Current epoch
        """
        torch.save(self.autoencoder.state_dict(), os.path.join(self.model_dir, f"autoencoder_epoch_{epoch}.pth"))
        logger.info(f"Saved autoencoder at epoch {epoch}")
    
    def load_autoencoder(self, epoch):
        """
        Load the autoencoder model.
        
        Parameters:
        -----------
        epoch : int
            Epoch to load
            
        Returns:
        --------
        bool
            Whether the model was loaded successfully
        """
        model_path = os.path.join(self.model_dir, f"autoencoder_epoch_{epoch}.pth")
        
        if os.path.exists(model_path):
            self.autoencoder.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded autoencoder from epoch {epoch}")
            return True
        else:
            logger.warning(f"Could not load autoencoder from epoch {epoch}")
            return False
    
    def compute_reconstruction_errors(self, dataloader):
        """
        Compute reconstruction errors for a dataset.
        
        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            Dataloader for the dataset
            
        Returns:
        --------
        tuple
            (errors, labels) - Reconstruction errors and corresponding labels
        """
        logger.info("Computing reconstruction errors")
        
        # Set autoencoder to eval mode
        self.autoencoder.eval()
        
        # Lists to store errors and labels
        errors = []
        labels = []
        
        # Compute errors
        with torch.no_grad():
            for inputs, batch_labels in tqdm(dataloader, desc="Computing errors"):
                # Move to device
                inputs = inputs.to(self.device)
                
                # Forward pass
                outputs, _ = self.autoencoder(inputs)
                
                # Compute error
                error = torch.mean((outputs - inputs) ** 2, dim=(1, 2)).cpu().numpy()
                
                # Store error and label
                errors.extend(error)
                labels.extend(batch_labels.numpy())
        
        # Convert to numpy arrays
        errors = np.array(errors)
        labels = np.array(labels)
        
        return errors, labels
    
    def train_svm(self, nu=0.1):
        """
        Train a one-class SVM on reconstruction errors of non-transit windows.
        
        Parameters:
        -----------
        nu : float
            Nu parameter for one-class SVM (controls the fraction of outliers)
            
        Returns:
        --------
        OneClassSVM
            Trained SVM model
        """
        logger.info(f"Training one-class SVM with nu={nu}")
        
        # Compute reconstruction errors for non-transit windows
        errors, _ = self.compute_reconstruction_errors(self.non_transit_loader)
        
        # Reshape errors for SVM
        errors = errors.reshape(-1, 1)
        
        # Scale errors
        errors_scaled = self.scaler.fit_transform(errors)
        
        # Create and train SVM
        self.svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        self.svm.fit(errors_scaled)
        
        logger.info("SVM training completed")
        
        return self.svm
    
    def save_svm(self):
        """
        Save the SVM model and scaler.
        """
        if self.svm is not None:
            import joblib
            joblib.dump(self.svm, os.path.join(self.model_dir, "one_class_svm.joblib"))
            joblib.dump(self.scaler, os.path.join(self.model_dir, "error_scaler.joblib"))
            logger.info("Saved SVM model and scaler")
    
    def load_svm(self):
        """
        Load the SVM model and scaler.
        
        Returns:
        --------
        bool
            Whether the model was loaded successfully
        """
        svm_path = os.path.join(self.model_dir, "one_class_svm.joblib")
        scaler_path = os.path.join(self.model_dir, "error_scaler.joblib")
        
        if os.path.exists(svm_path) and os.path.exists(scaler_path):
            import joblib
            self.svm = joblib.load(svm_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded SVM model and scaler")
            return True
        else:
            logger.warning("Could not load SVM model and scaler")
            return False
    
    def evaluate_anomaly_detection(self):
        """
        Evaluate the anomaly detection performance.
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating anomaly detection performance")
        
        # Compute reconstruction errors for non-transit windows
        non_transit_errors, _ = self.compute_reconstruction_errors(self.non_transit_loader)
        
        # Compute reconstruction errors for transit windows
        transit_errors, _ = self.compute_reconstruction_errors(self.transit_loader)
        
        # Scale errors
        non_transit_errors_scaled = self.scaler.transform(non_transit_errors.reshape(-1, 1)).flatten()
        transit_errors_scaled = self.scaler.transform(transit_errors.reshape(-1, 1)).flatten()
        
        # Predict anomaly scores
        if self.svm is not None:
            non_transit_scores = self.svm.decision_function(non_transit_errors.reshape(-1, 1))
            transit_scores = self.svm.decision_function(transit_errors.reshape(-1, 1))
            
            # Higher score = more normal, lower score = more anomalous
            # Invert scores for ROC curve (higher = more anomalous)
            non_transit_scores = -non_transit_scores
            transit_scores = -transit_scores
            
            # Combine scores and labels
            all_scores = np.concatenate([non_transit_scores, transit_scores])
            all_labels = np.concatenate([np.zeros_like(non_transit_scores), np.ones_like(transit_scores)])
            
            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(all_labels, all_scores)
            roc_auc = auc(fpr, tpr)
            
            # Compute precision-recall curve and average precision
            precision, recall, _ = precision_recall_curve(all_labels, all_scores)
            avg_precision = average_precision_score(all_labels, all_scores)
            
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
            plt.savefig(os.path.join(self.plot_dir, "anomaly_detection_performance.png"))
            plt.close()
            
            # Plot error distributions
            plt.figure(figsize=(10, 5))
            plt.hist(non_transit_errors, bins=50, alpha=0.5, label='Non-Transit')
            plt.hist(transit_errors, bins=50, alpha=0.5, label='Transit')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Frequency')
            plt.title('Reconstruction Error Distributions')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.plot_dir, "reconstruction_error_distributions.png"))
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
            plt.savefig(os.path.join(self.plot_dir, "anomaly_score_distributions.png"))
            plt.close()
            
            # Compile evaluation metrics
            metrics = {
                "roc_auc": roc_auc,
                "average_precision": avg_precision,
                "non_transit_mean_error": np.mean(non_transit_errors),
                "transit_mean_error": np.mean(transit_errors),
                "non_transit_mean_score": np.mean(non_transit_scores),
                "transit_mean_score": np.mean(transit_scores)
            }
            
            logger.info(f"Evaluation metrics: {metrics}")
            
            return metrics
        else:
            logger.warning("SVM model not trained, cannot evaluate")
            return {}
    
    def visualize_reconstructions(self, num_examples=5):
        """
        Visualize original and reconstructed windows.
        
        Parameters:
        -----------
        num_examples : int
            Number of examples to visualize
            
        Returns:
        --------
        None
        """
        logger.info(f"Visualizing {num_examples} reconstructions")
        
        # Set autoencoder to eval mode
        self.autoencoder.eval()
        
        # Get examples from non-transit dataset
        non_transit_examples = []
        for i in range(min(num_examples, len(self.non_transit_dataset))):
            flux, _ = self.non_transit_dataset[i]
            non_transit_examples.append(flux)
        
        # Get examples from transit dataset
        transit_examples = []
        for i in range(min(num_examples, len(self.transit_dataset))):
            flux, _ = self.transit_dataset[i]
            transit_examples.append(flux)
        
        # Reconstruct examples
        with torch.no_grad():
            # Non-transit reconstructions
            non_transit_inputs = torch.stack(non_transit_examples).to(self.device)
            non_transit_outputs, _ = self.autoencoder(non_transit_inputs)
            
            # Transit reconstructions
            transit_inputs = torch.stack(transit_examples).to(self.device)
            transit_outputs, _ = self.autoencoder(transit_inputs)
        
        # Plot non-transit reconstructions
        plt.figure(figsize=(15, 10))
        for i in range(len(non_transit_examples)):
            # Original
            plt.subplot(2, num_examples, i + 1)
            plt.plot(non_transit_inputs[i, 0].cpu().numpy())
            plt.title(f"Original (Non-Transit {i+1})")
            plt.grid(True)
            
            # Reconstruction
            plt.subplot(2, num_examples, i + 1 + num_examples)
            plt.plot(non_transit_outputs[i, 0].cpu().numpy())
            plt.title(f"Reconstructed (Non-Transit {i+1})")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "non_transit_reconstructions.png"))
        plt.close()
        
        # Plot transit reconstructions
        plt.figure(figsize=(15, 10))
        for i in range(len(transit_examples)):
            # Original
            plt.subplot(2, num_examples, i + 1)
            plt.plot(transit_inputs[i, 0].cpu().numpy())
            plt.title(f"Original (Transit {i+1})")
            plt.grid(True)
            
            # Reconstruction
            plt.subplot(2, num_examples, i + 1 + num_examples)
            plt.plot(transit_outputs[i, 0].cpu().numpy())
            plt.title(f"Reconstructed (Transit {i+1})")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "transit_reconstructions.png"))
        plt.close()
        
        logger.info("Visualization completed")
    
    def detect_anomalies(self, dataloader, threshold=None):
        """
        Detect anomalies in a dataset.
        
        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            Dataloader for the dataset
        threshold : float, optional
            Threshold for anomaly detection. If None, use the SVM decision boundary.
            
        Returns:
        --------
        tuple
            (anomalies, scores, labels) - Anomaly flags, anomaly scores, and true labels
        """
        logger.info("Detecting anomalies")
        
        # Compute reconstruction errors
        errors, labels = self.compute_reconstruction_errors(dataloader)
        
        # Scale errors
        errors_scaled = self.scaler.transform(errors.reshape(-1, 1)).flatten()
        
        # Predict anomaly scores
        if self.svm is not None:
            scores = -self.svm.decision_function(errors.reshape(-1, 1))  # Invert scores (higher = more anomalous)
            
            # Detect anomalies
            if threshold is None:
                # Use SVM decision boundary
                anomalies = self.svm.predict(errors.reshape(-1, 1)) == -1  # -1 for anomaly, 1 for normal
            else:
                # Use custom threshold
                anomalies = scores > threshold
            
            logger.info(f"Detected {np.sum(anomalies)} anomalies out of {len(anomalies)} samples")
            
            return anomalies, scores, labels
        else:
            logger.warning("SVM model not trained, cannot detect anomalies")
            return None, None, labels
    
    def run_anomaly_detection_pipeline(self, num_epochs=50, nu=0.1):
        """
        Run the complete anomaly detection pipeline.
        
        Parameters:
        -----------
        num_epochs : int
            Number of epochs to train the autoencoder
        nu : float
            Nu parameter for one-class SVM
            
        Returns:
        --------
        dict
            Dictionary containing pipeline results
        """
        logger.info("Starting anomaly detection pipeline")
        
        # Step 1: Train autoencoder
        losses = self.train_autoencoder(num_epochs=num_epochs)
        
        # Step 2: Train SVM
        self.train_svm(nu=nu)
        
        # Step 3: Save models
        self.save_svm()
        
        # Step 4: Evaluate performance
        metrics = self.evaluate_anomaly_detection()
        
        # Step 5: Visualize reconstructions
        self.visualize_reconstructions()
        
        # Compile pipeline results
        pipeline_results = {
            "autoencoder_training": {
                "num_epochs": num_epochs,
                "final_loss": losses[-1] if losses else None
            },
            "svm_training": {
                "nu": nu
            },
            "evaluation_metrics": metrics
        }
        
        # Save pipeline results
        with open(os.path.join(self.data_dir, "anomaly_detection_pipeline_results.txt"), "w") as f:
            for section, results in pipeline_results.items():
                f.write(f"{section}:\n")
                for key, value in results.items():
                    f.write(f"  {key}: {value}\n")
        
        logger.info("Anomaly detection pipeline completed")
        logger.info(f"Pipeline results: {pipeline_results}")
        
        return pipeline_results


if __name__ == "__main__":
    # Example usage
    detector = AnomalyDetector()
    
    # For testing, run with fewer epochs
    pipeline_results = detector.run_anomaly_detection_pipeline(num_epochs=10)
    print(pipeline_results)
