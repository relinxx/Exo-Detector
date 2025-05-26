import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import glob
from tqdm import tqdm
import logging
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LightCurveDataset(Dataset):
    """Dataset class for light curves."""
    
    def __init__(self, windows, labels=None, transform=None):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        windows : list
            List of light curve windows
        labels : list or None
            List of labels (0 for non-transit, 1 for transit)
        transform : callable or None
            Optional transform to apply to the data
        """
        self.windows = windows
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        if self.transform:
            window = self.transform(window)
        
        if self.labels is not None:
            return window, self.labels[idx]
        else:
            return window

class ConvAutoencoder(nn.Module):
    """1D Convolutional Autoencoder for light curves."""
    
    def __init__(self, window_size=200, latent_dim=8):
        """
        Initialize the autoencoder.
        
        Parameters:
        -----------
        window_size : int
            Size of input window
        latent_dim : int
            Size of latent dimension
        """
        super(ConvAutoencoder, self).__init__()
        
        # Encoder - Streamlined with fewer filters and more aggressive pooling
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # window_size / 2
            
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # window_size / 4
            
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # window_size / 8
        )
        
        # Flatten layer
        self.flatten_size = window_size // 8 * 16
        
        # Bottleneck
        self.fc1 = nn.Linear(self.flatten_size, latent_dim)
        self.fc2 = nn.Linear(latent_dim, self.flatten_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def encode(self, x):
        """Encode input to latent space."""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x
    
    def decode(self, x):
        """Decode from latent space."""
        x = self.fc2(x)
        x = x.view(x.size(0), 16, -1)  # Reshape
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Forward pass."""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.0001):
        """
        Initialize early stopping.
        
        Parameters:
        -----------
        patience : int
            Number of epochs to wait after min has been hit
        min_delta : float
            Minimum change in loss to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        """
        Check if training should stop.
        
        Parameters:
        -----------
        val_loss : float
            Validation loss
            
        Returns:
        --------
        bool
            Whether to stop training
        """
        if val_loss < self.min_loss - self.min_delta:
            self.min_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

class AnomalyDetector:
    """Anomaly detector for light curves."""
    
    def __init__(self, data_dir="data", window_size=200, latent_dim=8, batch_size=128):
        """
        Initialize the anomaly detector.
        
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
        """
        # Convert to absolute path
        self.data_dir = os.path.abspath(data_dir)
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
        # Define directories
        self.transit_windows_dir = os.path.join(self.data_dir, "transit_windows")
        self.non_transit_windows_dir = os.path.join(self.data_dir, "non_transit_windows")
        self.models_dir = os.path.join(self.data_dir, "models")
        self.results_dir = os.path.join(self.data_dir, "results")
        self.validation_dir = os.path.join(self.data_dir, "validation")
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.validation_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.autoencoder = ConvAutoencoder(window_size=window_size, latent_dim=latent_dim).to(self.device)
        self.svm = None
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized AnomalyDetector with window_size={window_size}, latent_dim={latent_dim}, batch_size={batch_size}")
    
    def load_window_files(self, directory, label=None, limit=None):
        """
        Load window files from a directory.
        
        Parameters:
        -----------
        directory : str
            Directory containing window files
        label : int or None
            Label to assign to windows (0 for non-transit, 1 for transit)
        limit : int or None
            Maximum number of files to load
            
        Returns:
        --------
        tuple
            (windows, labels)
        """
        logger.info(f"Looking for .csv files in: {directory}")
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        
        if limit is not None:
            csv_files = csv_files[:limit]
        
        logger.info(f"Found {len(csv_files)} .csv files: {csv_files}")
        
        # Load first few files to check structure
        for i in range(min(3, len(csv_files))):
            df = pd.read_csv(csv_files[i])
            print(f"Contents of {csv_files[i]}:")
            print(df.head())
        
        # Load windows
        windows = []
        labels = []
        
        for csv_file in csv_files:
            try:
                # Load CSV file
                df = pd.read_csv(csv_file)
                
                # Extract flux column
                if 'flux' in df.columns:
                    flux = df['flux'].values
                    
                    # Pad or truncate to window_size
                    if len(flux) < self.window_size:
                        # Pad with zeros
                        flux = np.pad(flux, (0, self.window_size - len(flux)), 'constant')
                    elif len(flux) > self.window_size:
                        # Truncate
                        flux = flux[:self.window_size]
                    
                    # Add to windows
                    windows.append(flux)
                    
                    # Add label if provided
                    if label is not None:
                        labels.append(label)
            
            except Exception as e:
                logger.error(f"Error loading window file {csv_file}: {str(e)}")
        
        logger.info(f"Found {len(windows)} window files in {directory}" + (f" with label {label}" if label is not None else ""))
        
        return windows, labels
    
    def prepare_data(self, test_size=0.2, limit=None):
        """
        Prepare data for training and testing.
        
        Parameters:
        -----------
        test_size : float
            Fraction of data to use for testing
        limit : int or None
            Maximum number of files to load per class
            
        Returns:
        --------
        tuple
            (train_loader, test_loader, test_windows, test_labels)
        """
        # Load non-transit windows (label 0)
        non_transit_windows, non_transit_labels = self.load_window_files(
            self.non_transit_windows_dir, label=0, limit=limit
        )
        
        # Load transit windows (label 1)
        transit_windows, transit_labels = self.load_window_files(
            self.transit_windows_dir, label=1, limit=limit
        )
        
        # Combine windows and labels
        windows = non_transit_windows + transit_windows
        labels = non_transit_labels + transit_labels
        
        # Convert to numpy arrays
        windows = np.array(windows)
        labels = np.array(labels)
        
        # Split into train and test sets
        train_windows, test_windows, train_labels, test_labels = train_test_split(
            windows, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create PyTorch datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(train_windows).unsqueeze(1),  # Add channel dimension
            torch.FloatTensor(train_labels)
        )
        
        test_dataset = TensorDataset(
            torch.FloatTensor(test_windows).unsqueeze(1),  # Add channel dimension
            torch.FloatTensor(test_labels)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        return train_loader, test_loader, test_windows, test_labels
    
    def train_autoencoder(self, train_loader, epochs=50, learning_rate=0.001, save_interval=10):
        """
        Train the autoencoder.
        
        Parameters:
        -----------
        train_loader : torch.utils.data.DataLoader
            Data loader for training data
        epochs : int
            Number of epochs to train for
        learning_rate : float
            Learning rate for optimizer
        save_interval : int
            Interval for saving model checkpoints
            
        Returns:
        --------
        list
            List of losses
        """
        logger.info(f"Training autoencoder for {epochs} epochs")
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
        
        # Training loop
        losses = []
        
        for epoch in range(1, epochs + 1):
            # Set model to training mode
            self.autoencoder.train()
            
            # Initialize epoch loss
            epoch_loss = 0.0
            
            # Process batches
            for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
                # Move data to device
                data = data.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = self.autoencoder(data)
                
                # Calculate loss
                loss = criterion(output, data)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Update epoch loss
                epoch_loss += loss.item()
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(train_loader)
            losses.append(avg_epoch_loss)
            
            # Log progress
            logger.info(f"[{epoch}/{epochs}] Loss: {avg_epoch_loss:.6f}")
            
            # Save model checkpoint
            if epoch % save_interval == 0:
                torch.save(self.autoencoder.state_dict(), os.path.join(self.models_dir, f"autoencoder_epoch_{epoch}.pt"))
                logger.info(f"Saved autoencoder at epoch {epoch}")
            
            # Check for early stopping
            if early_stopping(avg_epoch_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Save final model
        torch.save(self.autoencoder.state_dict(), os.path.join(self.models_dir, "autoencoder_final.pt"))
        logger.info("Saved final autoencoder model")
        
        return losses
    
    def compute_reconstruction_errors(self, data_loader):
        """
        Compute reconstruction errors for a dataset.
        
        Parameters:
        -----------
        data_loader : torch.utils.data.DataLoader
            Data loader for the dataset
            
        Returns:
        --------
        tuple
            (reconstruction_errors, labels)
        """
        # Set model to evaluation mode
        self.autoencoder.eval()
        
        # Initialize lists for errors and labels
        reconstruction_errors = []
        labels = []
        
        # Disable gradient computation
        with torch.no_grad():
            # Process batches
            for data, label in data_loader:
                # Move data to device
                data = data.to(self.device)
                
                # Forward pass
                output = self.autoencoder(data)
                
                # Calculate reconstruction error (MSE)
                error = torch.mean((output - data) ** 2, dim=(1, 2))
                
                # Add to lists
                reconstruction_errors.extend(error.cpu().numpy())
                labels.extend(label.cpu().numpy())
        
        return np.array(reconstruction_errors), np.array(labels)
    
    def train_svm(self, reconstruction_errors, labels):
        """
        Train a One-Class SVM on reconstruction errors.
        
        Parameters:
        -----------
        reconstruction_errors : numpy.ndarray
            Array of reconstruction errors
        labels : numpy.ndarray
            Array of labels
            
        Returns:
        --------
        sklearn.svm.OneClassSVM
            Trained SVM model
        """
        # Extract non-transit (normal) samples
        normal_errors = reconstruction_errors[labels == 0]
        
        # Scale the data
        normal_errors_scaled = self.scaler.fit_transform(normal_errors.reshape(-1, 1))
        
        # Initialize and train SVM
        # Using linear kernel for faster inference, tuned nu parameter
        svm = OneClassSVM(kernel='linear', nu=0.1, gamma='scale')
        svm.fit(normal_errors_scaled)
        
        return svm
    
    def predict_anomalies(self, reconstruction_errors):
        """
        Predict anomalies using the trained SVM.
        
        Parameters:
        -----------
        reconstruction_errors : numpy.ndarray
            Array of reconstruction errors
            
        Returns:
        --------
        tuple
            (predictions, anomaly_scores)
        """
        # Scale the data
        errors_scaled = self.scaler.transform(reconstruction_errors.reshape(-1, 1))
        
        # Get decision function values (distance from hyperplane)
        # Multiply by -1 so that higher values indicate more anomalous
        anomaly_scores = -1 * self.svm.decision_function(errors_scaled)
        
        # Get predictions (1 for inlier, -1 for outlier)
        # Convert to 0 for normal, 1 for anomaly
        predictions = (self.svm.predict(errors_scaled) == -1).astype(int)
        
        return predictions, anomaly_scores.flatten()
    
    def evaluate_anomaly_detection(self, predictions, anomaly_scores, true_labels):
        """
        Evaluate anomaly detection performance.
        
        Parameters:
        -----------
        predictions : numpy.ndarray
            Array of predicted labels (0 for normal, 1 for anomaly)
        anomaly_scores : numpy.ndarray
            Array of anomaly scores
        true_labels : numpy.ndarray
            Array of true labels (0 for normal, 1 for anomaly)
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(true_labels, anomaly_scores)
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(true_labels, anomaly_scores)
        
        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        
        # Calculate average precision
        ap = average_precision_score(true_labels, anomaly_scores)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == true_labels)
        
        # Compile metrics
        metrics = {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'average_precision': float(ap),
            'precision_recall_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
            },
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist() if len(roc_thresholds) > 0 else []
            }
        }
        
        return metrics
    
    def plot_evaluation_curves(self, metrics, save_dir=None):
        """
        Plot evaluation curves.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of evaluation metrics
        save_dir : str or None
            Directory to save plots
            
        Returns:
        --------
        tuple
            (pr_fig, roc_fig)
        """
        # Create precision-recall curve
        pr_fig, pr_ax = plt.subplots(figsize=(10, 8))
        pr_ax.plot(
            metrics['precision_recall_curve']['recall'],
            metrics['precision_recall_curve']['precision'],
            label=f"AP = {metrics['average_precision']:.3f}"
        )
        pr_ax.set_xlabel('Recall')
        pr_ax.set_ylabel('Precision')
        pr_ax.set_title('Precision-Recall Curve')
        pr_ax.legend()
        pr_ax.grid(True)
        
        # Create ROC curve
        roc_fig, roc_ax = plt.subplots(figsize=(10, 8))
        roc_ax.plot(
            metrics['roc_curve']['fpr'],
            metrics['roc_curve']['tpr'],
            label=f"AUC = {metrics['roc_auc']:.3f}"
        )
        roc_ax.plot([0, 1], [0, 1], 'k--')
        roc_ax.set_xlabel('False Positive Rate')
        roc_ax.set_ylabel('True Positive Rate')
        roc_ax.set_title('ROC Curve')
        roc_ax.legend()
        roc_ax.grid(True)
        
        # Save plots if directory is provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            pr_fig.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
            roc_fig.savefig(os.path.join(save_dir, 'roc_curve.png'))
        
        return pr_fig, roc_fig
    
    def run_anomaly_detection_pipeline(self, epochs=50, test_size=0.2, limit=None):
        """
        Run the complete anomaly detection pipeline.
        
        Parameters:
        -----------
        epochs : int
            Number of epochs to train the autoencoder
        test_size : float
            Fraction of data to use for testing
        limit : int or None
            Maximum number of files to load per class
            
        Returns:
        --------
        dict
            Dictionary containing pipeline results
        """
        start_time = time.time()
        logger.info("Starting anomaly detection pipeline")
        
        # Step 1: Prepare data
        train_loader, test_loader, test_windows, test_labels = self.prepare_data(
            test_size=test_size, limit=limit
        )
        
        # Step 2: Train autoencoder
        losses = self.train_autoencoder(
            train_loader, epochs=epochs, learning_rate=0.001, save_interval=10
        )
        
        # Step 3: Compute reconstruction errors
        reconstruction_errors, labels = self.compute_reconstruction_errors(test_loader)
        
        # Step 4: Train SVM
        self.svm = self.train_svm(reconstruction_errors, labels)
        
        # Step 5: Predict anomalies
        predictions, anomaly_scores = self.predict_anomalies(reconstruction_errors)
        
        # Step 6: Evaluate performance
        metrics = self.evaluate_anomaly_detection(predictions, anomaly_scores, labels)
        
        # Step 7: Plot evaluation curves
        self.plot_evaluation_curves(metrics, save_dir=self.validation_dir)
        
        # Save metrics
        with open(os.path.join(self.results_dir, 'anomaly_detection_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save SVM model
        import joblib
        joblib.dump(self.svm, os.path.join(self.models_dir, 'anomaly_svm.pkl'))
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'anomaly_scaler.pkl'))
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Compile pipeline results
        pipeline_results = {
            'elapsed_time': elapsed_time,
            'num_epochs': epochs,
            'final_loss': float(losses[-1]) if losses else None,
            'metrics': metrics,
            'num_samples': {
                'train': len(train_loader.dataset),
                'test': len(test_loader.dataset)
            },
            'class_distribution': {
                'normal': int(np.sum(labels == 0)),
                'anomaly': int(np.sum(labels == 1))
            },
            'prediction_distribution': {
                'normal': int(np.sum(predictions == 0)),
                'anomaly': int(np.sum(predictions == 1))
            }
        }
        
        logger.info("Anomaly detection pipeline completed")
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        logger.info(f"Final loss: {losses[-1]:.6f}" if losses else "No loss data")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Average Precision: {metrics['average_precision']:.4f}")
        
        return pipeline_results
    
    def load_autoencoder(self, epoch):
        """
        Load a trained autoencoder model from a specified file.
        
        Parameters:
        -----------
        epoch : int
            The epoch number of the model to load
            
        Returns:
        --------
        bool
            Whether the model was loaded successfully
        """
        model_path = os.path.join(self.models_dir, f"autoencoder_epoch_{epoch}.pth")
        if os.path.exists(model_path):
            self.autoencoder.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded autoencoder model from {model_path}")
            return True
        else:
            logger.error(f"Autoencoder model file {model_path} does not exist.")
            return False


def run_anomaly_detection(data_dir="data", window_size=200, latent_dim=8, batch_size=128, epochs=50, limit=None):
    """
    Run the anomaly detection pipeline.
    
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
    # Initialize anomaly detector
    detector = AnomalyDetector(
        data_dir=data_dir,
        window_size=window_size,
        latent_dim=latent_dim,
        batch_size=batch_size
    )
    
    # Run pipeline
    results = detector.run_anomaly_detection_pipeline(
        epochs=epochs,
        test_size=0.2,
        limit=limit
    )
    
    return results


if __name__ == "__main__":
    # Run anomaly detection pipeline
    results = run_anomaly_detection(
        window_size=200,
        latent_dim=8,
        batch_size=128,
        epochs=50
    )
    print(results)
