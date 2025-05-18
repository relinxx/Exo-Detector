import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
import logging
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../data/gan_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TransitDataset(Dataset):
    """Dataset class for transit windows."""
    
    def __init__(self, data_dir, window_size=200, normalize=True, synthetic=False):
        """
        Initialize the transit dataset.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing transit window files
        window_size : int
            Size of the window to use (will pad or truncate)
        normalize : bool
            Whether to normalize the flux values
        synthetic : bool
            Whether the data is synthetic
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.normalize = normalize
        self.synthetic = synthetic
        
        # Get all transit window files
        self.transit_files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        logger.info(f"Found {len(self.transit_files)} transit window files in {data_dir}")
    
    def __len__(self):
        """Return the number of transit windows."""
        return len(self.transit_files)
    
    def __getitem__(self, idx):
        """
        Get a transit window.
        
        Parameters:
        -----------
        idx : int
            Index of the transit window
            
        Returns:
        --------
        tuple
            (flux_tensor, label)
        """
        # Load the transit window
        data = np.load(self.transit_files[idx])
        flux = data['flux']
        
        # Get label (1 for transit, 0 for non-transit)
        label = 1 if self.synthetic else data.get('label', 1)
        
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

class TransitClassifier(nn.Module):
    """CNN classifier for transit validation."""
    
    def __init__(self, input_size=200):
        """
        Initialize the classifier.
        
        Parameters:
        -----------
        input_size : int
            Size of the input window
        """
        super(TransitClassifier, self).__init__()
        
        self.input_size = input_size
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            # Block 1: (1, input_size) -> (16, input_size/2)
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            
            # Block 2: (16, input_size/2) -> (32, input_size/4)
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            
            # Block 3: (32, input_size/4) -> (64, input_size/8)
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            
            # Block 4: (64, input_size/8) -> (128, input_size/16)
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128)
        )
        
        # Calculate flattened size
        self.flattened_size = 128 * (input_size // 16)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(64, 2)  # 2 classes: real vs. synthetic
        )
    
    def forward(self, x):
        """
        Forward pass of the classifier.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input transit window
            
        Returns:
        --------
        torch.Tensor
            Class logits
        """
        # Convolutional layers
        x = self.conv_blocks(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x

class GANValidator:
    """Class for validating GAN outputs."""
    
    def __init__(self, data_dir="../data", window_size=200, batch_size=32, lr=0.001):
        """
        Initialize the GAN validator.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the data
        window_size : int
            Size of the transit window
        batch_size : int
            Batch size for training
        lr : float
            Learning rate
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.batch_size = batch_size
        
        # Define directories
        self.transit_dir = os.path.join(data_dir, "transit_windows")
        self.non_transit_dir = os.path.join(data_dir, "non_transit_windows")
        self.synthetic_dir = os.path.join(data_dir, "synthetic_transits")
        self.validation_dir = os.path.join(data_dir, "validation")
        
        # Create validation directory if it doesn't exist
        os.makedirs(self.validation_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create datasets and dataloaders
        self.real_dataset = TransitDataset(self.transit_dir, window_size=window_size, synthetic=False)
        self.synthetic_dataset = TransitDataset(self.synthetic_dir, window_size=window_size, synthetic=True)
        
        # Create classifier
        self.classifier = TransitClassifier(input_size=window_size).to(self.device)
        
        # Create optimizer and loss function
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Initialized GANValidator with window_size={window_size}, batch_size={batch_size}")
    
    def train_classifier(self, num_epochs=10):
        """
        Train a classifier to distinguish between real and synthetic transits.
        
        Parameters:
        -----------
        num_epochs : int
            Number of epochs to train
            
        Returns:
        --------
        dict
            Dictionary containing training statistics
        """
        logger.info(f"Training classifier for {num_epochs} epochs")
        
        # Create combined dataset
        real_size = len(self.real_dataset)
        synthetic_size = len(self.synthetic_dataset)
        
        # Balance dataset sizes
        if real_size > synthetic_size:
            real_indices = np.random.choice(real_size, synthetic_size, replace=False)
            real_subset = torch.utils.data.Subset(self.real_dataset, real_indices)
            combined_dataset = torch.utils.data.ConcatDataset([real_subset, self.synthetic_dataset])
        elif synthetic_size > real_size:
            synthetic_indices = np.random.choice(synthetic_size, real_size, replace=False)
            synthetic_subset = torch.utils.data.Subset(self.synthetic_dataset, synthetic_indices)
            combined_dataset = torch.utils.data.ConcatDataset([self.real_dataset, synthetic_subset])
        else:
            combined_dataset = torch.utils.data.ConcatDataset([self.real_dataset, self.synthetic_dataset])
        
        # Create dataloader
        dataloader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        # Lists to store losses and accuracies
        losses = []
        accuracies = []
        
        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                # Move to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.classifier(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = 100 * correct / total
            
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)
            
            logger.info(f"[{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%")
        
        # Save the classifier
        torch.save(self.classifier.state_dict(), os.path.join(self.validation_dir, "transit_classifier.pth"))
        
        # Plot training statistics
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(accuracies)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Training Accuracy")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.validation_dir, "classifier_training.png"))
        plt.close()
        
        logger.info("Classifier training completed")
        
        # Return training statistics
        return {
            "losses": losses,
            "accuracies": accuracies,
            "final_loss": losses[-1],
            "final_accuracy": accuracies[-1]
        }
    
    def evaluate_classifier(self):
        """
        Evaluate the classifier on a test set.
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating classifier")
        
        # Create test datasets
        real_size = len(self.real_dataset)
        synthetic_size = len(self.synthetic_dataset)
        
        # Use 20% of data for testing
        real_test_size = int(0.2 * real_size)
        synthetic_test_size = int(0.2 * synthetic_size)
        
        # Create test indices
        real_test_indices = np.random.choice(real_size, real_test_size, replace=False)
        synthetic_test_indices = np.random.choice(synthetic_size, synthetic_test_size, replace=False)
        
        # Create test subsets
        real_test_subset = torch.utils.data.Subset(self.real_dataset, real_test_indices)
        synthetic_test_subset = torch.utils.data.Subset(self.synthetic_dataset, synthetic_test_indices)
        
        # Create combined test dataset
        test_dataset = torch.utils.data.ConcatDataset([real_test_subset, synthetic_test_subset])
        
        # Create test dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        # Set classifier to eval mode
        self.classifier.eval()
        
        # Lists to store predictions and labels
        all_preds = []
        all_labels = []
        
        # Evaluate
        with torch.no_grad():
            for inputs, labels in tqdm(test_dataloader, desc="Evaluating"):
                # Move to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.classifier(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                # Store predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        # Set classifier back to train mode
        self.classifier.train()
        
        # Compile evaluation metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def compare_distributions(self):
        """
        Compare the distributions of real and synthetic transits.
        
        Returns:
        --------
        dict
            Dictionary containing distribution comparison metrics
        """
        logger.info("Comparing distributions of real and synthetic transits")
        
        # Load real and synthetic transits
        real_transits = []
        for i in range(len(self.real_dataset)):
            flux, _ = self.real_dataset[i]
            real_transits.append(flux.squeeze().numpy())
        
        synthetic_transits = []
        for i in range(len(self.synthetic_dataset)):
            flux, _ = self.synthetic_dataset[i]
            synthetic_transits.append(flux.squeeze().numpy())
        
        # Convert to numpy arrays
        real_transits = np.array(real_transits)
        synthetic_transits = np.array(synthetic_transits)
        
        # Calculate statistics
        real_mean = np.mean(real_transits, axis=0)
        real_std = np.std(real_transits, axis=0)
        synthetic_mean = np.mean(synthetic_transits, axis=0)
        synthetic_std = np.std(synthetic_transits, axis=0)
        
        # Calculate metrics
        mse = np.mean((real_mean - synthetic_mean) ** 2)
        correlation = np.corrcoef(real_mean, synthetic_mean)[0, 1]
        
        # Plot comparison
        plt.figure(figsize=(12, 10))
        
        # Plot means
        plt.subplot(2, 2, 1)
        plt.plot(real_mean, label='Real')
        plt.plot(synthetic_mean, label='Synthetic')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.title('Mean Transit Shape')
        plt.legend()
        plt.grid(True)
        
        # Plot standard deviations
        plt.subplot(2, 2, 2)
        plt.plot(real_std, label='Real')
        plt.plot(synthetic_std, label='Synthetic')
        plt.xlabel('Time')
        plt.ylabel('Standard Deviation')
        plt.title('Transit Shape Variability')
        plt.legend()
        plt.grid(True)
        
        # Plot histograms of mean flux
        plt.subplot(2, 2, 3)
        plt.hist(real_mean, bins=30, alpha=0.5, label='Real')
        plt.hist(synthetic_mean, bins=30, alpha=0.5, label='Synthetic')
        plt.xlabel('Flux')
        plt.ylabel('Frequency')
        plt.title('Histogram of Mean Flux')
        plt.legend()
        plt.grid(True)
        
        # Plot histograms of standard deviation
        plt.subplot(2, 2, 4)
        plt.hist(real_std, bins=30, alpha=0.5, label='Real')
        plt.hist(synthetic_std, bins=30, alpha=0.5, label='Synthetic')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Frequency')
        plt.title('Histogram of Flux Variability')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.validation_dir, "distribution_comparison.png"))
        plt.close()
        
        # Compile distribution comparison metrics
        metrics = {
            "mse": mse,
            "correlation": correlation
        }
        
        logger.info(f"Distribution comparison metrics: {metrics}")
        
        return metrics
    
    def visualize_latent_space(self):
        """
        Visualize the latent space of real and synthetic transits using dimensionality reduction.
        
        Returns:
        --------
        None
        """
        logger.info("Visualizing latent space")
        
        # Load real and synthetic transits
        real_transits = []
        for i in range(min(500, len(self.real_dataset))):
            flux, _ = self.real_dataset[i]
            real_transits.append(flux.squeeze().numpy())
        
        synthetic_transits = []
        for i in range(min(500, len(self.synthetic_dataset))):
            flux, _ = self.synthetic_dataset[i]
            synthetic_transits.append(flux.squeeze().numpy())
        
        # Convert to numpy arrays
        real_transits = np.array(real_transits)
        synthetic_transits = np.array(synthetic_transits)
        
        # Combine data
        combined_data = np.vstack((real_transits, synthetic_transits))
        labels = np.array([0] * len(real_transits) + [1] * len(synthetic_transits))
        
        # Apply PCA
        pca = PCA(n_components=50)
        pca_result = pca.fit_transform(combined_data)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(pca_result)
        
        # Plot t-SNE visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Class (0=Real, 1=Synthetic)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization of Transit Data')
        plt.grid(True)
        plt.savefig(os.path.join(self.validation_dir, "latent_space_visualization.png"))
        plt.close()
        
        logger.info("Latent space visualization completed")
    
    def run_validation(self):
        """
        Run the complete validation process.
        
        Returns:
        --------
        dict
            Dictionary containing validation results
        """
        logger.info("Starting validation process")
        
        # Step 1: Compare distributions
        distribution_metrics = self.compare_distributions()
        
        # Step 2: Train and evaluate classifier
        training_stats = self.train_classifier()
        evaluation_metrics = self.evaluate_classifier()
        
        # Step 3: Visualize latent space
        self.visualize_latent_space()
        
        # Compile validation results
        validation_results = {
            "distribution_comparison": distribution_metrics,
            "classifier_training": {
                "final_loss": training_stats["final_loss"],
                "final_accuracy": training_stats["final_accuracy"]
            },
            "classifier_evaluation": evaluation_metrics
        }
        
        # Save validation results to file
        with open(os.path.join(self.validation_dir, "validation_results.json"), "w") as f:
            json.dump(validation_results, f, indent=4)
        
        # Generate validation summary
        validation_summary = {
            "mse": distribution_metrics["mse"],
            "correlation": distribution_metrics["correlation"],
            "classifier_accuracy": evaluation_metrics["accuracy"],
            "classifier_f1": evaluation_metrics["f1"]
        }
        
        # Save validation summary to file
        with open(os.path.join(self.validation_dir, "validation_summary.txt"), "w") as f:
            for key, value in validation_summary.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Validation process completed")
        logger.info(f"Summary: {validation_summary}")
        
        return validation_results


if __name__ == "__main__":
    # Run validation
    validator = GANValidator()
    validation_results = validator.run_validation()
    print(json.dumps(validation_results, indent=4))
