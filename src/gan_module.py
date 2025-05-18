import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../data/gan_training.log"),
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

class TransitDataset(Dataset):
    """Dataset class for transit windows."""
    
    def __init__(self, data_dir, window_size=200, normalize=True):
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
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.normalize = normalize
        
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
        torch.Tensor
            Tensor containing the transit window
        """
        # Load the transit window
        data = np.load(self.transit_files[idx])
        flux = data['flux']
        
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
        
        return flux_tensor

class Generator(nn.Module):
    """Generator network for the GAN."""
    
    def __init__(self, latent_dim=100, output_size=200):
        """
        Initialize the generator.
        
        Parameters:
        -----------
        latent_dim : int
            Dimension of the latent space
        output_size : int
            Size of the output window
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        # Calculate initial size
        self.initial_size = output_size // 16
        
        # Initial linear layer
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.initial_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            # Block 1: (128, initial_size) -> (64, initial_size*2)
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2: (64, initial_size*2) -> (32, initial_size*4)
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3: (32, initial_size*4) -> (16, initial_size*8)
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 4: (16, initial_size*8) -> (1, initial_size*16 = output_size)
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, z):
        """
        Forward pass of the generator.
        
        Parameters:
        -----------
        z : torch.Tensor
            Latent vector
            
        Returns:
        --------
        torch.Tensor
            Generated transit window
        """
        # Linear layer
        x = self.linear(z)
        
        # Reshape for convolutional layers
        x = x.view(x.size(0), 128, self.initial_size)
        
        # Convolutional layers
        x = self.conv_blocks(x)
        
        return x

class Discriminator(nn.Module):
    """Discriminator network for the GAN."""
    
    def __init__(self, input_size=200):
        """
        Initialize the discriminator.
        
        Parameters:
        -----------
        input_size : int
            Size of the input window
        """
        super(Discriminator, self).__init__()
        
        self.input_size = input_size
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            # Block 1: (1, input_size) -> (16, input_size/2)
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # Block 2: (16, input_size/2) -> (32, input_size/4)
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # Block 3: (32, input_size/4) -> (64, input_size/8)
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # Block 4: (64, input_size/8) -> (128, input_size/16)
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        
        # Calculate flattened size
        self.flattened_size = 128 * (input_size // 16)
        
        # Final linear layer
        self.linear = nn.Sequential(
            nn.Linear(self.flattened_size, 1),
            nn.Sigmoid()  # Output in range [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass of the discriminator.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input transit window
            
        Returns:
        --------
        torch.Tensor
            Probability that the input is real
        """
        # Convolutional layers
        x = self.conv_blocks(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Linear layer
        x = self.linear(x)
        
        return x

class TransitGAN:
    """Class for training and using the GAN."""
    
    def __init__(self, data_dir="../data", window_size=200, latent_dim=100, batch_size=32, lr=0.0002, beta1=0.5):
        """
        Initialize the GAN.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the data
        window_size : int
            Size of the transit window
        latent_dim : int
            Dimension of the latent space
        batch_size : int
            Batch size for training
        lr : float
            Learning rate
        beta1 : float
            Beta1 parameter for Adam optimizer
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
        for directory in [self.synthetic_dir, self.model_dir, self.plot_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create dataset and dataloader
        self.dataset = TransitDataset(self.transit_dir, window_size=window_size)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Create generator and discriminator
        self.generator = Generator(latent_dim=latent_dim, output_size=window_size).to(self.device)
        self.discriminator = Discriminator(input_size=window_size).to(self.device)
        
        # Initialize weights
        self._init_weights(self.generator)
        self._init_weights(self.discriminator)
        
        # Create optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        logger.info(f"Initialized TransitGAN with window_size={window_size}, latent_dim={latent_dim}, batch_size={batch_size}")
    
    def _init_weights(self, model):
        """
        Initialize weights of the model.
        
        Parameters:
        -----------
        model : torch.nn.Module
            Model to initialize
        """
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
            if model.bias is not None:
                nn.init.constant_(model.bias.data, 0)
    
    def train(self, num_epochs=100, save_interval=10):
        """
        Train the GAN.
        
        Parameters:
        -----------
        num_epochs : int
            Number of epochs to train
        save_interval : int
            Interval for saving models and generating samples
            
        Returns:
        --------
        dict
            Dictionary containing training statistics
        """
        logger.info(f"Starting GAN training for {num_epochs} epochs")
        
        # Lists to store losses
        G_losses = []
        D_losses = []
        D_accuracies = []
        
        # Fixed noise for generating samples
        fixed_noise = torch.randn(16, self.latent_dim, device=self.device)
        
        # Training loop
        for epoch in range(num_epochs):
            for i, real_transits in enumerate(tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                ############################
                # Update Discriminator
                ############################
                self.discriminator.zero_grad()
                
                # Format real batch
                real_transits = real_transits.to(self.device)
                batch_size = real_transits.size(0)
                label = torch.full((batch_size, 1), 1, dtype=torch.float, device=self.device)
                
                # Forward pass real batch through D
                output = self.discriminator(real_transits)
                
                # Calculate loss on real batch
                errD_real = self.criterion(output, label)
                
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()
                
                # Generate batch of latent vectors
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                
                # Generate fake transit batch with G
                fake = self.generator(noise)
                label.fill_(0)
                
                # Classify fake batch with D
                output = self.discriminator(fake.detach())
                
                # Calculate D's loss on the fake batch
                errD_fake = self.criterion(output, label)
                
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                
                # Add the gradients from the real and fake batches
                errD = errD_real + errD_fake
                
                # Update D
                self.optimizer_D.step()
                
                ############################
                # Update Generator
                ############################
                self.generator.zero_grad()
                label.fill_(1)  # Fake labels are real for generator cost
                
                # Since we just updated D, perform another forward pass of fake batch through D
                output = self.discriminator(fake)
                
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                
                # Update G
                self.optimizer_G.step()
                
                # Save losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                
                # Calculate discriminator accuracy
                pred_real = (output > 0.5).float()
                accuracy = (pred_real == label).float().mean().item()
                D_accuracies.append(accuracy)
            
            # Output training stats
            logger.info(f"[{epoch+1}/{num_epochs}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f} Acc: {accuracy:.4f}")
            
            # Save models and generate samples at specified intervals
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
                self.save_models(epoch + 1)
                self.generate_samples(fixed_noise, epoch + 1)
                self.plot_losses(G_losses, D_losses, D_accuracies, epoch + 1)
        
        logger.info("GAN training completed")
        
        # Return training statistics
        return {
            "G_losses": G_losses,
            "D_losses": D_losses,
            "D_accuracies": D_accuracies
        }
    
    def save_models(self, epoch):
        """
        Save the models.
        
        Parameters:
        -----------
        epoch : int
            Current epoch
        """
        torch.save(self.generator.state_dict(), os.path.join(self.model_dir, f"generator_epoch_{epoch}.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.model_dir, f"discriminator_epoch_{epoch}.pth"))
        logger.info(f"Saved models at epoch {epoch}")
    
    def load_models(self, epoch):
        """
        Load the models.
        
        Parameters:
        -----------
        epoch : int
            Epoch to load
            
        Returns:
        --------
        bool
            Whether the models were loaded successfully
        """
        generator_path = os.path.join(self.model_dir, f"generator_epoch_{epoch}.pth")
        discriminator_path = os.path.join(self.model_dir, f"discriminator_epoch_{epoch}.pth")
        
        if os.path.exists(generator_path) and os.path.exists(discriminator_path):
            self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))
            logger.info(f"Loaded models from epoch {epoch}")
            return True
        else:
            logger.warning(f"Could not load models from epoch {epoch}")
            return False
    
    def generate_samples(self, noise=None, epoch=None, num_samples=16):
        """
        Generate and save samples.
        
        Parameters:
        -----------
        noise : torch.Tensor, optional
            Noise to use for generation. If None, random noise is used.
        epoch : int, optional
            Current epoch. If None, samples are not saved.
        num_samples : int
            Number of samples to generate
            
        Returns:
        --------
        torch.Tensor
            Generated samples
        """
        # Set generator to eval mode
        self.generator.eval()
        
        with torch.no_grad():
            # Generate noise if not provided
            if noise is None:
                noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            
            # Generate samples
            fake = self.generator(noise)
            
            # Plot samples
            plt.figure(figsize=(10, 10))
            for i in range(min(16, fake.size(0))):
                plt.subplot(4, 4, i + 1)
                plt.plot(fake[i, 0].cpu().numpy())
                plt.title(f"Sample {i+1}")
                plt.ylim(-1.5, 1.5)
                plt.grid(True)
            
            plt.tight_layout()
            
            # Save plot if epoch is provided
            if epoch is not None:
                plt.savefig(os.path.join(self.plot_dir, f"samples_epoch_{epoch}.png"))
            
            plt.close()
        
        # Set generator back to train mode
        self.generator.train()
        
        return fake
    
    def plot_losses(self, G_losses, D_losses, D_accuracies, epoch):
        """
        Plot and save the losses.
        
        Parameters:
        -----------
        G_losses : list
            Generator losses
        D_losses : list
            Discriminator losses
        D_accuracies : list
            Discriminator accuracies
        epoch : int
            Current epoch
        """
        plt.figure(figsize=(10, 8))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(G_losses, label='Generator')
        plt.plot(D_losses, label='Discriminator')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Losses')
        plt.grid(True)
        
        # Plot discriminator accuracy
        plt.subplot(2, 1, 2)
        plt.plot(D_accuracies)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Discriminator Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"losses_epoch_{epoch}.png"))
        plt.close()
    
    def generate_synthetic_dataset(self, num_samples=1000, epoch=None):
        """
        Generate a synthetic dataset.
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to generate
        epoch : int, optional
            Epoch to use for generation. If None, the latest epoch is used.
            
        Returns:
        --------
        list
            List of file paths to the generated samples
        """
        logger.info(f"Generating {num_samples} synthetic transit samples")
        
        # Load the latest model if epoch is not specified
        if epoch is None:
            # Find the latest epoch
            generator_files = glob.glob(os.path.join(self.model_dir, "generator_epoch_*.pth"))
            if len(generator_files) == 0:
                logger.error("No generator models found")
                return []
            
            latest_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in generator_files])
            self.load_models(latest_epoch)
        else:
            self.load_models(epoch)
        
        # Set generator to eval mode
        self.generator.eval()
        
        # Generate samples
        synthetic_files = []
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, self.batch_size), desc="Generating synthetic transits"):
                # Determine batch size
                batch_size = min(self.batch_size, num_samples - i)
                
                # Generate noise
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                
                # Generate samples
                fake = self.generator(noise)
                
                # Save each sample
                for j in range(batch_size):
                    # Convert to numpy
                    sample = fake[j, 0].cpu().numpy()
                    
                    # Save to file
                    file_path = os.path.join(self.synthetic_dir, f"synthetic_transit_{i+j+1}.npz")
                    np.savez(
                        file_path,
                        flux=sample,
                        label=1,  # 1 for transit
                        synthetic=True
                    )
                    
                    synthetic_files.append(file_path)
        
        # Set generator back to train mode
        self.generator.train()
        
        logger.info(f"Generated {len(synthetic_files)} synthetic transit samples")
        
        return synthetic_files
    
    def validate_synthetic_samples(self, num_samples=100):
        """
        Validate synthetic samples against real samples.
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to validate
            
        Returns:
        --------
        dict
            Dictionary containing validation results
        """
        logger.info(f"Validating {num_samples} synthetic samples")
        
        # Get real transit files
        real_files = glob.glob(os.path.join(self.transit_dir, "*.npz"))
        if len(real_files) == 0:
            logger.error("No real transit files found")
            return {}
        
        # Get synthetic transit files
        synthetic_files = glob.glob(os.path.join(self.synthetic_dir, "*.npz"))
        if len(synthetic_files) == 0:
            logger.error("No synthetic transit files found")
            return {}
        
        # Sample files
        real_samples = np.random.choice(real_files, min(num_samples, len(real_files)), replace=False)
        synthetic_samples = np.random.choice(synthetic_files, min(num_samples, len(synthetic_files)), replace=False)
        
        # Load samples
        real_data = []
        for file in real_samples:
            data = np.load(file)
            real_data.append(data['flux'])
        
        synthetic_data = []
        for file in synthetic_samples:
            data = np.load(file)
            synthetic_data.append(data['flux'])
        
        # Convert to numpy arrays
        real_data = np.array(real_data)
        synthetic_data = np.array(synthetic_data)
        
        # Calculate statistics
        real_mean = np.mean(real_data, axis=0)
        real_std = np.std(real_data, axis=0)
        synthetic_mean = np.mean(synthetic_data, axis=0)
        synthetic_std = np.std(synthetic_data, axis=0)
        
        # Calculate metrics
        mse = np.mean((real_mean - synthetic_mean) ** 2)
        correlation = np.corrcoef(real_mean, synthetic_mean)[0, 1]
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        # Plot means
        plt.subplot(2, 1, 1)
        plt.plot(real_mean, label='Real')
        plt.plot(synthetic_mean, label='Synthetic')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.title('Mean Transit Shape')
        plt.legend()
        plt.grid(True)
        
        # Plot standard deviations
        plt.subplot(2, 1, 2)
        plt.plot(real_std, label='Real')
        plt.plot(synthetic_std, label='Synthetic')
        plt.xlabel('Time')
        plt.ylabel('Standard Deviation')
        plt.title('Transit Shape Variability')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "synthetic_validation.png"))
        plt.close()
        
        # Plot sample comparison
        plt.figure(figsize=(12, 8))
        
        # Plot real samples
        plt.subplot(2, 1, 1)
        for i in range(min(10, len(real_data))):
            plt.plot(real_data[i])
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.title('Real Transit Samples')
        plt.grid(True)
        
        # Plot synthetic samples
        plt.subplot(2, 1, 2)
        for i in range(min(10, len(synthetic_data))):
            plt.plot(synthetic_data[i])
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.title('Synthetic Transit Samples')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "sample_comparison.png"))
        plt.close()
        
        # Compile validation results
        validation_results = {
            "num_real_samples": len(real_data),
            "num_synthetic_samples": len(synthetic_data),
            "mse": mse,
            "correlation": correlation
        }
        
        logger.info(f"Validation results: {validation_results}")
        
        return validation_results
    
    def run_gan_pipeline(self, num_epochs=100, num_synthetic_samples=1000):
        """
        Run the complete GAN pipeline.
        
        Parameters:
        -----------
        num_epochs : int
            Number of epochs to train
        num_synthetic_samples : int
            Number of synthetic samples to generate
            
        Returns:
        --------
        dict
            Dictionary containing pipeline results
        """
        logger.info("Starting GAN pipeline")
        
        # Step 1: Train the GAN
        training_stats = self.train(num_epochs=num_epochs)
        
        # Step 2: Generate synthetic dataset
        synthetic_files = self.generate_synthetic_dataset(num_samples=num_synthetic_samples)
        
        # Step 3: Validate synthetic samples
        validation_results = self.validate_synthetic_samples()
        
        # Compile pipeline results
        pipeline_results = {
            "training_stats": {
                "num_epochs": num_epochs,
                "final_G_loss": training_stats["G_losses"][-1] if training_stats["G_losses"] else None,
                "final_D_loss": training_stats["D_losses"][-1] if training_stats["D_losses"] else None,
                "final_D_accuracy": training_stats["D_accuracies"][-1] if training_stats["D_accuracies"] else None
            },
            "synthetic_dataset": {
                "num_samples": len(synthetic_files)
            },
            "validation_results": validation_results
        }
        
        # Save pipeline results
        with open(os.path.join(self.data_dir, "gan_pipeline_results.txt"), "w") as f:
            for section, results in pipeline_results.items():
                f.write(f"{section}:\n")
                for key, value in results.items():
                    f.write(f"  {key}: {value}\n")
        
        logger.info("GAN pipeline completed")
        logger.info(f"Pipeline results: {pipeline_results}")
        
        return pipeline_results


if __name__ == "__main__":
    # Example usage
    gan = TransitGAN()
    
    # For testing, run with fewer epochs and samples
    pipeline_results = gan.run_gan_pipeline(num_epochs=10, num_synthetic_samples=100)
    print(pipeline_results)
