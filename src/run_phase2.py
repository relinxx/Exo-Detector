# src/run_phase2.py

import os
import logging
import argparse
import json
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import glob

from advanced_augmentation import AdvancedTransitAugmentation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_from_windows(window_dir, window_size):
    """Loads and prepares data from window files for the CVAE-GAN."""
    all_flux_data = []
    window_files = glob.glob(os.path.join(window_dir, "*.csv"))

    if not window_files:
        raise FileNotFoundError(f"No window files found in {window_dir}. Run preprocessing first.")
        
    for f in window_files:
        df = pd.read_csv(f)
        flux = df['flux'].values
        if len(flux) < window_size:
            flux = np.pad(flux, (0, window_size - len(flux)), 'constant', constant_values=1.0)
        elif len(flux) > window_size:
            flux = flux[:window_size]
        all_flux_data.append(flux)
        
    flux_tensor = torch.tensor(np.array(all_flux_data), dtype=torch.float32).unsqueeze(1)
    dummy_conditions = torch.randn(len(flux_tensor), 10)
    
    return TensorDataset(flux_tensor, dummy_conditions)

def run_phase2(data_dir="data", num_epochs=50, num_synthetic_samples=1000, batch_size=32):
    """Runs the advanced augmentation pipeline using a CVAE-GAN."""
    logger.info("--- Starting Phase 2: Advanced Data Augmentation ---")
    
    window_size = 256
    data_dir = os.path.abspath(data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    transit_window_dir = os.path.join(data_dir, "transit_windows")
    try:
        train_dataset = load_data_from_windows(transit_window_dir, window_size=window_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        logger.info(f"Loaded {len(train_dataset)} transit windows for training.")
    except FileNotFoundError as e:
        logger.error(e)
        return

    augmenter = AdvancedTransitAugmentation(data_dir=data_dir, device=device, window_size=window_size)
    
    logger.info(f"Training CVAE-GAN for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_losses = {'vae_loss': [], 'd_loss': []}
        for real_data, conditions in train_loader:
            real_data, conditions = real_data.to(device), conditions.to(device)
            fake_conditions = torch.randn_like(conditions)
            
            losses = augmenter.train_step(real_data, conditions, fake_conditions)
            epoch_losses['vae_loss'].append(losses['vae_loss'])
            epoch_losses['d_loss'].append(losses['d_loss'])
            
        avg_vae_loss = np.mean(epoch_losses['vae_loss'])
        avg_d_loss = np.mean(epoch_losses['d_loss'])
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], VAE Loss: {avg_vae_loss:.4f}, Disc Loss: {avg_d_loss:.4f}")

    logger.info(f"Generating {num_synthetic_samples} synthetic samples...")
    synthetic_dir = os.path.join(data_dir, "synthetic_transits")
    os.makedirs(synthetic_dir, exist_ok=True)
    
    model = augmenter.model.to(device)
    model.eval()
    with torch.no_grad():
        for i in range(num_synthetic_samples):
            z = torch.randn(1, model.latent_dim).to(device)
            c = torch.randn(1, model.condition_dim).to(device)
            synthetic_flux = model.decoder(z, c).squeeze().cpu().numpy()
            
            df = pd.DataFrame({'flux': synthetic_flux})
            df.to_csv(os.path.join(synthetic_dir, f"synthetic_sample_{i}.csv"), index=False)
            
    logger.info("Phase 2 pipeline completed successfully.")
    summary = {'timestamp': datetime.now().isoformat(), 'model_used': 'CVAE-GAN'}
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Advanced Augmentation (Phase 2)")
    # *** DEFINITIVE FIX PART 1 ***
    # This section correctly defines the command-line arguments.
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples to generate")
    args = parser.parse_args()

    # *** DEFINITIVE FIX PART 2 ***
    # This section correctly passes your command-line arguments into the main function.
    run_phase2(
        data_dir=args.data_dir, 
        num_epochs=args.epochs, 
        num_synthetic_samples=args.samples
    )
