# src/transformer_anomaly_detection.py

import os
import glob
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

# Configure logger
logger = logging.getLogger(__name__)

# --- Transformer Model Components ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = nn.Linear(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, src):
        src = self.encoder(src) * np.sqrt(self.encoder.out_features)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

# --- Main Anomaly Detector Class ---
class EnhancedAnomalyDetector:
    def __init__(self, data_dir="data", window_size=256):
        self.data_dir = os.path.abspath(data_dir)
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.transformer_ae = TransformerAutoencoder().to(self.device)
        self.isolation_forest = IsolationForest(contamination='auto', random_state=42)
        
        # Directories for loading data
        self.non_transit_dir = os.path.join(self.data_dir, "non_transit_windows")
        self.transit_dir = os.path.join(self.data_dir, "transit_windows")
        self.synthetic_dir = os.path.join(self.data_dir, "synthetic_transits")
        
        self.normal_training_data = None
        logger.info(f"EnhancedAnomalyDetector initialized for device: {self.device}")

    def _load_windows_from_dir(self, directory):
        """Helper function to load all window files from a directory."""
        files = glob.glob(os.path.join(directory, "*.csv"))
        windows = []
        for f in files:
            df = pd.read_csv(f)
            flux = df['flux'].values
            if len(flux) < self.window_size:
                flux = np.pad(flux, (0, self.window_size - len(flux)), 'constant', constant_values=1.0)
            windows.append(flux[:self.window_size])
        return np.array(windows)

    def prepare_data(self):
        """
        Loads normal and anomaly data, splits it, and creates PyTorch DataLoaders.
        This is the missing function that caused the error.
        """
        logger.info("Preparing data for anomaly detection...")
        
        # Load normal data (non-transits)
        normal_data = self._load_windows_from_dir(self.non_transit_dir)
        if len(normal_data) == 0:
            raise FileNotFoundError("No normal data (non-transit windows) found. Please run preprocessing.")
        
        # Load anomaly data (real transits + synthetic transits)
        real_anomalies = self._load_windows_from_dir(self.transit_dir)
        synth_anomalies = self._load_windows_from_dir(self.synthetic_dir)
        anomaly_data = np.concatenate((real_anomalies, synth_anomalies), axis=0)
        
        # Create labels: 0 for normal, 1 for anomaly
        normal_labels = np.zeros(len(normal_data))
        anomaly_labels = np.ones(len(anomaly_data))

        # Split normal data: 80% for training, 20% for testing
        train_normal, test_normal, _, _ = train_test_split(
            normal_data, normal_labels, test_size=0.2, random_state=42)
        
        self.normal_training_data = train_normal # Save for later use

        # Combine test data
        test_data = np.concatenate((test_normal, anomaly_data))
        test_labels = np.concatenate((np.zeros(len(test_normal)), np.ones(len(anomaly_data))))
        
        # Create PyTorch Datasets
        train_dataset = TensorDataset(torch.from_numpy(train_normal).float().unsqueeze(-1))
        test_dataset = TensorDataset(torch.from_numpy(test_data).float().unsqueeze(-1), torch.from_numpy(test_labels).float())

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"Data prepared: {len(train_dataset)} training samples (normal), {len(test_dataset)} testing samples (mixed).")
        return train_loader, test_loader
        
    def get_normal_data_array(self):
        """Returns the normal training data for classical models."""
        if self.normal_training_data is None:
            raise ValueError("Normal training data not available. Run prepare_data() first.")
        return self.normal_training_data

    def train_transformer_ae(self, train_loader, epochs=50):
        """Trains the Transformer Autoencoder on normal data."""
        optimizer = torch.optim.Adam(self.transformer_ae.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.transformer_ae.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                data = batch[0].to(self.device)
                optimizer.zero_grad()
                output = self.transformer_ae(data)
                loss = criterion(output, data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")

    def train_classical_detectors(self, normal_data):
        """Trains the Isolation Forest on the normal data."""
        logger.info("Training Isolation Forest...")
        self.isolation_forest.fit(normal_data)

    def detect_anomalies(self, test_loader):
        """Detects anomalies using the trained models."""
        self.transformer_ae.eval()
        all_labels = []
        transformer_scores = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                reconstructed = self.transformer_ae(data)
                loss = torch.mean((data - reconstructed) ** 2, dim=1).squeeze()
                transformer_scores.extend(loss.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Get scores from Isolation Forest
        if_scores = -self.isolation_forest.score_samples(test_loader.dataset.tensors[0].squeeze().numpy())

        # Combine scores (simple averaging)
        final_scores = (np.array(transformer_scores) + if_scores) / 2
        return final_scores, np.array(all_labels)

