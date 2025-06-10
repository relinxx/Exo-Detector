import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoConfig

class TransferLearningTransitDetector:
    """Transfer learning from Kepler/PLATO to TESS data."""
    
    def __init__(self, source_domain="kepler", target_domain="tess"):
        self.source_domain = source_domain
        self.target_domain = target_domain
        
        # Initialize feature extractor
        self.feature_extractor = self._build_feature_extractor()
        self.domain_classifier = self._build_domain_classifier()
        self.transit_classifier = self._build_transit_classifier()
        
    def _build_feature_extractor(self):
        """Build shared feature extractor."""
        return nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128)
        )
        
    def train_with_domain_adaptation(self, source_loader, target_loader, epochs=100):
        """Train with domain adaptation using DANN (Domain Adversarial Neural Networks)."""
        optimizer_feat = torch.optim.Adam(self.feature_extractor.parameters(), lr=0.001)
        optimizer_transit = torch.optim.Adam(self.transit_classifier.parameters(), lr=0.001)
        optimizer_domain = torch.optim.Adam(self.domain_classifier.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            # Training loop with gradient reversal
            self._train_epoch_dann(
                source_loader, target_loader, 
                optimizer_feat, optimizer_transit, optimizer_domain,
                epoch, epochs
            )
            
    def _train_epoch_dann(self, source_loader, target_loader, 
                         opt_feat, opt_transit, opt_domain, epoch, total_epochs):
        """Single epoch of DANN training."""
        # Calculate lambda for gradient reversal layer
        p = float(epoch) / total_epochs
        lambda_domain = 2. / (1. + np.exp(-10 * p)) - 1
        
        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
            # Combine source and target data
            combined_data = torch.cat([source_data, target_data], dim=0)
            batch_size = source_data.size(0)
            
            # Domain labels
            domain_labels = torch.cat([
                torch.zeros(batch_size),  # Source domain = 0
                torch.ones(target_data.size(0))   # Target domain = 1
            ], dim=0)
            
            # Feature extraction
            features = self.feature_extractor(combined_data)
            
            # Transit classification (only for source data)
            source_features = features[:batch_size]
            transit_pred = self.transit_classifier(source_features)
            transit_loss = F.binary_cross_entropy_with_logits(
                transit_pred.squeeze(), source_labels.float()
            )
            
            # Domain classification with gradient reversal
            domain_pred = self.domain_classifier(
                GradientReversalLayer.apply(features, lambda_domain)
            )
            domain_loss = F.binary_cross_entropy_with_logits(
                domain_pred.squeeze(), domain_labels.float()
            )
            
            # Total loss
            total_loss = transit_loss + domain_loss
            
            # Backward pass
            opt_feat.zero_grad()
            opt_transit.zero_grad()
            opt_domain.zero_grad()
            
            total_loss.backward()
            
            opt_feat.step()
            opt_transit.step()
            opt_domain.step()

class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for domain adaptation."""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None
