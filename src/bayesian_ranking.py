from venv import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class BayesianTransitClassifier(nn.Module):
    """Bayesian neural network for transit classification."""
    
    def __init__(self, input_dim=200, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        # Define layers with weight uncertainty
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                BayesianLinear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(BayesianLinear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, num_samples=1):
        """Forward pass with uncertainty sampling."""
        if num_samples == 1:
            return torch.sigmoid(self.network(x))
        else:
            outputs = []
            for _ in range(num_samples):
                output = torch.sigmoid(self.network(x))
                outputs.append(output)
            return torch.stack(outputs, dim=0)

class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty."""
    
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 2)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 2)
        
        # Prior parameters
        self.prior_std = prior_std
        
    def forward(self, x):
        # Sample weights and biases
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
        
        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        
        return F.linear(x, weight, bias)
        
    def kl_divergence(self):
        """Compute KL divergence from prior."""
        weight_kl = self._kl_divergence(self.weight_mu, self.weight_logvar)
        bias_kl = self._kl_divergence(self.bias_mu, self.bias_logvar)
        return weight_kl + bias_kl
        
    def _kl_divergence(self, mu, logvar):
        """KL divergence between Gaussian and standard normal prior."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class BayesianEnsembleRanker:
    """Bayesian ensemble for candidate ranking with uncertainty quantification."""
    
    def __init__(self, input_dim=200, num_models=5):
        self.num_models = num_models
        self.models = []
        self.gp_regressors = []
        
        # Initialize Bayesian neural networks
        for _ in range(num_models):
            model = BayesianTransitClassifier(input_dim)
            self.models.append(model)
            
        # Initialize Gaussian Process regressors for different features
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        for _ in range(3):  # For period, depth, duration
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
            self.gp_regressors.append(gp)
            
    def train_ensemble(self, train_loader, epochs=100):
        """Train ensemble of Bayesian models."""
        optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) 
                     for model in self.models]
        
        for epoch in range(epochs):
            total_losses = [0.0] * self.num_models
            
            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.view(data.size(0), -1)  # Flatten
                labels = labels.float()
                
                for i, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output = model(data).squeeze()
                    
                    # Likelihood loss
                    likelihood_loss = F.binary_cross_entropy(output, labels)
                    
                    # KL divergence loss
                    kl_loss = 0
                    for layer in model.network:
                        if isinstance(layer, BayesianLinear):
                            kl_loss += layer.kl_divergence()
                    
                    kl_loss /= len(train_loader.dataset)
                    
                    # Total loss
                    total_loss = likelihood_loss + 0.01 * kl_loss
                    total_loss.backward()
                    optimizer.step()
                    
                    total_losses[i] += total_loss.item()
                    
            if epoch % 10 == 0:
                avg_losses = [loss / len(train_loader) for loss in total_losses]
                logger.info(f"Epoch {epoch}, Average losses: {avg_losses}")
                
    def predict_with_uncertainty(self, data, num_samples=100):
        """Predict with uncertainty quantification."""
        data = data.view(data.size(0), -1)  # Flatten
        
        # Collect predictions from all models
        ensemble_predictions = []
        ensemble_uncertainties = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                # Sample multiple predictions
                predictions = model(data, num_samples=num_samples)
                
                # Calculate mean and uncertainty
                mean_pred = torch.mean(predictions, dim=0)
                uncertainty = torch.std(predictions, dim=0)
                
                ensemble_predictions.append(mean_pred)
                ensemble_uncertainties.append(uncertainty)
                
        # Aggregate ensemble results
        final_predictions = torch.stack(ensemble_predictions, dim=0)
        final_uncertainties = torch.stack(ensemble_uncertainties, dim=0)
        
        # Model averaging
        ensemble_mean = torch.mean(final_predictions, dim=0)
        ensemble_uncertainty = torch.sqrt(
            torch.mean(final_uncertainties**2, dim=0) +  # Aleatoric uncertainty
            torch.var(final_predictions, dim=0)          # Epistemic uncertainty
        )
        
        return ensemble_mean, ensemble_uncertainty
        
    def rank_candidates(self, candidates_data, features):
        """Rank candidates using Bayesian ensemble."""
        # Neural network predictions
        predictions, uncertainties = self.predict_with_uncertainty(candidates_data)
        
        # Gaussian Process predictions for physical features
        gp_predictions = []
        gp_uncertainties = []
        
        feature_names = ['period', 'depth', 'duration']
        for i, (gp, feature_name) in enumerate(zip(self.gp_regressors, feature_names)):
            if feature_name in features.columns:
                feature_values = features[feature_name].values.reshape(-1, 1)
                gp_pred, gp_std = gp.predict(feature_values, return_std=True)
                gp_predictions.append(gp_pred)
                gp_uncertainties.append(gp_std)
                
        # Combine neural network and GP predictions
        combined_scores = predictions.cpu().numpy()
        combined_uncertainties = uncertainties.cpu().numpy()
        
        # Add GP contributions if available
        if gp_predictions:
            gp_weight = 0.3
            nn_weight = 0.7
            
            gp_ensemble = np.mean(gp_predictions, axis=0)
            combined_scores = nn_weight * combined_scores + gp_weight * gp_ensemble
            
        # Calculate ranking scores with uncertainty penalty
        uncertainty_penalty = 0.1
        ranking_scores = combined_scores - uncertainty_penalty * combined_uncertainties
        
        # Create ranking with confidence intervals
        results = []
        for i, score in enumerate(ranking_scores):
            results.append({
                'candidate_id': i,
                'score': float(score),
                'uncertainty': float(combined_uncertainties[i]),
                'lower_bound': float(score - 1.96 * combined_uncertainties[i]),
                'upper_bound': float(score + 1.96 * combined_uncertainties[i])
            })
            
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
