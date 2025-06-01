# src/advanced_augmentation.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

class TransitEncoder(nn.Module):
    """Encoder for VAE component."""
    def __init__(self, input_size, latent_dim=64, condition_dim=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
        )
        
        # Correctly calculate flattened size based on the actual input_size
        self.flatten_size = 128 * (input_size // 8)
        self.fc_input = nn.Linear(self.flatten_size + condition_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x, conditions):
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        h = torch.cat([h, conditions], dim=1)
        h = F.relu(self.fc_input(h))
        return self.fc_mu(h), self.fc_logvar(h)

class TransitDecoder(nn.Module):
    """Decoder for VAE component."""
    def __init__(self, latent_dim, condition_dim, output_size):
        super().__init__()
        self.initial_size = output_size // 8
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * self.initial_size),
            nn.ReLU()
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z, conditions):
        h = torch.cat([z, conditions], dim=1)
        h = self.fc(h)
        h = h.view(h.size(0), 128, self.initial_size)
        return self.deconv_layers(h)

class TransitDiscriminator(nn.Module):
    """Discriminator for Conditional VAE-GAN."""
    def __init__(self, input_size, condition_dim=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
        )
        self.flatten_size = 128 * (input_size // 8)
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size + condition_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x, conditions):
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        h = torch.cat([h, conditions], dim=1)
        return self.fc(h)

class ConditionalVAEGAN(nn.Module):
    """Conditional VAE-GAN for transit augmentation."""
    def __init__(self, input_size, latent_dim=64, condition_dim=10):
        super().__init__()
        self.encoder = TransitEncoder(input_size, latent_dim, condition_dim)
        self.decoder = TransitDecoder(latent_dim, condition_dim, input_size)
        self.discriminator = TransitDiscriminator(input_size, condition_dim)
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, conditions):
        mu, logvar = self.encoder(x, conditions)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, conditions)
        return recon_x, mu, logvar, z

class AdvancedTransitAugmentation:
    """Wrapper class for the CVAE-GAN training process."""
    def __init__(self, data_dir="data", device="cuda", window_size=256): # Accept window_size
        self.data_dir = data_dir
        self.device = device
        
        # *** DEFINITIVE FIX PART 2 ***
        # Initialize the model with the correct window_size
        self.model = ConditionalVAEGAN(input_size=window_size).to(device)
        
        self.optimizer_vae = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
            lr=0.001, betas=(0.9, 0.999)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=0.0004, betas=(0.5, 0.999)
        )

    def train_step(self, real_data, conditions, fake_conditions):
        """Single training step."""
        batch_size = real_data.size(0)
        
        # VAE/Generator training
        self.optimizer_vae.zero_grad()
        recon_data, mu, logvar, z = self.model(real_data, conditions)
        
        recon_loss = F.mse_loss(recon_data, real_data)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        z_fake = torch.randn(batch_size, self.model.latent_dim).to(self.device)
        fake_data = self.model.decoder(z_fake, fake_conditions)
        fake_pred_gen = self.model.discriminator(fake_data, fake_conditions)
        g_loss = F.binary_cross_entropy_with_logits(fake_pred_gen, torch.ones_like(fake_pred_gen))
        
        vae_loss = recon_loss + 0.1 * kl_loss + 0.01 * g_loss
        vae_loss.backward()
        self.optimizer_vae.step()
        
        # Discriminator training
        self.optimizer_disc.zero_grad()
        real_pred = self.model.discriminator(real_data.detach(), conditions)
        d_loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        
        fake_pred = self.model.discriminator(fake_data.detach(), fake_conditions)
        d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.optimizer_disc.step()
        
        return {'vae_loss': vae_loss.item(), 'd_loss': d_loss.item()}
