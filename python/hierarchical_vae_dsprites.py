"""
Hierarchical VAE Experiments for Proximal Dominance Validation

This script implements a two-layer hierarchical VAE on dSprites to validate
the Proximal Dominance Principle: MFVI failure is determined by proximal
coupling, not distal structure.

Requirements:
    pip install torch torchvision numpy matplotlib tqdm

Data:
    Download dSprites: https://github.com/deepmind/dsprites-dataset
    wget https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HierarchicalVAEConfig:
    """Configuration for hierarchical VAE experiments."""
    # Architecture
    z1_dim: int = 10          # Proximal latent dimension
    z2_dim: int = 5           # Distal latent dimension
    hidden_dim: int = 256     # Hidden layer dimension
    
    # Proximal coupling control
    # Higher values = stronger coupling from z2 to z1
    # proximal_coupling = 0 means z1 prior is independent of z2
    proximal_coupling: float = 1.0
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 50
    beta: float = 1.0         # KL weight (beta-VAE)
    
    # Data
    data_path: str = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    image_size: int = 64
    
    # Reproducibility
    seed: int = 42


# =============================================================================
# Model Components
# =============================================================================

class Encoder(nn.Module):
    """
    Encoder: x -> (z1, z2)
    
    Outputs parameters for both latent levels.
    """
    def __init__(self, config: HierarchicalVAEConfig):
        super().__init__()
        self.config = config
        
        # Convolutional encoder
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 8 -> 4
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent dimension: 256 * 4 * 4 = 4096
        conv_out_dim = 256 * 4 * 4
        
        # z2 (distal) parameters
        self.fc_z2_mu = nn.Linear(conv_out_dim, config.z2_dim)
        self.fc_z2_logvar = nn.Linear(conv_out_dim, config.z2_dim)
        
        # z1 (proximal) parameters - depend on z2
        self.fc_z1_mu = nn.Linear(conv_out_dim + config.z2_dim, config.z1_dim)
        self.fc_z1_logvar = nn.Linear(conv_out_dim + config.z2_dim, config.z1_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Images [batch, 1, 64, 64]
        
        Returns:
            Dictionary with z1_mu, z1_logvar, z2_mu, z2_logvar
        """
        h = self.conv(x)
        
        # z2 (distal) - depends only on x
        z2_mu = self.fc_z2_mu(h)
        z2_logvar = self.fc_z2_logvar(h)
        
        # Sample z2 for z1 conditioning
        z2 = self.reparameterize(z2_mu, z2_logvar)
        
        # z1 (proximal) - depends on both x and z2
        h_z1 = torch.cat([h, z2], dim=-1)
        z1_mu = self.fc_z1_mu(h_z1)
        z1_logvar = self.fc_z1_logvar(h_z1)
        
        return {
            'z1_mu': z1_mu,
            'z1_logvar': z1_logvar,
            'z2_mu': z2_mu,
            'z2_logvar': z2_logvar,
            'z2_sample': z2
        }
    
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class HierarchicalPrior(nn.Module):
    """
    Hierarchical prior: p(z2) = N(0, I), p(z1 | z2) = N(f(z2), sigma^2 I)
    
    The proximal_coupling parameter controls sigma^2:
    - High coupling (low sigma^2): z1 is strongly determined by z2
    - Low coupling (high sigma^2): z1 is nearly independent of z2
    """
    def __init__(self, config: HierarchicalVAEConfig):
        super().__init__()
        self.config = config
        
        # p(z1 | z2) mean function: f(z2) = W @ z2
        self.z2_to_z1 = nn.Linear(config.z2_dim, config.z1_dim, bias=False)
        
        # Initialize to scaled identity-like mapping
        nn.init.orthogonal_(self.z2_to_z1.weight)
        
    def get_z1_prior(self, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute p(z1 | z2) parameters.
        
        Args:
            z2: Samples from z2 [batch, z2_dim]
        
        Returns:
            z1_prior_mu: [batch, z1_dim]
            z1_prior_logvar: [batch, z1_dim] (constant based on coupling)
        """
        # Mean depends on z2
        z1_prior_mu = self.z2_to_z1(z2) * self.config.proximal_coupling
        
        # Variance is constant, controlled by proximal_coupling
        # Higher coupling -> lower variance -> z1 more determined by z2
        # We parameterize as: sigma^2 = 1 / (1 + kappa^2) where kappa = proximal_coupling
        variance = 1.0 / (1.0 + self.config.proximal_coupling ** 2)
        z1_prior_logvar = torch.full_like(z1_prior_mu, np.log(variance))
        
        return z1_prior_mu, z1_prior_logvar


class Decoder(nn.Module):
    """
    Decoder: z1 -> x
    
    Note: Only z1 (proximal) generates observations.
    z2 (distal) influences x only through z1.
    """
    def __init__(self, config: HierarchicalVAEConfig):
        super().__init__()
        self.config = config
        
        self.fc = nn.Linear(config.z1_dim, 256 * 4 * 4)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),     # 32 -> 64
            nn.Sigmoid()
        )
    
    def forward(self, z1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: Proximal latent [batch, z1_dim]
        
        Returns:
            x_recon: Reconstructed images [batch, 1, 64, 64]
        """
        h = self.fc(z1)
        h = h.view(-1, 256, 4, 4)
        return self.deconv(h)


class HierarchicalVAE(nn.Module):
    """
    Two-layer Hierarchical VAE for Proximal Dominance experiments.
    
    Architecture:
        Prior: p(z2) = N(0, I), p(z1 | z2) = N(f(z2), sigma^2 I)
        Likelihood: p(x | z1) = Bernoulli(decoder(z1))
        Posterior: q(z1, z2 | x) approx q(z1 | x, z2) q(z2 | x)
    
    The key parameter is proximal_coupling:
        - Controls how strongly z2 influences z1
        - When 0: z1 and z2 are independent in the prior
        - When high: z1 is largely determined by z2
    """
    def __init__(self, config: HierarchicalVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.prior = HierarchicalPrior(config)
        self.decoder = Decoder(config)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass computing reconstruction and latent distributions."""
        # Encode
        enc_out = self.encoder(x)
        
        # Sample z1
        z1 = self.encoder.reparameterize(enc_out['z1_mu'], enc_out['z1_logvar'])
        
        # Decode
        x_recon = self.decoder(z1)
        
        # Get prior parameters for z1
        z1_prior_mu, z1_prior_logvar = self.prior.get_z1_prior(enc_out['z2_sample'])
        
        return {
            'x_recon': x_recon,
            'z1': z1,
            'z2': enc_out['z2_sample'],
            'z1_mu': enc_out['z1_mu'],
            'z1_logvar': enc_out['z1_logvar'],
            'z2_mu': enc_out['z2_mu'],
            'z2_logvar': enc_out['z2_logvar'],
            'z1_prior_mu': z1_prior_mu,
            'z1_prior_logvar': z1_prior_logvar
        }
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss with hierarchical KL terms.
        
        ELBO = E_q[log p(x|z1)] - KL(q(z1|x,z2) || p(z1|z2)) - KL(q(z2|x) || p(z2))
        """
        # Reconstruction loss (binary cross-entropy for dSprites)
        recon_loss = F.binary_cross_entropy(
            outputs['x_recon'], x, reduction='sum'
        ) / x.shape[0]
        
        # KL for z2: KL(q(z2|x) || p(z2)) where p(z2) = N(0, I)
        kl_z2 = -0.5 * torch.sum(
            1 + outputs['z2_logvar'] - outputs['z2_mu'].pow(2) - outputs['z2_logvar'].exp(),
            dim=-1
        ).mean()
        
        # KL for z1: KL(q(z1|x,z2) || p(z1|z2))
        # Both are Gaussian, so closed form
        kl_z1 = self._gaussian_kl(
            outputs['z1_mu'], outputs['z1_logvar'],
            outputs['z1_prior_mu'], outputs['z1_prior_logvar']
        ).mean()
        
        # Total loss
        total_loss = recon_loss + self.config.beta * (kl_z1 + kl_z2)
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_z1': kl_z1,
            'kl_z2': kl_z2
        }
    
    @staticmethod
    def _gaussian_kl(mu1, logvar1, mu2, logvar2):
        """KL divergence between two diagonal Gaussians."""
        var1 = logvar1.exp()
        var2 = logvar2.exp()
        kl = 0.5 * (
            logvar2 - logvar1 
            + var1 / var2 
            + (mu1 - mu2).pow(2) / var2 
            - 1
        )
        return kl.sum(dim=-1)


# =============================================================================
# Metrics
# =============================================================================

def compute_active_units(model: HierarchicalVAE, dataloader: DataLoader, 
                         threshold: float = 0.01) -> Dict[str, int]:
    """
    Count active units in each latent layer.
    
    A unit is active if its variance across the dataset exceeds threshold.
    (Following Burda et al., 2015)
    """
    model.eval()
    z1_samples = []
    z2_samples = []
    
    with torch.no_grad():
        for x, in dataloader:
            x = x.to(next(model.parameters()).device)
            outputs = model(x)
            z1_samples.append(outputs['z1_mu'].cpu())
            z2_samples.append(outputs['z2_mu'].cpu())
    
    z1_all = torch.cat(z1_samples, dim=0)
    z2_all = torch.cat(z2_samples, dim=0)
    
    # Variance across dataset
    z1_var = z1_all.var(dim=0)
    z2_var = z2_all.var(dim=0)
    
    active_z1 = (z1_var > threshold).sum().item()
    active_z2 = (z2_var > threshold).sum().item()
    
    return {
        'active_z1': active_z1,
        'active_z2': active_z2,
        'z1_var': z1_var.numpy(),
        'z2_var': z2_var.numpy()
    }


def compute_cf_between_layers(model: HierarchicalVAE, dataloader: DataLoader) -> float:
    """
    Compute Circulatory Fidelity between z1 and z2 samples.
    
    Uses the Linfoot correlation as CF proxy: r_L = |rho| for Gaussians.
    """
    model.eval()
    z1_samples = []
    z2_samples = []
    
    with torch.no_grad():
        for x, in dataloader:
            x = x.to(next(model.parameters()).device)
            outputs = model(x)
            z1_samples.append(outputs['z1'].cpu())
            z2_samples.append(outputs['z2'].cpu())
    
    z1_all = torch.cat(z1_samples, dim=0).numpy()
    z2_all = torch.cat(z2_samples, dim=0).numpy()
    
    # Compute maximum absolute correlation between any z1 and z2 dimension
    max_corr = 0.0
    for i in range(z1_all.shape[1]):
        for j in range(z2_all.shape[1]):
            corr = np.corrcoef(z1_all[:, i], z2_all[:, j])[0, 1]
            if np.isfinite(corr):
                max_corr = max(max_corr, abs(corr))
    
    return max_corr


# =============================================================================
# Training
# =============================================================================

def train_epoch(model: HierarchicalVAE, dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl_z1 = 0
    total_kl_z2 = 0
    
    for x, in dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        
        outputs = model(x)
        losses = model.compute_loss(x, outputs)
        
        losses['loss'].backward()
        optimizer.step()
        
        total_loss += losses['loss'].item()
        total_recon += losses['recon_loss'].item()
        total_kl_z1 += losses['kl_z1'].item()
        total_kl_z2 += losses['kl_z2'].item()
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'recon_loss': total_recon / n,
        'kl_z1': total_kl_z1 / n,
        'kl_z2': total_kl_z2 / n
    }


def run_experiment(config: HierarchicalVAEConfig, device: torch.device) -> Dict:
    """
    Run single experiment with given configuration.
    
    Returns metrics for Proximal Dominance analysis.
    """
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Load data
    data = np.load(config.data_path, allow_pickle=True)
    images = data['imgs'].astype(np.float32)
    images = images[:, np.newaxis, :, :]  # Add channel dim
    
    # Use subset for faster experiments
    n_samples = min(100000, len(images))
    images = images[:n_samples]
    
    dataset = TensorDataset(torch.from_numpy(images))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Create model
    model = HierarchicalVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    history = []
    for epoch in tqdm(range(config.num_epochs), desc=f"kappa={config.proximal_coupling}"):
        metrics = train_epoch(model, dataloader, optimizer, device)
        history.append(metrics)
    
    # Final evaluation
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    active_units = compute_active_units(model, eval_loader)
    cf = compute_cf_between_layers(model, eval_loader)
    
    return {
        'config': config,
        'history': history,
        'final_loss': history[-1]['loss'],
        'final_kl_z1': history[-1]['kl_z1'],
        'final_kl_z2': history[-1]['kl_z2'],
        'active_z1': active_units['active_z1'],
        'active_z2': active_units['active_z2'],
        'cf_z1_z2': cf
    }


# =============================================================================
# Main Experiment: Proximal Dominance Sweep
# =============================================================================

def run_proximal_dominance_sweep(coupling_values: List[float] = [0.0, 0.5, 1.0, 2.0, 4.0],
                                  data_path: str = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"):
    """
    Main experiment: vary proximal coupling strength and measure effects.
    
    Hypothesis (Proximal Dominance Principle):
    - Low proximal coupling -> z2 collapses (low active units, KL_z2 -> 0)
    - High proximal coupling -> z2 is utilized
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = []
    
    for coupling in coupling_values:
        print(f"\n{'='*50}")
        print(f"Proximal Coupling kappa = {coupling}")
        print('='*50)
        
        config = HierarchicalVAEConfig(
            proximal_coupling=coupling,
            data_path=data_path,
            num_epochs=30  # Reduced for faster experimentation
        )
        
        result = run_experiment(config, device)
        results.append(result)
        
        print(f"Final Loss: {result['final_loss']:.4f}")
        print(f"KL(z1): {result['final_kl_z1']:.4f}, KL(z2): {result['final_kl_z2']:.4f}")
        print(f"Active z1: {result['active_z1']}/{config.z1_dim}, Active z2: {result['active_z2']}/{config.z2_dim}")
        print(f"CF(z1, z2): {result['cf_z1_z2']:.4f}")
    
    return results


def plot_results(results: List[Dict], save_path: str = "proximal_dominance_results.png"):
    """Visualize Proximal Dominance experiment results."""
    
    couplings = [r['config'].proximal_coupling for r in results]
    kl_z2 = [r['final_kl_z2'] for r in results]
    active_z2 = [r['active_z2'] for r in results]
    cf = [r['cf_z1_z2'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Panel A: KL divergence for z2
    axes[0].plot(couplings, kl_z2, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Proximal Coupling (kappa)')
    axes[0].set_ylabel('KL(q(z2|x) || p(z2))')
    axes[0].set_title('(A) Distal KL Divergence')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Panel B: Active units in z2
    axes[1].bar(range(len(couplings)), active_z2, tick_label=[str(c) for c in couplings])
    axes[1].set_xlabel('Proximal Coupling (kappa)')
    axes[1].set_ylabel('Active Units in z2')
    axes[1].set_title('(B) Distal Layer Utilization')
    axes[1].axhline(results[0]['config'].z2_dim, color='red', linestyle='--', 
                    label=f'Max ({results[0]["config"].z2_dim})')
    axes[1].legend()
    
    # Panel C: CF between layers
    axes[2].plot(couplings, cf, 's-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Proximal Coupling (kappa)')
    axes[2].set_ylabel('CF(z1, z2)')
    axes[2].set_title('(C) Inter-Layer Coupling')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to {save_path}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hierarchical VAE Proximal Dominance Experiments")
    parser.add_argument('--data', type=str, 
                        default="dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
                        help="Path to dSprites dataset")
    parser.add_argument('--couplings', type=float, nargs='+', 
                        default=[0.0, 0.5, 1.0, 2.0, 4.0],
                        help="Proximal coupling values to test")
    args = parser.parse_args()
    
    # Check data exists
    if not Path(args.data).exists():
        print("dSprites dataset not found. Download with:")
        print("wget https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
        exit(1)
    
    # Run experiments
    results = run_proximal_dominance_sweep(args.couplings, args.data)
    
    # Visualize
    plot_results(results)
    
    # Summary table
    print("\n" + "="*70)
    print("PROXIMAL DOMINANCE EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'kappa':<8} {'KL(z2)':<12} {'Active z2':<12} {'CF(z1,z2)':<12}")
    print("-"*70)
    for r in results:
        print(f"{r['config'].proximal_coupling:<8.1f} {r['final_kl_z2']:<12.4f} {r['active_z2']:<12} {r['cf_z1_z2']:<12.4f}")
    print("="*70)
    print("\nPrediction: Low kappa -> z2 collapse (KL->0, few active units)")
    print("           High kappa -> z2 utilized (KL>0, more active units)")
