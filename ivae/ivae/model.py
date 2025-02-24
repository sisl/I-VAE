import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class StateEncoder(nn.Module):
    def __init__(self, state_channels, latent_dim, dropout_rate):
        super(StateEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(state_channels, 32, kernel_size=4, stride=2, padding=1),  # 80x80 -> 40x40
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 40x40 -> 20x20
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 20x20 -> 10x10
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 2 * latent_dim)  # Output mean and log variance
        )

    def forward(self, state):
        params = self.encoder(state)
        mean, log_var = torch.chunk(params, 2, dim=1)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, state_channels, dropout_rate):
        super(Decoder, self).__init__()
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128 * 10 * 10),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 10x10 -> 20x20
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 20x20 -> 40x40
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.ConvTranspose2d(32, state_channels, kernel_size=4, stride=2, padding=1),  # 40x40 -> 80x80
            nn.Sigmoid()  # Assuming binary images
        )

    def forward(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 10, 10)
        x = self.decoder_conv(x)
        return x


class ObsEncoder(nn.Module):
    def __init__(self, obs_channels, latent_dim, dropout_rate):
        super(ObsEncoder, self).__init__()
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(in_channels=obs_channels, out_channels=16, kernel_size=5, stride=1, padding=2),  # (200, 200) -> (200, 200)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=2),  # (200, 200) -> (100, 100)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # (100, 100) -> (100, 100)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=2),  # (100, 100) -> (50, 50)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),  # (50, 50) -> (46, 46)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=2),  # (46, 46) -> (23, 23)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),  # (23, 23) -> (21, 21)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=2),  # (21, 21) -> (10, 10)

            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 2 * latent_dim)  # Output mean and log variance
        )

    def forward(self, obs):
        params = self.obs_encoder(obs)
        mean, log_var = torch.chunk(params, 2, dim=1)
        return mean, log_var


class IVAE(nn.Module):
    def __init__(self, state_channels, obs_channels, latent_dim, dropout_rate):
        super(IVAE, self).__init__()
        self.state_encoder = StateEncoder(state_channels, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, state_channels, dropout_rate)
        self.obs_encoder = ObsEncoder(obs_channels, latent_dim, dropout_rate)
        self.latent_dim = latent_dim

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, state, obs):
        qs_mean, qs_log_var = self.state_encoder(state) # State Encoder: q(z|s)
        z = self.reparameterize(qs_mean, qs_log_var)    # Reparameterization trick
        recon_state = self.decoder(z)                   # Decoder: p(s|z)
        qo_mean, qo_log_var = self.obs_encoder(obs)     # Prior: q(z|o)
        return recon_state, qs_mean, qs_log_var, qo_mean, qo_log_var

    def sample(self, obs_or_belief, m=1, thresholded=True):
        # Example of sampling from the model after training
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(obs_or_belief, tuple):
                qo_mean, qo_log_var = obs_or_belief
            else:
                obs = obs_or_belief
                obs = obs.to(device)
                # Get prior parameters from observation
                qo_mean, qo_log_var = self.obs_encoder(obs)
            if m > 1:
                qo_mean = qo_mean.repeat(m, 1)
                qo_log_var = qo_log_var.repeat(m, 1)
            # Sample z from the prior
            z = self.reparameterize(qo_mean, qo_log_var)
            # Generate state sample
            sampled_states = self.decoder(z)
        if thresholded:
            return (sampled_states > 0.5).float()
        else:
            return sampled_states

    def update(self, obs):
        # Example of sampling from the self after training
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            obs = obs.to(device)
            # Get prior parameters from observation
            qo_mean, qo_log_var = self.obs_encoder(obs)
        return qo_mean, qo_log_var


# Define the VAE Loss Function
def ivae_loss(recon_state, state, qs_mean, qs_log_var, qo_mean, qo_log_var, beta=1.0):
    """
    Computes the loss for the I-VAE.

    Args:
        recon_state (torch.Tensor): Reconstructed state
        state (torch.Tensor): Original state
        qs_mean (torch.Tensor): Mean of state encoder distribution
        qs_log_var (torch.Tensor): Log variance of state encoder distribution
        qo_mean (torch.Tensor): Mean of observation encoder distribution
        qo_log_var (torch.Tensor): Log variance of observation encoder distribution
    Returns:
        loss (torch.Tensor): Total loss
    """
    # Reconstruction loss (binary cross-entropy for binary images)
    recon_loss = F.binary_cross_entropy(recon_state, state, reduction='sum')

    # KL divergence between q(z|s) and q(z|o)
    kl_div = 0.5 * torch.sum(
        qo_log_var - qs_log_var +
        (torch.exp(qs_log_var) + (qs_mean - qo_mean) ** 2) / torch.exp(qo_log_var) - 1
    )
    return recon_loss + beta*kl_div


def train(model, dataloader, val_dataloader, optimizer, device, epoch, epochs, best_loss=None, freq=1):
    model.train()
    total_loss = 0
    progress_trigger = epoch % freq == 0
    if progress_trigger:
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader
    for (state, obs) in pbar:
        state = state.to(device)
        obs = obs.to(device)
        optimizer.zero_grad()
        recon_state, qs_mean, qs_log_var, qo_mean, qo_log_var = model(state, obs)
        loss = ivae_loss(recon_state, state, qs_mean, qs_log_var, qo_mean, qo_log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if progress_trigger:
            pbar.set_postfix({f'[{epoch}/{epochs}] Loss': total_loss / len(dataloader.dataset)})
    average_loss = total_loss / len(dataloader.dataset)

    ######################
    # Validate the model #
    ######################
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        if progress_trigger:
            pbar = tqdm(val_dataloader)
        else:
            pbar = val_dataloader
        for (state, obs) in pbar:
            state = state.to(device)
            obs = obs.to(device)
            recon_state, qs_mean, qs_log_var, qo_mean, qo_log_var = model(state, obs)
            loss = ivae_loss(recon_state, state, qs_mean, qs_log_var, qo_mean, qo_log_var)
            val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader.dataset)
            if progress_trigger:
                if best_loss is not None:
                    pbar.set_postfix({f'[{epoch}/{epochs}] Loss': average_loss, 'Val. Loss': avg_val_loss, 'Best Val. Loss': best_loss})
                else:
                    pbar.set_postfix({f'[{epoch}/{epochs}] Loss': average_loss, 'Val. Loss': avg_val_loss})

    return avg_val_loss
