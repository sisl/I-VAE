import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

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
            nn.Linear(128 * 10 * 10 + latent_dim, 256),
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

    def forward(self, z, obs_h):
        z = torch.cat([z, obs_h], dim=1)
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
            # nn.Linear(128 * 10 * 10, 2 * latent_dim)
        )
        self.obs_fc = nn.Linear(128 * 10 * 10, 2 * latent_dim)  # Output mean and log variance

    def forward(self, obs):
        h = self.obs_encoder(obs)
        params = self.obs_fc(h)
        mean, log_var = torch.chunk(params, 2, dim=1)
        return h, mean, log_var


class IVAE(nn.Module):
    def __init__(self, state_channels, obs_channels, latent_dim, dropout_rate):
        super(IVAE, self).__init__()
        self.state_encoder = StateEncoder(state_channels, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, state_channels, dropout_rate)
        self.obs_encoder = ObsEncoder(obs_channels, latent_dim, dropout_rate)
        self.latent_dim = latent_dim
        self.tau = 1.0

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        std_T = std * math.sqrt(self.tau)
        return mean + std_T * epsilon

    def forward(self, state, obs):
        qs_mean, qs_log_var = self.state_encoder(state) # State Encoder: q(z|s)
        z = self.reparameterize(qs_mean, qs_log_var)    # Reparameterization trick
        obs_h, qo_mean, qo_log_var = self.obs_encoder(obs)     # Prior: q(z|o)
        recon_state = self.decoder(z, obs_h)                   # Decoder: p(s|z)
        return recon_state, qs_mean, qs_log_var, qo_mean, qo_log_var

    def sample(self, obs_or_belief, m=1, thresholded=True):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(obs_or_belief, tuple):
                obs_h, qo_mean, qo_log_var = obs_or_belief
            else:
                obs = obs_or_belief
                obs = obs.to(device)
                obs_h, qo_mean, qo_log_var = self.obs_encoder(obs)
            if m > 1:
                obs_h = obs_h.repeat(m, 1)
                qo_mean = qo_mean.repeat(m, 1)
                qo_log_var = qo_log_var.repeat(m, 1)
            z = self.reparameterize(qo_mean, qo_log_var)
            sampled_states = self.decoder(z, obs_h)
        if thresholded:
            return (sampled_states > 0.5).float()
        else:
            return sampled_states

    def update(self, obs):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            obs = obs.to(device)
            obs_h, qo_mean, qo_log_var = self.obs_encoder(obs)
        return obs_h, qo_mean, qo_log_var


def ivae_loss(recon_state, state, qs_mean, qs_log_var, qo_mean, qo_log_var, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon_state, state, reduction='sum')
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
    model.eval()
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


def trajectory_contrastive_loss(z, tau=0.07, max_offset=3):
    B, T, D = z.shape
    z_norm = F.normalize(z, dim=-1)
    z_flat = z_norm.view(B*T, D)

    offsets = torch.randint(1, max_offset+1, (B,), device=z.device)
    anchor_idx = []
    pos_idx = []
    for b, k in enumerate(offsets):
        t = torch.arange(0, T-k, device=z.device)
        anchor_idx.append(b*T + t)
        pos_idx.append(b*T + t + k)
    anchor_idx = torch.cat(anchor_idx)
    pos_idx = torch.cat(pos_idx)

    anchors = z_flat[anchor_idx]
    logits = anchors @ z_flat.T / tau
    row_idx = torch.arange(anchor_idx.size(0), device=z.device)
    logits[row_idx, anchor_idx] = float('-inf')
    loss = F.cross_entropy(logits, pos_idx)

    return loss

def finetune(model, traj_dataloader, T=101, alpha=1e3, lmbda=1, max_offset=25, tau=0.1, epochs=5, lr_proj=1e-3, lr_enc=5e-6, output_filename='ivae_finetuned.pth'):
    decoder = model.decoder
    state_encoder = model.state_encoder
    obs_encoder = model.obs_encoder

    # before epoch 1 unfreeze
    for p in state_encoder.parameters(): p.requires_grad = False
    for p in decoder.parameters():       p.requires_grad = False

    device = next(decoder.parameters()).device

    proj_head = nn.Sequential(
        nn.Linear(model.latent_dim, 2 * model.latent_dim),
        nn.ReLU(inplace=True),
        nn.Linear(2 * model.latent_dim, 2 * model.latent_dim),
        nn.ReLU(inplace=True),
        nn.Linear(2 * model.latent_dim, model.latent_dim),
        nn.LayerNorm(model.latent_dim),
    ).to(device)

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        total_loss = 0
        total_loss_elbo = 0
        total_loss_tcl = 0
        pbar = tqdm(traj_dataloader)

        if epoch == 0:
            opt = torch.optim.Adam(
                list(proj_head.parameters()),
                lr=lr_proj,
            )
        else:
            opt = torch.optim.Adam([
                {'params': proj_head.parameters(),
                'lr': lr_proj},
                {'params': 
                    list(obs_encoder.parameters()),
                'lr': lr_enc
                },
            ])

        for (state, obs_seq) in pbar:
            state = state.to(device)
            obs_seq = obs_seq.to(device)
            z_seq = []
            elbos = []
            for t in range(T):
                qs_mean, qs_log_var = state_encoder(state)
                h_t, qo_mean, qo_log_var  = obs_encoder(obs_seq[:,t])
                z_t = model.reparameterize(qo_mean, qo_log_var)
                z_t = proj_head(qo_mean) # mean instead of z_t
                z_seq.append(z_t)
                recon_s_t = decoder(z_t, h_t)
                elbo = ivae_loss(recon_s_t, state, qs_mean, qs_log_var, qo_mean, qo_log_var)
                elbos.append(elbo)

            z_seq = torch.stack(z_seq, dim=1)

            tcl = trajectory_contrastive_loss(z_seq, tau=tau, max_offset=max_offset)
            elbo = torch.stack(elbos).mean()

            loss = alpha*tcl + lmbda*elbo
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_loss_elbo += (lmbda*elbo).item()
            total_loss_tcl += (alpha*tcl).item()
            pbar.set_postfix({f'[{epoch+1}/{epochs}] Loss': total_loss / len(traj_dataloader.dataset), 'TCL': total_loss_tcl / len(traj_dataloader.dataset), 'ELBO': total_loss_elbo / len(traj_dataloader.dataset)})

    torch.save(model.state_dict(), output_filename)
