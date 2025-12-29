#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import re
import numpy as np

from dataset_tokenized import TokenizedReactionsDataset, collate_fn
from model_cggm_t import SMILESEncoder, CGGMT_VAE
import loss_constraints

# -------------------------
# Módulo de Diversidad (Minibatch Discrimination)
# -------------------------
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, intermediate_features=16):
        super().__init__()
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, intermediate_features))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        matrices = torch.mm(x, self.T.view(self.T.shape[0], -1))
        matrices = matrices.view(-1, self.T.shape[1], self.T.shape[2])
        M_i, M_j = matrices.unsqueeze(0), matrices.unsqueeze(1)
        exp_dist = torch.exp(-torch.abs(M_i - M_j).sum(3))
        out = exp_dist.sum(0) - 1
        return torch.cat([x, out], 1)

class Discriminator_Robust(nn.Module):
    def __init__(self, context_dim, temp_dim=1, hidden_dim=512, n_layers=3, dropout=0.2):
        super().__init__()
        curr_dim = context_dim + temp_dim
        layers = []
        for i in range(n_layers):
            layers.append(nn.utils.spectral_norm(nn.Linear(curr_dim, hidden_dim)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            curr_dim = hidden_dim
            if i == 0:
                self.mbd = MinibatchDiscrimination(hidden_dim, 32)
                curr_dim = hidden_dim + 32
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(curr_dim, 1)

    def forward(self, c, t):
        x = torch.cat([c, t.view(-1, 1).float()], dim=1)
        for i in range(3): x = self.features[i](x)
        x = self.mbd(x)
        for i in range(3, len(self.features)): x = self.features[i](x)
        return self.output(x)

# -------------------------
# Utils: Checkpoint & GP
# -------------------------
def find_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir): return None
    files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
    pattern = re.compile(r".*epoch_(\d+)\.pth")
    latest_epoch, latest_path = -1, None
    for f in files:
        m = pattern.match(f)
        if m and int(m.group(1)) > latest_epoch:
            latest_epoch, latest_path = int(m.group(1)), os.path.join(save_dir, f)
    return latest_path

def compute_gradient_penalty(D, c, T_real, T_fake, device):
    batch_size = T_real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    interpolates = (alpha * T_real + ((1 - alpha) * T_fake)).requires_grad_(True)
    d_int = D(c, interpolates)
    grads = torch.autograd.grad(outputs=d_int, inputs=interpolates, grad_outputs=torch.ones_like(d_int),
                                create_graph=True, retain_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

# -------------------------
# Main Loop
# -------------------------
def main(args):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Iniciando Entrenamiento Híbrido (VAE Baseline Style) | Device: {device}")

    dataset = TokenizedReactionsDataset(csv_path=args.csv_path, max_rows=args.max_rows, vocab_path=args.vocab_path)
    T_norm_min, T_norm_max = dataset.T_norm_min, dataset.T_norm_max
    train_size = int(0.9 * len(dataset))
    train_ds, _ = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # --- Arquitectura idéntica a tu VAE ---
    CONTEXT_DIM = 32
    G_encoder = SMILESEncoder(vocab_size=dataset.vocab_size).to(device)
    G_reducer = nn.Sequential(
        nn.Linear(G_encoder.output_dim, CONTEXT_DIM),
        nn.LayerNorm(CONTEXT_DIM),
        nn.ReLU(),
        nn.Dropout(p=0.2)
    ).to(device)

    G_vae = CGGMT_VAE(
        context_dim=CONTEXT_DIM, 
        latent_dim=args.latent_dim, 
        hidden_dim=args.hidden_dim_g,
        n_res_blocks=args.vae_n_res_blocks,
        dropout=args.vae_dropout
    ).to(device)
    
    D = Discriminator_Robust(context_dim=CONTEXT_DIM, hidden_dim=args.hidden_dim_d, n_layers=args.n_layers_d).to(device)

    # Configuración de KL Adaptativo igual a tu VAE
    G_vae.set_beta(args.beta)
    if args.kl_adapt:
        G_vae.set_target_kl(args.target_kl, adapt_rate=args.kl_adapt_rate)

    # Optimizadores
    params_G = list(G_encoder.parameters()) + list(G_reducer.parameters()) + list(G_vae.parameters())
    opt_G = optim.AdamW(params_G, lr=args.lr_g, weight_decay=1e-4)
    opt_D = optim.AdamW(D.parameters(), lr=args.lr_d, betas=(0.5, 0.9))
    
    # Carga de Checkpoints Protegida
    resume_path = find_latest_checkpoint(args.save_path)
    start_epoch = 1
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        G_encoder.load_state_dict(ckpt["encoder_state_dict"])
        G_reducer.load_state_dict(ckpt["reducer_state_dict"])
        G_vae.load_state_dict(ckpt["vae_state_dict"])
        if "discriminator_state_dict" in ckpt:
            D.load_state_dict(ckpt["discriminator_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        G_vae.set_beta(ckpt.get("beta", args.beta))
        print(f"🔄 Reanudado desde época {start_epoch}")

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    for epoch in range(start_epoch, args.epochs + 1):
        G_encoder.train(); G_reducer.train(); G_vae.train(); D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        gan_weight = min(1.0, epoch / 10.0)

        for tokens, lengths, T_true in pbar:
            tokens, lengths = tokens.to(device), lengths.to(device)
            T_true = T_true.to(device).view(-1, 1).float()

            # --- D Step ---
            for _ in range(args.k_steps_D):
                opt_D.zero_grad()
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    c = G_reducer(G_encoder(tokens, lengths)).detach()
                    T_fake, _, _ = G_vae(T_true, c, is_sampling=True)
                    L_D = D(c, T_fake.detach()).mean() - D(c, T_true).mean()
                    gp = compute_gradient_penalty(D, c, T_true, T_fake.detach(), device)
                    total_D = L_D + (args.lambda_gp * gp)
                scaler.scale(total_D).backward(); scaler.step(opt_D); scaler.update()

            # --- G Step (Fusionado con tu lógica ELBO) ---
            opt_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                c = G_reducer(G_encoder(tokens, lengths))
                T_pred, mu, logvar = G_vae(T_true, c, is_sampling=False)
                T_stoch, _, _ = G_vae(T_true, c, is_sampling=True)
                
                # 1. Pérdida ELBO (Igual a tu VAE)
                recon = F.mse_loss(T_pred, T_true)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                kl_penalty = torch.clamp(kl, min=args.free_bits_tau)
                l_cons = loss_constraints.constraint_loss(T_pred, T_norm_min, T_norm_max)
                
                L_G_base = recon + (G_vae.beta * kl_penalty) + (args.lambda_cons * l_cons)
                
                # 2. Pérdida Adversaria
                L_adv = -D(c, T_stoch).mean()
                L_G_total = L_G_base + (args.lambda_adv * gan_weight * L_adv)

            scaler.scale(L_G_total).backward(); scaler.step(opt_G); scaler.update()
            
            # Adaptación de Beta (como en tu VAE)
            if args.kl_adapt:
                G_vae.adapt_beta(kl.item())
            
            pbar.set_postfix({'Rec': f"{recon.item():.4f}", 'KL': f"{kl.item():.2f}", 'Beta': f"{G_vae.beta:.3f}"})

        # --- Guardado Seguro ---
        if epoch % args.save_every == 0 or epoch == args.epochs:
            # Crea la carpeta si no existe o si Drive se desconectó
            os.makedirs(args.save_path, exist_ok=True)
            
            ckpt_path = os.path.join(args.save_path, f"hybrid_v3_epoch_{epoch}.pth")
            try:
                torch.save({
                    "encoder_state_dict": G_encoder.state_dict(),
                    "reducer_state_dict": G_reducer.state_dict(),
                    "vae_state_dict": G_vae.state_dict(),
                    "discriminator_state_dict": D.state_dict(),
                    "epoch": epoch,
                    "beta": G_vae.beta
                }, ckpt_path)
                print(f"\n💾 Checkpoint guardado exitosamente en: {ckpt_path}")
            except Exception as e:
                print(f"\n❌ Error crítico al guardar en Drive: {e}")
                # Intento de guardado local en Colab por si falla el almacenamiento en la nube
                torch.save(G_vae.state_dict(), f"emergency_vae_epoch_{epoch}.pth")
                print("⚠️ Se ha guardado una copia de emergencia en el almacenamiento local de Colab.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hiperparámetros de tu VAE
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--target_kl", type=float, default=2.0)
    parser.add_argument("--kl_adapt", action="store_true", default=True)
    parser.add_argument("--kl_adapt_rate", type=float, default=0.1)
    parser.add_argument("--free_bits_tau", type=float, default=0.5)
    parser.add_argument("--lambda_cons", type=float, default=5.0)
    parser.add_argument("--vae_dropout", type=float, default=0.1)
    # Hiperparámetros de GAN
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--k_steps_D", type=int, default=2)
    parser.add_argument("--lambda_adv", type=float, default=0.01) # Empezar suave
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    # Rutas y Arquitectura básica
    parser.add_argument("--csv_path", type=str, default="/content/drive/MyDrive/CGGM/data/ord/ord_tokenized_reactions.csv.gz")
    parser.add_argument("--vocab_path", type=str, default="/content/drive/MyDrive/CGGM/data/ord/smiles_vocabulary.json")
    parser.add_argument("--save_path", type=str, default="/content/drive/MyDrive/CGGM/checkpoints_hybrid_v3")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim_g", type=int, default=128)
    parser.add_argument("--vae_n_res_blocks", type=int, default=3)
    parser.add_argument("--hidden_dim_d", type=int, default=512)
    parser.add_argument("--n_layers_d", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
