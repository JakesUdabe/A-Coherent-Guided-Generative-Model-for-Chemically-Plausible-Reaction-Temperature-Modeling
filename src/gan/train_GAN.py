#!/usr/bin/env python3
# train_gan_pure.py
# Modificación: GAN Puro (WGAN-GP) eliminando la componente VAE (KL/ELBO)

import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import re

from dataset_tokenized import TokenizedReactionsDataset, collate_fn
from model_cggm_t import SMILESEncoder, CGGMT_VAE 
from model_hybrid_simple import Discriminator_Simple
import loss_constraints

# -------------------------
# Utilidad de Reanudación
# -------------------------
def find_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir): return None
    files = os.listdir(save_dir)
    pattern = re.compile(r"gan_pure_epoch_(\d+)\.pth")
    latest_epoch = -1
    latest_path = None
    for file in files:
        match = pattern.match(file)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = os.path.join(save_dir, file)
    return latest_path

# -------------------------
# Loss utils (GAN + GP)
# -------------------------
def compute_gradient_penalty(D, c_real, T_real, c_fake, T_fake, device, lambda_gp=10.0):
    batch = c_real.size(0)
    alpha = torch.rand(batch, 1, device=device)
    c_interpol = (alpha * c_real + (1 - alpha) * c_fake).requires_grad_(True)
    T_interpol = (alpha * T_real + (1 - alpha) * T_fake).requires_grad_(True)
    
    d_interpolates = D(c_interpol, T_interpol)
    fake_labels = torch.ones_like(d_interpolates, device=device)
    
    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=[c_interpol, T_interpol],
        grad_outputs=fake_labels,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )
    g_c = grads[0].view(batch, -1)
    g_t = grads[1].view(batch, -1)
    gradients = torch.cat([g_c, g_t], dim=1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gradient_penalty

# -------------------------
# Entrenamiento principal
# -------------------------
def main(args):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Modo: GAN Puro (WGAN-GP) | Device: {device}")

    # 1. Dataset
    dataset = TokenizedReactionsDataset(csv_path=args.csv_path, max_rows=args.max_rows, vocab_path=args.vocab_path)
    T_norm_min, T_norm_max = dataset.T_norm_min, dataset.T_norm_max

    n_train = int(0.8 * len(dataset))
    train_ds, _ = random_split(dataset, [n_train, len(dataset)-n_train], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    # 2. Arquitectura
    CONTEXT_DIM = 32
    LATENT_DIM = args.latent_dim 
    
    G_encoder = SMILESEncoder(vocab_size=dataset.vocab_size).to(device)
    G_reducer = torch.nn.Sequential(
        torch.nn.Linear(G_encoder.output_dim, CONTEXT_DIM),
        torch.nn.LayerNorm(CONTEXT_DIM),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.2)
    ).to(device)

    G_gen = CGGMT_VAE(context_dim=CONTEXT_DIM, latent_dim=LATENT_DIM, hidden_dim=args.hidden_dim_g,
                      n_res_blocks=args.vae_n_res_blocks, dropout=args.vae_dropout).to(device)

    D = Discriminator_Simple(context_dim=CONTEXT_DIM, temp_dim=1,
                             hidden_dim=args.hidden_dim_d, n_layers=args.n_layers_d,
                             dropout=args.dropout_d).to(device)
    
    # 3. Optimización
    params_G = list(G_encoder.parameters()) + list(G_reducer.parameters()) + list(G_gen.parameters())
    opt_G = optim.AdamW(params_G, lr=args.lr_g, betas=(0.5, 0.9))
    opt_D = optim.AdamW(D.parameters(), lr=args.lr_d, betas=(0.5, 0.9))
    
    start_epoch = 1
    scaler_G = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Reanudación
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    resume_path = find_latest_checkpoint(args.save_path)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        G_encoder.load_state_dict(ckpt["encoder_state_dict"])
        G_reducer.load_state_dict(ckpt["reducer_state_dict"])
        G_gen.load_state_dict(ckpt["vae_state_dict"])
        D.load_state_dict(ckpt["discriminator_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"✅ Reanudado en época {start_epoch}")

    print("--- 🚀 Iniciando entrenamiento GAN Puro ---")
    for epoch in range(start_epoch, args.epochs + 1):
        train_G_enabled = epoch > args.d_warmup_epochs
        logs_G = {'adv': 0.0, 'cons': 0.0}
        logs_D = {'loss': 0.0, 'real': 0.0, 'fake': 0.0}

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")
        for i, (tokens, lengths, T_true) in pbar:
            tokens, lengths, T_true = tokens.to(device), lengths.to(device), T_true.to(device)

            c = G_reducer(G_encoder(tokens, lengths))
            
            # --- Train Discriminator ---
            opt_D.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                T_fake, _, _ = G_gen(T_true, c.detach(), is_sampling=True)
                D_real = D(c.detach(), T_true)
                D_fake = D(c.detach(), T_fake.detach())
                gp = compute_gradient_penalty(D, c.detach(), T_true, c.detach(), T_fake.detach(), device, args.lambda_gp)
                L_D = D_fake.mean() - D_real.mean() + gp

            L_D.backward()
            opt_D.step()

            logs_D['loss'] += L_D.item()
            logs_D['real'] += D_real.mean().item()
            logs_D['fake'] += D_fake.mean().item()

            # --- Train Generator ---
            if train_G_enabled:
                opt_G.zero_grad()
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    T_gen, _, _ = G_gen(T_true, c, is_sampling=True)
                    D_fake_G = D(c, T_gen)
                    L_adv = -D_fake_G.mean()
                    L_cons = loss_constraints.constraint_loss(T_gen, T_norm_min, T_norm_max)
                    L_total_G = (args.lambda_adv * L_adv) + (args.lambda_cons * L_cons)

                scaler_G.scale(L_total_G).backward()
                scaler_G.step(opt_G)
                scaler_G.update()

                logs_G['adv'] += L_adv.item()
                logs_G['cons'] += L_cons.item()

            pbar.set_postfix({'D_loss': f"{L_D.item():.4f}", 'G_adv': f"{logs_G['adv']/(i+1):.4f}"})

        # Resumen Época
        separation = (logs_D['real'] - logs_D['fake']) / len(train_loader)
        print(f"\n📊 Resumen Epoch {epoch}: Separation: {separation:.4f}, G_Adv: {logs_G['adv']/len(train_loader):.4f}")

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_path, f"gan_pure_epoch_{epoch}.pth")
            torch.save({
                "encoder_state_dict": G_encoder.state_dict(),
                "reducer_state_dict": G_reducer.state_dict(),
                "vae_state_dict": G_gen.state_dict(),
                "discriminator_state_dict": D.state_dict(),
                "epoch": epoch,
                "separation": separation
            }, ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN Puro para Predicción de Temperatura")
    # RUTAS CRÍTICAS (Ajustadas a tu Drive)
    parser.add_argument("--csv_path", type=str, default="/content/drive/MyDrive/CGGM/data/ord/ord_tokenized_reactions.csv.gz")
    parser.add_argument("--vocab_path", type=str, default="/content/drive/MyDrive/CGGM/data/ord/smiles_vocabulary.json")
    parser.add_argument("--save_path", type=str, default="/content/drive/MyDrive/CGGM/checkpoints_gan")
    
    # Parámetros Generales
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    
    # Hiperparámetros GAN
    parser.add_argument("--lr_g", type=float, default=1e-5)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--k_steps_D", type=int, default=1) # 1 suele bastar con AdamW y GP
    parser.add_argument("--d_warmup_epochs", type=int, default=0)
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--lambda_cons", type=float, default=5.0)
    
    # Arquitectura
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim_g", type=int, default=64)
    parser.add_argument("--vae_n_res_blocks", type=int, default=3)
    parser.add_argument("--vae_dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dim_d", type=int, default=768)
    parser.add_argument("--n_layers_d", type=int, default=4)
    parser.add_argument("--dropout_d", type=float, default=0.1)

    args, _ = parser.parse_known_args()
    main(args)