#!/usr/bin/env python3
# train_cggm_vae.py
# Entrenamiento CGGMT VAE (Residual, KL Adaptativo, StandardScaler Compatible)

import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import re

# Importar componentes locales
from dataset_tokenized import TokenizedReactionsDataset, collate_fn
from model_cggm_t import SMILESEncoder, CGGMT_VAE 
import loss_constraints

# -------------------------
# Utilidad de Reanudación
# -------------------------
def find_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir):
        return None
    files = os.listdir(save_dir)
    pattern = re.compile(r"cggm_vae_epoch_(\d+)\.pth")
    
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
# Loss (ELBO) - Corregida para estabilidad
# -------------------------
def elbo_loss(T_pred, T_true, mu, logvar, T_min, T_max,
              lambda_cons=5.0, beta=0.05, free_bits_tau=1.0):
    """
    Calcula la pérdida ELBO.
    Nota: Con StandardScaler, T_true y T_pred pueden ser negativos. MSE funciona bien.
    """
    T_pred = T_pred.view_as(T_true)
    
    # 1. Reconstrucción: MSE es ideal para datos estandarizados (Media 0, Var 1)
    recon = F.mse_loss(T_pred, T_true, reduction='mean')
    
    # 2. KL Divergence (Cálculo analítico)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    # Free-bits (Previene el colapso del espacio latente)
    kl_penalty = torch.clamp(kl, min=free_bits_tau)
    
    # 3. Restricción física (Usa los límites normalizados del dataset)
    l_cons = loss_constraints.constraint_loss(T_pred, T_min, T_max)
    
    # ELBO Total
    total_loss = recon + (beta * kl_penalty) + (lambda_cons * l_cons)
    
    return total_loss, recon, kl, l_cons

# -------------------------
# Entrenamiento principal
# -------------------------
def main(args):
    # Configuración de hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    # Crear carpeta de salida si no existe
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    # Carga de datos
    print(f"📂 Cargando dataset...")
    dataset = TokenizedReactionsDataset(
        csv_path=args.csv_path, 
        max_rows=args.max_rows, 
        vocab_path=args.vocab_path
    )
    
    # Extraer límites de normalización (Z-Scores)
    T_norm_min = dataset.T_norm_min
    T_norm_max = dataset.T_norm_max
    print(f"✅ Dataset cargado. Límites físicos (Z-score): [{T_norm_min:.2f}, {T_norm_max:.2f}]")

    # Split Train/Val
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    # --- Inicialización de Modelos ---
    CONTEXT_DIM = 32
    G_encoder = SMILESEncoder(vocab_size=dataset.vocab_size).to(device)
    G_reducer = torch.nn.Sequential(
        torch.nn.Linear(G_encoder.output_dim, CONTEXT_DIM),
        torch.nn.LayerNorm(CONTEXT_DIM), # Añadida estabilidad
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.2)
    ).to(device)

    G_vae = CGGMT_VAE(
        context_dim=CONTEXT_DIM, 
        latent_dim=args.latent_dim, 
        hidden_dim=args.hidden_dim_g,
        n_res_blocks=args.vae_n_res_blocks, 
        dropout=args.vae_dropout
    ).to(device)

    # Optimizador
    params = list(G_encoder.parameters()) + list(G_reducer.parameters()) + list(G_vae.parameters())
    optimizer = optim.AdamW(params, lr=args.lr_g, weight_decay=1e-4)
    
    G_vae.set_beta(args.beta)
    if args.kl_adapt:
        G_vae.set_target_kl(args.target_kl, adapt_rate=args.kl_adapt_rate)

    # Reanudación
    start_epoch = 1
    checkpoint_path = find_latest_checkpoint(args.save_path)
    if checkpoint_path:
        print(f"🔄 Cargando checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        G_encoder.load_state_dict(ckpt["encoder_state_dict"])
        G_reducer.load_state_dict(ckpt["reducer_state_dict"])
        G_vae.load_state_dict(ckpt["vae_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_g_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        G_vae.set_beta(ckpt.get("beta", args.beta))

    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    print(f"🚀 Iniciando entrenamiento desde Época {start_epoch}")
    for epoch in range(start_epoch, args.epochs + 1):
        G_encoder.train(); G_reducer.train(); G_vae.train()
        
        running_recon, running_kl, running_cons = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for tokens, lengths, T_true in pbar:
            tokens, lengths, T_true = tokens.to(device), lengths.to(device), T_true.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                # Encoding molecular
                context = G_reducer(G_encoder(tokens, lengths))
                
                # VAE Forward
                T_pred, mu, logvar = G_vae(T_true, context, is_sampling=False)
                
                # Loss
                loss, recon, kl, cons = elbo_loss(
                    T_pred, T_true, mu, logvar, T_norm_min, T_norm_max,
                    args.lambda_cons, G_vae.beta, args.free_bits_tau
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_recon += recon.item()
            running_kl += kl.item()
            running_cons += cons.item()
            
            pbar.set_postfix({
                'Recon': f"{recon.item():.4f}", 
                'KL': f"{kl.item():.3f}",
                'Beta': f"{G_vae.beta:.3f}"
            })

        # --- Fin de Época ---
        avg_recon = running_recon / len(train_loader)
        avg_kl = running_kl / len(train_loader)
        avg_cons = running_cons / len(train_loader)

        # Adaptación de Beta
        if args.kl_adapt:
            G_vae.adapt_beta(avg_kl)

        print(f"\n📊 Resumen Epoch {epoch}: Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Cons={avg_cons:.4f}, Beta={G_vae.beta:.4f}")

        # Guardado de seguridad
        if epoch % args.save_every == 0:
            ckpt_name = os.path.join(args.save_path, f"cggm_vae_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": G_encoder.state_dict(),
                "reducer_state_dict": G_reducer.state_dict(),
                "vae_state_dict": G_vae.state_dict(),
                "optimizer_g_state_dict": optimizer.state_dict(),
                "beta": G_vae.beta
            }, ckpt_name)
            print(f"💾 Checkpoint guardado en: {ckpt_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Cambia estos defaults a tus rutas reales en Drive
    parser.add_argument("--csv_path", type=str, default="/content/drive/MyDrive/CGGM/data/ord/ord_tokenized_reactions.csv.gz")
    parser.add_argument("--vocab_path", type=str, default="/content/drive/MyDrive/CGGM/data/ord/smiles_vocabulary.json")
    parser.add_argument("--save_path", type=str, default="/content/drive/MyDrive/CGGM/checkpoints")
    parser.add_argument("--baseline_model_path", type=str, default=None)
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_g", type=float, default=1e-4) # Subido un poco para estabilidad inicial
    parser.add_argument("--lambda_cons", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=0.01) # Beta inicial más baja
    parser.add_argument("--target_kl", type=float, default=2.0) # Objetivo de información
    parser.add_argument("--kl_adapt", action="store_true", default=True)
    parser.add_argument("--kl_adapt_rate", type=float, default=0.1)
    parser.add_argument("--free_bits_tau", type=float, default=0.5)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim_g", type=int, default=128)
    parser.add_argument("--vae_n_res_blocks", type=int, default=3)
    parser.add_argument("--vae_dropout", type=float, default=0.1)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true", default=True)

    args, _ = parser.parse_known_args()
    main(args)