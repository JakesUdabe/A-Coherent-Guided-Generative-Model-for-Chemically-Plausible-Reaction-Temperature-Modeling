#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import pandas as pd
import numpy as np
import json
from argparse import Namespace
from torch.utils.data import DataLoader

# 1. PATHS Y MÓDULOS
PROJECT_ROOT = "/content/drive/MyDrive/CGGM/src"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    for sub in ["vae", "gan", "hybrid"]: sys.path.append(os.path.join(PROJECT_ROOT, sub))

try:
    from dataset_tokenized import TokenizedReactionsDataset, collate_fn
    from model_cggm_t import SMILESEncoder, CGGMT_VAE 
    print("✅ Módulos cargados correctamente.")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)

# 2. CONFIGURACIÓN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DATA_DIR = "/content/drive/MyDrive/CGGM/data"
CKPT_DIR = os.path.join(BASE_DATA_DIR, "checkpoints")
SCALER_PARAMS_PATH = os.path.join(BASE_DATA_DIR, "ord/scaler_params.json")

CHECKPOINTS = {
    "GAN": os.path.join(CKPT_DIR, "GAN.pth"),
    "VAE": os.path.join(CKPT_DIR, "VAE.pth"),
    "Hybrid": os.path.join(CKPT_DIR, "Hybrid.pth")
}

def get_denormalizer():
    if os.path.exists(SCALER_PARAMS_PATH):
        with open(SCALER_PARAMS_PATH, 'r') as f:
            params = json.load(f)
        return lambda x: ((x * params['std_k']) + params['mean_k']) - 273.15
    return lambda x: x # Si no hay scaler, devuelve el valor normalizado

# 3. CARGA DINÁMICA DE MODELOS
def load_model_autoconfig(ckpt_path, vocab_size):
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint no encontrado: {ckpt_path}")
        return None

    # Probamos con las dos configuraciones que tienes en tus archivos
    for h_dim in [128, 64]:
        try:
            args = Namespace(latent_dim=64, hidden_dim_g=h_dim, vae_n_res_blocks=3, context_dim=32, vae_dropout=0.1)
            
            encoder = SMILESEncoder(vocab_size=vocab_size).to(DEVICE)
            reducer = torch.nn.Sequential(
                torch.nn.Linear(encoder.output_dim, args.context_dim),
                torch.nn.LayerNorm(args.context_dim),
                torch.nn.ReLU()
            ).to(DEVICE)
            
            vae = CGGMT_VAE(context_dim=args.context_dim, latent_dim=args.latent_dim,
                            hidden_dim=args.hidden_dim_g, n_res_blocks=args.vae_n_res_blocks, dropout=args.vae_dropout).to(DEVICE)

            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            reducer.load_state_dict(ckpt["reducer_state_dict"])
            
            key = "vae_state_dict" if "vae_state_dict" in ckpt else "model_state_dict"
            vae.load_state_dict(ckpt[key])
            
            encoder.eval(); reducer.eval(); vae.eval()
            print(f"✅ Cargado exitosamente: {os.path.basename(ckpt_path)} (hidden_dim={h_dim})")
            return (encoder, reducer, vae)
            
        except RuntimeError as e:
            if "size mismatch" in str(e):
                continue # Probamos con la siguiente dimensión
            else:
                print(f"❌ Error inesperado en {ckpt_path}: {e}")
                return None
    
    print(f"❌ No se pudo cargar {ckpt_path} con ninguna configuración (64 o 128).")
    return None

# 4. MAIN
@torch.no_grad()
def main():
    DATA_CSV = os.path.join(BASE_DATA_DIR, "ord/ord_tokenized_reactions.csv.gz")
    VOCAB_JSON = os.path.join(BASE_DATA_DIR, "ord/smiles_vocabulary.json")

    dataset = TokenizedReactionsDataset(csv_path=DATA_CSV, vocab_path=VOCAB_JSON)
    denorm = get_denormalizer()
    loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    
    models = {name: load_model_autoconfig(path, dataset.vocab_size) for name, path in CHECKPOINTS.items()}
    models = {k: v for k, v in models.items() if v is not None}

    if not models:
        print("❌ Ningún modelo cargado. Revisa tus archivos .pth")
        return

    results = []
    print(f"\n🚀 Procesando comparación...")
    
    for i, (tokens, lengths, T_real) in enumerate(loader):
        T_real_c = denorm(T_real.numpy().flatten())
        results.append(pd.DataFrame({"Temp": T_real_c, "Type": "Real Data"}))

        for name, (enc, red, vae) in models.items():
            c = red(enc(tokens.to(DEVICE), lengths.cpu()))
            T_gen, _, _ = vae(T_real.to(DEVICE), c, is_sampling=True)
            results.append(pd.DataFrame({"Temp": denorm(T_gen.cpu().numpy().flatten()), "Type": name}))

        if (i + 1) * 128 >= 10000: break

    pd.concat(results).to_csv("comparison_results.csv", index=False)
    print("\n✨ Archivo 'comparison_results.csv' generado con éxito.")

if __name__ == "__main__":
    main()
