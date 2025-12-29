#!/usr/bin/env python3
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

# --- 1. CONFIGURACIÓN DE RUTAS ---
PROJECT_ROOT = "/content/drive/MyDrive/CGGM/src/hybrid"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from dataset_tokenized import TokenizedReactionsDataset, collate_fn
    from model_cggm_t import SMILESEncoder, CGGMT_VAE
    print("✅ Módulos cargados correctamente.")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)

# --- 2. PARÁMETROS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_CSV = "/content/drive/MyDrive/CGGM/data/ord/ord_tokenized_reactions.csv.gz"
VOCAB_JSON = "/content/drive/MyDrive/CGGM/data/ord/smiles_vocabulary.json"
# Ruta a tu mejor checkpoint del Hybrid
HYBRID_CKPT = "/content/drive/MyDrive/CGGM/data/checkpoints/Hybrid.pth" 
MAX_SAMPLES = 1000  # Aumentado para mayor densidad

@torch.no_grad()
def get_latent_data(loader, encoder, reducer, vae, max_samples):
    latent_vecs, temps = [], []
    print(f"🔭 Extrayendo vectores latentes (objetivo: {max_samples})...")
    
    for tokens, lengths, T_real in loader:
        tokens, lengths = tokens.to(DEVICE), lengths.to(DEVICE)
        
        # Extraer contexto y mu
        context = reducer(encoder(tokens, lengths))
        _, mu, _ = vae(T_real.to(DEVICE), context, is_sampling=False)
        
        latent_vecs.append(mu.cpu().numpy())
        temps.append(T_real.numpy())
        
        current_count = np.concatenate(latent_vecs).shape[0]
        if current_count >= max_samples: 
            break
            
    return np.concatenate(latent_vecs)[:max_samples], np.concatenate(temps)[:max_samples]

def main():
    # 1. Cargar datos
    dataset = TokenizedReactionsDataset(csv_path=DATA_CSV, vocab_path=VOCAB_JSON)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

    # 2. Re-instanciar arquitectura Hybrid v3
    encoder = SMILESEncoder(vocab_size=dataset.vocab_size).to(DEVICE)
    reducer = torch.nn.Sequential(
        torch.nn.Linear(encoder.output_dim, 32),
        torch.nn.LayerNorm(32),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2)
    ).to(DEVICE)
    vae = CGGMT_VAE(context_dim=32, latent_dim=64, hidden_dim=128, n_res_blocks=3).to(DEVICE)

    # 3. Cargar pesos
    if not os.path.exists(HYBRID_CKPT):
        print(f"❌ Checkpoint no encontrado en: {HYBRID_CKPT}")
        return

    ckpt = torch.load(HYBRID_CKPT, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    reducer.load_state_dict(ckpt["reducer_state_dict"])
    vae.load_state_dict(ckpt["vae_state_dict"])
    encoder.eval(); reducer.eval(); vae.eval()

    # 4. Obtener datos y calcular t-SNE
    z, t = get_latent_data(loader, encoder, reducer, vae, MAX_SAMPLES)
    
    print(f"🎨 Calculando t-SNE para {MAX_SAMPLES} puntos...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=1000)
    z_2d = tsne.fit_transform(z)

    # 5. Exportar a CSV
    df_export = pd.DataFrame({
        'tsne_1': z_2d[:, 0],
        'tsne_2': z_2d[:, 1],
        'temp_zscore': t.flatten()
    })
    csv_filename = "latent_space_data.csv"
    df_export.to_csv(csv_filename, index=False)
    print(f"💾 Datos exportados a: {csv_filename}")

    # 6. Graficar
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=t, cmap='coolwarm', s=5, alpha=0.5)
    plt.colorbar(sc, label='Temperatura (Z-Score)')
    plt.title(f"Espacio Latente Detallado - Hybrid Model (n={MAX_SAMPLES})", fontsize=15)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    
    plot_filename = "latent_space_dense.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"🖼️ Gráfica guardada como: {plot_filename}")
    plt.show()

if __name__ == "__main__":
    main()