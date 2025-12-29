#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd  # Añadido para manejo de CSV

# --- 1. CONFIGURACIÓN DE RUTAS ---
PROJECT_ROOT = "/content/drive/MyDrive/CGGM/src/hybrid"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from model_cggm_t import CGGMT_VAE 
    print("✅ Módulo 'model_cggm_t' cargado correctamente.")
except ImportError:
    print(f"❌ Error: No se encontró 'model_cggm_t.py' en {PROJECT_ROOT}")
    sys.exit(1)

# --- 2. PARÁMETROS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HYBRID_CKPT = "/content/drive/MyDrive/CGGM/checkpoints_hybrid_v3/hybrid_v3_epoch_10.pth"

@torch.no_grad()
def main():
    vae = CGGMT_VAE(
        context_dim=32, 
        latent_dim=64, 
        hidden_dim=128, 
        n_res_blocks=3
    ).to(DEVICE)

    if not os.path.exists(HYBRID_CKPT):
        print(f"❌ Checkpoint no encontrado en: {HYBRID_CKPT}")
        return

    ckpt = torch.load(HYBRID_CKPT, map_location=DEVICE)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.eval()
    print("🚀 Modelo Hybrid cargado.")

    # --- 3. CAMINATA LATENTE ---
    steps = 50
    z_start = torch.full((1, 64), -2.5).to(DEVICE)
    z_end = torch.full((1, 64), 2.5).to(DEVICE)
    dummy_context = torch.zeros((1, 32)).to(DEVICE)
    
    alphas = np.linspace(0, 1, steps)
    predicted_temps = []

    print("🚶 Generando trayectoria térmica...")
    for alpha in alphas:
        z_interp = torch.lerp(z_start, z_end, alpha)
        
        # Réplica del decoder interno
        combined = torch.cat([z_interp, dummy_context], dim=1)
        h_dec = vae.decoder_body(combined)
        t_pred = vae.dec_mu(h_dec)
        
        predicted_temps.append(t_pred.item())

    # --- 4. EXPORTAR A CSV ---
    # Creamos un DataFrame con los resultados
    df_traversal = pd.DataFrame({
        'Alpha_Interpolation': alphas,
        'Predicted_Temperature_ZScore': predicted_temps
    })
    
    csv_filename = "latent_traversal_data.csv"
    df_traversal.to_csv(csv_filename, index=False)
    print(f"📊 Datos exportados exitosamente a: {csv_filename}")

    # --- 5. VISUALIZACIÓN ---
    plt.figure(figsize=(12, 7))
    plt.plot(alphas, predicted_temps, color='#2ecc71', linewidth=4, label='Hybrid Model (Generative Path)')
    plt.fill_between(alphas, predicted_temps, min(predicted_temps)-0.5, color='#2ecc71', alpha=0.1)
    
    plt.title("Figura 4: Continuidad del Espacio Latente - Latent Traversal", fontsize=16, fontweight='bold')
    plt.xlabel("Interpolación en el Espacio Latente (0 = Frío, 1 = Caliente)", fontsize=13)
    plt.ylabel("Temperatura Predicha (Z-Score)", fontsize=13)
    
    plt.axhspan(-3, -1, color='blue', alpha=0.07, label='Régimen Criogénico')
    plt.axhspan(1, 4, color='red', alpha=0.07, label='Régimen de Reflujo/Térmico')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    
    img_filename = "latent_traversal_hybrid_final.png"
    plt.savefig(img_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ ¡Gráfica generada con éxito como '{img_filename}'!")

if __name__ == "__main__":
    main()