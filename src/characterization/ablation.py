#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_peak_ablation(csv_path="comparison_results.csv"):
    # 1. Cargar datos
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {csv_path}")
        return

    # Configuración de estilo científico
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Paleta de colores consistente con tus gráficas anteriores
    palette = {
        "Real Data": "#2c3e50", # Gris oscuro/Negro
        "GAN": "#3498db",       # Azul
        "VAE": "#e74c3c",       # Rojo
        "Hybrid": "#2ecc71"     # Verde
    }

    # --- PANEL 1: Distribución Global ---
    sns.kdeplot(data=df, x="Temp", hue="Type", palette=palette, 
                fill=True, common_norm=False, alpha=0.1, linewidth=2, ax=axes[0])
    axes[0].set_title("A) Distribución Global (Z-Score)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Temperatura (Z-Score)")
    axes[0].set_ylabel("Densidad")

    # --- PANEL 2: Zoom en Temperatura Ambiente (~25°C) ---
    # En Z-Score, la media (0) suele estar cerca de la temperatura ambiente
    sns.kdeplot(data=df, x="Temp", hue="Type", palette=palette, 
                linewidth=2.5, common_norm=False, ax=axes[1])
    axes[1].set_xlim(-1.5, 0.5) # Rango centrado en el pico principal
    axes[1].set_title("B) Zoom: Temperatura Ambiente", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Temperatura (Z-Score)")
    axes[1].get_legend().remove()

    # --- PANEL 3: Zoom en Zona de Reflujo (~100-110°C) ---
    # Esta zona suele estar en los valores positivos del Z-Score (cola derecha)
    sns.kdeplot(data=df, x="Temp", hue="Type", palette=palette, 
                linewidth=2.5, common_norm=False, ax=axes[2])
    axes[2].set_xlim(1.5, 3.5) # Rango de la "joroba" de alta temperatura
    axes[2].set_ylim(0, 0.05)  # Ajuste de escala para ver el detalle de la cola
    axes[2].set_title("C) Zoom: Zona de Reflujo", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Temperatura (Z-Score)")
    axes[2].get_legend().remove()

    # Anotaciones de evidencia
    axes[1].annotate('VAE Suavizado', xy=(-0.2, 0.12), xytext=(-1.2, 0.15),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    
    plt.tight_layout()
    output_name = "peak_ablation_analysis.png"
    plt.savefig(output_name, dpi=300)
    print(f"✅ Figura de ablación guardada como: {output_name}")
    plt.show()

if __name__ == "__main__":
    plot_peak_ablation()