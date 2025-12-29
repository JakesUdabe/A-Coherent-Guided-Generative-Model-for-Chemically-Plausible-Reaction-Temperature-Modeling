#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import mean_absolute_error
import torch

# Si no tienes instalada la librería de Optimal Transport:
# !pip install POT

def calculate_distribution_metrics(y_real, y_pred):
    """
    Calcula métricas de error y similitud de distribuciones.
    """
    # 1. MAE (Error Absoluto Medio)
    mae = mean_absolute_error(y_real, y_pred)
    
    # 2. Distancia de Wasserstein (W1) - Métrica estrella para GANs
    w1 = wasserstein_distance(y_real, y_pred)
    
    # Para métricas de densidad (KL y Log-Likelihood), creamos histogramas normalizados (PDF)
    # Usamos un rango común para que las comparaciones sean válidas
    bins = 100
    range_min = min(y_real.min(), y_pred.min())
    range_max = max(y_real.max(), y_pred.max())
    
    p_real, _ = np.histogram(y_real, bins=bins, range=(range_min, range_max), density=True)
    p_pred, _ = np.histogram(y_pred, bins=bins, range=(range_min, range_max), density=True)
    
    # Evitar ceros para cálculos logarítmicos
    p_real = np.where(p_real == 0, 1e-10, p_real)
    p_pred = np.where(p_pred == 0, 1e-10, p_pred)
    
    # 3. Kullback-Leibler Divergence
    kl_div = entropy(p_real, p_pred)
    
    # 4. Log-Likelihood Aproximado
    # Mide qué tan probable es ver los datos reales bajo la distribución predicha
    log_likelihood = np.mean(np.log(p_pred))
    
    return {
        "MAE": mae,
        "W1_Distance": w1,
        "KL_Divergence": kl_div,
        "Log_Likelihood": log_likelihood
    }

def main():
    # 1. Cargar el CSV generado anteriormente con las predicciones
    # Asegúrate de tener el archivo 'comparison_results.csv' de los pasos anteriores
    try:
        df = pd.read_csv("comparison_results.csv")
    except FileNotFoundError:
        print("❌ Error: No se encontró 'comparison_results.csv'. Ejecuta primero el script de comparación.")
        return

    # 2. Separar los datos por tipo de modelo
    real_data = df[df["Type"] == "Real Data"]["Temp"].values
    
    model_names = [m for m in df["Type"].unique() if m != "Real Data"]
    all_metrics = []

    for model in model_names:
        pred_data = df[df["Type"] == model]["Temp"].values
        
        # Ajustar tamaños si hay diferencias por el batching
        min_len = min(len(real_data), len(pred_data))
        m = calculate_distribution_metrics(real_data[:min_len], pred_data[:min_len])
        m["Model"] = model
        all_metrics.append(m)

    # 3. Crear y mostrar la tabla comparativa
    df_metrics = pd.DataFrame(all_metrics).set_index("Model")
    
    # Reordenar columnas para que coincidan con tu propuesta
    df_metrics = df_metrics[["MAE", "Log_Likelihood", "W1_Distance", "KL_Divergence"]]
    
    print("\n📊 TABLA DE MÉTRICAS DE ERROR DE DISTRIBUCIÓN")
    print("="*60)
    print(df_metrics.round(4))
    print("="*60)
    
    # Exportar a CSV para tu artículo
    df_metrics.to_csv("dist_metrics_table.csv")
    print("💾 Tabla guardada como 'dist_metrics_table.csv'")

if __name__ == "__main__":
    main()