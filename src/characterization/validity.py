#!/usr/bin/env python3
import pandas as pd
import numpy as np

def calculate_physical_validity_zscore(csv_path="comparison_results.csv"):
    # 1. Cargar datos
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {csv_path}")
        return

    # --- AJUSTE PARA Z-SCORE ---
    # En una distribución normal:
    # <-3.0 es el 0.1% más frío (Anomalía)
    # >4.0 es el extremo de alta temperatura (Anomalía)
    LOWER_LIMIT_Z = -3.0 
    UPPER_LIMIT_Z = 4.0

    results = []
    model_names = df["Type"].unique()

    for model in model_names:
        data = df[df["Type"] == model]["Temp"]
        total = len(data)
        
        # A. Errores Críticos (NaN o Inf)
        critical_errors = data.isna().sum() + np.isinf(data).sum()
        perc_critical = (critical_errors / total) * 100
        
        # B. Fuera de Rango (Leyes de la estadística/química en Z-Score)
        valid_data = data.dropna()
        out_of_range = ((valid_data < LOWER_LIMIT_Z) | (valid_data > UPPER_LIMIT_Z)).sum()
        perc_out = (out_of_range / total) * 100
        
        results.append({
            "Modelo": model,
            "% Fuera de Rango (Z)": f"{perc_out:.2f}%",
            "% Errores Críticos": f"{perc_critical:.2f}%"
        })

    # 2. Crear Tabla
    df_validity = pd.DataFrame(results)
    
    # 3. Mostrar y Guardar
    print("\n⚖️ TABLA DE VALIDEZ FÍSICA (AJUSTE Z-SCORE)")
    print("=" * 60)
    print(df_validity.to_string(index=False))
    print("=" * 60)
    print(f"Nota: Fuera de rango definido como Z < {LOWER_LIMIT_Z} o Z > {UPPER_LIMIT_Z}")
    
    df_validity.to_csv("physical_constraints_zscore.csv", index=False)
    return df_validity

if __name__ == "__main__":
    calculate_physical_validity_zscore()