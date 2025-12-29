#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

def classify_reaction(smiles):
    """Clasifica la reacción basándose en patrones de texto en reaction_smiles_sequence."""
    if pd.isna(smiles): return "Desconocida"
    smiles = str(smiles).lower()
    
    # Suzuki: Ácidos borónicos o ésteres (B-O-H o B-O-C)
    if "b(o)o" in smiles or "b1oo" in smiles or "ob(o)" in smiles:
        return "Suzuki Coupling"
    
    # Buchwald-Hartwig: Aminas (N) + Haluros de arilo + presencia común de Pd
    # Buscamos nitrógeno unido a carbono y presencia de haluros
    elif "n" in smiles and any(h in smiles for h in ["cl", "br", "i"]) and "c1" in smiles:
        return "Buchwald-Hartwig"
    
    # Esterificación: Ácido + Alcohol -> C(=O)O
    elif "c(=o)o" in smiles and (smiles.count("o") >= 3 or ".o" in smiles):
        return "Esterificación"
    
    return "Otras"

def calculate_coherence_table(csv_results="comparison_results.csv", original_data_csv="/content/drive/MyDrive/CGGM/data/ord/ord_tokenized_reactions.csv.gz"):
    try:
        df_res = pd.read_csv(csv_results)
        df_orig = pd.read_csv(original_data_csv)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Usamos la columna específica detectada
    col_smiles = "reaction_smiles_sequence"
    if col_smiles not in df_orig.columns:
        print(f"❌ Error: La columna {col_smiles} no existe.")
        return

    print(f"🔬 Clasificando reacciones usando '{col_smiles}'...")
    
    # Mapeo de tipos de modelo
    model_types = df_res["Type"].unique()
    real_data_mask = (df_res["Type"] == "Real Data")
    sample_size = len(df_res[real_data_mask])
    
    # Clasificar las muestras
    df_orig_sample = df_orig.head(sample_size).copy()
    df_orig_sample["Reaction_Class"] = df_orig_sample[col_smiles].apply(classify_reaction)

    results_table = []
    clases_interes = ["Suzuki Coupling", "Buchwald-Hartwig", "Esterificación"]

    for r_class in clases_interes:
        # Obtener los índices de las muestras que pertenecen a esta clase
        class_indices = df_orig_sample[df_orig_sample["Reaction_Class"] == r_class].index
        
        if len(class_indices) < 5:
            print(f"⚠️ Clase '{r_class}' con pocas muestras ({len(class_indices)}), saltando...")
            continue

        row = {"Clase de Reacción": r_class, "Muestras": len(class_indices)}
        
        # Temperatura Real (Z-Score)
        y_true = df_res[real_data_mask].iloc[class_indices]["Temp"].values

        for m_type in model_types:
            if m_type == "Real Data": continue
            
            # Predicciones del modelo para esos mismos índices
            y_pred = df_res[df_res["Type"] == m_type].iloc[class_indices]["Temp"].values
            
            # Calculamos R2 (Coeficiente de determinación)
            r2 = r2_score(y_true, y_pred)
            row[f"R2 {m_type}"] = round(r2, 3)
            
        results_table.append(row)

    # 3. Mostrar y Guardar
    if results_table:
        df_final = pd.DataFrame(results_table)
        print("\n⚖️ TABLA 5: COHERENCIA QUÍMICA POR CATEGORÍA (R²)")
        print("================================================================")
        print(df_final.to_string(index=False))
        print("================================================================")
        df_final.to_csv("chemical_coherence_r2.csv", index=False)
        print("✅ Tabla guardada como 'chemical_coherence_r2.csv'")
    else:
        print("❌ No se encontraron suficientes muestras para las clases especificadas.")

if __name__ == "__main__":
    calculate_coherence_table()