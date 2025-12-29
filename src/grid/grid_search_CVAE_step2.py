#!/usr/bin/env python3
# vae_grid_search_stage2.py
# Etapa 2: Optimización del Aprendizaje (LR) y Ajuste de Restricciones Físicas (Lambda)

import itertools
import pandas as pd
import time
import os
import re
import subprocess
import numpy as np

# --- 1. VALORES GANADORES ETAPA 1 (FIJOS AHORA) ---
BEST_BETA = 0.05
BEST_TAU = 1.0
BEST_LATENT = 64
BEST_RES_BLOCKS = 3

# --- 2. CONFIGURACIÓN DE LA GRILLA ETAPA 2 ---
# Exploramos el ritmo de aprendizaje y la fuerza de la restricción física
LR_VALUES = [5e-5, 1e-5, 5e-6] 
LAMBDA_CONS_VALUES = [5.0, 10.0, 20.0, 50.0] 

# --- CONFIGURACIÓN DE EJECUCIÓN ---
OUTPUT_CSV = "vae_grid_results_stage2.csv"
FIXED_EPOCHS = 5  # Aumentamos un poco las épocas para ver mejor la convergencia del LR

FIXED_ARGS = {
    'epochs': FIXED_EPOCHS,
    'kl_adapt': 'False',
    'base_save_path': "/content/drive/MyDrive/cggm_project/stage2",
}

# -------------------------
# Utilidad de Extracción (Mantenida igual para consistencia)
# -------------------------
NUMBER_PATTERN = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?" 

def extract_final_metrics(log_output):
    pattern = re.compile(
        r"VAE_Loss:\s*(?P<vae_loss>" + NUMBER_PATTERN + r")\s*\[Recon:\s*(?P<recon_loss>" + NUMBER_PATTERN + r"),\s*KL:\s*(?P<kl_loss>" + NUMBER_PATTERN + r"),\s*Cons:\s*(?P<cons_loss>" + NUMBER_PATTERN + r")\]"
    )
    matches = list(pattern.finditer(log_output))
    if not matches:
        return np.nan, np.nan, np.nan, np.nan
        
    final_match = matches[-1] 
    return (
        float(final_match.group('vae_loss')),
        float(final_match.group('recon_loss')),
        float(final_match.group('kl_loss')),
        float(final_match.group('cons_loss'))
    )

# -------------------------
# Carga y Reanudación
# -------------------------
def load_or_initialize_results(csv_path):
    columns = ['run_id', 'lr_g', 'lambda_cons', 'beta', 'latent_dim', 'vae_n_res_blocks', 'free_bits_tau', 
               'VAE_Loss_Final', 'Recon_Loss_Final', 'KL_Loss_Final', 'Cons_Loss_Final']
    
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        # Limpieza de fallidos
        results_df = results_df[results_df['VAE_Loss_Final'] > 0]
        start_id = results_df['run_id'].max() + 1 if not results_df.empty else 1
        return results_df, start_id
    return pd.DataFrame(columns=columns), 1

# -------------------------
# Ejecución del Job
# -------------------------
def run_training_job(lr, l_cons, run_id):
    run_save_path = os.path.join(FIXED_ARGS['base_save_path'], f"run_st2_{run_id}")
    
    cmd = [
        "python3", "train_cggm_vae.py", 
        "--epochs", str(FIXED_ARGS['epochs']),
        "--save_path", run_save_path, 
        "--lr_g", f"{lr:.1e}", 
        "--lambda_cons", str(l_cons), 
        "--beta", str(BEST_BETA), 
        "--latent_dim", str(BEST_LATENT), 
        "--vae_n_res_blocks", str(BEST_RES_BLOCKS),
        "--free_bits_tau", str(BEST_TAU),
        "--kl_adapt", "False"
    ]
    
    print(f"\n[STAGE 2] Run {run_id} -> LR: {lr}, Lambda_Cons: {l_cons}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600) 
        vae_l, recon_l, kl_l, cons_l = extract_final_metrics(result.stdout)
        status = "✅ OK"
    except Exception as e:
        print(f"❌ Error en run {run_id}: {e}")
        vae_l, recon_l, kl_l, cons_l = np.nan, np.nan, np.nan, np.nan
        status = "❌ FALLÓ"

    return {
        'run_id': run_id, 'lr_g': lr, 'lambda_cons': l_cons,
        'beta': BEST_BETA, 'latent_dim': BEST_LATENT, 'vae_n_res_blocks': BEST_RES_BLOCKS, 'free_bits_tau': BEST_TAU,
        'VAE_Loss_Final': vae_l, 'Recon_Loss_Final': recon_l, 'KL_Loss_Final': kl_l, 'Cons_Loss_Final': cons_l
    }

# -------------------------
# Bucle Principal
# -------------------------
if __name__ == "__main__":
    results_df, start_id = load_or_initialize_results(OUTPUT_CSV)
    
    all_combinations = list(itertools.product(LR_VALUES, LAMBDA_CONS_VALUES))
    pending = all_combinations[start_id - 1:]

    for i, (lr, l_cons) in enumerate(pending):
        run_id = start_id + i 
        res = run_training_job(lr, l_cons, run_id)
        results_df = pd.concat([results_df, pd.DataFrame([res])], ignore_index=True)
        results_df.to_csv(OUTPUT_CSV, index=False)
            
    print(f"\n--- Etapa 2 Completada. Resultados en {OUTPUT_CSV} ---")