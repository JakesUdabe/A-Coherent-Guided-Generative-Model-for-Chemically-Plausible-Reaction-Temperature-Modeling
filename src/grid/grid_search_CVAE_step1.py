# vae_full_grid_search.py
# Etapa 1: Búsqueda Jerárquica para Estabilidad (Beta y Tau) y Capacidad (Latent Dim y Res Blocks)

import itertools
import pandas as pd
import time
import os
import re
import subprocess
import numpy as np
from datetime import datetime

# --- CONFIGURACIÓN DE LA GRILLA (ETAPA 1: 81 Combinaciones) ---

BETA_VALUES = [0.01] #Lo he modificado para este ultimo experimento, antes era [0.05, 0.1, 0.5] 
TAU_VALUES = [0.0, 1.0, 2.0] 
LATENT_DIM_VALUES = [64, 128, 256] 
RES_BLOCKS_VALUES = [1, 3, 5] 

# --- CONFIGURACIÓN FIJA ---
OUTPUT_CSV = f"vae_grid_results_stage1.csv"
FIXED_EPOCHS = 5

FIXED_ARGS = {
    'epochs': FIXED_EPOCHS,
    'kl_adapt': 'False',
    'lr_g': 1.00e-05,
    'lambda_cons': 10.0,
    # Ruta base asumida a partir de logs anteriores.
    'base_save_path': "/content/drive/MyDrive/cggm_project",
}

# -------------------------
# Utilidad de Extracción de Métricas (Manteniendo la robustez)
# -------------------------

NUMBER_PATTERN = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?" 

def extract_final_metrics(log_output):
    """
    Extrae las métricas de la última época del log de entrenamiento, 
    soportando notación científica y manejando errores de parsing.
    """
    pattern = re.compile(
        r"VAE_Loss:\s*(?P<vae_loss>" + NUMBER_PATTERN + r")\s*\[Recon:\s*(?P<recon_loss>" + NUMBER_PATTERN + r"),\s*KL:\s*(?P<kl_loss>" + NUMBER_PATTERN + r"),\s*Cons:\s*(?P<cons_loss>" + NUMBER_PATTERN + r")\]"
    )
    matches = list(pattern.finditer(log_output))
    
    if not matches:
        print("⚠️ No se pudieron extraer las métricas finales del log. Verifique el formato de salida.")
        print("--- Últimas 10 líneas del log para depuración ---")
        print('\n'.join(log_output.splitlines()[-10:]))
        print("-------------------------------------------------")
        return np.nan, np.nan, np.nan 
        
    final_match = matches[-1] 
    
    try:
        vae_loss = float(final_match.group('vae_loss'))
        recon_loss = float(final_match.group('recon_loss'))
        kl_loss = float(final_match.group('kl_loss'))
        
        if recon_loss <= 0.0001: 
            print("\n🚨 ¡ALERTA DE COLAPSO! Recon Loss es CERO (o cercano a cero).")
            # Imprimir el inicio del log para ver la causa del colapso (ej. NaN en los gradientes)
            print("--- INICIO DEL LOG COMPLETO para confirmar el colapso ---")
            # Mostrar los primeros 3000 caracteres
            print(log_output[:3000]) 
            print("--- FIN DEL LOG DE COLAPSO ---\n")
        
        return vae_loss, recon_loss, kl_loss
    except ValueError:
        print("❌ ERROR DE PARSEO: La expresión regular capturó texto no convertible a float.")
        return np.nan, np.nan, np.nan

# -------------------------
# Carga y Reanudación
# -------------------------
def load_or_initialize_results(csv_path):
    """Carga resultados previos o inicializa un nuevo DataFrame."""
    columns = ['run_id', 'lr_g', 'beta', 'latent_dim', 'lambda_cons', 'vae_n_res_blocks', 'free_bits_tau', 'VAE_Loss_Final', 'Recon_Loss_Final', 'KL_Loss_Final']
    
    if os.path.exists(csv_path):
        try:
            results_df = pd.read_csv(csv_path)
            for col in columns:
                if col not in results_df.columns:
                    results_df[col] = np.nan
                    
            # Elimina filas con 0.0s o NaN en la pérdida principal para reintentar
            results_df = results_df[results_df['VAE_Loss_Final'] != 0.0] 
            results_df.dropna(subset=['VAE_Loss_Final'], inplace=True)
            start_id = results_df['run_id'].max() + 1 if not results_df.empty else 1
            print(f"🔄 Reanudando. {start_id - 1} ejecuciones encontradas. Iniciando desde el ID {start_id}.")
            return results_df, start_id
        except Exception as e:
            print(f"❌ ERROR al cargar CSV ({e}), iniciando desde ID 1.")
            return pd.DataFrame(columns=columns), 1
    else:
        print("🆕 Archivo de resultados no encontrado. Iniciando Grid Search desde el ID 1.")
        return pd.DataFrame(columns=columns), 1

# -------------------------
# Ejecución del Entrenamiento REAL (CORRECCIÓN CLAVE AQUÍ)
# -------------------------
def run_training_job(beta, latent_dim, res_blocks, tau, run_id):
    """Ejecuta el script train_cggm_vae.py con los hiperparámetros dados."""
    
    # 🎯 CORRECCIÓN CLAVE: Generar una ruta de guardado única para este run.
    # Esto aísla el checkpoint guardado (cggm_vae_epoch_5.pth) de otros runs,
    # impidiendo la reanudación indeseada.
    run_save_path = os.path.join(FIXED_ARGS['base_save_path'], f"vae_grid_run_{run_id}")
    
    # 1. Construir el comando de ejecución
    cmd = [
        "python3", "train_cggm_vae.py", 
        "--epochs", str(FIXED_ARGS['epochs']),
        "--kl_adapt", FIXED_ARGS['kl_adapt'],
        
        # Usar la ruta de guardado ÚNICA, que es el único argumento para influir en el checkpointing
        "--save_path", run_save_path, 
        
        # VALORES FIJOS Y VARIABLES
        "--lr_g", f"{FIXED_ARGS['lr_g']:.1e}", 
        "--lambda_cons", str(FIXED_ARGS['lambda_cons']), 
        "--beta", str(beta), 
        "--latent_dim", str(latent_dim), 
        "--vae_n_res_blocks", str(res_blocks),
        "--free_bits_tau", str(tau)
        # Nota: vae_dropout no es parte del grid, así que no se pasa a menos que se fije
    ]
    
    total_combinations = len(globals().get('all_combinations', []))
        
    print(f"\n--- Ejecutando Configuración {run_id} de {total_combinations} ---")
    print(f"Comando: {' '.join(cmd)}")
    print(f"El checkpoint se guardará en: {run_save_path}/cggm_vae_epoch_5.pth")
    
    start_time = time.time()
    
    # 2. Ejecutar el script real y capturar la salida
    try:
        # Se añade un timeout para evitar bloqueos eternos (25 min por seguridad)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3000) 
        end_time = time.time()
        print(f"✅ Ejecución completada en {end_time - start_time:.2f}s.")
        
        # 3. Extraer métricas de la salida
        vae_loss, recon_loss, kl_loss = extract_final_metrics(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"❌ FALLÓ la ejecución {run_id} (Código {e.returncode}).")
        
        # --- DEBUGGING AÑADIDO: Imprimir el error completo (STDERR) ---
        print("\n--- ERROR COMPLETO (STDERR) ---")
        print(e.stderr)
        print("-------------------------------")

        vae_loss, recon_loss, kl_loss = np.nan, np.nan, np.nan
    
    # 4. Devolver resultados
    return {
        'run_id': run_id, 
        'lr_g': FIXED_ARGS['lr_g'], 
        'lambda_cons': FIXED_ARGS['lambda_cons'],
        'beta': beta, 
        'latent_dim': latent_dim, 
        'vae_n_res_blocks': res_blocks,
        'free_bits_tau': tau,
        'VAE_Loss_Final': vae_loss, 
        'Recon_Loss_Final': recon_loss, 
        'KL_Loss_Final': kl_loss
    }


# -------------------------
# Bucle Principal del Grid Search
# -------------------------
if __name__ == "__main__":
    
    results_df, start_id = load_or_initialize_results(OUTPUT_CSV)
    
    # 2. Generar todas las combinaciones (3 * 3 * 3 * 3 = 81)
    all_combinations = list(itertools.product(
        BETA_VALUES, LATENT_DIM_VALUES, RES_BLOCKS_VALUES, TAU_VALUES
    ))
    
    total_combinations = len(all_combinations)
    print(f"Total de configuraciones en la grilla: {total_combinations}")
    
    pending_combinations = all_combinations[start_id - 1:]
    print(f"Configuraciones restantes para ejecutar: {len(pending_combinations)}")

    for i, (beta, latent_dim, res_blocks, tau) in enumerate(pending_combinations):
        
        run_id = start_id + i 
        
        result = run_training_job(beta, latent_dim, res_blocks, tau, run_id)
        
        results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)

        # Checkpoint: Guardar cada run
        if run_id % 1 == 0 or run_id == total_combinations:
            results_df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n[CHECKPOINT] Progreso guardado en '{OUTPUT_CSV}' ({run_id}/{total_combinations}).")
            
    print("\n--- Grid Search Completado ---")
    print(f"Resultados finales guardados en {OUTPUT_CSV}")