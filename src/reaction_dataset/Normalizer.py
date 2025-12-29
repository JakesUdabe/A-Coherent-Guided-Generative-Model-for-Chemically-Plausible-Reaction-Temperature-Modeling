#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalizer.py (VERSIÓN LIMPIA)
Solución: Filtrado estricto de outliers (-100°C a 500°C) y StandardScaler.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import logging

# ==============================
# Configuración
# ==============================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("normalizer")

# Rutas de archivo para Colab
BASE_DIR = "/content/drive/MyDrive/CGGM/data/ord"
INPUT_FILENAME = "ord_filtered_key_reactions.csv.gz"
OUTPUT_FILENAME = "ord_normalized_key_reactions.csv.gz"

INPUT_PATH = os.path.join(BASE_DIR, INPUT_FILENAME)
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_FILENAME)

# Nombres de columnas
TEMP_C_COL = 'temperature_c'
TEMP_K_COL = 'temperature_k'
TEMP_NORM_COL = 'temperature_normalized'
ID_COL = 'reaction_id'
TEST_SIZE = 0.2
RANDOM_SEED = 42

# 🚨 FILTRO DE OUTLIERS: Basado en tu análisis previo
MIN_TEMP_C = -100.0
MAX_TEMP_C = 500.0

# ==============================
# Pipeline
# ==============================

def load_data(file_path: str) -> pd.DataFrame:
    """Carga datos y elimina outliers fuera del rango químico realista."""
    logger.info(f"Cargando datos desde: {file_path}")
    df = pd.read_csv(file_path, compression='gzip')
    
    # Asegurar IDs
    if ID_COL not in df.columns:
        logger.warning(f"Columna '{ID_COL}' no encontrada. Generando IDs.")
        df[ID_COL] = [f"reaction_{i}" for i in range(len(df))]
        
    # Limpieza de nulos y conversión a número
    df[TEMP_C_COL] = pd.to_numeric(df[TEMP_C_COL], errors='coerce')
    df.dropna(subset=[TEMP_C_COL], inplace=True)
    
    # --- FILTRADO DE OUTLIERS ---
    initial_rows = len(df)
    # Aplicamos el filtro estricto
    df = df[(df[TEMP_C_COL] >= MIN_TEMP_C) & (df[TEMP_C_COL] <= MAX_TEMP_C)]
    removed = initial_rows - len(df)
    
    logger.info(f"Filas iniciales: {initial_rows}")
    logger.info(f"Outliers eliminados (>500°C o <-100°C): {removed}")
    logger.info(f"Filas restantes: {len(df)}")
    
    return df

def convert_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte a Kelvin y aplica StandardScaler (Media 0, Desv 1)."""
    
    # 1. Conversión a Kelvin
    df[TEMP_K_COL] = df[TEMP_C_COL] + 273.15
    
    # 2. División para evitar 'Data Leakage'
    indices = df.index
    train_indices, test_indices = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    df_train = df.loc[train_indices].copy()
    df_test = df.loc[test_indices].copy()
    
    # 3. StandardScaler
    # Ahora que no hay outliers, la desviación estándar será pequeña y útil.
    scaler = StandardScaler()
    scaler.fit(df_train[[TEMP_K_COL]])
    
    logger.info(f"Estadísticas del Scaler: Media={scaler.mean_[0]:.2f}K, Desv={scaler.scale_[0]:.2f}K")

    # Transformación
    df_train[TEMP_NORM_COL] = scaler.transform(df_train[[TEMP_K_COL]])
    df_test[TEMP_NORM_COL] = scaler.transform(df_test[[TEMP_K_COL]])

    # 4. Re-unión manteniendo el orden original de las filas filtradas
    df_final = pd.concat([df_train, df_test]).sort_index()
    
    return df_final

def main():
    try:
        if not os.path.exists(BASE_DIR):
             os.makedirs(BASE_DIR)

        df_raw = load_data(INPUT_PATH)
        df_norm = convert_and_normalize(df_raw)
        
        # Guardado
        logger.info(f"Guardando archivo limpio en: {OUTPUT_PATH}")
        df_norm.to_csv(OUTPUT_PATH, index=False, compression='gzip')
        
        logger.info("✅ ¡Proceso de limpieza y normalización exitoso!")
        
        # Verificación visual
        print("\nDistribución final de Temperatura Normalizada:")
        print(df_norm[TEMP_NORM_COL].describe())
        print("\nPrimeras muestras:")
        print(df_norm[[ID_COL, TEMP_C_COL, TEMP_NORM_COL]].head())

    except Exception as e:
        logger.critical(f"Error fatal durante el proceso: {e}")

if __name__ == "__main__":
    main()