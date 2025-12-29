#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
structural_encoder.py (CORREGIDO)
Paso 1.6: Codificación Estructural de SMILES.
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from typing import List, Tuple, Dict
from rdkit import Chem
from collections import Counter

# (Configuración y Constantes - Igual que antes)
# ==============================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("structural_encoder")

BASE_DIR = "/content/drive/MyDrive/CGGM/data/ord"
INPUT_FILENAME = "ord_normalized_key_reactions.csv.gz"
OUTPUT_FILENAME = "ord_tokenized_reactions.csv.gz"
VOCAB_FILENAME = "smiles_vocabulary.json"

INPUT_PATH = os.path.join(BASE_DIR, INPUT_FILENAME)
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_FILENAME)
VOCAB_PATH = os.path.join(BASE_DIR, VOCAB_FILENAME)

REACTION_SEPARATOR = '.' 
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>' 
EOS_TOKEN = '<EOS>' 
TOKEN_REACTION_SEPARATOR = '<SEP>'

# ==============================
# Funciones RDKit y Tokenización (Funciones auxiliares, sin cambios)
# ==============================

def canonize_smiles(smiles_list: List[str]) -> List[str]:
    # ... (código canonize_smiles de la respuesta anterior, sin cambios)
    canon_smiles = []
    for s in smiles_list:
        if s:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    canon = Chem.MolToSmiles(mol, isomericSmiles=True)
                    canon_smiles.append(canon)
                else:
                    canon_smiles.append(None)
            except Exception:
                canon_smiles.append(None)
        else:
            canon_smiles.append(None)
            
    return sorted(list(set([s for s in canon_smiles if s is not None])))


def preprocess_smiles_column(df: pd.DataFrame, col_name: str) -> pd.Series:
    # ... (código preprocess_smiles_column de la respuesta anterior, sin cambios)
    logger.info(f"Iniciando canonización para la columna '{col_name}'...")
    
    def process_row(smiles_json):
        try:
            smiles_list = json.loads(smiles_json)
        except (TypeError, json.JSONDecodeError):
            smiles_list = []
            
        return REACTION_SEPARATOR.join(canonize_smiles(smiles_list))

    processed_series = df[col_name].apply(process_row)
    processed_series.replace('', np.nan, inplace=True)
    
    return processed_series


def create_and_tokenize_sequence(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, int]]:
    # ... (código create_and_tokenize_sequence de la respuesta anterior, sin cambios)
    logger.info("Concatenando secuencias de reacción (Reactivos <SEP> Productos)...")
    
    df['reaction_smiles_sequence'] = (
        df['reactant_smiles_canon']
        .astype(str)
        .str.cat(
            df['product_smiles_canon'].astype(str), 
            sep=TOKEN_REACTION_SEPARATOR
        )
        .replace('nan'+TOKEN_REACTION_SEPARATOR+'nan', np.nan)
    )
    
    logger.info("Creando vocabulario y tokenizando secuencias...")
    
    def tokenize_smiles(smiles_sequence: str) -> List[str]:
        # Tokenizador simplificado
        tokens = []
        i = 0
        while i < len(smiles_sequence):
            if smiles_sequence[i:i+5] == TOKEN_REACTION_SEPARATOR:
                 tokens.append(TOKEN_REACTION_SEPARATOR)
                 i += len(TOKEN_REACTION_SEPARATOR)
            elif smiles_sequence[i] == '[' and i + 1 < len(smiles_sequence):
                j = smiles_sequence.find(']', i)
                if j != -1:
                    tokens.append(smiles_sequence[i:j+1])
                    i = j + 1
                else:
                    tokens.append(smiles_sequence[i])
                    i += 1
            else:
                tokens.append(smiles_sequence[i])
                i += 1
        return [t for t in tokens if t != REACTION_SEPARATOR and t != ' ']

    all_tokens = []
    valid_sequences = df['reaction_smiles_sequence'].dropna().astype(str)
    
    for seq in valid_sequences:
        all_tokens.extend(tokenize_smiles(seq))

    token_counts = Counter(all_tokens)
    
    vocab = {
        PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3, TOKEN_REACTION_SEPARATOR: 4
    }
    
    next_index = len(vocab)
    for token, _ in token_counts.most_common():
        if token not in vocab:
            vocab[token] = next_index
            next_index += 1
            
    token_to_idx = vocab
    
    def sequence_to_ids(smiles_sequence):
        if pd.isna(smiles_sequence):
            return []
        tokens = [SOS_TOKEN] + tokenize_smiles(smiles_sequence) + [EOS_TOKEN]
        ids = [token_to_idx.get(token, token_to_idx[UNK_TOKEN]) for token in tokens]
        return ids

    token_id_series = df['reaction_smiles_sequence'].apply(sequence_to_ids)
    
    return token_id_series, vocab


def run_structural_encoding():
    """Ejecuta el pipeline de codificación estructural."""
    logger.info("Iniciando pipeline de Codificación Estructural (Paso 1.6)")
    
    try:
        df = pd.read_csv(INPUT_PATH, compression='gzip')
    except Exception as e:
        logger.error(f"Error al cargar el archivo de entrada: {e}")
        return

    initial_rows = len(df)
    
    # 2. Canonizar SMILES
    df['reactant_smiles_canon'] = preprocess_smiles_column(df, 'reactant_smiles_json')
    df['product_smiles_canon'] = preprocess_smiles_column(df, 'product_smiles_json')

    # Limpieza de filas que no se pudieron canonizar
    df.dropna(subset=['reactant_smiles_canon', 'product_smiles_canon'], how='any', inplace=True)
    
    logger.info(f"Filas restantes después de canonización y limpieza: {len(df)}")
    
    # 3. Concatenar y Tokenizar
    token_ids, vocab = create_and_tokenize_sequence(df)
    
    df['smiles_token_ids'] = token_ids
    
    # 4. Guardar vocabulario
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    logger.info(f"Vocabulario SMILES guardado con {len(vocab)} tokens en {VOCAB_PATH}")
    
    # 5. Guardar el DataFrame final
    df['smiles_token_ids_json'] = df['smiles_token_ids'].apply(json.dumps)
    
    # **COLUMNAS FINALES A MANTENER:**
    cols_to_keep = [
        'reaction_id', 
        'reactant_smiles_json', 
        'product_smiles_json',
        'reactant_smiles_canon', 
        'product_smiles_canon',
        'reaction_smiles_sequence', 
        'smiles_token_ids_json', # <- Secuencia de IDs (Entrada del Encoder)
        'temperature_normalized', # <- Condición (Entrada del Decoder/Codificador Condicional)
    ]
    
    # 🚨 **Corrección Asegurada:** Seleccionamos solo las columnas que sabemos que existen.
    # Eliminamos columnas intermedias y mantenemos la estructura final para el modelo.
    final_df = df[cols_to_keep].copy()
    
    final_df.to_csv(OUTPUT_PATH, index=False, compression='gzip')
    logger.info(f"Datos tokenizados guardados en {OUTPUT_PATH}")

    # ==============================
    # 6. Diagnóstico
    # ==============================
    logger.info("=" * 50)
    logger.info("DIAGNÓSTICO DE CODIFICACIÓN ESTRUCTURAL")
    logger.info(f"Filas válidas tokenizadas: {len(final_df)}")
    logger.info(f"Tamaño del Vocabulario: {len(vocab)} tokens")
    
    sequence_lengths = df['smiles_token_ids'].apply(len)
    logger.info(f"Longitud Media de Secuencia: {sequence_lengths.mean():.2f}")
    
    logger.info("\nPrimeras 5 líneas del DataFrame final:")
    print(final_df[['reaction_smiles_sequence', 'smiles_token_ids_json', 'temperature_normalized']].head(5).to_string(index=False))
    
    logger.info("=" * 50)

if __name__ == "__main__":
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        logger.warning(f"Se ha creado el directorio base: {BASE_DIR}")
        
    run_structural_encoding()