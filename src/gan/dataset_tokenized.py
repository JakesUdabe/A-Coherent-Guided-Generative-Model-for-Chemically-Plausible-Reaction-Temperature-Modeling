# ============================================================
# dataset_tokenized.py - Versión Corregida
# ============================================================

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import json
import os
import ast # Importar para manejar la conversión de strings de lista/JSON

class TokenizedReactionsDataset(Dataset):
    """
    Dataset para reacciones tokenizadas (IDs de tokens y temperatura normalizada).
    """
    
    # Se añade vocab_path=None (compatible con train_cggm_t.py) y se corrige max_rows=0
    def __init__(self, csv_path, max_rows=0, vocab_path=None):
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ No se encontró el archivo: {csv_path}")

        print(f"📂 Cargando dataset desde: {csv_path}")
        
        # Usar compression="gzip" si el archivo es .gz
        try:
            self.df = pd.read_csv(csv_path, compression="gzip")
        except Exception:
            # Intentar sin compresión si falla (para mayor robustez)
            self.df = pd.read_csv(csv_path)

        # 1. Límite de filas
        if max_rows > 0:
            self.df = self.df.head(max_rows)
            print(f"📉 Se usaron solo las primeras {max_rows} filas.")

        # 2. Validación de columnas
        SMILES_COL = "smiles_token_ids_json"
        TEMP_COL = "temperature_normalized"

        if SMILES_COL not in self.df.columns:
            raise ValueError(f"❌ Falta la columna '{SMILES_COL}' en el CSV.")
        if TEMP_COL not in self.df.columns:
            raise ValueError(f"❌ Falta la columna '{TEMP_COL}' en el CSV.")

        # 3. Conversión de tokens (usando ast.literal_eval)
        # Esto maneja tanto strings JSON como strings Python de listas (ej: "['C', '=']")
        self.data = self.df[SMILES_COL].apply(ast.literal_eval).tolist()
        
        # 4. Temperaturas
        self.temperatures = torch.tensor(self.df[TEMP_COL].values, dtype=torch.float32)

        # 5. Vocabulario: cargar o generar
        if vocab_path and os.path.exists(vocab_path):
            print(f"📘 Vocab cargado desde {vocab_path}")
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
            # Asumiendo que el vocabulario cargado ya tiene el ID máximo + 1
            self.vocab_size = max(self.vocab.values()) + 1 if self.vocab else 1
        else:
            # Si el vocabulario no existe o no se pasa, lo generamos (si los tokens son strings)
            # Ya que usamos IDs pre-calculados, el vocab_size es el máximo ID + 1.
            print("🧠 Calculando vocab_size a partir de los IDs...")
            all_ids = [idx for seq in self.data for idx in seq]
            self.vocab_size = max(all_ids) + 1 if all_ids else 1
            # Nota: No guardamos el vocabulario JSON aquí, solo calculamos el tamaño.

        print(f"✅ Dataset cargado: {len(self.df)} muestras, vocab_size={self.vocab_size}")
        
        # 6. Definir límites de normalización (para pasar al script de entrenamiento si es necesario)
        self.T_norm_min = -3.0 # Equivale a temperaturas muy frías (~ -100°C)
        self.T_norm_max = 4.5  # Equivale a temperaturas muy altas (~ 300°C)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Aseguramos que los tokens sean un tensor long
        token_tensor = torch.tensor(self.data[idx], dtype=torch.long)
        T_norm = self.temperatures[idx].unsqueeze(0) # (1,) para consistencia con temps
        
        # Devolvemos el tensor de tokens y la T_norm
        return token_tensor, T_norm


def collate_fn(batch):
    """
    Combina ejemplos individuales en un batch.
    Hace padding de las secuencias al máximo largo del batch.
    """
    # Descomponemos el batch: toks = token_tensor, temps = T_norm
    token_tensors, temps = zip(*batch)
    
    # Calculamos lengths y apilamos temps
    lengths = torch.tensor([len(t) for t in token_tensors], dtype=torch.long)
    temps = torch.stack(temps) # (B, 1)

    # Padding de las secuencias al máximo largo del batch
    padded_tokens = pad_sequence(token_tensors, batch_first=True, padding_value=0)
    
    return padded_tokens, lengths, temps