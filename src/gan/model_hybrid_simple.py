# model_hybrid_simple.py
# Discriminator_Simple mejorado: spectral norm + residual blocks + LayerNorm + dropout

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class ResidualBlock(nn.Module):
    """Bloque residual con spectral norm, LayerNorm y Dropout."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # Usamos spectral_norm para controlar la constante de Lipschitz del discriminador
        self.fc1 = spectral_norm(nn.Linear(dim, dim))
        self.fc2 = spectral_norm(nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # Sumamos la conexión residual antes de la última activación
        out = out + residual
        return self.act(out)

class Discriminator_Simple(nn.Module):
    """
    Discriminador optimizado para regresión condicional.
    Compatible con valores de temperatura estandarizados (Z-score).
    """
    def __init__(self, context_dim=32, temp_dim=1, hidden_dim=768, n_layers=4, dropout=0.1):
        super().__init__()
        input_dim = context_dim + temp_dim

        layers = []
        
        # Capa de entrada: Proyecta la combinación [Molécula + Temperatura]
        layers.append(spectral_norm(nn.Linear(input_dim, hidden_dim)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Cuerpo de la red: Bloques residuales o densos
        if n_layers > 2:
            for _ in range(n_layers - 2):
                layers.append(ResidualBlock(hidden_dim, dropout=dropout))
        else:
            # Fallback para arquitecturas poco profundas
            for _ in range(max(0, n_layers - 2)):
                layers.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Capa de salida: Retorna un score escalar (logit)
        # No usamos Sigmoid al final si planeas usar WGAN-GP o BinaryCrossEntropyWithLogits
        layers.append(spectral_norm(nn.Linear(hidden_dim, 1)))
        
        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier Uniform es ideal para LeakyReLU
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, context_c, temp_T):
        """
        context_c: Vector de la molécula (B, context_dim)
        temp_T: Temperatura normalizada (B, 1) o (B,)
        """
        # Asegurar que la temperatura tenga dimensión (B, 1)
        if temp_T.dim() == 1:
            temp_T = temp_T.unsqueeze(-1)
        
        # Concatenar condición (molécula) y valor (temperatura)
        x = torch.cat([context_c, temp_T], dim=1)
        return self.model(x)