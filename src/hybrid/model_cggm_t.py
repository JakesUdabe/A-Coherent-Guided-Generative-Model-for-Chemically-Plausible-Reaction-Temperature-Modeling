import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------------------------------------------------
# Bloque de Red Estructural (Residual Block)
# ------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
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
        out = out + residual  # Skip connection
        return self.act(out)

# ------------------------------------------------------------------
# 🧠 SMILESEncoder
# ------------------------------------------------------------------
class SMILESEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.output_dim = hidden_dim * 2
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, tokens, lengths):
        embedded = self.dropout_layer(self.embedding(tokens))
        # enforce_sorted=False es vital ya que los batches no suelen venir ordenados por longitud
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, h_n = self.gru(packed_embedded)
        # Extraer estados finales de ambas direcciones
        h_fwd = h_n[-2, :, :]
        h_bwd = h_n[-1, :, :]
        context_vector = torch.cat([h_fwd, h_bwd], dim=1)
        return context_vector

# ------------------------------------------------------------------
# 🌡️ CGGMT_VAE (Conditional VAE)
# ------------------------------------------------------------------
class CGGMT_VAE(nn.Module):
    def __init__(self, context_dim=32, latent_dim=32, hidden_dim=64, n_res_blocks=2, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.beta = 1.0
        self.target_kl = None
        self.kl_adapt_rate = 0.0

        # === ENCODER ===
        input_enc_dim = context_dim + 1 
        enc_layers = [nn.Linear(input_enc_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(n_res_blocks):
            enc_layers.append(ResidualBlock(hidden_dim, dropout))
        self.encoder_body = nn.Sequential(*enc_layers)
        
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # === DECODER ===
        input_dec_dim = latent_dim + context_dim
        dec_layers = [nn.Linear(input_dec_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(n_res_blocks):
            dec_layers.append(ResidualBlock(hidden_dim, dropout))
        self.decoder_body = nn.Sequential(*dec_layers)
        
        self.dec_mu = nn.Linear(hidden_dim, 1) 
        # logvar inicial de -2.0 es conservador, permitiendo que el modelo aprenda la precisión
        self.dec_logvar = nn.Parameter(torch.tensor([-2.0]), requires_grad=True) 

    def forward(self, T_true_norm, context_c, is_sampling=False):
        # Asegurar dimensiones: T_true_norm debe ser (B, 1)
        if T_true_norm.dim() == 1:
            T_true_norm = T_true_norm.unsqueeze(-1)
        
        batch_size = context_c.size(0)
        device = context_c.device

        # --- 1. ENCODER ---
        if not is_sampling:
            combined_input = torch.cat([T_true_norm, context_c], dim=1) 
            h_encode = self.encoder_body(combined_input)
            mu = self.enc_mu(h_encode)
            logvar = self.enc_logvar(h_encode)
        else:
            # Si muestreamos, el encoder no se usa. 
            # Retornamos ceros para mu/logvar para evitar errores de referencia.
            mu = torch.zeros(batch_size, self.latent_dim).to(device)
            logvar = torch.zeros(batch_size, self.latent_dim).to(device)

        # --- 2. REPARAMETRIZACIÓN ---
        if not is_sampling:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # En inferencia pura, z viene de la distribución base N(0, 1)
            z = torch.randn(batch_size, self.latent_dim).to(device)
        
        # --- 3. DECODER ---
        combined_decode = torch.cat([z, context_c], dim=1)
        h_decode = self.decoder_body(combined_decode)
        
        T_pred = self.dec_mu(h_decode)
        return T_pred, mu, logvar

    # --- Métodos para Beta Adaptativa ---
    def set_beta(self, beta):
        self.beta = beta

    def set_target_kl(self, target, adapt_rate):
        self.target_kl = target
        self.kl_adapt_rate = adapt_rate

    def adapt_beta(self, measured_kl):
        if self.target_kl is None:
            return self.beta
            
        # Error relativo es más estable que el absoluto
        kl_error = (measured_kl - self.target_kl)
        
        # Corrección: Asegurar que el cálculo de la nueva beta ocurra en el mismo dispositivo
        # Usamos un paso de actualización proporcional
        new_beta = self.beta * math.exp(self.kl_adapt_rate * kl_error)

        self.beta = min(max(new_beta, 1e-4), 10.0) # Límites de beta para evitar colapso
        return self.beta

    def load_state_dict_compat(self, state_dict):
        """Carga pesos ignorando discrepancias en capas residuales si vienes de un modelo MLP."""
        new_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in new_state_dict and param.size() == new_state_dict[name].size():
                new_state_dict[name].copy_(param)
        self.load_state_dict(new_state_dict)