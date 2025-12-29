#!/usr/bin/env python3
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from dataset_tokenized import TokenizedReactionsDataset, collate_fn
from model_cggm_t import SMILESEncoder, CGGMT_VAE
from model_hybrid_simple import Discriminator_Simple
import loss_constraints

# --------------------------------------------------
# ELBO (idéntica a training)
# --------------------------------------------------
def elbo_loss(T_pred, T_true, mu, logvar, T_min, T_max,
              lambda_cons, beta, free_bits_tau):

    T_pred = T_pred.view_as(T_true)

    recon = F.mse_loss(T_pred, T_true, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_free = torch.clamp(kl, min=free_bits_tau)
    cons = loss_constraints.constraint_loss(T_pred, T_min, T_max)

    elbo = recon + beta * kl_free + lambda_cons * cons
    return elbo, recon, kl


# --------------------------------------------------
# Evaluación adversarial
# --------------------------------------------------
@torch.no_grad()
def evaluate_hybrid(encoder, reducer, vae, D, loader, device,
                    T_min, T_max, lambda_cons, beta, free_bits_tau):

    encoder.eval()
    reducer.eval()
    vae.eval()
    D.eval()

    totals = {
        "elbo": 0.0,
        "adv": 0.0,
        "g_loss": 0.0,
        "d_real": 0.0,
        "d_fake": 0.0
    }

    n_batches = 0

    for tokens, lengths, T_true in loader:
        tokens = tokens.to(device)
        lengths = lengths.to(device)
        T_true = T_true.to(device)

        c = reducer(encoder(tokens, lengths))

        # Generator forward
        T_pred, mu, logvar = vae(T_true, c, is_sampling=False)

        elbo, _, _ = elbo_loss(
            T_pred, T_true, mu, logvar,
            T_min, T_max,
            lambda_cons, beta, free_bits_tau
        )

        # Adversarial terms
        d_real = D(c, T_true).mean()
        d_fake = D(c, T_pred).mean()
        adv_loss = -d_fake

        totals["elbo"] += elbo.item()
        totals["adv"] += adv_loss.item()
        totals["g_loss"] += (elbo + adv_loss).item()
        totals["d_real"] += d_real.item()
        totals["d_fake"] += d_fake.item()

        n_batches += 1

    for k in totals:
        totals[k] /= n_batches

    delta_sep = totals["d_real"] - totals["d_fake"]

    return {
        "ELBO_Final": totals["elbo"],
        "Adv_Loss": totals["adv"],
        "G_Loss": totals["g_loss"],
        "Delta_Separation": delta_sep
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    # -------- PATHS --------
    CHECKPOINT_DIR = "/content/drive/MyDrive/cggm_project/stage3"
    GRID_CSV = "/content/drive/MyDrive/cggm_project/stage3/adv_grid.csv"
    OUTPUT_CSV = f"{CHECKPOINT_DIR}/adv_grid_evaluated.csv"

    CSV_PATH = "../data/ord_tokenized_reactions.csv.gz"
    VOCAB_PATH = "../data/vocab.json"

    # -------- FIXED PARAMS (baseline VAE) --------
    BATCH_SIZE = 64
    CONTEXT_DIM = 32
    LATENT_DIM = 64
    HIDDEN_DIM_G = 64

    BETA = 0.05
    FREE_BITS_TAU = 1.0
    LAMBDA_CONS = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- DATASET --------
    dataset = TokenizedReactionsDataset(
        csv_path=CSV_PATH,
        vocab_path=VOCAB_PATH
    )

    T_min = dataset.T_norm_min
    T_max = dataset.T_norm_max

    n = len(dataset)
    n_train = int(0.8 * n)
    _, val_ds = random_split(
        dataset, [n_train, n - n_train],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # -------- GRID --------
    grid = pd.read_csv(GRID_CSV)
    results = []

    for _, row in grid.iterrows():

        run_id = int(row["run_id"])
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"H{run_id}.pth"
        )

        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint no encontrado: {ckpt_path}")
            continue

        print(f"Evaluating run {run_id}")

        # Models
        encoder = SMILESEncoder(vocab_size=dataset.vocab_size).to(device)
        reducer = torch.nn.Sequential(
            torch.nn.Linear(encoder.output_dim, CONTEXT_DIM),
            torch.nn.ReLU()
        ).to(device)

        vae = CGGMT_VAE(
            context_dim=CONTEXT_DIM,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM_G,
            n_res_blocks=3
        ).to(device)

        D = Discriminator_Simple(
            context_dim=CONTEXT_DIM,
            temp_dim=1,
            hidden_dim=768,
            n_layers=4,
            dropout=0.1
        ).to(device)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        reducer.load_state_dict(ckpt["reducer_state_dict"])

        try:
            vae.load_state_dict(ckpt["vae_state_dict"])
        except Exception:
            vae.load_state_dict_compat(ckpt["vae_state_dict"])

        D.load_state_dict(ckpt["discriminator_state_dict"])

        metrics = evaluate_hybrid(
            encoder, reducer, vae, D,
            val_loader, device,
            T_min, T_max,
            lambda_cons=LAMBDA_CONS,
            beta=BETA,
            free_bits_tau=FREE_BITS_TAU
        )

        results.append({
            **row.to_dict(),
            **metrics
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Resultados guardados en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
