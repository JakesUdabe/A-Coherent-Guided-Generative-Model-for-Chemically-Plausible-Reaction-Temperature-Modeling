#!/usr/bin/env python3
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from dataset_tokenized import TokenizedReactionsDataset, collate_fn
from model_cggm_t import SMILESEncoder, CGGMT_VAE
import loss_constraints


# -------------------------
# ELBO (idéntica a training)
# -------------------------
def elbo_loss(T_pred, T_true, mu, logvar,
              T_min, T_max,
              lambda_cons, beta, free_bits_tau):

    T_pred = T_pred.view_as(T_true)

    recon = F.mse_loss(T_pred, T_true, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_free = torch.clamp(kl, min=free_bits_tau)

    l_cons = loss_constraints.constraint_loss(T_pred, T_min, T_max)

    L = recon + beta * kl_free + lambda_cons * l_cons
    return L, recon, kl, l_cons


# -------------------------
# Evaluación
# -------------------------
def evaluate(encoder, reducer, vae, loader, device,
             T_min, T_max, lambda_cons, free_bits_tau):

    encoder.eval()
    reducer.eval()
    vae.eval()

    totals = {"recon": 0.0, "kl": 0.0, "cons": 0.0, "elbo": 0.0}
    n_batches = 0

    with torch.no_grad():
        for tokens, lengths, T_true in loader:
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            T_true = T_true.to(device)

            c = reducer(encoder(tokens, lengths))
            T_pred, mu, logvar = vae(T_true, c, is_sampling=False)

            L, recon, kl, cons = elbo_loss(
                T_pred, T_true, mu, logvar,
                T_min, T_max,
                lambda_cons, vae.beta, free_bits_tau
            )

            totals["recon"] += recon.item()
            totals["kl"] += kl.item()
            totals["cons"] += cons.item()
            totals["elbo"] += L.item()
            n_batches += 1

    for k in totals:
        totals[k] /= n_batches

    return totals


# -------------------------
# MAIN
# -------------------------
def main():

    # -------- PATHS --------
    CHECKPOINT_DIR = "/content/drive/MyDrive/cggm_project/checkpoints"
    GRID_CSV = os.path.join(CHECKPOINT_DIR, "vae_grid_results_stage2.csv")
    OUTPUT_CSV = os.path.join(CHECKPOINT_DIR, "vae_grid_evaluated2.csv")

    CSV_PATH = "../data/ord_tokenized_reactions.csv.gz"
    VOCAB_PATH = "../data/vocab.json"

    # -------- FIXED PARAMS --------
    BATCH_SIZE = 64
    CONTEXT_DIM = 32
    HIDDEN_DIM = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

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

    # -------- MODELOS BASE --------
    encoder = SMILESEncoder(
        vocab_size=dataset.vocab_size
    ).to(device)

    reducer = torch.nn.Sequential(
        torch.nn.Linear(encoder.output_dim, CONTEXT_DIM),
        torch.nn.ReLU()
    ).to(device)

    # -------- GRID --------
    grid = pd.read_csv(GRID_CSV)

    results = []
    skipped = []

    for _, row in grid.iterrows():

        run_id = int(row["run_id"])
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"VAE{run_id}.pth")

        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint no encontrado: {ckpt_path}")
            skipped.append((run_id, "file_not_found"))
            continue

        print(f"Evaluating run {run_id}")

        # ----- construir VAE correcto -----
        vae = CGGMT_VAE(
            context_dim=CONTEXT_DIM,
            latent_dim=int(row["latent_dim"]),
            hidden_dim=HIDDEN_DIM,
            n_res_blocks=int(row["vae_n_res_blocks"])
        ).to(device)

        # ----- cargar pesos -----
        try:
            ckpt = torch.load(ckpt_path, map_location=device)

            encoder.load_state_dict(ckpt["encoder_state_dict"])
            reducer.load_state_dict(ckpt["reducer_state_dict"])

            # Carga estricta: arquitectura debe coincidir
            vae.load_state_dict(ckpt["vae_state_dict"], strict=True)

        except RuntimeError as e:
            print(f"⏭️ Skip run {run_id} (arquitectura incompatible)")
            print(f"    Motivo: {str(e).splitlines()[0]}")
            skipped.append((run_id, "arch_mismatch"))
            continue

        # ----- parámetros -----
        vae.set_beta(float(row["beta"]))
        free_bits_tau = float(row["free_bits_tau"])

        # ----- evaluación -----
        metrics = evaluate(
            encoder, reducer, vae,
            val_loader, device,
            T_min, T_max,
            lambda_cons=float(row["lambda_cons"]),
            free_bits_tau=free_bits_tau
        )

        results.append({
            **row.to_dict(),
            "recon_loss": metrics["recon"],
            "kl_loss": metrics["kl"],
            "cons_loss": metrics["cons"],
            "elbo": metrics["elbo"]
        })

    # -------- SAVE --------
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Resultados guardados en {OUTPUT_CSV}")

    if skipped:
        print(f"⏭️ Runs skippeados: {len(skipped)}")
        for r, reason in skipped:
            print(f"   - run {r}: {reason}")


if __name__ == "__main__":
    main()
