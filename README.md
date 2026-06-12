# A Coherent Guided Generative Model for Chemically Plausible Reaction Temperature Modeling

[![Published in: Artificial Intelligence Chemistry](https://img.shields.io/badge/Published%20in-Artificial%20Intelligence%20Chemistry-blue)](URL_AL_PAPER_AQUI)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official repository for the paper **"A Coherent Guided Generative Model for Chemically Plausible Reaction Temperature Modeling"**, published in *Artificial Intelligence Chemistry*.

This repository provides the code implementations for the diverse generative architectures (VAE, GAN, and a novel Hybrid Model) developed to predict and model chemically valid reaction temperatures based on molecular structures and reaction conditions.

## 📌 Overview

Determining the optimal reaction temperature is critical in chemical synthesis. This project introduces a Coherent Guided Generative Model (CGGM) that not only predicts reaction temperatures but also captures the underlying physical distributions, separating common room-temperature reactions from cryogenic or reflux regimes. 

We compare three generative approaches:
1. **CGGM-VAE:** A Conditional Variational Autoencoder.
2. **CGGM-GAN:** A Conditional Generative Adversarial Network (WGAN-GP).
3. **CGGM-Hybrid:** Our proposed model combining the generative capacity of VAEs with the distributional matching of GANs.

## 📂 Repository Structure

To ensure maximum reproducibility and ease of use, this repository is organized into **self-contained subdirectories**. 

Instead of a heavily interdependent structure, each model's folder (e.g., `vae/`, `gan/`, `hybrid/`) contains all the necessary architectural files (`model_cggm_t.py`, `dataset_tokenized.py`, etc.) and loss constraint scripts. This design choice allows researchers to download and run specific experiments independently without dealing with complex path configurations (`PYTHONPATH`) or cross-dependencies.

```text
📦 CGGM-Temperature-Modeling
 ┣ 📂 data_processing/       # Scripts for ETL, normalization, and SMILES tokenization of the ORD dataset
 ┣ 📂 models/
 ┃ ┣ 📂 vae/                 # Self-contained CGGM-VAE implementation and training scripts
 ┃ ┣ 📂 gan/                 # Self-contained CGGM-GAN implementation and training scripts
 ┃ ┗ 📂 hybrid/              # Self-contained CGGM-Hybrid implementation and training scripts
 ┣ 📂 evaluation/            # Scripts for chemical coherence, ablation studies, and figure generation (Fig 3, etc.)
 ┗ 📜 README.md
```

## 📊 Data Availability
To keep this repository lightweight, the raw datasets (.csv), parsed Open Reaction Database (ORD) files, and training logs (.txt) are not included directly in this repository.

Please place the downloaded data in the corresponding data/ directories as expected by the training scripts before running them.

## 🚀 Getting Started
1. Requirements
Install the required dependencies:

Bash
pip install -r requirements.txt
(Core dependencies include PyTorch, RDKit, Pandas, Scikit-learn, and Seaborn).

2. Running an Experiment
Because the folders are self-contained, simply navigate to the model you wish to train and run the script. For example, to train the Hybrid model:

Bash
cd models/hybrid
python train_cggm_hybrid_simple.py --epochs 50 --batch_size 128
## 📎 Citation
If you use this code or the associated models in your research, please cite our work:

Fragmento de código
@article{cggm_temp_2026,
  title={A Coherent Guided Generative Model for Chemically Plausible Reaction Temperature Modeling},
  author={[Your Name / Authors]},
  journal={Artificial Intelligence Chemistry},
  year={2026},
  doi={[Insert DOI Here]}
}
