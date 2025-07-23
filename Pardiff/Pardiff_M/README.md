PARDiff: An Order-Agnostic Autoregressive Diffusion Model for Graph Generation
This repository provides the official PyTorch implementation of:

PARDiff: An Order-Agnostic Autoregressive Diffusion Model for Graph Generation
Usman Ahmad Usmani, Arunava Roy, Junzo Watada

🧠 Overview
PARDiff is a novel hybrid framework for graph generation that synergizes the strengths of autoregressive models and discrete diffusion processes, while remaining invariant to node permutations.

Unlike traditional autoregressive models that rely on fixed or handcrafted node orderings, PARDiff leverages a block-wise structural decomposition that preserves local dependency modeling via diffusion and enables global generative control through a structural autoregressive factorization. This design ensures that the model remains order-agnostic, scalable, and interpretable.

✨ Key Contributions
Hybrid Generative Framework: Merges autoregressive likelihood factorization with a shared discrete diffusion process at the block level.

Permutation Invariance: Maintains invariance through a permutation-consistent node ordering and block-wise masking strategies, eliminating the need for symmetry-breaking heuristics.

Block-wise Parallelism: Achieves high efficiency by structuring the graph into blocks and generating them sequentially while modeling each block in parallel.

Transformer-Based Backbone: Utilizes a higher-order graph transformer trained in a GPT-style causal manner, enabling flexible conditioning and generalization.

State-of-the-Art Results: Demonstrates strong performance on diverse benchmarks including QM9, ZINC250K, and MOSES, achieving superior generation quality and diversity compared to prior approaches.

📂 Repository Structure
bash
Copy
Edit
PARDiff/
│
├── models/                # Transformer and denoising models
├── diffusion/             # Forward and reverse discrete diffusion processes
├── blocks/                # Block-wise graph decomposition and masking
├── training/              # Unified training and generation routines
├── datasets/              # Preprocessing and loaders for QM9, ZINC, MOSES
├── utils/                 # Auxiliary tools and logging
├── main.py                # Entry point for training and evaluation
└── config.yaml            # Configuration file for experiments
📦 Installation
bash
Copy
Edit
git clone https://github.com/yourusername/PARDiff.git
cd PARDiff
pip install -r requirements.txt
Requirements include:

PyTorch ≥ 1.12

NetworkX

RDKit

PyTorch Geometric

tqdm, numpy, matplotlib

🚀 Getting Started
To train the model:

bash
Copy
Edit
python main.py --config config.yaml
To generate new graphs:

bash
Copy
Edit
python main.py --generate --checkpoint path/to/model.ckpt
📊 Datasets
We support multiple standard graph datasets:

QM9 (molecular graphs with 13 atom types)

ZINC250K (drug-like molecules)

MOSES (scaffold diversity benchmarking)

Preprocessing scripts are available in the datasets/ folder.
