PARDiff: An Order-Agnostic Autoregressive Diffusion Model for Graph Generation
This repository provides the official PyTorch implementation of:

PARDiff: An Order-Agnostic Autoregressive Diffusion Model for Graph Generation
Usman Ahmad Usmani, Arunava Roy, Junzo Watada

🧠 Overview
PARDiff is a novel hybrid graph generative framework that integrates the strengths of autoregressive modeling and discrete diffusion processes in a unified and order-agnostic manner.

Traditional autoregressive models often depend on fixed or handcrafted node orderings, which introduces sensitivity to permutation and limits scalability. In contrast, PARDiff leverages a block-wise structural decomposition that models local dependencies via discrete diffusion while enforcing global coherence through autoregressive factorization. This structure ensures the model remains permutation-invariant, scalable to large graphs, and interpretable in its generation process.

✨ Key Contributions
Hybrid Generative Framework
Combines autoregressive likelihood decomposition with a shared discrete diffusion process at the block level.

Permutation Invariance
Achieves full invariance to node reordering through permutation-consistent node ranking and block-wise masking, avoiding symmetry-breaking heuristics.

Block-wise Parallelism
Structures graphs into sequential blocks while enabling parallel denoising of each block, offering both scalability and efficiency.

Transformer-Based Backbone
Utilizes a higher-order graph transformer trained in a GPT-style causal manner, supporting flexible conditioning and expressive representation learning.

State-of-the-Art Results
Demonstrates superior performance on benchmark datasets including QM9, ZINC250K, and MOSES, achieving high sample quality and diversity compared to leading graph generative models.

📂 Repository Structure
bash
Copy
Edit
PARDiff/
├── models/       # Transformer and denoising networks
├── diffusion/    # Forward and reverse discrete diffusion processes
├── blocks/       # Block-wise graph decomposition and masking logic
├── training/     # Unified training and generation routines
├── datasets/     # Preprocessing and loaders for QM9, ZINC, and MOSES
├── utils/        # Logging, evaluation, and miscellaneous utilities
├── main.py       # Entry point for training and generation
└── config.yaml   # Configuration file for experimental settings
📦 Installation
bash
Copy
Edit
git clone https://github.com/yourusername/PARDiff.git
cd PARDiff
pip install -r requirements.txt
Dependencies
PyTorch ≥ 1.12

PyTorch Geometric

RDKit

NetworkX

numpy, tqdm, matplotlib

🚀 Getting Started
To train the model:

bash
Copy
Edit
python main.py --config config.yaml
To generate new graphs from a trained checkpoint:

bash
Copy
Edit
python main.py --generate --checkpoint path/to/model.ckpt
📊 Supported Datasets
PARDiff supports standard molecular and structural graph datasets:

QM9: Molecular graphs with 13 atom types

ZINC250K: Drug-like molecule graphs

MOSES: Scaffold-diverse chemical structures

Dataset preparation and preprocessing scripts are available under the datasets/ directory.

