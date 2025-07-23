# PARDiff: An Order-Agnostic Autoregressive Diffusion Model for Graph Generation

Official PyTorch implementation of:

> **PARDiff: An Order-Agnostic Autoregressive Diffusion Model for Graph Generation**  
> Usman Ahmad Usmani, Arunava Roy, Junzo Watada  
>

---

## 🧠 About

**PARDiff** introduces a hybrid graph generation framework that combines the benefits of **autoregressive modeling** and **discrete diffusion** in a unified and **order-agnostic** way. Unlike prior approaches that rely on handcrafted node orderings or symmetry-breaking features, PARDiff models graphs **block-wise**, capturing local dependencies via a shared diffusion process and maintaining global coherence through a structural autoregressive factorization.

This repository contains the **official implementation** by the first author and primary contributor of the paper.

### 🔍 Highlights

- 🚀 Combines autoregressive (AR) and diffusion models for scalable and controllable graph generation
- 🔁 Maintains permutation-invariance through structural block decomposition
- 🧠 Uses a higher-order graph transformer with GPT-style training for parallelization
- 🧪 Outperforms state-of-the-art baselines on molecular (QM9, ZINC250K, MOSES) and structural datasets

