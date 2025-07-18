# SurpriseNet

Draft – will be refined before publication

SurpriseNet is an experimental framework for studying **internal network dynamics** when deep neural networks encounter **surprising or unexpected events** in naturalistic video streams.  The codebase combines multiple spatial auto-encoders with a temporal LSTM head and instruments the model so that per-layer weights and activations can be logged whenever the network detects a surprise.

## Purpose and White-Paper Scope
This repository supports a white-paper proposing new research directions in computational neuroscience and machine learning:

* Quantitatively characterising how internal representations shift when a model is surprised.
* Relating those shifts to hypotheses about predictive processing in biological brains.
* Providing open, reproducible tooling that anyone can extend with alternative architectures or stimulus modalities.

## Model Architecture
```
Video ➜ Frame-wise Encoder (Conv AE x N) ➜ Latent sequences ➜ LSTM ➜ Surprise detector
```
Key components:
1. **Frame Auto-Encoders** – N independent convolutional auto-encoders trained for high-fidelity reconstruction.
2. **Temporal LSTM** – Processes latent sequences, predicting the next latent and estimating prediction error.
3. **Surprise Trigger** – When reconstruction / prediction error exceeds an adaptive threshold, internal weights and activations are snap-shotted for subsequent analysis.

## Repository Layout (initial)
```
whitenet/
├── data/                # DO NOT commit – raw & processed datasets (git-ignored)
├── models/              # Network definitions & checkpoints
├── notebooks/           # Exploratory analysis / prototyping
├── scripts/             # Training, evaluation & instrumentation scripts
└── README.md            # This file
```

## Quick Start
1. Clone the repo (after this draft is pushed).
2. Create a virtual environment with Python ≥3.9.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download or generate a video dataset and place it under `data/raw/` (ignored by git).
5. Run the preprocessing & training pipeline:
   ```bash
   python scripts/preprocess.py
   python scripts/train.py
   ```

## Acknowledgements
The project is inspired by research on predictive coding, brain imaging analysis of surprise, and recent work on self-supervised video representation learning.

---
*This README is a draft. Feel free to edit, expand, or reshape before the first public release.*
