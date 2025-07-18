# SurpriseNet

SurpriseNet is an experimental framework for studying **internal network dynamics** when deep neural networks encounter **surprising or unexpected events** in naturalistic video streams.  The codebase combines multiple spatial auto-encoders with a temporal LSTM head and instruments the model so that per-layer weights and activations can be logged whenever the network detects a surprise. The project is inspired by research on predictive coding, and a desire to perform brain imaging analysis of surprise using neural networks, and my existing work on EEG video analysis. This project is an extention of the work I performed at Vanderbilt University and a proposal for future research.

Please see the white-paper for my exploratory analysis and explanations of the network design.
[📄 View the Whitepaper](docs/Surprise_Whitepaper.pdf)

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

## Repository Layout
```
whitenet/
├── encoder_models/        # Trained frame-encoder checkpoints (git-ignored)
├── LSTM_models/           # Trained LSTM checkpoints (git-ignored)
├── preprocessed_data/     # Intermediate tensors (git-ignored)
├── Videos/                # Source video clips (git-ignored)
├── figures/               # Figures used in the white-paper (git-ignored)
├── image_reconstructions/ # Reconstruction outputs & visualisations (git-ignored)
├── *.py                   # Training / inference / visualisation scripts
├── README.md              # Project overview (this file)
├── .gitignore             # Ignore patterns for data & checkpoints
└── requirements.txt       # Python dependencies
```

## Quick Start  
(GPU strongly recommended)

1. Create and activate a Python ≥3.9 virtual environment then install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Place four baseline videos in `Videos/` with the exact filenames
   ```text
   Videos/
   ├── City_Video.mp4
   ├── Forest_Video.mp4
   ├── Ocean_Video.mp4
   └── Stock_Video.mp4
   ```
   (Any additional clips are fine – update the lists in the dataset modules if you add more.)
3. Compute dataset normalisation statistics (creates `preprocessed_data/encoder_mean_std.pth`):
   ```bash
   python VideoData.py
   ```
4. Train convolutional auto-encoders (checkpoints saved to `encoder_models/`):
   ```bash
   python AutoEncoder.py
   ```
5. Train the temporal LSTM classifier (checkpoints saved to `LSTM_models/`):
   ```bash
   python EncoderLSTM.py
   ```
6. Run inference & visualisation:
   ```bash
   python Inference.py           # interactive plots
   # or
   python VideoInference.py      # saves probabilities & activations
   ```

Large files (videos, checkpoints, tensors) are excluded from version control via `.gitignore`.