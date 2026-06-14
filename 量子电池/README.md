# Quantum Battery Efficiency Prediction via LSTM

Predicting energy evolution and efficiency of quantum batteries coupled to structured (non-Markovian) reservoirs using LSTM neural networks.

## Overview

This project implements a full pipeline:
1. **Physics simulation** — Generate time-series density matrix data via QuTiP HEOM solver for N-TLS quantum batteries in Lorentzian-structured reservoirs.
2. **LSTM model** — Train a PyTorch LSTM to predict future energy states from past evolution windows.
3. **Physics-informed constraints** — Loss function enforces trace conservation, hermiticity, and positivity.
4. **Three experiments** — Non-Markovian boosting, storage lifetime prediction, and cross-parameter generalization.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate training data
```bash
python src/data/generate.py
```

### 2. Train the LSTM model
```bash
python src/training/train.py
```

### 3. Run experiments
```bash
python experiments/experiment_a_feedback.py
python experiments/experiment_b_shelflife.py
python experiments/experiment_c_extrapolation.py
```

## Configuration

Edit `config.yaml` to adjust physics parameters, model architecture, and training hyperparameters.

## Project Structure

```
mlqb/
├── config.yaml              # Central configuration
├── src/
│   ├── physics/             # Hamiltonian, spectral density, HEOM dynamics
│   ├── data/                # Data generation pipeline & PyTorch Dataset
│   ├── model/               # LSTM model & physics-informed loss
│   ├── training/            # Training loop & evaluation metrics
│   └── visualization/       # Plotting utilities
├── experiments/             # Three core research experiments
└── notebooks/               # Jupyter demo notebook
```
