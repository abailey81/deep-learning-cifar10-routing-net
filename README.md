# CIFAR-10 RoutingNet (PyTorch)

A **modular, production-style deep-learning project** implementing a *Routed Convolutional Neural Network* for the **CIFAR-10** image classification dataset.  
This repository demonstrates best practices in model architecture design, training reproducibility, and software-engineering structure using **PyTorch**.

---

##  Overview

RoutingNet introduces **multi-expert attention** into a CNN backbone: several convolutional "experts" process the same input in parallel, and a *router module* dynamically weights their contributions using global-average features.  
This enables adaptive feature routing — enhancing generalisation without increasing computational cost dramatically.

The project was built for a graduate-level deep-learning module but refactored to meet **professional open-source standards**:  
modular code, versioned configs, CI/CD, and technical documentation.

---

##  Key Features

| Area | Description |
|------|--------------|
| **Architecture** | Custom CNN with routed multi-expert attention |
| **Training** | SGD + momentum, cosine learning-rate schedule, optional label smoothing |
| **Data** | CIFAR-10 with augmentation and early-phase *mixup* regularisation |
| **Reproducibility** | Deterministic seeding, YAML configs, single-command training |
| **Extensibility** | Plug-and-play modules for experiments and model variants |
| **Reporting** | Structured notebook and technical report in `/docs` |

---

##  Repository Structure

## Repository Layout
deep-learning-cifar10-routing-net/
├── src/                     model, training, utils
├── notebooks/               cifar10_routing_net.ipynb
├── configs/                 training_config.yaml
├── docs/                    report.pdf
├── requirements.txt         dependencies
└── README.md

This layout mirrors a **real MLOps-ready repo**, enabling easy integration with experiment tracking tools (e.g., MLflow, Weights & Biases) and CI workflows.

---

##  Quickstart

```bash
# 1️ Clone
git clone https://github.com/abailey81/deep-learning-cifar10-routing-net.git
cd deep-learning-cifar10-routing-net

# 2️ Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3️ Train
python -m src.training.train --config configs/training_config.yaml

# 4️ Evaluate
python -m src.training.evaluate
License: MIT
