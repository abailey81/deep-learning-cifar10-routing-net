<div align="center">

# RoutingNet CIFAR-10

**Routed Multi-Expert CNN for Image Classification**

A dynamic attention-weighted expert routing architecture built with PyTorch, featuring mixup augmentation, cosine LR scheduling, and fully reproducible YAML-driven training.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[![CI](https://github.com/abailey81/deep-learning-cifar10-routing-net/actions/workflows/lint-and-test.yml/badge.svg)](https://github.com/abailey81/deep-learning-cifar10-routing-net/actions/workflows/lint-and-test.yml)
[![GitHub stars](https://img.shields.io/github/stars/abailey81/deep-learning-cifar10-routing-net?style=social)](https://github.com/abailey81/deep-learning-cifar10-routing-net/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/abailey81/deep-learning-cifar10-routing-net?style=social)](https://github.com/abailey81/deep-learning-cifar10-routing-net/network/members)

</div>

---

## Highlights

<table>
<tr>
<td align="center" width="25%">
<br>
<strong>Architecture Innovation</strong>
<br><br>
Multi-expert routing with learned softmax attention weights &#8212; parallel convolutional experts dynamically combined per input.
<br><br>
</td>
<td align="center" width="25%">
<br>
<strong>Training Pipeline</strong>
<br><br>
Mixup regularization (first 40% of epochs), cosine annealing LR, SGD with momentum, and Kaiming initialization.
<br><br>
</td>
<td align="center" width="25%">
<br>
<strong>Reproducibility</strong>
<br><br>
Deterministic seeding, YAML-driven configuration, single-command training, and pinned dependencies.
<br><br>
</td>
<td align="center" width="25%">
<br>
<strong>Documentation</strong>
<br><br>
Interactive Jupyter notebook with visualizations, technical PDF report, and comprehensive code comments.
<br><br>
</td>
</tr>
</table>

---

## Architecture

RoutingNet introduces **multi-expert attention** into a CNN backbone. Multiple convolutional experts process input feature maps in parallel, and a learned router dynamically weights their contributions using global-average-pooled features passed through a lightweight MLP with softmax gating.

```
                         Input (3 x 32 x 32)
                                |
                          +----------+
                          |   Stem   |    Conv2d(3 -> 48) + BN + ReLU
                          +----------+
                                |
                         (48 x 32 x 32)
                                |
                 +--------------+--------------+
                 |                             |
          +------------+                +------------+
          |  Expert 0  |                |  Expert 1  |    Conv2d(48 -> 48) + BN + ReLU
          +------------+                +------------+
                 |                             |
                 +-------+       +-------------+
                         |       |
                    +----------+
                    |  Router  |    GAP -> FC(48->12) -> ReLU -> FC(12->2) -> Softmax
                    +----------+
                         |
                  Weighted Sum + BN + Dropout(0.1)
                         |
                 +---------------+
                 |     Head      |    GAP -> Flatten -> Dropout(0.2) -> Linear(48 -> 10)
                 +---------------+
                         |
                   Logits (10)
```

### Model Details

| Component | Configuration |
|-----------|--------------|
| **Stem** | Conv2d(3 &#8594; 48, 3x3, pad=1) + BatchNorm2d + ReLU |
| **Experts** | k=2 parallel Conv2d(48 &#8594; 48, 3x3, pad=1) + BatchNorm2d + ReLU |
| **Router** | AdaptiveAvgPool2d &#8594; FC(48 &#8594; 12) &#8594; ReLU &#8594; FC(12 &#8594; 2) &#8594; Softmax |
| **Router Norm** | BatchNorm2d(48) + Dropout(0.1) |
| **Head** | AdaptiveAvgPool2d &#8594; Flatten &#8594; Dropout(0.2) &#8594; Linear(48 &#8594; 10) |
| **Initialization** | Kaiming Normal (all Conv2d and Linear layers) |

---

## Training Configuration

All hyperparameters are managed through [`configs/training_config.yaml`](configs/training_config.yaml):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `seed` | 42 | Deterministic reproducibility |
| `batch_size` | 256 | Training batch size |
| `epochs` | 40 | Total training epochs |
| `optimizer` | SGD | With momentum=0.9, weight_decay=5e-4 |
| `learning_rate` | 0.1 | Initial learning rate |
| `scheduler` | Cosine Annealing | Smooth LR decay to zero |
| `mixup_alpha` | 0.1 | Beta distribution parameter (first 40% of epochs) |
| `augmentation` | RandomCrop + HFlip | Standard CIFAR-10 augmentations |

### Key Techniques

- **Mixup Augmentation** -- Convex interpolation of training pairs during the first 40% of epochs provides regularization and smoother decision boundaries.
- **Cosine Annealing LR** -- Learning rate decays smoothly following a cosine curve, eliminating the need for manual milestone tuning.
- **Kaiming Initialization** -- All convolutional and linear weights initialized with `kaiming_normal_` for stable ReLU-network training.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/abailey81/deep-learning-cifar10-routing-net.git
cd deep-learning-cifar10-routing-net

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python -m src.training.train --config configs/training_config.yaml

# Evaluate on CIFAR-10 test set
python -m src.training.evaluate
```

> **Note:** Training automatically downloads CIFAR-10 on first run. A GPU is recommended but not required -- the code falls back to CPU via `device: auto`.

---

## Project Structure

```
deep-learning-cifar10-routing-net/
├── .github/
│   ├── workflows/
│   │   └── lint-and-test.yml        # CI: smoke test (model forward pass)
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml           # Bug report template
│   │   ├── feature_request.yml      # Feature request template
│   │   └── config.yml               # Issue template config
│   └── PULL_REQUEST_TEMPLATE.md     # PR template
├── configs/
│   └── training_config.yaml         # All hyperparameters (YAML)
├── docs/
│   └── report.pdf                   # Technical report
├── notebooks/
│   └── cifar10_routing_net.ipynb    # Full training pipeline + visualizations
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── routing_net.py           # RoutingNet architecture
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                 # Training loop with mixup
│   │   └── evaluate.py              # Test-set evaluation
│   └── utils/
│       ├── __init__.py
│       └── my_utils.py              # Shared utilities
├── .gitignore
├── CONTRIBUTING.md                  # Contribution guidelines
├── LICENSE                          # MIT License
├── README.md
├── SECURITY.md                      # Security policy
└── requirements.txt                 # Python dependencies
```

---

<details>
<summary><strong>Detailed Architecture Explanation</strong></summary>

### How Expert Routing Works

Traditional CNNs apply a single convolutional pathway to every input. RoutingNet instead maintains **k=2 parallel expert pathways**, each a full Conv-BN-ReLU block operating on the same feature map. The key innovation is the **router module**, which decides how much each expert should contribute to the final representation.

**Step 1 -- Feature Extraction (Stem)**
The stem converts the raw 3-channel RGB input into a 48-channel feature map using a single convolutional layer with batch normalization and ReLU activation.

**Step 2 -- Parallel Expert Processing**
Each expert independently processes the 48-channel feature map through its own Conv-BN-ReLU block. Because experts have separate learned weights, they can specialize in detecting different feature patterns.

**Step 3 -- Attention-Weighted Routing**
The router computes input-dependent attention weights:
1. Global Average Pooling compresses the spatial dimensions to a 48-d vector.
2. A two-layer MLP (48 &#8594; 12 &#8594; 2) produces raw routing logits.
3. Softmax normalizes these logits into attention weights that sum to 1.
4. Expert outputs are combined via weighted sum using these attention weights.

This mechanism allows the network to **dynamically allocate capacity** -- for some inputs, Expert 0 may dominate; for others, Expert 1 contributes more. The routing weights are fully differentiable, so they are learned end-to-end via backpropagation.

**Step 4 -- Classification Head**
The routed feature map passes through adaptive average pooling, dropout (p=0.2), and a linear classifier producing 10-class logits.

### Why This Matters

- **Conditional Computation**: Different inputs activate different expert combinations, increasing effective model capacity without proportionally increasing inference cost.
- **Smooth Gating**: Unlike hard routing (e.g., Mixture of Experts with top-k selection), softmax gating ensures all experts receive gradients, leading to stable training.
- **Minimal Overhead**: The router MLP is lightweight (48 &#8594; 12 &#8594; 2 = 626 parameters), adding negligible computation to the forward pass.

</details>

---

<div align="center">

**[Technical Report](docs/report.pdf)** &#183; **[Training Notebook](notebooks/cifar10_routing_net.ipynb)** &#183; **[Configuration](configs/training_config.yaml)**

MIT License &#169; 2025 Tamer Atesyakar

</div>
