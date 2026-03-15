# Contributing to RoutingNet CIFAR-10

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Getting Started

1. **Fork** the repository and clone your fork locally.
2. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Create a new branch for your feature or fix:

```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Configuration

All training hyperparameters are managed through `configs/training_config.yaml`. When proposing changes to training behavior, modify the YAML configuration rather than hardcoding values in Python.

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Keep functions focused and well-documented.
- Use type hints where practical.

### Reproducibility

This project prioritizes deterministic, reproducible results:

- All random seeds are controlled via the `seed` field in the YAML config.
- CuDNN deterministic mode is enabled during training.
- When adding new stochastic components, ensure they respect the global seed.

### Testing

Before submitting a pull request, verify that the model builds and runs correctly:

```bash
python -c "
from src.models.routing_net import RoutingNet
import torch
m = RoutingNet()
x = torch.randn(2, 3, 32, 32)
y = m(x)
assert y.shape == (2, 10)
print('Smoke test passed')
"
```

The CI pipeline runs this smoke test automatically on every push and pull request.

### Project Structure

```
src/
  models/routing_net.py      # Model architecture (Stem, Expert, Router, RoutingNet)
  training/train.py          # Training loop with mixup augmentation
  training/evaluate.py       # Test-set evaluation
  utils/my_utils.py          # Shared utilities
configs/
  training_config.yaml       # All hyperparameters
notebooks/
  cifar10_routing_net.ipynb  # Interactive training notebook
```

## Submitting Changes

1. Ensure your code passes the smoke test and any additional tests you have written.
2. Write a clear commit message describing your changes.
3. Open a pull request against the `main` branch.
4. Fill out the pull request template with a description of your changes, motivation, and testing steps.

## Types of Contributions

We welcome contributions in these areas:

- **Architecture improvements**: New expert types, routing mechanisms, or head designs.
- **Training enhancements**: Additional augmentation strategies, schedulers, or optimizers.
- **Evaluation tools**: Per-class metrics, confusion matrices, visualization utilities.
- **Documentation**: Improved docstrings, tutorials, or explanations.
- **Bug fixes**: Correctness issues, edge cases, or compatibility fixes.

## Reporting Issues

Use the [issue templates](https://github.com/abailey81/deep-learning-cifar10-routing-net/issues/new/choose) provided:

- **Bug Report**: For unexpected behavior or errors.
- **Feature Request**: For new capabilities or improvements.

## Code of Conduct

Be respectful, constructive, and inclusive in all interactions. We are committed to providing a welcoming environment for everyone.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
