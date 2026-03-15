"""RoutingNet architecture for CIFAR-10 image classification.

A dynamic attention-weighted expert routing CNN where multiple convolutional
experts process features in parallel, and a learned router dynamically weights
their contributions using softmax gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stem(nn.Module):
    """Initial convolutional stem that extracts low-level features.

    Converts 3-channel RGB input to a higher-dimensional feature map
    using a single Conv-BN-ReLU block.
    """

    def __init__(self, out_ch: int = 48) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class Expert(nn.Module):
    """Single convolutional expert in the mixture-of-experts routing block.

    Each expert independently learns to detect different feature patterns
    through its own Conv-BN-ReLU pathway.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class Router(nn.Module):
    """Attention-weighted expert routing module.

    Maintains k parallel expert pathways and dynamically combines their
    outputs using input-dependent softmax attention weights computed by
    a lightweight MLP gating network.
    """

    def __init__(self, ch: int, k: int = 2) -> None:
        super().__init__()
        self.experts = nn.ModuleList([Expert(ch) for _ in range(k)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        hid = max(8, ch // 4)
        self.fc1 = nn.Linear(ch, hid)
        self.fc2 = nn.Linear(hid, k)
        self.norm = nn.BatchNorm2d(ch)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.stack([e(x) for e in self.experts], dim=0)
        g = self.pool(x).flatten(1)
        w = torch.softmax(self.fc2(torch.relu(self.fc1(g))), dim=-1)
        w = w.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        z = (w * feats).sum(dim=0)
        return self.drop(self.norm(z))


class RoutingNet(nn.Module):
    """Routed multi-expert CNN for CIFAR-10 classification.

    Architecture: Stem -> Router (k experts + attention gating) -> Head.
    All Conv2d and Linear layers use Kaiming normal initialization.

    Args:
        ch: Channel dimension for stem and expert layers.
        num_classes: Number of output classes.
        p_drop: Dropout probability before the final linear layer.
    """

    def __init__(self, ch: int = 48, num_classes: int = 10, p_drop: float = 0.2) -> None:
        super().__init__()
        self.stem = Stem(ch)
        self.router = Router(ch)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(p_drop), nn.Linear(ch, num_classes)
        )
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.router(x)
        return self.head(x)
