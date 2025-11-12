import torch, torch.nn as nn, torch.nn.functional as F

class Stem(nn.Module):
    def __init__(self, out_ch=48):
        super().__init__()
        self.conv = nn.Conv2d(3, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class Expert(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class Router(nn.Module):
    def __init__(self, ch, k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(ch) for _ in range(k)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        hid = max(8, ch // 4)
        self.fc1 = nn.Linear(ch, hid)
        self.fc2 = nn.Linear(hid, k)
        self.norm = nn.BatchNorm2d(ch)
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        feats = torch.stack([e(x) for e in self.experts], dim=0)
        g = self.pool(x).flatten(1)
        w = torch.softmax(self.fc2(torch.relu(self.fc1(g))), dim=-1)
        w = w.transpose(0,1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        z = (w * feats).sum(dim=0)
        return self.drop(self.norm(z))

class RoutingNet(nn.Module):
    def __init__(self, ch=48, num_classes=10, p_drop=0.2):
        super().__init__()
        self.stem = Stem(ch)
        self.router = Router(ch)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(p_drop), nn.Linear(ch, num_classes)
        )
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    def forward(self, x):
        x = self.stem(x)
        x = self.router(x)
        return self.head(x)
