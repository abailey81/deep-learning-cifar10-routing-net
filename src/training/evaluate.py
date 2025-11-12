import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models.routing_net import RoutingNet

def evaluate(weights="outputs/routingnet_cifar10.pt"):
    tf = transforms.ToTensor()
    ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    model = RoutingNet().cuda() if torch.cuda.is_available() else RoutingNet()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    device = next(model.parameters()).device
    model.eval(); total=0; correct=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); total += x.size(0)
    acc = correct/total
    print(f"Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    evaluate()
