import argparse, os, random, yaml, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.routing_net import RoutingNet

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def get_device(cfg):
    if cfg.get("device","auto")=="auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])

def mixup_data(x, y, alpha):
    if alpha <= 0: return x, y, None
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam

def mixup_criterion(criterion, pred, targets):
    y_a, y_b, lam = targets
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def build_dataloaders(cfg):
    aug = []
    if cfg["data"]["aug"].get("random_crop", True):
        aug += [transforms.RandomCrop(32, padding=4)]
    if cfg["data"]["aug"].get("random_flip", True):
        aug += [transforms.RandomHorizontalFlip()]
    aug += [transforms.ToTensor()]
    train_tf = transforms.Compose(aug)
    test_tf  = transforms.Compose([transforms.ToTensor()])
    root = cfg["data"]["root"]
    train_ds = datasets.CIFAR10(root=root, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)
    bs = cfg["data"]["batch_size"]; nw = cfg["data"]["num_workers"]
    return (DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True),
            DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True))

def build_model(cfg):
    ch = cfg["model"]["channels"]; num_classes = cfg["model"]["num_classes"]
    return RoutingNet(ch=ch, num_classes=num_classes)

def build_opt_sched(model, cfg, steps_per_epoch):
    opt_cfg = cfg["train"]["optimizer"]; sch_cfg = cfg["train"]["scheduler"]
    if opt_cfg["name"].lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=opt_cfg["lr"], momentum=opt_cfg["momentum"], weight_decay=opt_cfg["weight_decay"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    if sch_cfg["name"].lower() == "cosine":
        total_steps = cfg["train"]["epochs"] * steps_per_epoch
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    else:
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(0.6*cfg["train"]["epochs"]), int(0.85*cfg["train"]["epochs"])], gamma=0.1)
    return opt, sch

def evaluate(model, loader, device, criterion):
    model.eval(); total = 0; correct = 0; loss_sum = 0.0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item()*x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred==y).sum().item(); total += x.size(0)
    return loss_sum/total, correct/total

def train(cfg):
    set_seed(cfg["seed"]); device = get_device(cfg)
    train_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    steps_per_epoch = math.ceil(len(train_loader.dataset)/cfg["data"]["batch_size"])
    opt, sch = build_opt_sched(model, cfg, steps_per_epoch)
    criterion = nn.CrossEntropyLoss()
    mix_alpha = float(cfg["data"]["aug"].get("mixup_alpha", 0.0))

    step = 0
    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        for i,(x,y) in enumerate(train_loader, start=1):
            x,y = x.to(device), y.to(device)
            if mix_alpha>0 and epoch<=int(0.4*cfg["train"]["epochs"]):
                x,(y_a,y_b),lam = mixup_data(x,y,mix_alpha)
                logits = model(x); loss = mixup_criterion(criterion, logits, (y_a,y_b,lam))
            else:
                logits = model(x); loss = criterion(logits,y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            step += 1
            if i % cfg["log"]["interval"] == 0:
                print(f"epoch {epoch:02d} step {i:04d}/{steps_per_epoch} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.5f}")
        val_loss, val_acc = evaluate(model, test_loader, device, criterion)
        print(f"[val] epoch {epoch:02d}: loss {val_loss:.4f} acc {val_acc:.4f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/routingnet_cifar10.pt")
    print("Saved weights to outputs/routingnet_cifar10.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training_config.yaml")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
