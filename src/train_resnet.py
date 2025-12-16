import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.slice_dataset import MRISliceDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.backbones import build_resnet18, build_efficientnet_b0, adapt_first_conv_to_grayscale

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_acc, total_loss, n = 0.0, 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += (logits.argmax(1) == y).float().sum().item()
        n += bs
    return total_loss / n, total_acc / n

def train(
    splits_dir="dataset/OASIS1/splits",
    slices_root="dataset/OASIS1/processed_slices",
    backbone="resnet18",  # or "effb0"
    img_size=224,
    batch_size=32,
    lr=1e-4,
    epochs=10,
    patience=3,
    out_dir="artifacts"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    train_ds = MRISliceDataset(os.path.join(splits_dir, "train.csv"), slices_root, transform=get_train_transforms(img_size))
    val_ds   = MRISliceDataset(os.path.join(splits_dir, "val.csv"),   slices_root, transform=get_eval_transforms(img_size))
    test_ds  = MRISliceDataset(os.path.join(splits_dir, "test.csv"),  slices_root, transform=get_eval_transforms(img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if backbone == "resnet18":
        model = build_resnet18(num_classes=2, pretrained=True)
        model = adapt_first_conv_to_grayscale(model)
    else:
        model = build_efficientnet_b0(num_classes=2, pretrained=True)
        # EfficientNet expects 3ch; easiest: replicate channel in dataset OR keep 3ch.
        # If you want true 1ch for effnet, we can adapt similarly, but keep resnet first.
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    bad = 0
    best_path = os.path.join(out_dir, f"best_{backbone}.pt")

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for x, y, _ in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        val_loss, val_acc = eval_one_epoch(model, val_loader, device)
        print(f"Epoch {ep} - TrainLoss~{running_loss/len(train_loader):.4f} - ValLoss {val_loss:.4f} - ValAcc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad = 0
            torch.save({"model": model.state_dict(), "backbone": backbone, "img_size": img_size}, best_path)
            print(f"[INFO] New best saved: {best_path} (val_acc={best_val_acc:.4f})")
        else:
            bad += 1
            if bad >= patience:
                print("[EARLY STOP] No improvement.")
                break

    # Test best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = eval_one_epoch(model, test_loader, device)
    print(f"[TEST] Loss {test_loss:.4f} Acc {test_acc:.4f}")

if __name__ == "__main__":
    train()
