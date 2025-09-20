# src/train.py
import torch, json
from torch import nn
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict

from .config import CFG
from .data import make_loaders
from .model import build_model
from .metrics import evaluate, expected_calibration_error
from .utils import seed_everything, RACE_IDX2STR, GENDER_IDX2STR

def compute_class_weights(train_loader, num_classes=9, device="cpu"):
    counts = Counter()
    for _, y, _, _ in train_loader:
        for yi in y.tolist(): counts[int(yi)] += 1
    total = sum(counts.values())
    weights = torch.ones(num_classes, device=device)
    for c in range(num_classes):
        freq = counts[c]/total if total>0 else 1.0
        weights[c] = 1.0/max(freq, 1e-6)
    weights = weights/weights.mean()
    return weights

def compute_group_weights(train_loader, device="cpu"):
    counts = Counter()
    for _, y, r, g in train_loader:
        for ri,gi in zip(r.tolist(), g.tolist()):
            counts[(ri,gi)] += 1
    total = sum(counts.values())
    gw = {k: 1.0 / max(v/total, 1e-6) for k,v in counts.items()}
    # normalize
    mean_w = sum(gw.values())/len(gw)
    gw = {k: v/mean_w for k,v in gw.items()}
    # convert to tensor map resolved at batch time
    return gw

def groupdro_loss_per_batch(losses, group_ids, temperature=5.0):
    # Smooth approximation to max over groups via LogSumExp
    # losses: tensor [B], group_ids: list of tuples (r,g) length B
    device = losses.device
    # aggregate mean loss per group in batch
    group_to_losses = defaultdict(list)
    for li, gid in zip(losses.tolist(), group_ids):
        group_to_losses[gid].append(li)
    group_means = torch.tensor([sum(v)/len(v) for v in group_to_losses.values()], device=device)
    lse = torch.logsumexp(temperature*group_means, dim=0) / temperature
    return lse

def train(cfg: CFG):
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(cfg)
    model = build_model(cfg.model_name, num_classes=9).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss(reduction="none")  # we'll weight it

    class_weights = compute_class_weights(train_loader, num_classes=9, device=device) if cfg.use_reweighting else None
    group_weights = compute_group_weights(train_loader, device=device) if cfg.use_reweighting else None

    best_val = -1; patience = 0
    for epoch in range(1, cfg.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for x, y, r, g in pbar:
            x=x.to(device); y=y.to(device)
            logits = model(x)
            per_ex_loss = ce(logits, y)  # [B]

            # --- Reweighting (class + group) ---
            if cfg.use_reweighting:
                cw = class_weights[y]  # class weights per example
                gw = []
                for ri, gi in zip(r.tolist(), g.tolist()):
                    gw.append(group_weights[(int(ri), int(gi))])
                gw = torch.tensor(gw, device=device, dtype=cw.dtype)
                weights = cw * gw
                loss = (per_ex_loss * weights).mean()
            # --- Group DRO ---
            elif cfg.use_groupdro:
                group_ids = [(int(ri), int(gi)) for ri,gi in zip(r.tolist(), g.tolist())]
                loss = groupdro_loss_per_batch(per_ex_loss.detach(), group_ids, temperature=cfg.dro_temperature)
                # Add a supervised term to keep it stable
                loss = 0.5*loss + 0.5*per_ex_loss.mean()
            else:
                loss = per_ex_loss.mean()

            opt.zero_grad(); loss.backward(); opt.step()

        # ---- validation
        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics["overall_acc"]
        print(f"Val acc: {val_acc:.4f} | Worst-group acc: {val_metrics['per_group_acc'][val_metrics['worst_group']]:.4f} | Δgap: {val_metrics['acc_gap']:.3f}")

        if val_acc > best_val:
            best_val = val_acc; patience = 0
            torch.save(model.state_dict(), cfg.models_dir / "model_groupdro.pt")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("Early stopping."); break

    # ---- test & save report
    model.load_state_dict(torch.load(cfg.models_dir / "model_groupdro.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    ece = expected_calibration_error(test_metrics["y_all"], test_metrics["p_all"], test_metrics["conf_all"], n_bins=15)
    report = {
        "overall_acc": test_metrics["overall_acc"],
        "worst_group": test_metrics["worst_group"],
        "acc_gap": test_metrics["acc_gap"],
        "per_group_acc": test_metrics["per_group_acc"],
        "ece": ece,
    }
    with open(cfg.reports_dir / "groupdro_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved:", cfg.reports_dir / "groupdro_report.json")
