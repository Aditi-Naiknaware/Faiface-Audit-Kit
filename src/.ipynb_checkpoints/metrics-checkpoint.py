# src/metrics.py
import torch, numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple

@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    y_all, p_all, conf_all = [], [], []
    races, genders = [], []
    for x, y, race, gender in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1)
        y_all.extend(y.cpu().tolist())
        p_all.extend(preds.cpu().tolist())
        conf_all.extend(probs.max(1).values.cpu().tolist())
        races.extend(race.tolist()); genders.extend(gender.tolist())

    overall_acc = accuracy_score(y_all, p_all)

    buckets = defaultdict(lambda: {"y": [], "p": [], "conf": []})
    for yi, pi, ci, r, g in zip(y_all, p_all, conf_all, races, genders):
        key = f"r{r}|g{g}"
        buckets[key]["y"].append(yi); buckets[key]["p"].append(pi); buckets[key]["conf"].append(ci)

    per_group_acc = {k: accuracy_score(v["y"], v["p"]) for k,v in buckets.items()}
    worst_group = min(per_group_acc, key=per_group_acc.get)
    acc_gap = max(per_group_acc.values()) - min(per_group_acc.values())

    return {
        "overall_acc": overall_acc,
        "per_group_acc": per_group_acc,
        "worst_group": worst_group,
        "acc_gap": acc_gap,
        "y_all": y_all, "p_all": p_all, "conf_all": conf_all,
        "races": races, "genders": genders,
    }

def expected_calibration_error(y_true, y_pred, conf, n_bins=15):
    # Multiclass ECE: bin by confidence of predicted class
    y_true = np.array(y_true); y_pred=np.array(y_pred); conf=np.array(conf)
    bins = np.linspace(0,1,n_bins+1)
    ece=0.0; N=len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask): continue
        acc = (y_true[mask]==y_pred[mask]).mean()
        gap = abs(acc - conf[mask].mean())
        ece += (mask.mean())*gap
    return float(ece)
