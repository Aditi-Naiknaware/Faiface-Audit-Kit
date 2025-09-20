# src/calibration.py
import torch
from torch import nn
from torch.optim import LBFGS

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        T = self.temperature.clamp(1e-2, 10.0)
        return logits / T

@torch.no_grad()
def logits_on_loader(model, loader, device="cpu"):
    model.eval()
    all_logits, all_y = [], []
    for x,y,_,_ in loader:
        x=x.to(device); y=y.to(device)
        all_logits.append(model(x).detach().cpu())
        all_y.extend(y.cpu().tolist())
    return torch.cat(all_logits,0), torch.tensor(all_y)

def fit_temperature(model, val_loader, device="cpu"):
    scaler = TemperatureScaler().to(device)
    nll = nn.CrossEntropyLoss()
    logits, y = logits_on_loader(model, val_loader, device)
    logits = logits.to(device); y = y.to(device)

    optimizer = LBFGS([scaler.temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = nll(scaler(logits), y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return scaler
