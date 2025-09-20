# src/data.py
from datasets import load_dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch
from PIL import Image
from typing import Dict
from .config import CFG
from .utils import RACE_IDX2STR, GENDER_IDX2STR, AGE_IDX2STR

def get_transforms(train=True, sz=224):
    if train:
        return T.Compose([
            T.Resize((sz, sz)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2,0.2,0.2,0.1),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((sz, sz)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, tfm, target_col):
        self.ds = hf_split
        self.tfm = tfm
        self.target_col = target_col

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[int(idx)]
        img = row["image"].convert("RGB")
        x = self.tfm(img)
        y = int(row[self.target_col])
        race = int(row["race"]); gender = int(row["gender"])
        return x, y, race, gender

def make_loaders(cfg: CFG, batch_size=None, num_workers=None):
    ds = load_dataset(cfg.hf_dataset, cfg.hf_subset)  # has 'train' and 'validation' splits :contentReference[oaicite:3]{index=3}
    tf_train = get_transforms(True, cfg.img_size)
    tf_test  = get_transforms(False, cfg.img_size)

    train_set = HFDataset(ds["train"], tf_train, cfg.target)
    val_set   = HFDataset(ds["validation"], tf_test, cfg.target)  # we'll also draw test from here

    # split validation 50/50 into val/test for final report
    val_size = len(val_set)//2
    indices = torch.randperm(len(val_set)).tolist()
    val_idx, test_idx = indices[:val_size], indices[val_size:]
    val_subset  = torch.utils.data.Subset(val_set, val_idx)
    test_subset = torch.utils.data.Subset(val_set, test_idx)

    bs = batch_size or cfg.batch_size
    nw = num_workers or cfg.num_workers

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_subset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_subset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, val_loader, test_loader
