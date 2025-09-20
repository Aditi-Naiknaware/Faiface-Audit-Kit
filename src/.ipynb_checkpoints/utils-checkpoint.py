# src/utils.py
import random, os, numpy as np, torch
from pathlib import Path

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

AGE_IDX2STR = ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","more than 70"]
GENDER_IDX2STR = ["Male","Female"]
RACE_IDX2STR = ["East Asian","Indian","Black","White","Middle Eastern","Latino_Hispanic","Southeast Asian"]
