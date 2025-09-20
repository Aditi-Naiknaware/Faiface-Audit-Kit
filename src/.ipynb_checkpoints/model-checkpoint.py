# src/model.py
import timm, torch
from torch import nn

def build_model(model_name: str, num_classes: int = 9):
    m = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return m
