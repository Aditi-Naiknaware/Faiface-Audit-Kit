# src/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFG:
    project_dir: Path = Path.home() / "fairface-audit-kit"
    seed: int = 42

    # data
    hf_dataset: str = "HuggingFaceM4/FairFace"# official dataset id (train/validation) :contentReference[oaicite:2]{index=2}
    hf_subset: str = "0.25"  
    target: str = "age"  # we'll do 9-class age-group classification
    img_size: int = 224
    train_prop: float = 0.85  # from original 'train' we can keep as-is, we will use provided 'validation' for val/test
    batch_size: int = 64
    num_workers: int = 2

    # model
    model_name: str = "mobilenetv3_small_100"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    early_stop_patience: int = 3

    # mitigation toggles
    use_reweighting: bool = False
    use_groupdro: bool = False
    dro_temperature: float = 5.0  # higher = closer to max

    # dirs
    models_dir: Path = project_dir / "models"
    reports_dir: Path = project_dir / "reports"
    plots_dir: Path = project_dir / "reports" / "figs"
