
from dataclasses import dataclass, field
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class ExperimentConfig:
    model_dir: Path = Path("/root/autodl-tmp/Qwen2-Audio-7B-Instruct")
    data_dir: Path = Path("/root/autodl-tmp/iemocap_hf_dataset")
    results_dir: Path = Path("/root/autodl-tmp/waves_over_words_qwen_iemocap")
    label_scheme: str = "6way"
    train_sessions: tuple[int, ...] = (1, 2, 3)
    val_sessions: tuple[int, ...] = (4,)
    test_sessions: tuple[int, ...] = (5,)
    max_test_samples: int | None = None
    max_pert_samples: int | None = None
    max_conflict_samples: int | None = None
    rep_max_train: int = 1600
    rep_max_val: int = 400
    rep_max_test: int = 800
    rep_batch_size: int = 2
    mine_layers_stride: int = 4
    mine_pca_dim: int = 128
    mine_batch_size: int = 128
    mine_epochs: int = 50
    mine_lr: float = 2e-4
    mine_weight_decay: float = 1e-5
    mine_patience: int = 6
    intervene_layer: int = 16
    causal_max_samples: int = 400
    attn_calib_samples: int = 128
    ablation_eval_samples: int = 256
    seed: int = 42
    figure_dpi: int = 140
    savefig_dpi: int = 140
    cache_dir: Path = field(init=False)
    fig_dir: Path = field(init=False)
    table_dir: Path = field(init=False)
    materialized_audio_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        self.cache_dir = self.results_dir / "cache"
        self.fig_dir = self.results_dir / "figures"
        self.table_dir = self.results_dir / "tables"
        self.materialized_audio_dir = self.results_dir / "materialized_audio"

    @property
    def labels(self) -> list[str]:
        if self.label_scheme == "6way":
            return ["angry", "excited", "frustrated", "neutral", "sad", "happy"]
        if self.label_scheme == "4way":
            return ["angry", "happy", "neutral", "sad"]
        raise ValueError("label_scheme must be '6way' or '4way'")

    def ensure_dirs(self) -> None:
        for path in [self.results_dir, self.cache_dir, self.fig_dir, self.table_dir, self.materialized_audio_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def configure_matplotlib(self) -> None:
        plt.rcParams["figure.dpi"] = self.figure_dpi
        plt.rcParams["savefig.dpi"] = self.savefig_dpi

    def as_dict(self) -> dict:
        return {
            "model_dir": str(self.model_dir),
            "data_dir": str(self.data_dir),
            "results_dir": str(self.results_dir),
            "label_scheme": self.label_scheme,
            "labels": self.labels,
            "train_sessions": list(self.train_sessions),
            "val_sessions": list(self.val_sessions),
            "test_sessions": list(self.test_sessions),
            "max_test_samples": self.max_test_samples,
            "max_pert_samples": self.max_pert_samples,
            "max_conflict_samples": self.max_conflict_samples,
            "rep_max_train": self.rep_max_train,
            "rep_max_val": self.rep_max_val,
            "rep_max_test": self.rep_max_test,
            "rep_batch_size": self.rep_batch_size,
            "mine_layers_stride": self.mine_layers_stride,
            "mine_pca_dim": self.mine_pca_dim,
            "mine_batch_size": self.mine_batch_size,
            "mine_epochs": self.mine_epochs,
            "mine_lr": self.mine_lr,
            "mine_weight_decay": self.mine_weight_decay,
            "mine_patience": self.mine_patience,
            "intervene_layer": self.intervene_layer,
            "causal_max_samples": self.causal_max_samples,
            "attn_calib_samples": self.attn_calib_samples,
            "ablation_eval_samples": self.ablation_eval_samples,
            "seed": self.seed,
        }


def build_config(**overrides) -> ExperimentConfig:
    cfg = ExperimentConfig(**overrides)
    cfg.ensure_dirs()
    cfg.set_seed()
    cfg.configure_matplotlib()
    return cfg
