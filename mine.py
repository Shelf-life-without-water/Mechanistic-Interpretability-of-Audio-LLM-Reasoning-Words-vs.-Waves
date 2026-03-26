
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .config import ExperimentConfig


class MINECritic(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, y):
        z = torch.cat([x, y], dim=-1)
        return self.net(z).squeeze(-1)


def mine_dv_lower_bound(critic: MINECritic, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    joint = critic(x, y)
    idx = torch.randperm(y.size(0), device=y.device)
    y_marg = y[idx]
    marg = critic(x, y_marg)
    return joint.mean() - torch.logsumexp(marg, dim=0) + math.log(marg.numel())


def fit_mine(
    config: ExperimentConfig,
    device,
    x_tr,
    y_tr,
    x_val,
    y_val,
) -> tuple[MINECritic, float]:
    x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    critic = MINECritic(x_tr_t.shape[1], y_tr_t.shape[1]).to(device)
    opt = torch.optim.AdamW(
        critic.parameters(),
        lr=config.mine_lr,
        weight_decay=config.mine_weight_decay,
    )
    best_val = -1e9
    best_state = None
    patience = 0
    rng = np.random.default_rng(config.seed)
    for _ in range(config.mine_epochs):
        critic.train()
        order = rng.permutation(len(x_tr_t))
        for start in range(0, len(order), config.mine_batch_size):
            idx = order[start : start + config.mine_batch_size]
            xb = x_tr_t[idx]
            yb = y_tr_t[idx]
            lb = mine_dv_lower_bound(critic, xb, yb)
            loss = -lb
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            opt.step()
        critic.eval()
        with torch.inference_mode():
            val_lb = mine_dv_lower_bound(critic, x_val_t, y_val_t).item()
        if val_lb > best_val:
            best_val = val_lb
            best_state = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.mine_patience:
                break
    critic.load_state_dict(best_state)
    critic.eval()
    with torch.inference_mode():
        val_lb = mine_dv_lower_bound(critic, x_val_t, y_val_t).item()
    return critic.cpu(), float(val_lb)


def run_mine(
    config: ExperimentConfig,
    device,
    Htr: np.ndarray,
    S_tr: np.ndarray,
    U_tr: np.ndarray,
) -> pd.DataFrame:
    num_layers = Htr.shape[1]
    selected_layers = list(range(0, num_layers, config.mine_layers_stride))
    if selected_layers[-1] != num_layers - 1:
        selected_layers.append(num_layers - 1)
    mine_rows = []
    for lid in tqdm(selected_layers, desc="MINE layer loop"):
        Xtr = Htr[:, lid, :]
        x_scaler = StandardScaler()
        Xtr_z = x_scaler.fit_transform(Xtr)
        pca_dim = min(config.mine_pca_dim, Xtr_z.shape[0] - 1, Xtr_z.shape[1])
        pca = PCA(n_components=pca_dim, random_state=config.seed)
        Xtr_p = pca.fit_transform(Xtr_z).astype(np.float32)
        n_val = min(len(Xtr_p) - 1, max(64, int(0.15 * len(Xtr_p))))
        Xtr_m, Xval_m = Xtr_p[:-n_val], Xtr_p[-n_val:]
        Str_m, Sval_m = S_tr[:-n_val], S_tr[-n_val:]
        Utr_m, Uval_m = U_tr[:-n_val], U_tr[-n_val:]
        _, mi_s = fit_mine(config, device, Xtr_m, Str_m, Xval_m, Sval_m)
        _, mi_u = fit_mine(config, device, Xtr_m, Utr_m, Xval_m, Uval_m)
        mine_rows.append(
            {
                "layer": lid,
                "mi_semantic_lb": mi_s,
                "mi_acoustic_lb": mi_u,
            }
        )
    mine_df = pd.DataFrame(mine_rows)
    mine_df.to_csv(config.table_dir / f"mine_semantic_acoustic_{config.label_scheme}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mine_df["layer"], mine_df["mi_semantic_lb"], marker="o", label="I(h_l ; s) lower bound")
    ax.plot(mine_df["layer"], mine_df["mi_acoustic_lb"], marker="o", label="I(h_l ; u) lower bound")
    ax.set_xlabel("layer")
    ax.set_ylabel("MINE DV lower bound")
    ax.set_title("Layerwise mutual information trends")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.fig_dir / f"mine_semantic_acoustic_{config.label_scheme}.png", bbox_inches="tight")
    plt.close(fig)
    return mine_df
