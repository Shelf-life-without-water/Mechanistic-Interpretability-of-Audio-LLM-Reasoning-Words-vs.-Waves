
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import bootstrap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def bootstrap_ci_binary(x, n_resamples: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    x = np.asarray(x).astype(np.float64)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    res = bootstrap(
        (x,),
        np.mean,
        confidence_level=0.95,
        n_resamples=n_resamples,
        random_state=seed,
        method="basic",
    )
    return float(x.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high)


def paired_bootstrap_delta(metric_fn, y_true, pred_a, pred_b, n_resamples: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        da = metric_fn(y_true[idx], pred_a[idx])
        db = metric_fn(y_true[idx], pred_b[idx])
        deltas.append(db - da)
    deltas = np.asarray(deltas)
    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }


def jsd(p, q) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def plot_confmat(y_true, y_pred, labels: list[str], title: str, save_path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def maybe_limit_df(df: pd.DataFrame, max_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df.copy().reset_index(drop=True)
    out = []
    counts = df["label"].value_counts(normalize=True)
    for lab, frac in counts.items():
        n_lab = max(1, int(round(max_samples * frac)))
        dfl = df[df["label"] == lab]
        if len(dfl) <= n_lab:
            out.append(dfl)
        else:
            out.append(dfl.sample(n=n_lab, random_state=seed))
    out_df = pd.concat(out).sample(frac=1.0, random_state=seed).head(max_samples)
    return out_df.reset_index(drop=True)


def stratified_subset(df: pd.DataFrame, max_n: int | None, seed: int = 42) -> pd.DataFrame:
    return maybe_limit_df(df, max_samples=max_n, seed=seed)


def summarize_predictions(res_df: pd.DataFrame, name: str, seed: int = 42) -> pd.DataFrame:
    y_true = res_df["y_true"].values
    y_pred = res_df["pred"].values
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    _, acc_lo, acc_hi = bootstrap_ci_binary((y_true == y_pred).astype(np.float32), seed=seed)
    return pd.DataFrame(
        [
            {
                "name": name,
                "n": len(res_df),
                "acc": acc,
                "acc_ci_low": acc_lo,
                "acc_ci_high": acc_hi,
                "macro_f1": mf1,
            }
        ]
    )


def linear_cka(X, Y) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    var1 = np.linalg.norm(X.T @ X, ord="fro")
    var2 = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(hsic / (var1 * var2 + 1e-12))
