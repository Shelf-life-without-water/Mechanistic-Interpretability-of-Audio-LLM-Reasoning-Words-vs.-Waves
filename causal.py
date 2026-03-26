
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from .behavior import run_eval
from .config import ExperimentConfig
from .modeling import ModelBundle, load_audio, score_closed_set
from .prompts import build_audio_only_prompt, build_audio_text_prompt, build_text_only_prompt
from .utils import maybe_limit_df, paired_bootstrap_delta


def fit_subspace_basis(X: np.ndarray, Y: np.ndarray) -> dict:
    reg = RidgeCV(alphas=np.logspace(-2, 2, 9))
    reg.fit(X, Y)
    W = np.asarray(reg.coef_)
    if W.ndim == 1:
        W = W[None, :]
    W = W.T
    B, _ = np.linalg.qr(W)
    mu = X.mean(axis=0, keepdims=True).astype(np.float32)
    return {
        "basis": B.astype(np.float32),
        "mean": mu,
        "reg": reg,
    }


def get_layer_for_hook(bundle: ModelBundle, layer_idx: int):
    return bundle.lm_layers[layer_idx]


@contextmanager
def subspace_erasure_context(bundle: ModelBundle, layer_idx: int, basis_dict: dict):
    layer = get_layer_for_hook(bundle, layer_idx)
    B = torch.tensor(basis_dict["basis"], dtype=torch.float32, device=bundle.primary_device)
    mu = torch.tensor(basis_dict["mean"], dtype=torch.float32, device=bundle.primary_device)

    def pre_hook(module, inputs):
        h = inputs[0]
        dtype = h.dtype
        h_f = h.float()
        hc = h_f - mu
        proj = (hc @ B) @ B.T
        h_new = hc - proj + mu
        rest = inputs[1:]
        return (h_new.to(dtype),) + rest

    handle = layer.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def make_subspace_intervention(bundle: ModelBundle, config: ExperimentConfig, basis_dict: dict):
    def factory(full_inputs, prompt_lens):
        return subspace_erasure_context(bundle, config.intervene_layer, basis_dict)
    return factory


def run_eval_with_intervention(
    config: ExperimentConfig,
    bundle: ModelBundle,
    df: pd.DataFrame,
    intervention_factory,
    mode: str = "audio",
    cache_name: str = "tmp",
) -> pd.DataFrame:
    cache_path = config.cache_dir / f"{cache_name}_{config.label_scheme}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    rows = []
    dfx = df.copy().reset_index(drop=True)
    for _, row in tqdm(dfx.iterrows(), total=len(dfx), desc=f"intervene::{cache_name}"):
        wav = load_audio(row["audio_path"], bundle.sr) if mode != "text" else None
        if mode == "audio":
            prompt = build_audio_only_prompt(bundle.processor, config.labels)
        elif mode == "text":
            prompt = build_text_only_prompt(bundle.processor, row["transcription"], config.labels)
        elif mode == "audio_text":
            prompt = build_audio_text_prompt(bundle.processor, row["transcription"], config.labels)
        else:
            raise ValueError(mode)
        out = score_closed_set(
            bundle,
            prompt,
            config.labels,
            audio_array=wav,
            intervention_factory=intervention_factory,
        )
        rec = {
            "uid": row["uid"],
            "y_true": row["label"],
            "pred": out["pred"],
        }
        for lab, p in zip(config.labels, out["probs"]):
            rec[f"prob_{lab}"] = float(p)
        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_parquet(cache_path, index=False)
    return res


def run_causal_subspace_erasure(
    config: ExperimentConfig,
    bundle: ModelBundle,
    test_df: pd.DataFrame,
    Htr: np.ndarray,
    S_tr: np.ndarray,
    U_tr: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    Xtr_layer = Htr[:, config.intervene_layer, :].astype(np.float32)
    semantic_basis = fit_subspace_basis(Xtr_layer, S_tr)
    acoustic_basis = fit_subspace_basis(Xtr_layer, U_tr)
    causal_df = maybe_limit_df(test_df, max_samples=config.causal_max_samples, seed=config.seed)
    baseline_causal = run_eval(
        config,
        bundle,
        causal_df,
        mode="audio",
        max_samples=None,
        cache_path=config.cache_dir / f"causal_baseline_audio_{config.label_scheme}.parquet",
    )
    semantic_erase_res = run_eval_with_intervention(
        config,
        bundle,
        causal_df,
        make_subspace_intervention(bundle, config, semantic_basis),
        mode="audio",
        cache_name="causal_semantic_erasure",
    )
    acoustic_erase_res = run_eval_with_intervention(
        config,
        bundle,
        causal_df,
        make_subspace_intervention(bundle, config, acoustic_basis),
        mode="audio",
        cache_name="causal_acoustic_erasure",
    )
    base = baseline_causal[["uid", "y_true", "pred"]].rename(columns={"pred": "pred_base"})
    sem = semantic_erase_res[["uid", "pred"]].rename(columns={"pred": "pred_sem_erase"})
    acu = acoustic_erase_res[["uid", "pred"]].rename(columns={"pred": "pred_acu_erase"})
    cmp = base.merge(sem, on="uid").merge(acu, on="uid")
    summary_causal = pd.DataFrame(
        [
            {
                "condition": "baseline",
                "acc": accuracy_score(cmp["y_true"], cmp["pred_base"]),
                "macro_f1": f1_score(cmp["y_true"], cmp["pred_base"], average="macro"),
            },
            {
                "condition": "semantic_erasure",
                "acc": accuracy_score(cmp["y_true"], cmp["pred_sem_erase"]),
                "macro_f1": f1_score(cmp["y_true"], cmp["pred_sem_erase"], average="macro"),
            },
            {
                "condition": "acoustic_erasure",
                "acc": accuracy_score(cmp["y_true"], cmp["pred_acu_erase"]),
                "macro_f1": f1_score(cmp["y_true"], cmp["pred_acu_erase"], average="macro"),
            },
        ]
    )
    summary_causal.to_csv(config.table_dir / f"causal_subspace_erasure_{config.label_scheme}.csv", index=False)
    delta_sem = paired_bootstrap_delta(
        lambda yt, yp: accuracy_score(yt, yp),
        cmp["y_true"].values,
        cmp["pred_base"].values,
        cmp["pred_sem_erase"].values,
        seed=config.seed,
    )
    delta_acu = paired_bootstrap_delta(
        lambda yt, yp: accuracy_score(yt, yp),
        cmp["y_true"].values,
        cmp["pred_base"].values,
        cmp["pred_acu_erase"].values,
        seed=config.seed,
    )
    return summary_causal, {
        "cmp": cmp,
        "baseline_causal": baseline_causal,
        "semantic_erase_res": semantic_erase_res,
        "acoustic_erase_res": acoustic_erase_res,
        "delta_sem": delta_sem,
        "delta_acu": delta_acu,
        "semantic_basis": semantic_basis,
        "acoustic_basis": acoustic_basis,
    }


def collect_head_audio_attention_scores(
    config: ExperimentConfig,
    bundle: ModelBundle,
    df: pd.DataFrame,
    max_samples: int | None = None,
) -> np.ndarray:
    dfx = maybe_limit_df(df, max_samples=max_samples, seed=config.seed)
    head_scores = None
    head_counts = None
    rows = dfx.to_dict("records")
    for row in tqdm(rows, desc="head-audio attention calib"):
        wav = load_audio(row["audio_path"], bundle.sr)
        prompt = build_audio_only_prompt(bundle.processor, config.labels)
        batch = bundle.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        input_ids_cpu = batch["input_ids"].clone().cpu()
        attn_mask_cpu = batch["attention_mask"].clone().cpu()
        batch_dev = {k: v.to(bundle.primary_device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.inference_mode():
            outputs = bundle.model(
                **batch_dev,
                output_hidden_states=False,
                output_attentions=True,
                return_dict=True,
                use_cache=False,
            )
        attns = outputs.attentions
        if head_scores is None:
            n_layers = len(attns)
            n_heads = attns[0].shape[1]
            head_scores = np.zeros((n_layers, n_heads), dtype=np.float64)
            head_counts = np.zeros((n_layers, n_heads), dtype=np.float64)
        audio_pos = (input_ids_cpu[0] == bundle.audio_token_id).nonzero(as_tuple=False).squeeze(-1).numpy()
        if len(audio_pos) == 0:
            continue
        qpos = int(attn_mask_cpu[0].sum().item() - 1)
        for lid, attn in enumerate(attns):
            A = attn[0].detach().float().cpu().numpy()
            score = A[:, qpos, audio_pos].sum(axis=-1)
            head_scores[lid] += score
            head_counts[lid] += 1.0
        del outputs, attns, batch_dev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return head_scores / np.maximum(head_counts, 1e-12)


@contextmanager
def head_ablation_context(bundle: ModelBundle, layer_idx: int, head_ids: list[int]):
    layer = get_layer_for_hook(bundle, layer_idx)
    attn_mod = layer.self_attn
    num_heads = int(getattr(attn_mod, "num_heads"))
    head_dim = int(getattr(attn_mod, "head_dim", bundle.hidden_size // num_heads))

    def pre_hook(module, inputs):
        x = inputs[0]
        x = x.clone()
        for hid in head_ids:
            s = hid * head_dim
            e = (hid + 1) * head_dim
            x[..., s:e] = 0
        return (x,)

    handle = attn_mod.o_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def make_head_ablation_factory(bundle: ModelBundle, layer_idx: int, head_ids: list[int]):
    def factory(full_inputs, prompt_lens):
        return head_ablation_context(bundle, layer_idx, head_ids)
    return factory


@contextmanager
def token_position_zero_context(bundle: ModelBundle, layer_idx: int, positions: list[int]):
    pos_arr = np.array(sorted(set(int(p) for p in positions)), dtype=np.int64)
    layer = get_layer_for_hook(bundle, layer_idx)

    def pre_hook(module, inputs):
        h = inputs[0]
        h = h.clone()
        if len(pos_arr) > 0:
            h[:, pos_arr, :] = 0
        return (h,) + inputs[1:]

    handle = layer.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def make_audio_token_ablation_factory(bundle: ModelBundle, layer_idx: int = 0):
    def factory(full_inputs, prompt_lens):
        ids = full_inputs["input_ids"][0].cpu().numpy()
        pos = np.where(ids == bundle.audio_token_id)[0].tolist()
        return token_position_zero_context(bundle, layer_idx, pos)
    return factory


def make_random_text_token_ablation_factory(bundle: ModelBundle, layer_idx: int = 0, seed: int = 42):
    rng = np.random.default_rng(seed)

    def factory(full_inputs, prompt_lens):
        ids = full_inputs["input_ids"][0].cpu().numpy()
        pl = int(prompt_lens[0])
        audio_pos = set(np.where(ids == bundle.audio_token_id)[0].tolist())
        text_pos = [i for i in range(pl) if i not in audio_pos]
        n_audio = len(audio_pos)
        if n_audio == 0 or len(text_pos) == 0:
            chosen = []
        else:
            chosen = rng.choice(text_pos, size=min(n_audio, len(text_pos)), replace=False).tolist()
        return token_position_zero_context(bundle, layer_idx, chosen)
    return factory


def run_head_token_ablation(
    config: ExperimentConfig,
    bundle: ModelBundle,
    test_df: pd.DataFrame,
) -> dict:
    head_audio_scores = collect_head_audio_attention_scores(
        config,
        bundle,
        test_df,
        max_samples=config.attn_calib_samples,
    )
    np.save(config.cache_dir / f"head_audio_scores_{config.label_scheme}.npy", head_audio_scores)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(head_audio_scores, aspect="auto", cmap="viridis")
    ax.set_title("Mean attention mass from final prompt token to audio-token positions")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(config.fig_dir / f"head_audio_scores_{config.label_scheme}.png", bbox_inches="tight")
    plt.close(fig)
    top_heads_per_layer = []
    for lid in range(head_audio_scores.shape[0]):
        hid = int(np.argmax(head_audio_scores[lid]))
        top_heads_per_layer.append((lid, hid, float(head_audio_scores[lid, hid])))
    top_heads_global = sorted(
        [
            (lid, hid, float(head_audio_scores[lid, hid]))
            for lid in range(head_audio_scores.shape[0])
            for hid in range(head_audio_scores.shape[1])
        ],
        key=lambda x: x[2],
        reverse=True,
    )[:8]
    ablation_eval_df = maybe_limit_df(test_df, max_samples=config.ablation_eval_samples, seed=config.seed)
    head_ablation_rows = []
    baseline_head = run_eval(
        config,
        bundle,
        ablation_eval_df,
        mode="audio",
        max_samples=None,
        cache_path=config.cache_dir / f"head_ablation_baseline_{config.label_scheme}.parquet",
    )
    for rank, (lid, hid, score_val) in enumerate(top_heads_global):
        res = run_eval_with_intervention(
            config,
            bundle,
            ablation_eval_df,
            make_head_ablation_factory(bundle, lid, [hid]),
            mode="audio",
            cache_name=f"head_ablate_L{lid}_H{hid}",
        )
        merged = baseline_head[["uid", "y_true", "pred"]].rename(columns={"pred": "pred_base"}).merge(
            res[["uid", "pred"]].rename(columns={"pred": "pred_ablate"}),
            on="uid",
        )
        head_ablation_rows.append(
            {
                "rank": rank,
                "layer": lid,
                "head": hid,
                "audio_attention_score": score_val,
                "acc_base": accuracy_score(merged["y_true"], merged["pred_base"]),
                "acc_ablate": accuracy_score(merged["y_true"], merged["pred_ablate"]),
                "macro_f1_base": f1_score(merged["y_true"], merged["pred_base"], average="macro"),
                "macro_f1_ablate": f1_score(merged["y_true"], merged["pred_ablate"], average="macro"),
                "flip_rate": (merged["pred_base"] != merged["pred_ablate"]).mean(),
            }
        )
    head_ablation_df = pd.DataFrame(head_ablation_rows)
    head_ablation_df.to_csv(config.table_dir / f"head_ablation_results_{config.label_scheme}.csv", index=False)
    token_audio_res = run_eval_with_intervention(
        config,
        bundle,
        ablation_eval_df,
        make_audio_token_ablation_factory(bundle, layer_idx=0),
        mode="audio",
        cache_name="token_ablate_audio_tokens",
    )
    token_rand_res = run_eval_with_intervention(
        config,
        bundle,
        ablation_eval_df,
        make_random_text_token_ablation_factory(bundle, layer_idx=0, seed=config.seed),
        mode="audio",
        cache_name="token_ablate_random_text_tokens",
    )
    base_tok = baseline_head[["uid", "y_true", "pred"]].rename(columns={"pred": "pred_base"})
    aud_tok = token_audio_res[["uid", "pred"]].rename(columns={"pred": "pred_audio_token_zero"})
    rnd_tok = token_rand_res[["uid", "pred"]].rename(columns={"pred": "pred_random_text_zero"})
    tok_cmp = base_tok.merge(aud_tok, on="uid").merge(rnd_tok, on="uid")
    tok_summary = pd.DataFrame(
        [
            {
                "condition": "baseline",
                "acc": accuracy_score(tok_cmp["y_true"], tok_cmp["pred_base"]),
                "macro_f1": f1_score(tok_cmp["y_true"], tok_cmp["pred_base"], average="macro"),
            },
            {
                "condition": "zero_audio_tokens_layer0",
                "acc": accuracy_score(tok_cmp["y_true"], tok_cmp["pred_audio_token_zero"]),
                "macro_f1": f1_score(tok_cmp["y_true"], tok_cmp["pred_audio_token_zero"], average="macro"),
            },
            {
                "condition": "zero_random_text_tokens_layer0",
                "acc": accuracy_score(tok_cmp["y_true"], tok_cmp["pred_random_text_zero"]),
                "macro_f1": f1_score(tok_cmp["y_true"], tok_cmp["pred_random_text_zero"], average="macro"),
            },
        ]
    )
    tok_summary.to_csv(config.table_dir / f"token_ablation_results_{config.label_scheme}.csv", index=False)
    return {
        "head_audio_scores": head_audio_scores,
        "top_heads_per_layer": top_heads_per_layer,
        "top_heads_global": top_heads_global,
        "head_ablation_df": head_ablation_df,
        "tok_summary": tok_summary,
        "baseline_head": baseline_head,
        "token_audio_res": token_audio_res,
        "token_rand_res": token_rand_res,
    }
