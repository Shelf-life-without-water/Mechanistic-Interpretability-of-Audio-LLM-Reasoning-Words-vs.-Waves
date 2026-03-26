
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .modeling import ModelBundle, load_audio, move_batch_to_device
from .prompts import build_audio_only_prompt, build_audio_text_prompt, build_text_only_prompt
from .utils import linear_cka, stratified_subset


def prepare_representation_splits(
    config: ExperimentConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rep_train_df = stratified_subset(train_df, config.rep_max_train, seed=config.seed)
    rep_val_df = stratified_subset(val_df, config.rep_max_val, seed=config.seed)
    rep_test_df = stratified_subset(test_df, config.rep_max_test, seed=config.seed)
    rep_train_df.to_csv(config.table_dir / f"rep_train_meta_{config.label_scheme}.csv", index=False)
    rep_val_df.to_csv(config.table_dir / f"rep_val_meta_{config.label_scheme}.csv", index=False)
    rep_test_df.to_csv(config.table_dir / f"rep_test_meta_{config.label_scheme}.csv", index=False)
    return rep_train_df, rep_val_df, rep_test_df


def build_prompt_from_row(bundle: ModelBundle, row: dict, labels: list[str], mode: str) -> tuple[str, bool]:
    if mode == "audio":
        return build_audio_only_prompt(bundle.processor, labels), True
    if mode == "text":
        return build_text_only_prompt(bundle.processor, row["transcription"], labels), False
    if mode == "audio_text":
        return build_audio_text_prompt(bundle.processor, row["transcription"], labels), True
    raise ValueError(mode)


def extract_representations(
    config: ExperimentConfig,
    bundle: ModelBundle,
    df: pd.DataFrame,
    mode: str,
    batch_size: int | None = None,
    save_prefix: str = "audio",
):
    out_path = config.cache_dir / f"reps_{save_prefix}_{config.label_scheme}.npz"
    if out_path.exists():
        return np.load(out_path, allow_pickle=True)
    if batch_size is None:
        batch_size = config.rep_batch_size
    labels = config.labels
    all_reps = []
    all_uids = []
    rows = df.to_dict("records")
    for start in tqdm(range(0, len(rows), batch_size), desc=f"extract::{mode}"):
        batch_rows = rows[start : start + batch_size]
        texts = []
        audios = []
        uses_audio = []
        for row in batch_rows:
            prompt, use_audio = build_prompt_from_row(bundle, row, labels, mode)
            texts.append(prompt)
            uses_audio.append(use_audio)
            if use_audio:
                audios.append(load_audio(row["audio_path"], target_sr=bundle.sr))
        if all(uses_audio):
            batch = bundle.processor(text=texts, audio=audios, return_tensors="pt", padding=True)
        elif not any(uses_audio):
            batch = bundle.processor(text=texts, return_tensors="pt", padding=True)
        else:
            raise RuntimeError("Mixed audio and non-audio batches are not supported.")
        batch_dev = move_batch_to_device(batch, bundle.primary_device)
        with torch.inference_mode():
            outputs = bundle.model(
                **batch_dev,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
                use_cache=False,
            )
        hs = outputs.hidden_states
        attn_mask = batch["attention_mask"]
        last_idx = attn_mask.sum(dim=-1) - 1
        per_layer = []
        for lid in range(len(hs)):
            h = hs[lid].detach().float().cpu()
            rep = h[torch.arange(h.size(0)), last_idx, :]
            per_layer.append(rep.numpy().astype(np.float16))
        per_layer = np.stack(per_layer, axis=1)
        all_reps.append(per_layer)
        all_uids.extend([row["uid"] for row in batch_rows])
        del outputs, hs, batch_dev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    all_reps = np.concatenate(all_reps, axis=0)
    np.savez_compressed(out_path, reps=all_reps, uids=np.array(all_uids, dtype=object))
    return np.load(out_path, allow_pickle=True)


def run_representation_extraction(
    config: ExperimentConfig,
    bundle: ModelBundle,
    rep_train_df: pd.DataFrame,
    rep_val_df: pd.DataFrame,
    rep_test_df: pd.DataFrame,
) -> dict:
    reps_audio_train = extract_representations(config, bundle, rep_train_df, "audio", save_prefix="audio_train")
    reps_audio_val = extract_representations(config, bundle, rep_val_df, "audio", save_prefix="audio_val")
    reps_audio_test = extract_representations(config, bundle, rep_test_df, "audio", save_prefix="audio_test")
    reps_text_train = extract_representations(config, bundle, rep_train_df, "text", save_prefix="text_train")
    reps_text_val = extract_representations(config, bundle, rep_val_df, "text", save_prefix="text_val")
    reps_text_test = extract_representations(config, bundle, rep_test_df, "text", save_prefix="text_test")
    reps_at_train = extract_representations(config, bundle, rep_train_df, "audio_text", save_prefix="audio_text_train")
    reps_at_val = extract_representations(config, bundle, rep_val_df, "audio_text", save_prefix="audio_text_val")
    reps_at_test = extract_representations(config, bundle, rep_test_df, "audio_text", save_prefix="audio_text_test")
    return {
        "reps_audio_train": reps_audio_train,
        "reps_audio_val": reps_audio_val,
        "reps_audio_test": reps_audio_test,
        "reps_text_train": reps_text_train,
        "reps_text_val": reps_text_val,
        "reps_text_test": reps_text_test,
        "reps_at_train": reps_at_train,
        "reps_at_val": reps_at_val,
        "reps_at_test": reps_at_test,
    }


def run_cka(
    config: ExperimentConfig,
    rep_test_df: pd.DataFrame,
    reps_audio_test,
    reps_text_test,
    reps_at_test,
) -> dict:
    rep_test_meta = rep_test_df.copy().reset_index(drop=True)
    rep_test_meta["uid"] = rep_test_meta["uid"].astype(str)
    audio_test_uid_order = pd.Series(reps_audio_test["uids"]).astype(str).tolist()
    text_test_uid_order = pd.Series(reps_text_test["uids"]).astype(str).tolist()
    at_test_uid_order = pd.Series(reps_at_test["uids"]).astype(str).tolist()
    assert audio_test_uid_order == rep_test_meta["uid"].tolist()
    assert text_test_uid_order == rep_test_meta["uid"].tolist()
    assert at_test_uid_order == rep_test_meta["uid"].tolist()
    H_audio = reps_audio_test["reps"].astype(np.float32)
    H_text = reps_text_test["reps"].astype(np.float32)
    H_at = reps_at_test["reps"].astype(np.float32)
    num_layers = H_audio.shape[1]
    u_cols = ["speaking_rate", "pitch_mean", "pitch_std", "rms", "relative_db", "EmoAct", "EmoVal", "EmoDom"]
    U_test = rep_test_meta[u_cols].copy()
    U_test = U_test.fillna(U_test.median(numeric_only=True))
    U_test = U_test.to_numpy(dtype=np.float32)
    cka_audio_text = np.zeros((num_layers, num_layers), dtype=np.float32)
    for i in tqdm(range(num_layers), desc="CKA audio-text grid"):
        Xi = H_audio[:, i, :]
        for j in range(num_layers):
            Yj = H_text[:, j, :]
            cka_audio_text[i, j] = linear_cka(Xi, Yj)
    cka_audio_at = np.zeros((num_layers, num_layers), dtype=np.float32)
    for i in tqdm(range(num_layers), desc="CKA audio-audio+text grid"):
        Xi = H_audio[:, i, :]
        for j in range(num_layers):
            Yj = H_at[:, j, :]
            cka_audio_at[i, j] = linear_cka(Xi, Yj)
    cka_audio_u = np.array([linear_cka(H_audio[:, i, :], U_test) for i in range(num_layers)], dtype=np.float32)
    cka_text_u = np.array([linear_cka(H_text[:, i, :], U_test) for i in range(num_layers)], dtype=np.float32)
    cka_at_u = np.array([linear_cka(H_at[:, i, :], U_test) for i in range(num_layers)], dtype=np.float32)
    np.save(config.cache_dir / f"cka_audio_text_{config.label_scheme}.npy", cka_audio_text)
    np.save(config.cache_dir / f"cka_audio_at_{config.label_scheme}.npy", cka_audio_at)
    np.save(config.cache_dir / f"cka_audio_u_{config.label_scheme}.npy", cka_audio_u)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(cka_audio_text, cmap="magma")
    axes[0].set_title("CKA: audio-only layers vs text-only layers")
    axes[0].set_xlabel("text layer")
    axes[0].set_ylabel("audio layer")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(cka_audio_at, cmap="magma")
    axes[1].set_title("CKA: audio-only layers vs audio+transcript layers")
    axes[1].set_xlabel("audio+text layer")
    axes[1].set_ylabel("audio layer")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(config.fig_dir / f"cka_grids_{config.label_scheme}.png", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(num_layers), cka_audio_u, marker="o", label="audio-only vs acoustic U")
    ax.plot(np.arange(num_layers), cka_text_u, marker="o", label="text-only vs acoustic U")
    ax.plot(np.arange(num_layers), cka_at_u, marker="o", label="audio+text vs acoustic U")
    ax.set_xlabel("layer")
    ax.set_ylabel("linear CKA")
    ax.set_title("Layerwise alignment with acoustic feature matrix U")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.fig_dir / f"cka_acoustic_alignment_{config.label_scheme}.png", bbox_inches="tight")
    plt.close(fig)
    return {
        "rep_test_meta": rep_test_meta,
        "H_audio": H_audio,
        "H_text": H_text,
        "H_at": H_at,
        "cka_audio_text": cka_audio_text,
        "cka_audio_at": cka_audio_at,
        "cka_audio_u": cka_audio_u,
        "cka_text_u": cka_text_u,
        "cka_at_u": cka_at_u,
    }


def build_probe_targets(rep_train_meta: pd.DataFrame, rep_test_meta: pd.DataFrame, seed: int = 42) -> dict:
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    Xtxt_tr = tfidf.fit_transform(rep_train_meta["transcription"].tolist())
    Xtxt_te = tfidf.transform(rep_test_meta["transcription"].tolist())
    svd_dim = min(64, Xtxt_tr.shape[1] - 1) if Xtxt_tr.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    S_tr = svd.fit_transform(Xtxt_tr).astype(np.float32)
    S_te = svd.transform(Xtxt_te).astype(np.float32)
    u_cols = ["speaking_rate", "pitch_mean", "pitch_std", "rms", "relative_db", "EmoAct", "EmoVal", "EmoDom"]
    U_tr_df = rep_train_meta[u_cols].copy().fillna(rep_train_meta[u_cols].median(numeric_only=True))
    U_te_df = rep_test_meta[u_cols].copy().fillna(rep_train_meta[u_cols].median(numeric_only=True))
    u_scaler = StandardScaler()
    U_tr = u_scaler.fit_transform(U_tr_df.values.astype(np.float32)).astype(np.float32)
    U_te = u_scaler.transform(U_te_df.values.astype(np.float32)).astype(np.float32)
    y_tr = rep_train_meta["label"].values
    y_te = rep_test_meta["label"].values
    return {
        "tfidf": tfidf,
        "svd": svd,
        "u_scaler": u_scaler,
        "S_tr": S_tr,
        "S_te": S_te,
        "U_tr": U_tr,
        "U_te": U_te,
        "y_tr": y_tr,
        "y_te": y_te,
    }


def run_layer_probes(
    config: ExperimentConfig,
    rep_train_df: pd.DataFrame,
    rep_test_df: pd.DataFrame,
    reps_audio_train,
    reps_audio_test,
    targets: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    rep_train_meta = rep_train_df.copy().reset_index(drop=True)
    rep_test_meta = rep_test_df.copy().reset_index(drop=True)
    assert list(pd.Series(reps_audio_train["uids"]).astype(str)) == rep_train_meta["uid"].astype(str).tolist()
    assert list(pd.Series(reps_audio_test["uids"]).astype(str)) == rep_test_meta["uid"].astype(str).tolist()
    Htr = reps_audio_train["reps"].astype(np.float32)
    Hte = reps_audio_test["reps"].astype(np.float32)
    if targets is None:
        targets = build_probe_targets(rep_train_meta, rep_test_meta)
    S_tr = targets["S_tr"]
    S_te = targets["S_te"]
    U_tr = targets["U_tr"]
    U_te = targets["U_te"]
    y_tr = targets["y_tr"]
    y_te = targets["y_te"]
    num_layers = Htr.shape[1]
    probe_rows = []
    for lid in tqdm(range(num_layers), desc="Layer probes"):
        Xtr = Htr[:, lid, :]
        Xte = Hte[:, lid, :]
        sem_model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 2, 9)))
        sem_model.fit(Xtr, S_tr)
        S_hat = sem_model.predict(Xte)
        sem_r2 = r2_score(S_te, S_hat, multioutput="variance_weighted")
        ac_model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 2, 9)))
        ac_model.fit(Xtr, U_tr)
        U_hat = ac_model.predict(Xte)
        ac_r2 = r2_score(U_te, U_hat, multioutput="variance_weighted")
        clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-2, 2, 9)))
        clf.fit(Xtr, y_tr)
        y_hat = clf.predict(Xte)
        emo_f1 = f1_score(y_te, y_hat, average="macro")
        emo_acc = accuracy_score(y_te, y_hat)
        probe_rows.append(
            {
                "layer": lid,
                "semantic_r2": sem_r2,
                "acoustic_r2": ac_r2,
                "emotion_macro_f1": emo_f1,
                "emotion_acc": emo_acc,
            }
        )
    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(config.table_dir / f"layer_probes_{config.label_scheme}.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(probe_df["layer"], probe_df["semantic_r2"], marker="o", label="semantic probe (S)")
    axes[0].plot(probe_df["layer"], probe_df["acoustic_r2"], marker="o", label="acoustic probe (U)")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("R^2")
    axes[0].set_title("Layerwise semantic vs acoustic recoverability")
    axes[0].legend()
    axes[1].plot(probe_df["layer"], probe_df["emotion_macro_f1"], marker="o", label="emotion macro-F1")
    axes[1].plot(probe_df["layer"], probe_df["emotion_acc"], marker="o", label="emotion acc")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("score")
    axes[1].set_title("Layerwise task probe")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(config.fig_dir / f"layer_probes_{config.label_scheme}.png", bbox_inches="tight")
    plt.close(fig)
    return probe_df, {
        "rep_train_meta": rep_train_meta,
        "rep_test_meta": rep_test_meta,
        "Htr": Htr,
        "Hte": Hte,
        "targets": targets,
    }
