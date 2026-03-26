
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .modeling import ModelBundle, load_audio, score_closed_set
from .prompts import (
    build_audio_hint_prompt,
    build_audio_only_prompt,
    build_audio_text_prompt,
    build_text_only_prompt,
)
from .utils import jsd, maybe_limit_df, paired_bootstrap_delta, plot_confmat, summarize_predictions


def perturb_pitch(wav: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=n_steps).astype(np.float32)


def perturb_tempo(wav: np.ndarray, sr: int, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(wav, rate=rate).astype(np.float32)


def perturb_gain_db(wav: np.ndarray, sr: int, db: float) -> np.ndarray:
    scale = 10 ** (db / 20.0)
    out = wav * scale
    out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32)


def perturb_pause_stretch(wav: np.ndarray, sr: int, top_db: int = 30, stretch: float = 1.5) -> np.ndarray:
    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) <= 1:
        return wav.astype(np.float32)
    pieces = []
    prev = 0
    for start, end in intervals:
        if start > prev:
            sil = wav[prev:start]
            pieces.append(np.zeros(int(len(sil) * stretch), dtype=np.float32))
        pieces.append(wav[start:end].astype(np.float32))
        prev = end
    if prev < len(wav):
        sil = wav[prev:]
        pieces.append(np.zeros(int(len(sil) * stretch), dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32)


def get_perturbations() -> dict:
    return {
        "pitch_-2": lambda x, sr: perturb_pitch(x, sr, -2.0),
        "pitch_+2": lambda x, sr: perturb_pitch(x, sr, 2.0),
        "tempo_0.9": lambda x, sr: perturb_tempo(x, sr, 0.9),
        "tempo_1.1": lambda x, sr: perturb_tempo(x, sr, 1.1),
        "gain_-6db": lambda x, sr: perturb_gain_db(x, sr, -6.0),
        "gain_+6db": lambda x, sr: perturb_gain_db(x, sr, 6.0),
        "pause_x1.5": lambda x, sr: perturb_pause_stretch(x, sr, stretch=1.5),
    }


def run_eval(
    config: ExperimentConfig,
    bundle: ModelBundle,
    df: pd.DataFrame,
    mode: str = "audio",
    perturb_name: str | None = None,
    hint_map: dict | None = None,
    max_samples: int | None = None,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        return pd.read_parquet(cache_path)
    dfx = maybe_limit_df(df, max_samples=max_samples, seed=config.seed)
    labels = config.labels
    perturbations = get_perturbations()
    results = []
    for _, row in tqdm(dfx.iterrows(), total=len(dfx), desc=f"eval::{mode}::{perturb_name}"):
        transcription = row["transcription"]
        y_true = row["label"]
        if mode in ["audio", "audio_text", "audio_hint"]:
            wav = load_audio(row["audio_path"], target_sr=bundle.sr)
            if perturb_name is not None:
                wav = perturbations[perturb_name](wav, bundle.sr)
        else:
            wav = None
        if mode == "audio":
            prompt = build_audio_only_prompt(bundle.processor, labels)
        elif mode == "text":
            prompt = build_text_only_prompt(bundle.processor, transcription, labels)
        elif mode == "audio_text":
            prompt = build_audio_text_prompt(bundle.processor, transcription, labels)
        elif mode == "audio_hint":
            if hint_map is None:
                raise ValueError("hint_map is required when mode='audio_hint'")
            hint_label = hint_map[row["uid"]]
            prompt = build_audio_hint_prompt(bundle.processor, transcription, hint_label, labels)
        else:
            raise ValueError(mode)
        out = score_closed_set(bundle, prompt, labels, audio_array=wav)
        rec = {
            "uid": row["uid"],
            "file": row["file"],
            "split": row["split"],
            "y_true": y_true,
            "pred": out["pred"],
            "mode": mode,
            "perturb": perturb_name if perturb_name is not None else "none",
        }
        for lab, s, p in zip(labels, out["scores"], out["probs"]):
            rec[f"score_{lab}"] = float(s)
            rec[f"prob_{lab}"] = float(p)
        results.append(rec)
    res = pd.DataFrame(results)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        res.to_parquet(cache_path, index=False)
    return res


def run_behavior_audio_only(config: ExperimentConfig, bundle: ModelBundle, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audio_res = run_eval(
        config,
        bundle,
        test_df,
        mode="audio",
        max_samples=config.max_test_samples,
        cache_path=config.cache_dir / f"behavior_audio_only_{config.label_scheme}.parquet",
    )
    audio_summary = summarize_predictions(audio_res, "audio_only", seed=config.seed)
    audio_summary.to_csv(config.table_dir / f"behavior_audio_only_summary_{config.label_scheme}.csv", index=False)
    report = classification_report(audio_res["y_true"], audio_res["pred"], labels=config.labels, digits=4)
    (config.table_dir / f"behavior_audio_only_report_{config.label_scheme}.txt").write_text(report, encoding="utf-8")
    plot_confmat(
        audio_res["y_true"].values,
        audio_res["pred"].values,
        config.labels,
        f"Audio-only confusion ({config.label_scheme})",
        config.fig_dir / f"confmat_audio_only_{config.label_scheme}.png",
    )
    return audio_res, audio_summary


def run_behavior_perturbations(
    config: ExperimentConfig,
    bundle: ModelBundle,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    perturbations = get_perturbations()
    base_for_pert = maybe_limit_df(test_df, config.max_pert_samples, seed=config.seed)
    base_audio_res = run_eval(
        config,
        bundle,
        base_for_pert,
        mode="audio",
        max_samples=None,
        cache_path=config.cache_dir / f"pert_base_audio_{config.label_scheme}.parquet",
    )
    pert_summaries = []
    all_pert_results = []
    for pname in perturbations.keys():
        pres = run_eval(
            config,
            bundle,
            base_for_pert,
            mode="audio",
            perturb_name=pname,
            max_samples=None,
            cache_path=config.cache_dir / f"pert_{pname}_{config.label_scheme}.parquet",
        )
        merged = base_audio_res.merge(
            pres,
            on=["uid", "file", "split", "y_true"],
            suffixes=("_base", "_pert"),
        )
        flip_rate = (merged["pred_base"] != merged["pred_pert"]).mean()
        js_list = []
        for _, r in merged.iterrows():
            p = np.array([r[f"prob_{lab}_base"] for lab in config.labels], dtype=np.float64)
            q = np.array([r[f"prob_{lab}_pert"] for lab in config.labels], dtype=np.float64)
            js_list.append(jsd(p, q))
        summary = {
            "perturbation": pname,
            "n": len(merged),
            "acc_base": accuracy_score(merged["y_true"], merged["pred_base"]),
            "acc_pert": accuracy_score(merged["y_true"], merged["pred_pert"]),
            "macro_f1_base": f1_score(merged["y_true"], merged["pred_base"], average="macro"),
            "macro_f1_pert": f1_score(merged["y_true"], merged["pred_pert"], average="macro"),
            "flip_rate": flip_rate,
            "mean_jsd": float(np.mean(js_list)),
        }
        delta = paired_bootstrap_delta(
            lambda yt, yp: accuracy_score(yt, yp),
            merged["y_true"].values,
            merged["pred_base"].values,
            merged["pred_pert"].values,
            seed=config.seed,
        )
        summary.update(
            {
                "acc_delta_mean": delta["delta_mean"],
                "acc_delta_ci_low": delta["ci_low"],
                "acc_delta_ci_high": delta["ci_high"],
            }
        )
        pert_summaries.append(summary)
        all_pert_results.append(merged.assign(jsd=js_list, perturbation=pname))
    pert_summary_df = pd.DataFrame(pert_summaries).sort_values("acc_pert", ascending=False)
    pert_detail_df = pd.concat(all_pert_results, ignore_index=True)
    pert_summary_df.to_csv(config.table_dir / f"behavior_perturbation_summary_{config.label_scheme}.csv", index=False)
    pert_detail_df.to_parquet(config.cache_dir / f"behavior_perturbation_detail_{config.label_scheme}.parquet", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(np.arange(len(pert_summary_df)), pert_summary_df["flip_rate"].values)
    axes[0].set_xticks(np.arange(len(pert_summary_df)))
    axes[0].set_xticklabels(pert_summary_df["perturbation"].tolist(), rotation=45, ha="right")
    axes[0].set_title("Label flip rate under acoustic perturbations")
    axes[1].bar(np.arange(len(pert_summary_df)), pert_summary_df["mean_jsd"].values)
    axes[1].set_xticks(np.arange(len(pert_summary_df)))
    axes[1].set_xticklabels(pert_summary_df["perturbation"].tolist(), rotation=45, ha="right")
    axes[1].set_title("Mean JS divergence vs original")
    fig.tight_layout()
    fig.savefig(config.fig_dir / f"behavior_perturbation_summary_{config.label_scheme}.png", bbox_inches="tight")
    plt.close(fig)
    return pert_summary_df, pert_detail_df


def run_behavior_conflicts(
    config: ExperimentConfig,
    bundle: ModelBundle,
    test_df: pd.DataFrame,
    audio_res: pd.DataFrame | None = None,
) -> dict:
    if audio_res is None:
        audio_res = run_eval(
            config,
            bundle,
            test_df,
            mode="audio",
            max_samples=config.max_conflict_samples,
            cache_path=config.cache_dir / f"behavior_audio_only_for_conflict_{config.label_scheme}.parquet",
        )
    text_res = run_eval(
        config,
        bundle,
        test_df,
        mode="text",
        max_samples=config.max_conflict_samples,
        cache_path=config.cache_dir / f"behavior_text_only_{config.label_scheme}.parquet",
    )
    audio_text_res = run_eval(
        config,
        bundle,
        test_df,
        mode="audio_text",
        max_samples=config.max_conflict_samples,
        cache_path=config.cache_dir / f"behavior_audio_text_{config.label_scheme}.parquet",
    )
    merged_nat = (
        audio_res[["uid", "y_true", "pred"]]
        .rename(columns={"pred": "pred_audio"})
        .merge(text_res[["uid", "pred"]].rename(columns={"pred": "pred_text"}), on="uid", how="inner")
        .merge(audio_text_res[["uid", "pred"]].rename(columns={"pred": "pred_audio_text"}), on="uid", how="inner")
    )
    natural_conflict_a = merged_nat[
        (merged_nat["pred_audio"] == merged_nat["y_true"]) & (merged_nat["pred_text"] != merged_nat["y_true"])
    ].copy()
    natural_conflict_b = merged_nat[
        (merged_nat["pred_text"] == merged_nat["y_true"]) & (merged_nat["pred_audio"] != merged_nat["y_true"])
    ].copy()
    sdr_natural_a = (natural_conflict_a["pred_audio_text"] == natural_conflict_a["pred_text"]).mean() if len(natural_conflict_a) else np.nan
    retain_audio_a = (natural_conflict_a["pred_audio_text"] == natural_conflict_a["pred_audio"]).mean() if len(natural_conflict_a) else np.nan
    sdr_natural_b = (natural_conflict_b["pred_audio_text"] == natural_conflict_b["pred_text"]).mean() if len(natural_conflict_b) else np.nan
    retain_audio_b = (natural_conflict_b["pred_audio_text"] == natural_conflict_b["pred_audio"]).mean() if len(natural_conflict_b) else np.nan
    conflict_summary = pd.DataFrame(
        [
            {
                "subset": "A_audio_correct_text_wrong",
                "n": len(natural_conflict_a),
                "SDR_natural": sdr_natural_a,
                "retain_audio": retain_audio_a,
            },
            {
                "subset": "B_text_correct_audio_wrong",
                "n": len(natural_conflict_b),
                "SDR_natural": sdr_natural_b,
                "retain_audio": retain_audio_b,
            },
        ]
    )
    conflict_summary.to_csv(config.table_dir / f"natural_conflict_summary_{config.label_scheme}.csv", index=False)
    natural_conflict_a.to_csv(config.table_dir / f"natural_conflict_A_{config.label_scheme}.csv", index=False)
    natural_conflict_b.to_csv(config.table_dir / f"natural_conflict_B_{config.label_scheme}.csv", index=False)
    hint_df = (
        text_res[["uid", "pred"]]
        .rename(columns={"pred": "hint_label"})
        .merge(audio_res[["uid", "y_true", "pred"]].rename(columns={"pred": "pred_audio"}), on="uid")
    )
    eligible_hint = hint_df[hint_df["hint_label"] != hint_df["y_true"]].copy()
    if config.max_conflict_samples is not None and len(eligible_hint) > config.max_conflict_samples:
        eligible_hint = eligible_hint.sample(n=config.max_conflict_samples, random_state=config.seed)
    hint_map = dict(zip(eligible_hint["uid"], eligible_hint["hint_label"]))
    hint_test_df = test_df[test_df["uid"].isin(hint_map.keys())].copy()
    if len(hint_map) == 0:
        audio_hint_res = pd.DataFrame(columns=["uid", "pred"])
        synth = pd.DataFrame(columns=["uid", "y_true", "pred_audio", "pred_audio_hint", "hint_label"])
        synth_summary = pd.DataFrame(
            [
                {
                    "n": 0,
                    "SDR_synthetic": np.nan,
                    "retain_audio": np.nan,
                    "acc_audio": np.nan,
                    "acc_audio_hint": np.nan,
                }
            ]
        )
    else:
        audio_hint_res = run_eval(
            config,
            bundle,
            hint_test_df,
            mode="audio_hint",
            hint_map=hint_map,
            max_samples=None,
            cache_path=config.cache_dir / f"behavior_audio_hint_{config.label_scheme}.parquet",
        )
        synth = (
            hint_test_df[["uid", "label"]]
            .rename(columns={"label": "y_true"})
            .merge(audio_res[["uid", "pred"]].rename(columns={"pred": "pred_audio"}), on="uid")
            .merge(audio_hint_res[["uid", "pred"]].rename(columns={"pred": "pred_audio_hint"}), on="uid")
            .merge(pd.DataFrame({"uid": list(hint_map.keys()), "hint_label": list(hint_map.values())}), on="uid")
        )
        sdr_synth = (synth["pred_audio_hint"] == synth["hint_label"]).mean()
        retain_audio_synth = (synth["pred_audio_hint"] == synth["pred_audio"]).mean()
        acc_audio = accuracy_score(synth["y_true"], synth["pred_audio"])
        acc_audio_hint = accuracy_score(synth["y_true"], synth["pred_audio_hint"])
        synth_summary = pd.DataFrame(
            [
                {
                    "n": len(synth),
                    "SDR_synthetic": sdr_synth,
                    "retain_audio": retain_audio_synth,
                    "acc_audio": acc_audio,
                    "acc_audio_hint": acc_audio_hint,
                }
            ]
        )
    synth_summary.to_csv(config.table_dir / f"synthetic_conflict_summary_{config.label_scheme}.csv", index=False)
    synth.to_csv(config.table_dir / f"synthetic_conflict_detail_{config.label_scheme}.csv", index=False)
    return {
        "text_res": text_res,
        "audio_text_res": audio_text_res,
        "conflict_summary": conflict_summary,
        "natural_conflict_a": natural_conflict_a,
        "natural_conflict_b": natural_conflict_b,
        "audio_hint_res": audio_hint_res,
        "synth": synth,
        "synth_summary": synth_summary,
    }
