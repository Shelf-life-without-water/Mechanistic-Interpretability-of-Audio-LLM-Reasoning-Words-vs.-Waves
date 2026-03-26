
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Audio, load_from_disk
from tqdm.auto import tqdm

from .config import ExperimentConfig


def maybe_materialize_audio(audio_obj, uid: str, materialized_audio_dir: Path) -> str | None:
    if audio_obj is None:
        return None
    if isinstance(audio_obj, dict):
        path = audio_obj.get("path")
        bytes_ = audio_obj.get("bytes")
    else:
        path = None
        bytes_ = None
    if path is not None and Path(path).exists():
        return str(Path(path))
    if bytes_ is not None:
        out_path = materialized_audio_dir / f"{uid}.wav"
        if not out_path.exists():
            with open(out_path, "wb") as f:
                f.write(bytes_)
        return str(out_path)
    return None


def parse_session_from_file(file_str: str | None) -> float | int:
    if not isinstance(file_str, str):
        return np.nan
    match = re.search(r"Ses0?(\d)", Path(file_str).name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else np.nan


def parse_speaker_from_file(file_str: str | None) -> str | None:
    if not isinstance(file_str, str):
        return None
    return Path(file_str).stem


def assign_split(config: ExperimentConfig, session: int) -> str:
    if session in config.train_sessions:
        return "train"
    if session in config.val_sessions:
        return "val"
    if session in config.test_sessions:
        return "test"
    return "drop"


def load_raw_dataset(config: ExperimentConfig):
    raw_ds = load_from_disk(str(config.data_dir))
    for split in raw_ds.keys():
        raw_ds[split] = raw_ds[split].cast_column("audio", Audio(decode=False))
    return raw_ds


def build_metadata(config: ExperimentConfig) -> pd.DataFrame:
    raw_ds = load_raw_dataset(config)
    train_ds = raw_ds["train"]
    rows = []
    for i in tqdm(range(len(train_ds)), desc="Building metadata"):
        ex = train_ds[i]
        file_str = ex.get("file", f"row_{i}.wav")
        uid = Path(file_str).stem if isinstance(file_str, str) else f"row_{i}"
        audio_path = maybe_materialize_audio(ex.get("audio"), uid, config.materialized_audio_dir)
        row = {
            "row_id": i,
            "uid": uid,
            "file": file_str,
            "audio_path": audio_path,
            "session": parse_session_from_file(file_str),
            "speaker_key": parse_speaker_from_file(file_str),
            "gender": ex.get("gender"),
            "transcription": ex.get("transcription", ""),
            "major_emotion": ex.get("major_emotion"),
            "frustrated": ex.get("frustrated", np.nan),
            "angry": ex.get("angry", np.nan),
            "sad": ex.get("sad", np.nan),
            "disgust": ex.get("disgust", np.nan),
            "excited": ex.get("excited", np.nan),
            "fear": ex.get("fear", np.nan),
            "neutral": ex.get("neutral", np.nan),
            "surprise": ex.get("surprise", np.nan),
            "happy": ex.get("happy", np.nan),
            "EmoAct": ex.get("EmoAct", np.nan),
            "EmoVal": ex.get("EmoVal", np.nan),
            "EmoDom": ex.get("EmoDom", np.nan),
            "speaking_rate": ex.get("speaking_rate", np.nan),
            "pitch_mean": ex.get("pitch_mean", np.nan),
            "pitch_std": ex.get("pitch_std", np.nan),
            "rms": ex.get("rms", np.nan),
            "relative_db": ex.get("relative_db", np.nan),
        }
        rows.append(row)
    meta = pd.DataFrame(rows)
    meta = meta[meta["audio_path"].notna()].copy()
    meta = meta[meta["transcription"].notna()].copy()
    meta["transcription"] = meta["transcription"].astype(str).str.strip()
    meta = meta[meta["transcription"].str.len() > 0].copy()
    if config.label_scheme == "6way":
        keep = ["angry", "excited", "frustrated", "neutral", "sad", "happy"]
        meta = meta[meta["major_emotion"].isin(keep)].copy()
        meta["label"] = meta["major_emotion"]
    else:
        map4 = {
            "angry": "angry",
            "sad": "sad",
            "neutral": "neutral",
            "happy": "happy",
            "excited": "happy",
        }
        meta = meta[meta["major_emotion"].isin(map4.keys())].copy()
        meta["label"] = meta["major_emotion"].map(map4)
    meta = meta[meta["session"].notna()].copy()
    meta["session"] = meta["session"].astype(int)
    meta["split"] = meta["session"].apply(lambda x: assign_split(config, x))
    meta = meta[meta["split"] != "drop"].copy()
    num_cols = ["EmoAct", "EmoVal", "EmoDom", "speaking_rate", "pitch_mean", "pitch_std", "rms", "relative_db"]
    for col in num_cols:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta = meta.reset_index(drop=True)
    out_path = config.table_dir / f"metadata_{config.label_scheme}.csv"
    meta.to_csv(out_path, index=False)
    return meta


def split_metadata(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = meta[meta["split"] == "train"].copy().reset_index(drop=True)
    val_df = meta[meta["split"] == "val"].copy().reset_index(drop=True)
    test_df = meta[meta["split"] == "test"].copy().reset_index(drop=True)
    return train_df, val_df, test_df
