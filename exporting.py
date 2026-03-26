
import json

import pandas as pd

from .config import ExperimentConfig


def export_summary_bundle(config: ExperimentConfig) -> dict:
    summary_files = [
        config.table_dir / f"behavior_audio_only_summary_{config.label_scheme}.csv",
        config.table_dir / f"behavior_perturbation_summary_{config.label_scheme}.csv",
        config.table_dir / f"natural_conflict_summary_{config.label_scheme}.csv",
        config.table_dir / f"synthetic_conflict_summary_{config.label_scheme}.csv",
        config.table_dir / f"layer_probes_{config.label_scheme}.csv",
        config.table_dir / f"mine_semantic_acoustic_{config.label_scheme}.csv",
        config.table_dir / f"causal_subspace_erasure_{config.label_scheme}.csv",
        config.table_dir / f"head_ablation_results_{config.label_scheme}.csv",
        config.table_dir / f"token_ablation_results_{config.label_scheme}.csv",
    ]
    summary_bundle = {}
    for path in summary_files:
        if path.exists():
            summary_bundle[path.name] = pd.read_csv(path).to_dict(orient="records")
    out_path = config.results_dir / f"summary_bundle_{config.label_scheme}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_bundle, f, ensure_ascii=False, indent=2)
    return summary_bundle
