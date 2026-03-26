
import argparse
import json

from qwen_iemocap.behavior import run_behavior_audio_only, run_behavior_conflicts, run_behavior_perturbations
from qwen_iemocap.causal import run_causal_subspace_erasure, run_head_token_ablation
from qwen_iemocap.config import build_config
from qwen_iemocap.data import build_metadata, split_metadata
from qwen_iemocap.exporting import export_summary_bundle
from qwen_iemocap.mine import run_mine
from qwen_iemocap.modeling import load_model_bundle
from qwen_iemocap.representations import (
    build_probe_targets,
    prepare_representation_splits,
    run_cka,
    run_layer_probes,
    run_representation_extraction,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--label-scheme", type=str, choices=["4way", "6way"], default="6way")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-pert-samples", type=int, default=None)
    parser.add_argument("--max-conflict-samples", type=int, default=None)
    parser.add_argument("--rep-max-train", type=int, default=None)
    parser.add_argument("--rep-max-val", type=int, default=None)
    parser.add_argument("--rep-max-test", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {"label_scheme": args.label_scheme}
    if args.model_dir is not None:
        overrides["model_dir"] = args.model_dir
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.results_dir is not None:
        overrides["results_dir"] = args.results_dir
    if args.max_test_samples is not None:
        overrides["max_test_samples"] = args.max_test_samples
    if args.max_pert_samples is not None:
        overrides["max_pert_samples"] = args.max_pert_samples
    if args.max_conflict_samples is not None:
        overrides["max_conflict_samples"] = args.max_conflict_samples
    if args.rep_max_train is not None:
        overrides["rep_max_train"] = args.rep_max_train
    if args.rep_max_val is not None:
        overrides["rep_max_val"] = args.rep_max_val
    if args.rep_max_test is not None:
        overrides["rep_max_test"] = args.rep_max_test
    config = build_config(**overrides)
    with open(config.results_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config.as_dict(), f, ensure_ascii=False, indent=2)
    print("Config:", json.dumps(config.as_dict(), ensure_ascii=False, indent=2))
    meta = build_metadata(config)
    train_df, val_df, test_df = split_metadata(meta)
    print("Split sizes:", meta["split"].value_counts().to_dict())
    print("Label counts:", meta["label"].value_counts().to_dict())
    bundle = load_model_bundle(config)
    print("Primary device:", bundle.primary_device)
    print("Sampling rate:", bundle.sr)
    print("Audio token id:", bundle.audio_token_id)
    print("Num LM layers:", bundle.num_layers)
    print("Hidden size:", bundle.hidden_size)
    audio_res, audio_summary = run_behavior_audio_only(config, bundle, test_df)
    print(audio_summary)
    pert_summary_df, _ = run_behavior_perturbations(config, bundle, test_df)
    print(pert_summary_df)
    conflict_outputs = run_behavior_conflicts(config, bundle, test_df, audio_res=audio_res)
    print(conflict_outputs["conflict_summary"])
    print(conflict_outputs["synth_summary"])
    rep_train_df, rep_val_df, rep_test_df = prepare_representation_splits(config, train_df, val_df, test_df)
    rep_outputs = run_representation_extraction(config, bundle, rep_train_df, rep_val_df, rep_test_df)
    cka_outputs = run_cka(
        config,
        rep_test_df,
        rep_outputs["reps_audio_test"],
        rep_outputs["reps_text_test"],
        rep_outputs["reps_at_test"],
    )
    print("CKA complete:", {k: v.shape if hasattr(v, "shape") else type(v).__name__ for k, v in cka_outputs.items() if k.startswith("cka_")})
    targets = build_probe_targets(rep_train_df, rep_test_df, seed=config.seed)
    probe_df, probe_aux = run_layer_probes(
        config,
        rep_train_df,
        rep_test_df,
        rep_outputs["reps_audio_train"],
        rep_outputs["reps_audio_test"],
        targets=targets,
    )
    print(probe_df.head())
    mine_df = run_mine(
        config,
        bundle.primary_device if str(bundle.primary_device) != "cpu" else "cpu",
        probe_aux["Htr"],
        targets["S_tr"],
        targets["U_tr"],
    )
    print(mine_df)
    causal_summary, causal_aux = run_causal_subspace_erasure(
        config,
        bundle,
        test_df,
        probe_aux["Htr"],
        targets["S_tr"],
        targets["U_tr"],
    )
    print(causal_summary)
    print("Semantic erase delta:", causal_aux["delta_sem"])
    print("Acoustic erase delta:", causal_aux["delta_acu"])
    ablation_outputs = run_head_token_ablation(config, bundle, test_df)
    print(ablation_outputs["head_ablation_df"])
    print(ablation_outputs["tok_summary"])
    summary_bundle = export_summary_bundle(config)
    print("Summary bundle keys:", list(summary_bundle.keys()))
    print("Results dir:", config.results_dir)


if __name__ == "__main__":
    main()
