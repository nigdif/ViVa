#!/usr/bin/env python3
"""Evaluate one remaining-progress checkpoint on train/OOD success and failure."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from render_eval_batch import compute_dataset_state_stats
from viva_dataset import VivaDataset, load_episode_success_labels
from visualization import build_episode_table, infer_episode, load_model


TRAIN_PATH = "data/adapted/lerobot_v3/basket_place_a02_03_s_76_f_28_sf_state26"
OOD_PATH = "data/adapted/lerobot_v3/basket_place_a02_04_s_116_f_24_sf_state26"
TRUE_OOD_PATH = "data/adapted/lerobot_v3/basket_place_0428_s_37_sf_state26"
CASES = (
    ("train_success", TRAIN_PATH, 0),
    ("train_failure", TRAIN_PATH, 76),
    ("ood_success", OOD_PATH, 0),
    ("ood_failure", OOD_PATH, 116),
    ("true_ood_success", TRUE_OOD_PATH, 0),
)


def curve_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    error = predictions - targets
    correlation = np.corrcoef(predictions, targets)[0, 1]
    # Remaining progress should decrease. Allow tiny bf16-scale jitter.
    diffs = np.diff(predictions)
    n = len(predictions)
    tenth = max(1, n // 10)
    quartile_means = [
        float(chunk.mean())
        for chunk in np.array_split(predictions, 4)
    ]
    slope, intercept = np.polyfit(targets, predictions, 1)
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "correlation": float(correlation),
        "pred_first": float(predictions[0]),
        "pred_last": float(predictions[-1]),
        "pred_first_10pct_mean": float(predictions[:tenth].mean()),
        "pred_final_10pct_mean": float(predictions[-tenth:].mean()),
        "pred_range": float(predictions.max() - predictions.min()),
        "monotonic_increase_fraction": float((diffs > 0.01).mean()),
        "large_increase_fraction": float((diffs > 0.10).mean()),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "quartile_means": quartile_means,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument(
        "--cases", nargs="+", choices=[case[0] for case in CASES],
        default=[case[0] for case in CASES],
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(args.config)
    if config.dataset.value_target.mode != "remaining_progress":
        raise ValueError("This validator requires remaining_progress")

    train_paths = [
        path
        for task in config.dataset.tasks
        for path in task.data_paths
    ]
    state_dim = int(config.common.state_dim)
    state_config = config.dataset.get("state")
    state_stats = compute_dataset_state_stats(
        train_paths, state_dim=state_dim, state_config=state_config
    )
    model = load_model(args.checkpoint, config, args.device)
    t5_path = config.dataset.tasks[0].t5_embedding_path
    t5_embedding = torch.load(t5_path, map_location="cpu")

    # Evaluate all outcomes even when the checkpoint was trained success-only.
    value_config = OmegaConf.to_container(config.dataset.value_target, resolve=True)
    value_config["success_only"] = False

    datasets = {}
    results = []
    selected_cases = [case for case in CASES if case[0] in args.cases]
    for name, data_path, episode_index in selected_cases:
        if data_path not in datasets:
            datasets[data_path] = VivaDataset(
                data_paths=[data_path],
                video_height=int(config.common.video_height),
                video_width=int(config.common.video_width),
                state_stats=state_stats,
                state_dim=state_dim,
                value_target_config=value_config,
                state_config=state_config,
                require_success_labels=True,
            )
        dataset = datasets[data_path]
        episode_by_index = {
            item["episode_index"]: item for item in build_episode_table(dataset)
        }
        episode = episode_by_index[episode_index]
        predictions = infer_episode(
            model,
            dataset,
            episode,
            t5_embedding,
            args.device,
            args.num_inference_steps,
        )
        length = int(episode["episode_length"])
        targets = np.asarray([
            dataset._compute_value(frame_idx, length, True)
            for frame_idx in range(length)
        ], dtype=np.float32)
        outcome_labels = load_episode_success_labels(
            Path(data_path), len(dataset._subdataset_episode_lengths[0])
        )
        success = bool(outcome_labels[episode_index])
        metrics = curve_metrics(predictions, targets)
        values_path = output_dir / f"{name}_ep{episode_index:03d}_values.json"
        values_path.write_text(json.dumps({
            "name": name,
            "data_path": data_path,
            "episode_index": episode_index,
            "success": success,
            "targets": targets.tolist(),
            "predictions": predictions.tolist(),
        }))
        result = {
            "name": name,
            "data_path": data_path,
            "episode_index": episode_index,
            "episode_length": length,
            "success": success,
            "values": str(values_path),
            **metrics,
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "value_target_mode": "remaining_progress",
        "state_dim": state_dim,
        "num_inference_steps": args.num_inference_steps,
        "episodes": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
