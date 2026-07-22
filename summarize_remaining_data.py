#!/usr/bin/env python3
"""Summarize label, duration, and state-domain risks for remaining progress."""

import json
from pathlib import Path

import numpy as np


DATASETS = [
    Path("data/adapted/lerobot_v3/basket_place_a02_03_s_76_f_28_sf_state26"),
    Path("data/adapted/lerobot_v3/basket_place_a02_04_train80_s_93_f_19_sf_state26"),
    Path("data/adapted/lerobot_v3/basket_place_a02_04_s_116_f_24_sf_state26"),
]
AUDIT = Path("logs/gt_value_curves/basket_place_value_data_audit.json")
OUT = Path("logs/analysis/remaining_validation/data_audit.json")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def describe(values: list[int]) -> dict:
    array = np.asarray(values)
    return {
        "count": int(len(array)),
        "frames": int(array.sum()),
        "min": int(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p75": float(np.quantile(array, 0.75)),
        "max": int(array.max()),
    }


report = {"datasets": [], "state_domain_comparison": {}}
for dataset_path in DATASETS:
    episodes = read_jsonl(dataset_path / "meta/episodes.jsonl")
    outcomes = read_jsonl(dataset_path / "meta/episode_success.jsonl")
    outcome_by_ep = {int(row["episode_index"]): bool(row["success"]) for row in outcomes}
    success_lengths = []
    failure_lengths = []
    issues = []
    for row in episodes:
        episode_index = int(row["episode_index"])
        if episode_index not in outcome_by_ep:
            issues.append(f"episode {episode_index}: missing outcome")
            continue
        target = success_lengths if outcome_by_ep[episode_index] else failure_lengths
        target.append(int(row["length"]))
    if len(episodes) != len(outcomes):
        issues.append("episodes and outcome row counts differ")
    report["datasets"].append({
        "path": str(dataset_path),
        "episode_rows": len(episodes),
        "outcome_rows": len(outcomes),
        "issues": issues,
        "success": describe(success_lengths),
        "failure": describe(failure_lengths),
        "failure_frame_fraction": float(
            sum(failure_lengths) / (sum(success_lengths) + sum(failure_lengths))
        ),
    })

prior = json.loads(AUDIT.read_text())
train_a, train_b = prior["datasets"]
a_min = np.asarray(train_a["state_min_first26"])
a_max = np.asarray(train_a["state_max_first26"])
b_min = np.asarray(train_b["state_min_first26"])
b_max = np.asarray(train_b["state_max_first26"])
disjoint = np.where((a_max < b_min) | (b_max < a_min))[0]
constant_a = np.where(np.isclose(a_min, a_max))[0]
constant_b = np.where(np.isclose(b_min, b_max))[0]
report["state_domain_comparison"] = {
    "dataset_a": train_a["path"],
    "dataset_b": train_b["path"],
    "disjoint_state_dimensions": disjoint.tolist(),
    "constant_dimensions_dataset_a": constant_a.tolist(),
    "constant_dimensions_dataset_b": constant_b.tolist(),
    "dimension_details": [
        {
            "dimension": int(index),
            "dataset_a_range": [float(a_min[index]), float(a_max[index])],
            "dataset_b_range": [float(b_min[index]), float(b_max[index])],
        }
        for index in sorted(set(disjoint) | set(constant_a) | set(constant_b))
    ],
    "existing_audit_issues": {
        "dataset_a": train_a["issues"],
        "dataset_b": train_b["issues"],
    },
    "top_state_jumps_dataset_a": train_a["top_state_jumps"][:5],
    "top_state_jumps_dataset_b": train_b["top_state_jumps"][:5],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
