import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = [
    Path("data/adapted/lerobot_v3/basket_place_a02_03_s_76_f_28_sf_state26"),
    Path("data/adapted/lerobot_v3/basket_place_a02_04_train80_s_93_f_19_sf_state26"),
]
OUT = Path("logs/gt_value_curves/basket_place_value_data_audit.json")


def infer_counts(path):
    match = re.search(r"(?:^|_)s_(\d+)(?:_f_(\d+))?(?:_|$)", path.name)
    if not match:
        raise RuntimeError(f"Cannot infer success/failure from {path.name}")
    return int(match.group(1)), int(match.group(2) or 0)


def late_value(progress, success, warmup=0.5):
    if success:
        return 0.5 + 0.5 * progress
    if progress <= warmup:
        return 0.0
    return -((progress - warmup) / (1.0 - warmup))


def sep_value(progress, success):
    return 0.5 + 0.5 * progress if success else -0.5 - 0.5 * progress


def as_state26(series):
    return np.stack([np.asarray(x, dtype=np.float32)[:26] for x in series], axis=0)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {"datasets": [], "combined": {}}
    all_samples = []

    for ds_path in DATASETS:
        success_count, failure_count = infer_counts(ds_path)
        episodes = [
            json.loads(line)
            for line in (ds_path / "meta" / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        success_rows = [
            json.loads(line)
            for line in (ds_path / "meta" / "episode_success.jsonl").read_text().splitlines()
            if line.strip()
        ]
        parquet_files = sorted((ds_path / "data").glob("**/episode_*.parquet"))

        ds_report = {
            "path": str(ds_path),
            "expected_success": success_count,
            "expected_failure": failure_count,
            "meta_episodes": len(episodes),
            "success_file_rows": len(success_rows),
            "parquet_files": len(parquet_files),
            "issues": [],
            "state_dim_counts": {},
            "state_min_first26": None,
            "state_max_first26": None,
            "top_state_jumps": [],
            "value_ranges": {
                "outcome_separated_progress": {"success": [], "failure": []},
                "outcome_late_failure": {"success": [], "failure": []},
            },
        }

        if len(episodes) != success_count + failure_count:
            ds_report["issues"].append("episode count does not match s/f count")
        if len(success_rows) != len(episodes):
            ds_report["issues"].append("episode_success.jsonl row count mismatch")
        if len(parquet_files) != len(episodes):
            ds_report["issues"].append("parquet file count mismatch")

        state_min = np.full(26, np.inf, dtype=np.float64)
        state_max = np.full(26, -np.inf, dtype=np.float64)

        for ep_idx, file_path in enumerate(parquet_files):
            df = pd.read_parquet(file_path, columns=[
                "observation.state", "frame_index", "episode_index", "index"
            ])
            expected_len = int(episodes[ep_idx]["length"])
            success = ep_idx < success_count

            if len(df) != expected_len:
                ds_report["issues"].append(
                    f"episode {ep_idx}: parquet length {len(df)} != meta length {expected_len}"
                )
            if not np.array_equal(df["frame_index"].to_numpy(), np.arange(len(df))):
                ds_report["issues"].append(f"episode {ep_idx}: frame_index is not contiguous")
            if not (df["episode_index"].to_numpy() == ep_idx).all():
                ds_report["issues"].append(f"episode {ep_idx}: episode_index mismatch")

            dims = df["observation.state"].map(len).value_counts().to_dict()
            for dim, count in dims.items():
                ds_report["state_dim_counts"][str(int(dim))] = (
                    ds_report["state_dim_counts"].get(str(int(dim)), 0) + int(count)
                )

            state = as_state26(df["observation.state"])
            if not np.isfinite(state).all():
                ds_report["issues"].append(f"episode {ep_idx}: state has NaN or inf")
            state_min = np.minimum(state_min, state.min(axis=0))
            state_max = np.maximum(state_max, state.max(axis=0))

            if len(state) > 1:
                jumps = np.linalg.norm(np.diff(state, axis=0), axis=1)
                j = int(np.argmax(jumps))
                ds_report["top_state_jumps"].append({
                    "episode": ep_idx,
                    "frame": j,
                    "success": success,
                    "l2_jump": float(jumps[j]),
                })

            n = len(df)
            progress = np.linspace(0, 1, n, dtype=np.float32) if n > 1 else np.array([1.0])
            sep = np.array([sep_value(float(p), success) for p in progress])
            late = np.array([late_value(float(p), success) for p in progress])
            bucket = "success" if success else "failure"
            ds_report["value_ranges"]["outcome_separated_progress"][bucket].extend(
                [float(sep.min()), float(sep.max())]
            )
            ds_report["value_ranges"]["outcome_late_failure"][bucket].extend(
                [float(late.min()), float(late.max())]
            )

            # Uniformly sample a few states per episode for conflict diagnostics.
            take = np.linspace(0, n - 1, min(20, n), dtype=int)
            sampled_progress = progress[take]
            sampled_state = state[take]
            for svec, p in zip(sampled_state, sampled_progress):
                all_samples.append({
                    "dataset": ds_path.name,
                    "episode": ep_idx,
                    "success": success,
                    "progress": float(p),
                    "state": svec,
                    "sep_value": sep_value(float(p), success),
                    "late_value": late_value(float(p), success),
                })

        ds_report["state_min_first26"] = state_min.tolist()
        ds_report["state_max_first26"] = state_max.tolist()
        ds_report["top_state_jumps"] = sorted(
            ds_report["top_state_jumps"], key=lambda x: x["l2_jump"], reverse=True
        )[:10]
        for mode in ds_report["value_ranges"]:
            for bucket in ds_report["value_ranges"][mode]:
                vals = ds_report["value_ranges"][mode][bucket]
                ds_report["value_ranges"][mode][bucket] = (
                    [min(vals), max(vals)] if vals else None
                )

        report["datasets"].append(ds_report)

    states = np.stack([x["state"] for x in all_samples], axis=0)
    mean = states.mean(axis=0)
    std = states.std(axis=0) + 1e-6
    z = (states - mean) / std
    success_mask = np.array([x["success"] for x in all_samples], dtype=bool)
    early_mask = np.array([x["progress"] <= 0.5 for x in all_samples], dtype=bool)
    s_idx = np.where(success_mask & early_mask)[0]
    f_idx = np.where((~success_mask) & early_mask)[0]
    rng = np.random.RandomState(0)
    s_idx = rng.choice(s_idx, size=min(500, len(s_idx)), replace=False)
    f_idx = rng.choice(f_idx, size=min(500, len(f_idx)), replace=False)

    nearest = []
    if len(s_idx) and len(f_idx):
        sz = z[s_idx]
        fz = z[f_idx]
        for local_i, global_i in enumerate(f_idx):
            d = np.linalg.norm(sz - fz[local_i], axis=1)
            j = int(np.argmin(d))
            global_j = int(s_idx[j])
            fi = all_samples[int(global_i)]
            sj = all_samples[global_j]
            nearest.append({
                "distance_z_l2": float(d[j]),
                "failure": {
                    "dataset": fi["dataset"],
                    "episode": fi["episode"],
                    "progress": round(fi["progress"], 4),
                    "sep_value": round(fi["sep_value"], 4),
                    "late_value": round(fi["late_value"], 4),
                },
                "nearest_success": {
                    "dataset": sj["dataset"],
                    "episode": sj["episode"],
                    "progress": round(sj["progress"], 4),
                    "sep_value": round(sj["sep_value"], 4),
                    "late_value": round(sj["late_value"], 4),
                },
                "sep_label_gap": round(abs(fi["sep_value"] - sj["sep_value"]), 4),
                "late_label_gap": round(abs(fi["late_value"] - sj["late_value"]), 4),
            })
        nearest = sorted(nearest, key=lambda x: x["distance_z_l2"])[:20]

    report["combined"]["early_failure_to_success_nearest_state_examples"] = nearest
    report["combined"]["sample_count"] = len(all_samples)

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(OUT)
    print(json.dumps({
        "datasets": [
            {
                "path": d["path"],
                "issues_count": len(d["issues"]),
                "issues_first5": d["issues"][:5],
                "state_dim_counts": d["state_dim_counts"],
                "value_ranges": d["value_ranges"],
                "top_state_jumps_first3": d["top_state_jumps"][:3],
            }
            for d in report["datasets"]
        ],
        "nearest_conflicts_first5": nearest[:5],
    }, indent=2))


if __name__ == "__main__":
    main()
