import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET = Path("data/adapted/lerobot_v3/basket_place_a02_03_s_76_f_28_sf_state26")
OUT = Path("logs/gt_value_curves/basket_place_a02_03_gt_values.png")
SUMMARY = Path("logs/gt_value_curves/basket_place_a02_03_gt_values_summary.json")


def progress(i, n):
    return 1.0 if n <= 1 else i / (n - 1)


def remaining(i, n):
    return 0.5 if n <= 1 else (n - i - 1) / (n - 1)


def outcome_progress(i, n, success):
    p = progress(i, n)
    return p if success else p - 1.0


def outcome_separated(i, n, success):
    p = progress(i, n)
    return 0.5 + 0.5 * p if success else -0.5 - 0.5 * p


def outcome_late_failure(i, n, success, warmup=0.5):
    p = progress(i, n)
    if success:
        return 0.5 + 0.5 * p
    if p <= warmup:
        return 0.0
    return -((p - warmup) / (1.0 - warmup))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    match = re.search(r"(?:^|_)s_(\d+)(?:_f_(\d+))?(?:_|$)", DATASET.name)
    if not match:
        raise RuntimeError(f"Cannot infer success/failure counts from {DATASET.name}")
    success_count = int(match.group(1))
    failure_count = int(match.group(2) or 0)

    episodes = []
    with (DATASET / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                episodes.append(
                    {
                        "episode_index": int(row.get("episode_index", len(episodes))),
                        "length": int(row["length"]),
                    }
                )

    selected = [0, 1, success_count]
    selected = [idx for idx in selected if idx < len(episodes)]

    modes = [
        ("remaining_progress", lambda i, n, ok: remaining(i, n)),
        ("outcome_progress", outcome_progress),
        ("outcome_separated_progress", outcome_separated),
        ("outcome_late_failure", outcome_late_failure),
    ]

    fig, axes = plt.subplots(
        len(selected), 1, figsize=(12, 3.2 * len(selected)), sharex=False
    )
    if len(selected) == 1:
        axes = [axes]

    summary = {
        "dataset": str(DATASET),
        "success_count": success_count,
        "failure_count": failure_count,
        "selected": [],
    }

    for ax, ep_idx in zip(axes, selected):
        ep = episodes[ep_idx]
        n = ep["length"]
        success = ep_idx < success_count
        xs = list(range(n))

        for name, fn in modes:
            ys = [fn(i, n, success) for i in xs]
            lw = 2.8 if name == "outcome_late_failure" else 1.5
            alpha = 1.0 if name == "outcome_late_failure" else 0.75
            ax.plot(xs, ys, label=name, linewidth=lw, alpha=alpha)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.axhline(1, color="gray", linewidth=0.5, linestyle=":", alpha=0.35)
        ax.axhline(-1, color="gray", linewidth=0.5, linestyle=":", alpha=0.35)
        ax.set_ylim(-1.08, 1.08)
        ax.set_ylabel("GT value")
        label = "success" if success else "failure"
        ax.set_title(f"episode {ep_idx} | {label} | length={n}")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

        points = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
        summary["selected"].append(
            {
                "episode_index": ep_idx,
                "success": success,
                "length": n,
                "sample_points": [
                    {
                        "frame": int(i),
                        "progress": round(progress(i, n), 4),
                        "remaining_progress": round(remaining(i, n), 4),
                        "outcome_progress": round(outcome_progress(i, n, success), 4),
                        "outcome_separated_progress": round(
                            outcome_separated(i, n, success), 4
                        ),
                        "outcome_late_failure": round(
                            outcome_late_failure(i, n, success), 4
                        ),
                    }
                    for i in points
                ],
            }
        )

    axes[-1].set_xlabel("frame index")
    fig.suptitle(
        "GT value targets used for training: basket_place_a02_03_s_76_f_28_sf_state26",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.82, 0.97])
    fig.savefig(OUT, dpi=160)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUT)
    print(SUMMARY)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
