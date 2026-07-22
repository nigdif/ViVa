import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


AUDIT = Path("logs/gt_value_curves/basket_place_value_data_audit.json")
OUT_DIR = Path("logs/state_jump_clips")
WINDOW = 40
MAX_CLIPS_PER_DATASET = 4


def put_text(frame, lines):
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 26


def render_clip(dataset_path, case):
    ep = int(case["episode"])
    jump_frame = int(case["frame"])
    success = bool(case["success"])

    parquet = dataset_path / "data/chunk-000" / f"episode_{ep:06d}.parquet"
    video = dataset_path / "videos/chunk-000/observation.images.cam_high" / f"episode_{ep:06d}.mp4"
    if not parquet.exists() or not video.exists():
        return None

    df = pd.read_parquet(parquet, columns=["observation.state"])
    state = np.stack(
        [np.asarray(x, dtype=np.float32)[:26] for x in df["observation.state"]],
        axis=0,
    )
    delta = state[jump_frame + 1] - state[jump_frame]
    top = np.argsort(np.abs(delta))[::-1][:4]
    top_text = ", ".join(
        f"d{int(i)}:{state[jump_frame, i]:.1f}->{state[jump_frame + 1, i]:.1f}"
        for i in top
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = max(0, jump_frame - WINDOW)
    end = min(len(df) - 1, jump_frame + WINDOW)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label = "success" if success else "failure"
    out = OUT_DIR / f"{dataset_path.name}_ep{ep:06d}_frame{jump_frame:04d}_{label}.mp4"
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for frame_idx in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in {jump_frame, jump_frame + 1}:
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)
        put_text(
            frame,
            [
                f"{dataset_path.name}",
                f"episode={ep} {label} frame={frame_idx} jump={jump_frame}->{jump_frame + 1}",
                f"L2 jump={case['l2_jump']:.2f}",
                top_text,
            ],
        )
        writer.write(frame)

    cap.release()
    writer.release()
    return str(out)


def main():
    audit = json.loads(AUDIT.read_text())
    outputs = []
    for dataset in audit["datasets"]:
        dataset_path = Path(dataset["path"])
        for case in dataset["top_state_jumps"][:MAX_CLIPS_PER_DATASET]:
            rendered = render_clip(dataset_path, case)
            if rendered:
                outputs.append(rendered)

    summary = OUT_DIR / "summary.json"
    summary.write_text(json.dumps({"clips": outputs}, indent=2), encoding="utf-8")
    print(summary)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
