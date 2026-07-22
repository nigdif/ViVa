#!/usr/bin/env python3
"""Compare remaining-progress GT vs model predictions: pick vs place."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from render_eval_batch import resolve_state_stats
from visualization import (
    build_episode_table,
    infer_episode,
    load_model,
)
from viva_dataset import VivaDataset


def gt_remaining(ep_len: int) -> np.ndarray:
    if ep_len <= 1:
        return np.array([0.0], dtype=np.float32)
    t = np.arange(ep_len, dtype=np.float32)
    return (ep_len - t - 1) / (ep_len - 1)


def curve_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    disp = -pred  # visualization convention
    gt_disp = -gt
    return {
        "mae": float(np.mean(np.abs(pred - gt))),
        "pred_std": float(np.std(pred)),
        "pred_range": float(pred.max() - pred.min()),
        "disp_range": float(disp.max() - disp.min()),
        "corr_with_gt": float(np.corrcoef(pred, gt)[0, 1]) if len(pred) > 1 else 1.0,
        "first": float(pred[0]),
        "last": float(pred[-1]),
        "gt_first": float(gt[0]),
        "gt_last": float(gt[-1]),
        "disp_first": float(disp[0]),
        "disp_last": float(disp[-1]),
    }


def run_one(label, args, data_path, checkpoint, config_path, episode, device):
    config = OmegaConf.load(config_path)
    state_stats = resolve_state_stats(
        argparse.Namespace(
            state_txt=None,
            train_data_path=None,
            data_path=data_path,
        ),
        config,
    )
    dataset = VivaDataset(
        data_paths=[data_path],
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        state_stats=state_stats,
        state_dim=int(getattr(config.common, "state_dim", 14)),
        value_target_config=getattr(config.dataset, "value_target", None),
        state_config=getattr(config.dataset, "state", None),
        require_success_labels=False,
    )
    ep_info = next(
        ep for ep in build_episode_table(dataset) if ep["episode_index"] == episode
    )
    gt = gt_remaining(ep_info["episode_length"])

    model = load_model(checkpoint, config, device)
    t5 = None
    t5_path = args.t5_embedding
    if t5_path and os.path.exists(t5_path):
        import torch
        t5 = torch.load(t5_path, map_location="cpu")

    pred = infer_episode(
        model, dataset, ep_info, t5, device, args.num_inference_steps,
    )
    stats = curve_stats(pred, gt)
    stats.update({
        "label": label,
        "data_path": data_path,
        "episode": episode,
        "episode_length": int(ep_info["episode_length"]),
    })
    del model
    import torch
    torch.cuda.empty_cache()
    return pred, gt, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--output", default="logs/eval/remaining_pick_vs_place_compare.json")
    parser.add_argument("--t5_pick", default="data/t5_embedding/basket_pick.pt")
    parser.add_argument("--t5_place", default="data/t5_embedding/basket_place.pt")
    args = parser.parse_args()

    cases = [
        {
            "label": "pick_remaining_ckpt5000_ep0",
            "data_path": "data/adapted/lerobot_v3/basket_pick_0428_s_38_state26",
            "checkpoint": "checkpoints/basket_pick_remaining_state26/checkpoint_step_5000",
            "config": "config/train_basket_pick_remaining_state26.yaml",
            "episode": 0,
            "t5": args.t5_pick,
        },
        {
            "label": "place_remaining_ckpt5976_success_ep0",
            "data_path": "data/adapted/lerobot_v3/basket_place_a02_04_s_116_f_24_sf_state26",
            "checkpoint": "checkpoints/basket_place_remaining_state26/checkpoint_step_5976",
            "config": "config/train_basket_place_remaining_state26.yaml",
            "episode": 0,
            "t5": args.t5_place,
        },
        {
            "label": "place_remaining_ckpt5976_failure_ep116",
            "data_path": "data/adapted/lerobot_v3/basket_place_a02_04_s_116_f_24_sf_state26",
            "checkpoint": "checkpoints/basket_place_remaining_state26/checkpoint_step_5976",
            "config": "config/train_basket_place_remaining_state26.yaml",
            "episode": 116,
            "t5": args.t5_place,
        },
    ]

    results = []
    curves = {}
    for case in cases:
        print(f"\n=== {case['label']} ===")
        args.t5_embedding = case["t5"]
        pred, gt, stats = run_one(
            case["label"], args,
            case["data_path"], case["checkpoint"], case["config"],
            case["episode"], args.device,
        )
        print(json.dumps(stats, indent=2))
        results.append(stats)
        curves[case["label"]] = {
            "pred": pred.tolist(),
            "gt": gt.tolist(),
            "display_pred": (-pred).tolist(),
        }

    # Cross-compare place success vs failure predictions
    p0 = np.array(curves["place_remaining_ckpt5976_success_ep0"]["pred"])
    p116 = np.array(curves["place_remaining_ckpt5976_failure_ep116"]["pred"])
    t = min(len(p0), len(p116))
    place_sf_mae = float(np.mean(np.abs(p0[:t] - p116[:t])))
    results.append({
        "label": "place_success_vs_failure_pred_mae",
        "mae_between_preds": place_sf_mae,
        "note": "Lower = less visual discrimination between success/failure",
    })

    out = {"cases": results, "curves": curves}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
