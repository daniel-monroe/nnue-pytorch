"""One-line live status for the baseline vs pawn A/B: held-out val_loss for both,
plus pawn-feature weight magnitudes (mmap-read from the pawn checkpoint).

Usage: python report_ab.py
"""
import glob
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = "logs_baseline"
PAWN_DIR = "logs_pawn4k"


def latest_version(logdir):
    vs = sorted(
        glob.glob(os.path.join(logdir, "lightning_logs", "version_*")),
        key=os.path.getmtime,
    )
    return vs[-1] if vs else None


def last_val(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    v = latest_version(logdir)
    if not v:
        return None, None
    try:
        ea = EventAccumulator(v, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        tag = "val_loss_epoch" if "val_loss_epoch" in tags else (
            "val_loss" if "val_loss" in tags else None
        )
        if not tag:
            return None, None
        evs = ea.Scalars(tag)
        return evs[-1].value, evs[-1].step
    except Exception:
        return None, None


def pawn_weight_stats():
    import torch

    v = latest_version(PAWN_DIR)
    if not v:
        return None
    ck = os.path.join(v, "checkpoints", "last.ckpt")
    if not os.path.exists(ck):
        return None
    try:
        sd = torch.load(ck, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    except Exception:
        return None
    out = {}
    for idx, name in ((2, "L"), (3, "R")):
        k = f"model.input.features.{idx}.weight"
        if k not in sd:
            return None
        w = sd[k][:, :1024].float()  # L1 columns; PSQT cols are zero
        out[name] = (w.abs().mean().item(), w.abs().max().item())
    return out


def main():
    bv, bs = last_val(BASE_DIR)
    pv, ps = last_val(PAWN_DIR)
    pw = pawn_weight_stats()

    def fmt(v, s):
        return f"{v:.6f}@ep{s}" if v is not None else "  (pending) "

    line = f"BASE val={fmt(bv, bs)} | PAWN val={fmt(pv, ps)}"
    if pw:
        line += (
            f" | pawnL mean|w|={pw['L'][0]:.5f} max={pw['L'][1]:.4f}"
            f" | pawnR mean|w|={pw['R'][0]:.5f} max={pw['R'][1]:.4f}"
        )
    else:
        line += " | pawn weights (pending)"
    print(line, flush=True)


if __name__ == "__main__":
    main()
