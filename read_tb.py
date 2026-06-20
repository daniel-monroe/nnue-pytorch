"""Dump scalar series (esp. val_loss) from a Lightning TensorBoard events dir.

Usage: python read_tb.py <version_dir_or_logdir> [tag_substring]
"""
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else ""

ea = EventAccumulator(path, size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags().get("scalars", [])
print("scalar tags:", tags)
for tag in tags:
    if want and want not in tag:
        continue
    evs = ea.Scalars(tag)
    print(f"\n== {tag} ({len(evs)} points) ==")
    for e in evs:
        print(f"  step={e.step:>8}  value={e.value:.5f}")
