"""Summarize local TensorBoard scalar event files without changing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize(path: Path) -> dict:
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    scalars = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        scalars[tag] = {
            "count": len(events),
            "first_step": events[0].step,
            "last_step": events[-1].step,
            "last_value": events[-1].value,
            "maximum": max(event.value for event in events),
            "max_step": max(events, key=lambda event: event.value).step,
            "minimum": min(event.value for event in events),
            "min_step": min(events, key=lambda event: event.value).step,
            "selected_steps": {str(event.step): event.value for event in events
                               if event.step in (5, 18, 20, 22, 34, 65, 79)},
        }
    return {"run": path.parent.name, "file": path.name, "bytes": path.stat().st_size,
            "scalars": scalars, "other_tags": {key: value for key, value in acc.Tags().items()
                                               if key != "scalars" and value}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    if not directory.is_dir():
        parser.error(f"Not a directory: {directory}")
    paths = sorted(directory.rglob("events.out.tfevents*"))
    if not paths:
        parser.error(f"No event files under {directory}")
    print(json.dumps([summarize(path) for path in paths], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
