"""Build a leakage-audited timestamped MBHT benchmark from raw events.

The first supported source is Retailrocket v4.  The builder creates four
chronological partitions and never attaches timestamps to the legacy splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


BEHAVIOR_IDS = {"transaction": "0", "view": "1", "addtocart": "2"}
OUTPUT_HEADER = (
    "session_id:token\titem_id_list:token_seq\titem_type_list:token_seq\t"
    "item_time_list:float_seq\titem_id:token\ttarget_delta:float\t"
    "target_repeat:float\tvisitor_id:token\n"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["retailrocket"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="retail_time_beh")
    parser.add_argument(
        "--max-history",
        type=int,
        default=0,
        help=(
            "Maximum prefix length; 0 keeps the complete observable prefix "
            "and is the paper-aligned default."
        ),
    )
    parser.add_argument("--min-history", type=int, default=2)
    parser.add_argument("--min-item-train-events", type=int, default=5)
    return parser.parse_args()


def partition_for(timestamp: int, minimum: int, maximum: int) -> str:
    fraction = (timestamp - minimum) / max(maximum - minimum, 1)
    if fraction < 0.70:
        return "train"
    if fraction < 0.80:
        return "valid"
    if fraction < 0.90:
        return "development"
    return "confirmation"


def serialize_sample(sample: dict[str, object]) -> str:
    return "\t".join(
        [
            str(sample["session_id"]),
            " ".join(map(str, sample["items"])),
            " ".join(sample["behaviors"]),
            " ".join(f"{value:.6f}" for value in sample["times"]),
            str(sample["target"]),
            f"{sample['target_delta']:.6f}",
            f"{sample['target_repeat']:.1f}",
            str(sample["visitor_id"]),
        ]
    ) + "\n"


def build_retailrocket(args: argparse.Namespace) -> dict[str, object]:
    expected_header = "timestamp,visitorid,event,itemid,transactionid"
    with args.input.open("r", encoding="utf-8") as source:
        actual_header = source.readline().strip()
    if actual_header != expected_header:
        raise ValueError(f"Unexpected Retailrocket header: {actual_header!r}")

    frame = pd.read_csv(
        args.input,
        usecols=["timestamp", "visitorid", "event", "itemid"],
        dtype={
            "timestamp": "int64",
            "visitorid": "int64",
            "event": "category",
            "itemid": "int64",
        },
    )
    original_rows = int(len(frame))
    frame["original_row"] = range(original_rows)
    frame = frame.drop_duplicates(
        subset=["timestamp", "visitorid", "event", "itemid"], keep="first"
    )
    frame = frame[frame["event"].isin(BEHAVIOR_IDS)].copy()
    minimum = int(frame["timestamp"].min())
    maximum = int(frame["timestamp"].max())
    train_cutoff = minimum + int(0.70 * (maximum - minimum))
    train_item_counts = frame.loc[
        frame["timestamp"] < train_cutoff, "itemid"
    ].value_counts()
    retained_items = train_item_counts[
        train_item_counts >= args.min_item_train_events
    ].index
    frame = frame[frame["itemid"].isin(retained_items)].copy()
    frame.sort_values(
        ["visitorid", "timestamp", "original_row"], kind="mergesort", inplace=True
    )

    samples: dict[str, list[dict[str, object]]] = {
        "train": [],
        "valid": [],
        "development": [],
        "confirmation": [],
    }
    session_id = 1

    for visitor_id, events in frame.groupby("visitorid", sort=False, observed=True):
        timestamps = events["timestamp"].to_numpy()
        items = events["itemid"].to_numpy()
        behaviors = events["event"].astype("object").to_numpy()
        target_indices = [
            index
            for index, behavior in enumerate(behaviors)
            if behavior == "transaction" and index >= args.min_history
        ]
        # Every retained occurrence of the target behavior is a supervision
        # point. Each instance uses only the chronological prefix before its
        # own cutoff, so multiple targets from one visitor do not leak future
        # events into one another.
        for target_index in target_indices:
            split = partition_for(int(timestamps[target_index]), minimum, maximum)
            history_start = (
                max(0, target_index - args.max_history)
                if args.max_history > 0
                else 0
            )
            history_items = items[history_start:target_index].tolist()
            history_behaviors = [
                BEHAVIOR_IDS[str(value)]
                for value in behaviors[history_start:target_index]
            ]
            history_timestamps = timestamps[history_start:target_index]
            origin = int(history_timestamps[0])
            relative_hours = [
                (int(value) - origin) / 3_600_000.0 for value in history_timestamps
            ]
            target = int(items[target_index])
            target_delta = (
                int(timestamps[target_index]) - int(history_timestamps[-1])
            ) / 3_600_000.0
            sample = {
                "session_id": session_id,
                "items": history_items,
                "behaviors": history_behaviors,
                "times": relative_hours,
                "target": target,
                "target_delta": max(target_delta, 0.0),
                "target_repeat": float(target in history_items),
                "visitor_id": int(visitor_id),
            }
            samples[split].append(sample)
            session_id += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_hashes: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    repeat_rates: dict[str, float] = {}
    for split, records in samples.items():
        output_path = args.output_dir / f"{args.dataset_name}.{split}.inter"
        with output_path.open("w", encoding="utf-8", newline="") as target:
            target.write(OUTPUT_HEADER)
            for record in records:
                target.write(serialize_sample(record))
        split_hashes[split] = sha256_file(output_path)
        split_counts[split] = len(records)
        repeat_rates[split] = (
            sum(float(record["target_repeat"]) for record in records) / len(records)
            if records
            else 0.0
        )

    manifest = {
        "dataset_name": args.dataset_name,
        "source": "Retailrocket v4",
        "source_path": str(args.input.resolve()),
        "source_sha256": sha256_file(args.input),
        "original_rows": original_rows,
        "deduplicated_rows": int(len(frame)),
        "timestamp_unit": "Unix milliseconds",
        "output_time_unit": "hours from first retained history event",
        "split_rule": "absolute event-time range: 70/10/10/10",
        "tie_rule": "original row order",
        "max_history": args.max_history if args.max_history > 0 else None,
        "history_policy": (
            "complete observable prefix"
            if args.max_history <= 0
            else f"most recent {args.max_history} events"
        ),
        "min_history": args.min_history,
        "target_supervision_policy": "every eligible target-behavior occurrence",
        "min_item_train_events": args.min_item_train_events,
        "retained_item_count": int(len(retained_items)),
        "split_counts": split_counts,
        "target_repeat_rates": repeat_rates,
        "split_sha256": split_hashes,
    }
    manifest_path = args.output_dir / f"{args.dataset_name}.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2, sort_keys=True)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_retailrocket(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
