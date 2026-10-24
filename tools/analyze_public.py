#!/usr/bin/env python3
"""CPU 离线分析：公开聚合统计或自有 CSV 的区间并集。"""

import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path


def interval_summary(intervals):
    ordered = sorted(intervals)
    if not ordered:
        return {"records": 0, "span_us": 0, "union_us": 0, "uncovered_us": 0}
    left, right = ordered[0]
    union = 0.0
    for start, end in ordered[1:]:
        if start <= right:
            right = max(right, end)
        else:
            union += right - left
            left, right = start, end
    union += right - left
    span = max(end for _, end in ordered) - ordered[0][0]
    return {
        "records": len(ordered),
        "span_us": span,
        "union_us": union,
        "uncovered_us": max(0.0, span - union),
        "sum_durations_us": sum(end - start for start, end in ordered),
    }


def analyze_csv(args):
    opener = gzip.open if args.csv_path.suffix == ".gz" else open
    intervals = []
    skipped = 0
    with opener(args.csv_path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for name in (args.start_column, args.duration_column):
            if name not in (reader.fieldnames or []):
                raise ValueError(f"Missing required column: {name}")
        for row in reader:
            try:
                start = float(row[args.start_column])
                duration = float(row[args.duration_column])
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
                skipped += 1
                continue
            intervals.append((start, start + duration))
    return {**interval_summary(intervals), "skipped_records": skipped}


def verify_public():
    root = Path(__file__).resolve().parents[1] / "data"
    training = json.loads((root / "training-summary.json").read_text())
    aggregates = training["aggregates"]
    for mode in ("baseline", "adaptive"):
        for value in aggregates[mode].values():
            if not math.isclose(statistics.median(value["values"]), value["median"], rel_tol=1e-12):
                raise ValueError("Published median differs from its underlying values")
        for seconds, tokens_per_second in zip(
            aggregates[mode]["seconds"]["values"], aggregates[mode]["tokens_per_second"]["values"]
        ):
            expected = training["nonpadding_tokens_per_run"] / seconds
            if not math.isclose(expected, tokens_per_second, rel_tol=1e-12):
                raise ValueError("Token accounting differs from published throughput")
    gain = (
        aggregates["adaptive"]["tokens_per_second"]["median"]
        / aggregates["baseline"]["tokens_per_second"]["median"] - 1
    ) * 100
    reduction = (1 - aggregates["adaptive"]["seconds"]["median"] / aggregates["baseline"]["seconds"]["median"]) * 100
    if not math.isclose(gain, training["throughput_gain_percent"], rel_tol=1e-12):
        raise ValueError("Published gain is inconsistent")
    if not math.isclose(reduction, training["training_time_reduction_percent"], rel_tol=1e-12):
        raise ValueError("Published time reduction is inconsistent")
    trace = json.loads((root / "trace-summary.json").read_text())
    counts = {
        name: [round((1 - a / b) * 100, 6) for b, a in zip(values["baseline"], values["adaptive"])]
        for name, values in trace["counts"].items()
    }
    # 重叠区间相加会重复计时；接触的区间并为连续覆盖。
    assert interval_summary([(0, 10), (5, 15), (20, 25)])["union_us"] == 20
    assert interval_summary([(0, 10), (10, 15)])["uncovered_us"] == 0
    assert interval_summary([])["records"] == 0
    return {
        "published_aggregates_verified": True,
        "throughput_gain_percent": gain,
        "training_time_reduction_percent": reduction,
        "trace_count_reduction_percent_per_rank": counts,
        "raw_trace_or_public_workload_reproduction": False,
    }


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    intervals = commands.add_parser("intervals")
    intervals.add_argument("csv_path", type=Path)
    intervals.add_argument("--start-column", default="Start Time(us)")
    intervals.add_argument("--duration-column", default="Duration(us)")
    args = parser.parse_args()
    result = analyze_csv(args) if args.command == "intervals" else verify_public()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
