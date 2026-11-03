#!/usr/bin/env python3
"""Derive complete test-day calendars from checksum-verified source timestamps.

Only calendar metadata is exported. Power and counter values are never read.
This mirrors the frozen nearest-slot and elapsed-time fold rules, without using
the dates of observed evaluation episodes to truncate the sampling calendars.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

FRACTIONS = ((.40, .50, .65), (.55, .65, .80), (.70, .80, 1.0))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def derive_site(config, raw_root):
    zone = ZoneInfo(config["timestamp"]["timezone"])
    step = int(config["alignment"]["nominal_seconds"])
    tolerance = float(config["alignment"]["tolerance_seconds"])
    accepted, rejected, sources = [], 0, []
    for source in config["files"]:
        path = Path(raw_root) / source["path"]
        actual = sha256(path)
        if actual != source["sha256"]:
            raise ValueError(f"Source checksum mismatch for {source['path']}")
        records = 0
        with path.open(encoding=source.get("encoding", "utf-8-sig"), newline="") as stream:
            for row in csv.DictReader(stream, delimiter=source.get("delimiter", ";")):
                records += 1
                timestamp = datetime.fromisoformat(row[config["timestamp"]["column"]])
                if timestamp.tzinfo is None:
                    first, second = timestamp.replace(tzinfo=zone, fold=0), timestamp.replace(tzinfo=zone, fold=1)
                    if first.utcoffset() != second.utcoffset():
                        raise ValueError("Ambiguous or nonexistent local timestamp")
                    timestamp = first
                seconds = timestamp.timestamp()
                slot = math.floor(seconds / step + .5) * step
                if abs(slot - seconds) > tolerance:
                    rejected += 1
                    continue
                accepted.append(datetime.fromtimestamp(slot, timezone.utc).astimezone(timestamp.tzinfo))
        sources.append({"path": source["path"], "sha256": actual, "source_rows": records})
    if len(accepted) < 2:
        raise ValueError("Insufficient aligned timestamps")
    first, last = min(accepted), max(accepted)
    span = last - first
    folds = {}
    for fold, (train, validation, test) in enumerate(FRACTIONS, 1):
        train_end, validation_end, test_end = (first + span * q for q in (train, validation, test))
        # This reproduces the original reporter's inclusive day calendar. The
        # fold's final instant remains exclusive except for fold 3. An empty
        # endpoint date still belongs to the day resampling support.
        days = [(validation_end.date() + timedelta(days=i)).isoformat()
                for i in range((test_end.date() - validation_end.date()).days + 1)]
        folds[str(fold)] = {"fold_id": fold, "start": first.isoformat(),
                           "train_end_exclusive": train_end.isoformat(),
                           "validation_end_exclusive": validation_end.isoformat(),
                           "test_end": test_end.isoformat(), "inclusive_test_end": fold == 3,
                           "calendar_days": days}
    return {"sources": sources, "timezone": str(zone), "nominal_seconds": step,
            "alignment_tolerance_seconds": tolerance, "rejected_timestamps": rejected,
            "first_grid_time": first.isoformat(), "last_grid_time": last.isoformat(),
            "folds": folds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing the source paths in the frozen lock")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    lock_path = args.run_dir / "protocol.lock.json"
    lock = json.loads(lock_path.read_text())
    sites = {site: derive_site(lock["datasets"][site]["config"], args.raw_root) for site in lock["sites"]}
    for site in sites:
        if sites[site]["rejected_timestamps"] != lock["datasets"][site]["alignment_audit"]["n_rejected"]:
            raise ValueError(f"Alignment rejection count changed for {site}")
    output = {"schema_version": 1, "analysis_revision": "forecast-common-target-v2",
              "run_id": args.run_dir.name, "protocol_json_sha256": sha256(lock_path),
              "derivation": "Verified raw timestamp files; nearest-slot floor(t/300+0.5) within 90 seconds; full-grid endpoints; elapsed-time fold fractions; inclusive calendar dates from validation_end_exclusive through test_end.",
              "reference_implementation": ["src/energygap/data/timebase.py:nearest_slot", "src/energygap/protocols/splits.py:rolling_origin_splits", "src/energygap/reporting/conference.py:_calendars"],
              "derived_from_episode_dates": False, "raw_values_exported": False,
              "sites": sites}
    encoded = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json.exists() and args.output_json.read_text() != encoded:
        raise FileExistsError("Existing calendar manifest differs; create a new revision")
    args.output_json.write_text(encoded)
    print(f"Verified three raw source hashes. Calendar metadata written to {args.output_json}")


if __name__ == "__main__":
    main()
