#!/usr/bin/env python3
"""Verify the explicit public artifact inventory before packaging or release."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "ARTIFACT_MANIFEST.json").read_text())
    files = manifest["files_sha256"]
    for name, expected in files.items():
        path = ROOT / name
        if path.suffix in {".parquet", ".tex", ".text", ".bib", ".cls", ".sty", ".sqlite", ".pt", ".pth", ".ckpt"}:
            raise ValueError(f"Restricted artifact in public inventory: {name}")
        if set(path.relative_to(ROOT).parts) & {"manuscript", "article", "raw_data", "runs", "restricted_analysis"}:
            raise ValueError(f"Restricted directory in public inventory: {name}")
        if path.suffix == ".pdf" and not name.startswith("analysis/outputs/figures/"):
            raise ValueError(f"Only aggregate figure PDFs are public: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Public artifact checksum changed: {name}")
        if path.suffix == ".csv" and name.startswith(("analysis/", "results/")):
            with path.open() as stream:
                columns = next(csv.reader(stream))
            if set(columns) & {"gap_id", "day", "timestamp", "mask_seed", "reconnection_time", "target_time"}:
                raise ValueError(f"Individual-record field in public results: {name}")
    print(f"Verified {len(files)} allowlisted public artifacts; manuscript and individual records excluded.")


if __name__ == "__main__":
    main()
