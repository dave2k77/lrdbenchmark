#!/usr/bin/env python3
"""CLI helper to download pretrained LRDBenchmark assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lrdbenchmark import assets


def _format_manifest(manifest):
    rows = []
    for key, entries in manifest.items():
        rows.append(
            {
                "key": key,
                "filenames": [artifact.filename for artifact in entries],
                "description": entries[0].description,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pretrained LRDBenchmark model artifacts with checksum validation."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model keys to download (defaults to all).",
        choices=sorted(set(assets.list_available_artifacts().keys())),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available model keys and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output manifest/listing as JSON for machine consumption.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        manifest_rows = _format_manifest(assets.list_available_artifacts())
        if args.json:
            print(json.dumps(manifest_rows, indent=2))
        else:
            print("Available pretrained model keys:")
            for row in manifest_rows:
                print(f" - {row['key']}: {', '.join(row['filenames'])}")
        return 0

    resolved = assets.ensure_all_artifacts(args.models)
    if not resolved:
        print("No artifacts were downloaded. Check the logs for details.", file=sys.stderr)
        return 1

    cache_hint = assets.get_artifacts_cache_hint()
    if args.json:
        payload = {key: str(path) for key, path in resolved.items()}
        payload["_cache_dir"] = cache_hint
        print(json.dumps(payload, indent=2))
    else:
        print("Downloaded / verified pretrained models:")
        for key, path in resolved.items():
            print(f" - {key}: {path}")
        print(f"Artifacts cached under: {cache_hint}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

