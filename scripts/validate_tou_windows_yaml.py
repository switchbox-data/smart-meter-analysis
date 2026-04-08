#!/usr/bin/env python3
"""Validate that two YAML rate structures share identical TOU window definitions.

Loads two YAML configs, extracts their season date ranges and period hour
boundaries (stripping prices), and exits non-zero on any mismatch.

Usage::

    uv run python scripts/tmp/validate_tou_windows_yaml.py \\
        rate_structures/comed_stou_2026.yaml \\
        rate_structures/comed_dtou_2026.yaml

Exit codes:
    0 — windows are identical
    1 — mismatch found (diff printed to stdout)
    2 — input error (file not found or invalid YAML structure)
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent  # .../scripts/
sys.path.insert(0, str(_SCRIPTS_DIR))

from build_tariff_hourly_prices import compare_window_definitions, load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(
            "Usage: validate_tou_windows_yaml.py <yaml_a> <yaml_b>",
            file=sys.stderr,
        )
        return 2

    path_a, path_b = Path(args[0]), Path(args[1])

    try:
        cfg_a = load_config(path_a)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"ERROR loading {path_a}: {exc}", file=sys.stderr)
        return 2

    try:
        cfg_b = load_config(path_b)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"ERROR loading {path_b}: {exc}", file=sys.stderr)
        return 2

    ok, diff_msg = compare_window_definitions(
        cfg_a,
        cfg_b,
        name_a=str(path_a),
        name_b=str(path_b),
    )
    if ok:
        print(f"OK — window definitions match: {path_a.name} <-> {path_b.name}")
        return 0

    print(diff_msg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
