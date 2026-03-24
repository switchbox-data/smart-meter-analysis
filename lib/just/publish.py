#!/usr/bin/env python3
"""Publish rendered docs to the root docs/ directory for GitHub Pages.

Copies docs/ to ../../docs/<project_name>/, where <project_name> is
the name of the current directory (i.e. the report project).

Usage (from a report directory):
    uv run python -m lib.just.publish
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> None:
    docs = Path("docs")
    if not docs.is_dir():
        print("❌ No docs/ directory. Run 'just render' first.", file=sys.stderr)
        sys.exit(1)

    project = Path.cwd().name
    pub_path = Path("../..") / "docs" / project

    print(f"📦 Publishing docs/ → {pub_path}")
    if pub_path.exists():
        shutil.rmtree(pub_path)
    shutil.copytree(docs, pub_path)
    print(f"✅ Published to {pub_path}")


if __name__ == "__main__":
    main()
