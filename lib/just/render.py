#!/usr/bin/env python3
"""Render a Quarto report with baseline snapshot for diffing.

Snapshots current docs/ to .diff/baseline/ before rendering, inlines
any SVG figures into the HTML (removing the standalone .svg files),
and cleans up .ipynb artifacts from docs/ after rendering.

Usage (from a report directory):
    uv run python -m lib.just.render
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASELINE = Path(".diff/baseline")
INLINE_SVGS = Path("../.style/inline_svgs.py")


def _clean_quarto_artifacts(docs: Path) -> None:
    """Remove Quarto-generated intermediates from the source and docs trees."""
    removed = 0

    for pattern in ("**/*.out.ipynb", "**/*.embed.ipynb"):
        for f in Path(".").glob(pattern):
            f.unlink()
            removed += 1

    for f in docs.rglob("*.ipynb"):
        f.unlink()
        removed += 1

    if removed:
        print(f"🗑️  Removed {removed} .ipynb artifact(s)")


def main() -> None:
    docs = Path("docs")

    if docs.is_dir():
        print("📸 Snapshotting docs/ → .diff/baseline/")
        if BASELINE.exists():
            shutil.rmtree(BASELINE)
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(docs, BASELINE)

    render_failed = False
    print("📖 Rendering Quarto project...")
    try:
        result = subprocess.run(["quarto", "render", "."])
        if result.returncode != 0:
            render_failed = True
            print("💥 Quarto render failed!", file=sys.stderr)

        if not render_failed and INLINE_SVGS.exists():
            print("🖼️  Inlining SVGs into HTML...")
            result = subprocess.run([sys.executable, str(INLINE_SVGS), "docs"])
            if result.returncode != 0:
                render_failed = True

            svgs = list(docs.rglob("*.svg"))
            if svgs:
                for f in svgs:
                    f.unlink()
                print(f"🗑️  Removed {len(svgs)} standalone SVG file(s)")
    finally:
        _clean_quarto_artifacts(docs)

    if render_failed:
        sys.exit(1)

    print("✅ Render complete!")

    index = docs / "index.html"
    if index.exists():
        if platform.system() == "Darwin":
            subprocess.Popen(["open", str(index)])
        else:
            print(f"👉 Open {index} to view")


if __name__ == "__main__":
    main()
