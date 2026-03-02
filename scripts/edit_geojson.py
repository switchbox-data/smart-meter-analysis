import argparse
import glob
import json
import os
import shutil
from typing import Any, Optional

ANCHOR_MIN_ID = "__DOMAIN_ANCHOR_MIN__"
ANCHOR_MAX_ID = "__DOMAIN_ANCHOR_MAX__"


def _load_bound_sym(folder: str, bound_override: Optional[float]) -> float:
    if bound_override is not None:
        return float(bound_override)

    range_path = os.path.join(folder, "range_global.json")
    if os.path.exists(range_path):
        with open(range_path) as f:
            meta = json.load(f)
        if "bound_sym" not in meta:
            raise RuntimeError(f"{range_path} exists but has no 'bound_sym' key.")
        return float(meta["bound_sym"])

    # Fallback: last known bound from your run (range_global.json you pasted)
    return 24.143564317219816


def _has_anchor(features: list[dict[str, Any]]) -> bool:
    for feat in features:
        props = feat.get("properties") or {}
        geoid = props.get("geoid_bg")
        if geoid in (ANCHOR_MIN_ID, ANCHOR_MAX_ID):
            return True
    return False


def _make_anchor(geoid_bg: str, value: float) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": None,
        "properties": {
            "geoid_bg": geoid_bg,
            "n_households": 0,
            "mean_delta": None,
            "mean_delta_cap_global_sym": value,
            "is_domain_anchor": True,
        },
    }


def _warn_on_invariant(path: str, features: list[dict[str, Any]]) -> None:
    # Fail-loud would be better upstream; here we just warn.
    bad = 0
    for feat in features:
        props = feat.get("properties") or {}
        geoid = props.get("geoid_bg")
        if geoid in (ANCHOR_MIN_ID, ANCHOR_MAX_ID):
            continue
        nh = props.get("n_households")
        v = props.get("mean_delta_cap_global_sym")
        try:
            nh_i = int(nh) if nh is not None else 0
        except Exception:
            nh_i = 0
        if nh_i > 0:
            try:
                float(v)
            except Exception:
                bad += 1
    if bad:
        print(
            f"[WARN] {os.path.basename(path)}: {bad} features have n_households>0 but invalid mean_delta_cap_global_sym"
        )


def process_file(path: str, bound_sym: float, make_backup: bool) -> bool:
    with open(path) as f:
        data = json.load(f)

    if data.get("type") != "FeatureCollection":
        print(f"[SKIP] {path}: not a FeatureCollection")
        return False

    features = data.get("features")
    if not isinstance(features, list):
        print(f"[SKIP] {path}: missing/invalid 'features' array")
        return False

    if _has_anchor(features):
        print(f"[OK]   {os.path.basename(path)}: anchors already present (skipping)")
        _warn_on_invariant(path, features)
        return False

    features.append(_make_anchor(ANCHOR_MIN_ID, -bound_sym))
    features.append(_make_anchor(ANCHOR_MAX_ID, bound_sym))

    _warn_on_invariant(path, features)

    if make_backup:
        shutil.copy2(path, path + ".bak")

    # Write compact JSON (smaller files, fast upload). Use indent=2 if you prefer readable.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, path)

    print(f"[WRITE] {os.path.basename(path)}: added anchors ±{bound_sym:.6f}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Append ±bound_sym domain anchor features to all GeoJSONs in a folder.")
    ap.add_argument("folder", help="Folder containing *.geojson files (and optionally range_global.json).")
    ap.add_argument(
        "--bound-sym", type=float, default=None, help="Override bound_sym (else read range_global.json if present)."
    )
    ap.add_argument("--no-backup", action="store_true", help="Do not write .bak backups.")
    args = ap.parse_args()

    folder = os.path.abspath(os.path.expanduser(args.folder))
    if not os.path.isdir(folder):
        raise SystemExit(f"Not a directory: {folder}")

    bound_sym = _load_bound_sym(folder, args.bound_sym)
    geojson_paths = sorted(glob.glob(os.path.join(folder, "*.geojson")))

    if not geojson_paths:
        raise SystemExit(f"No *.geojson files found in {folder}")

    changed = 0
    for p in geojson_paths:
        if process_file(p, bound_sym, make_backup=(not args.no_backup)):
            changed += 1

    print(f"Done. Changed {changed} files out of {len(geojson_paths)}. bound_sym={bound_sym}")


if __name__ == "__main__":
    main()
