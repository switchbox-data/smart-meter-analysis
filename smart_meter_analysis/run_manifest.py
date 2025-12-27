# smart_meter_analysis/run_manifest.py
from __future__ import annotations

import json
import platform
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Stage2RunManifest:
    """Lightweight, reproducible metadata for a Stage 2 run."""

    created_utc: str
    python: str
    platform: str
    git_commit: str | None
    command: str
    output_dir: str

    # Inputs
    clusters_path: str
    crosswalk_path: str
    census_cache_path: str

    # Key parameters
    baseline_cluster: str | int | None
    min_obs_per_bg: int
    alpha: float
    weight_column: str | None

    # Predictor handling
    predictors_total_detected: int | None
    predictors_used: list[str]
    predictors_excluded_all_null: list[str]

    # Dataset sizes
    block_groups_total: int | None
    block_groups_after_min_obs: int | None
    block_groups_after_drop_null_predictors: int | None

    # Outputs
    regression_data_path: str | None
    regression_report_path: str | None
    run_log_path: str | None


def _safe_git_commit(repo_root: Path) -> str | None:
    """Best-effort git commit retrieval without depending on GitPython."""
    try:
        import subprocess

        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
        return None
    except Exception:
        return None


def write_stage2_manifest(
    *,
    output_dir: str | Path,
    command: str,
    repo_root: str | Path | None = None,
    clusters_path: str | Path,
    crosswalk_path: str | Path,
    census_cache_path: str | Path,
    baseline_cluster: str | int | None,
    min_obs_per_bg: int,
    alpha: float,
    weight_column: str | None,
    predictors_detected: int | None,
    predictors_used: Iterable[str],
    predictors_excluded_all_null: Iterable[str],
    block_groups_total: int | None,
    block_groups_after_min_obs: int | None,
    block_groups_after_drop_null_predictors: int | None,
    regression_data_path: str | Path | None,
    regression_report_path: str | Path | None,
    run_log_path: str | Path | None,
) -> Path:
    """Writes:
    - stage2_manifest.json : run metadata
    - predictors_used.txt  : final predictor list (stable across runs)
    - predictors_excluded_all_null.txt : excluded predictors with 100% nulls
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    predictors_used_list = sorted(dict.fromkeys(list(predictors_used)))
    excluded_list = sorted(dict.fromkeys(list(predictors_excluded_all_null)))

    # Persist predictor lists (the key stability artifact)
    (out / "predictors_used.txt").write_text("\n".join(predictors_used_list) + "\n", encoding="utf-8")
    (out / "predictors_excluded_all_null.txt").write_text("\n".join(excluded_list) + "\n", encoding="utf-8")

    # Build manifest
    created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    repo_root_path = Path(repo_root) if repo_root is not None else None
    git_commit = _safe_git_commit(repo_root_path) if repo_root_path else None

    manifest = Stage2RunManifest(
        created_utc=created_utc,
        python=sys.version.replace("\n", " "),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        git_commit=git_commit,
        command=command,
        output_dir=str(out),
        clusters_path=str(clusters_path),
        crosswalk_path=str(crosswalk_path),
        census_cache_path=str(census_cache_path),
        baseline_cluster=baseline_cluster,
        min_obs_per_bg=min_obs_per_bg,
        alpha=alpha,
        weight_column=weight_column,
        predictors_total_detected=predictors_detected,
        predictors_used=predictors_used_list,
        predictors_excluded_all_null=excluded_list,
        block_groups_total=block_groups_total,
        block_groups_after_min_obs=block_groups_after_min_obs,
        block_groups_after_drop_null_predictors=block_groups_after_drop_null_predictors,
        regression_data_path=str(regression_data_path) if regression_data_path else None,
        regression_report_path=str(regression_report_path) if regression_report_path else None,
        run_log_path=str(run_log_path) if run_log_path else None,
    )

    manifest_path = out / "stage2_manifest.json"
    manifest_path.write_text(json.dumps(manifest.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def write_run_manifest(
    *,
    output_dir: str | Path,
    command: str,
    repo_root: str | Path | None = None,
    run_name: str,
    year_month: str,
    num_files: int,
    sample_days: int,
    sample_households: int | None,
    day_strategy: str,
    k_min: int,
    k_max: int,
    n_init: int,
) -> Path:
    """Write a manifest file for a pipeline run.

    Records all parameters and metadata for reproducibility.
    Similar to Stage2RunManifest but for the full pipeline orchestrator.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    repo_root_path = Path(repo_root) if repo_root is not None else None
    git_commit = _safe_git_commit(repo_root_path) if repo_root_path else None

    manifest = {
        "created_utc": created_utc,
        "python": sys.version.replace("\n", " "),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "git_commit": git_commit,
        "command": command,
        "run_name": run_name,
        "output_dir": str(out),
        "parameters": {
            "year_month": year_month,
            "num_files": num_files,
            "sample_days": sample_days,
            "sample_households": sample_households,
            "day_strategy": day_strategy,
            "k_min": k_min,
            "k_max": k_max,
            "n_init": n_init,
        },
    }

    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def load_persisted_predictors(output_dir: str | Path) -> list[str] | None:
    """If `predictors_used.txt` exists, return that list to reuse exactly on a new run.
    This is the "stable across runs" option.
    """
    p = Path(output_dir) / "predictors_used.txt"
    if not p.exists():
        return None
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]
