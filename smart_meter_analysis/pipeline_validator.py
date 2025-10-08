# smart_meter_analysis/pipeline_validator.py
"""
Data quality validation at each pipeline step.
Ensures no data loss and correct transformations.
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


class PipelineValidator:
    """Track data through pipeline transformations"""

    def __init__(self) -> None:
        self.checkpoints: dict[str, dict[str, object]] = {}

    def checkpoint(
        self,
        step_name: str,
        df: pl.DataFrame,
        expected_cols: list[str] | None = None,
        required_cols: list[str] | None = None,
    ) -> dict[str, object]:  # Fixed: Dict -> dict
        """Validate data at a pipeline checkpoint."""
        report: dict[str, object] = {
            "step": step_name,
            "status": "PASS",
            "rows": df.height,
            "columns": len(df.columns),
            "issues": [],
            "warnings": [],
        }

        # Check required columns
        if required_cols:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                report["status"] = "FAIL"
                report["issues"].append(f"Missing required columns: {missing}")  # type: ignore

        # Check expected columns
        if expected_cols:
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                report["warnings"].append(f"Missing expected columns: {missing}")  # type: ignore

        # Check for nulls in key columns
        if required_cols:
            for col in required_cols:
                if col in df.columns:
                    null_count = df[col].null_count()
                    if null_count > 0:
                        pct = 100 * null_count / df.height
                        if pct > 50:
                            report["issues"].append(f"{col}: {null_count:,} nulls ({pct:.1f}%)")  # type: ignore
                        elif pct > 10:
                            report["warnings"].append(f"{col}: {null_count:,} nulls ({pct:.1f}%)")  # type: ignore

        # Check for duplicates on key columns
        if "account_identifier" in df.columns and "datetime" in df.columns:
            n_unique = df.select(["account_identifier", "datetime"]).unique().height
            if n_unique < df.height:
                duplicates = df.height - n_unique
                report["issues"].append(f"Found {duplicates:,} duplicate account/datetime rows")  # type: ignore

        # Store checkpoint
        self.checkpoints[step_name] = report

        # Log results
        if report["status"] == "FAIL":
            logger.error(f"❌ {step_name}: FAILED validation")
            for issue in report["issues"]:  # type: ignore
                logger.error(f"  - {issue}")
        else:
            logger.info(f"✅ {step_name}: {df.height:,} rows, {len(df.columns)} cols")

        if report["warnings"]:
            for warning in report["warnings"]:  # type: ignore
                logger.warning(f"  ⚠️  {warning}")

        return report

    def compare_checkpoints(self, step1: str, step2: str) -> dict[str, object]:  # Fixed: Dict -> dict
        """Compare two checkpoints to detect data loss"""
        if step1 not in self.checkpoints or step2 not in self.checkpoints:
            return {"status": "ERROR", "message": "Checkpoint not found"}

        cp1 = self.checkpoints[step1]
        cp2 = self.checkpoints[step2]

        row_change = cp2["rows"] - cp1["rows"]  # type: ignore[operator]
        row_pct = 100 * row_change / cp1["rows"] if cp1["rows"] > 0 else 0  # type: ignore[operator]

        report: dict[str, object] = {
            "from": step1,
            "to": step2,
            "row_change": row_change,
            "row_change_pct": row_pct,
            "status": "OK",
        }

        # Flag significant changes
        if abs(row_pct) > 10:
            report["status"] = "WARNING"
            report["message"] = f"Rows changed by {row_pct:.1f}%"

        logger.info(f"📊 {step1} → {step2}: {row_change:+,} rows ({row_pct:+.1f}%)")

        return report

    def summary(self) -> None:
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("PIPELINE VALIDATION SUMMARY")
        print("=" * 60)

        for step_name, report in self.checkpoints.items():
            status_icon = "✅" if report["status"] == "PASS" else "❌"
            print(f"\n{status_icon} {step_name}")
            print(f"  Rows: {report['rows']:,}")
            print(f"  Columns: {report['columns']}")

            if report["issues"]:
                print("  Issues:")
                for issue in report["issues"]:  # type: ignore[attr-defined]
                    print(f"    ❌ {issue}")

            if report["warnings"]:
                print("  Warnings:")
                for warning in report["warnings"]:  # type: ignore[attr-defined]
                    print(f"    ⚠️  {warning}")
