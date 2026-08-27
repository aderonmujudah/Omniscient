import logging
from typing import Iterable

from engine.calibration.harness import run_labeled_harness

logger = logging.getLogger(__name__)


def sweep_dispersion_threshold(fixture_path: str, screen_w: int, screen_h: int, diag_mm: float,
                               values: Iterable[float]) -> list[dict]:
    """
    Reports what each candidate dispersion threshold produces on one recording.

    It selects nothing. A sweep that returned the minimum of the curve would pick the value
    that best fits one session from one person, and that value would then enter the codebase
    carrying the authority of a measurement. The choice, the data it was made against and who
    made it are recorded outside the code.
    """
    rows = []
    for value in values:
        result = run_labeled_harness(fixture_path, screen_w, screen_h, diag_mm,
                                     dispersion_threshold=value)
        if result is None:
            rows.append({"threshold": value, "usable": False})
            continue
        rows.append({
            "threshold": value,
            "usable": True,
            "mean_error_deg": result["mean_error_deg"],
            "worst_error_deg": result["worst_error_deg"],
            "fit_points": result["fit_points"],
            "val_points": result["val_points"],
            "windows_diverged": result["windows_diverged"],
        })
    return rows


def format_sweep(rows: list[dict]) -> str:
    lines = ["threshold   mean_deg  worst_deg  fit  val  diverged"]
    for r in rows:
        if not r["usable"]:
            lines.append(f"{r['threshold']:<11.4f} unusable: too few accepted windows")
            continue
        lines.append(
            f"{r['threshold']:<11.4f} {r['mean_error_deg']:<9.3f} {r['worst_error_deg']:<10.3f} "
            f"{r['fit_points']:<4} {r['val_points']:<4} {r['windows_diverged']}"
        )
    return "\n".join(lines)
