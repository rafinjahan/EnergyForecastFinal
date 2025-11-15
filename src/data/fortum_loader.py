"""
Helpers for loading Fortum's Junction training workbook.

The CLI (main.py) expects two utilities:
- load_fortum_training: returns the core sheets as Pandas DataFrames.
- describe_frames: human-readable summary for quick sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

# Workbook contains several sheets but the hackathon spec only mentions three.
SHEET_ALIASES = {
    "consumption": "training_consumption",
    "groups": "groups",
    "prices": "training_prices",
}

# Filename published by Fortum; searching is case-sensitive on macOS but this
# keeps logic explicit. Extend list if organizers drop a new revision.
DEFAULT_FILENAMES = ["20251111_JUNCTION_training.xlsx"]


@dataclass(frozen=True)
class WorkbookSearchPath:
    """Small helper so we can debug quickly if future paths are added."""

    label: str
    path: Path


def load_fortum_training(xlsx_path: str | Path | None = None) -> Dict[str, pd.DataFrame]:
    """
    Load consumption, group, and price sheets from the Fortum workbook.

    Args:
        xlsx_path: Optional explicit path to the Excel file. When omitted we
                   search a few conventional locations such as project root,
                   ./data, ./data/raw, and ./Data.
    Raises:
        FileNotFoundError: if the workbook is not present in any search path.
        ValueError: if expected sheets are missing.
    """

    workbook = _resolve_workbook_path(xlsx_path)
    frames: Dict[str, pd.DataFrame] = {}

    with pd.ExcelFile(workbook) as xls:
        for alias, sheet in SHEET_ALIASES.items():
            if sheet not in xls.sheet_names:
                raise ValueError(f"Missing sheet '{sheet}' in {workbook}")
            frames[alias] = pd.read_excel(xls, sheet)

    return frames


def describe_frames(frames: Dict[str, pd.DataFrame]) -> str:
    """
    Provide a compact textual summary of loaded frames for CLI output.
    """

    if not frames:
        return "No frames loaded."

    lines = []
    for alias, df in frames.items():
        cols = ", ".join(map(str, df.columns[:6]))
        if len(df.columns) > 6:
            cols += ", ..."
        lines.append(
            f"- {alias}: shape={df.shape[0]}x{df.shape[1]} | columns=[{cols}]"
        )
    return "\n".join(lines)


def _resolve_workbook_path(xlsx_path: str | Path | None) -> Path:
    if xlsx_path:
        path = Path(xlsx_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Workbook not found at {path}")
        return path

    project_root = Path(__file__).resolve().parents[2]

    search_roots = [
        WorkbookSearchPath("project root", project_root),
        WorkbookSearchPath("data/", project_root / "data"),
        WorkbookSearchPath("data/raw/", project_root / "data" / "raw"),
        WorkbookSearchPath("Data/", project_root / "Data"),
    ]

    candidates = _candidate_paths(search_roots, DEFAULT_FILENAMES)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(f"{p.label}" for p in search_roots)
    raise FileNotFoundError(
        "Could not locate Fortum training workbook. "
        f"Looked for {DEFAULT_FILENAMES} under: {searched}. "
        "Pass --xlsx PATH to main.py to point at the file explicitly."
    )


def _candidate_paths(
    search_roots: Iterable[WorkbookSearchPath], filenames: Iterable[str]
) -> Iterable[Path]:
    for root in search_roots:
        for name in filenames:
            yield root.path / name
