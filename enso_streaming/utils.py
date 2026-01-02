# utils.py
from __future__ import annotations

import csv
import os
from typing import List, Optional

import numpy as np
import pandas as pd


# -----------------------
# ENSO utilities
# -----------------------
def load_enso_indices(path: str) -> pd.Series:
    """
    Read the ENSO txt data file and return a monthly Series starting 1870-01-01.
    Each line is assumed like: YEAR v1 v2 ... v12
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ENSO file not found: {path}")

    vals = []
    with open(path) as f:
        for line in f:
            toks = line.split()
            if len(toks) > 1:
                vals.extend(map(float, toks[1:]))

    s = pd.Series(vals, index=pd.date_range("1870-01-01", freq="MS", periods=len(vals)))
    return s


def clean_enso(series: pd.Series, missing_sentinel: float = -99.99) -> pd.Series:
    """
    Replace sentinel with NaN, drop missing, and normalize index to month-start.
    Uses period -> timestamp with how='S' (start of month) to avoid 'MS' issues.
    """
    s = series.replace(missing_sentinel, np.nan).dropna()
    s.index = pd.to_datetime(s.index)
    s.index = s.index.to_period("M").to_timestamp(how="S")
    s = s.sort_index()
    return s


# -----------------------
# CSV logging (timestamp + prediction only)
# -----------------------
class CsvLogger:
    """
    Utility for live CSV logging of predictions.

    Columns:
      timestamp, pred
    """

    def __init__(self, path: str):
        self.path = path
        self._initialized = False

    def init(self) -> None:
        """
        Create CSV with header if it doesn't exist or is empty.
        Header: timestamp, pred
        """
        needs_header = True
        if os.path.exists(self.path):
            try:
                needs_header = (os.path.getsize(self.path) == 0)
            except OSError:
                needs_header = True

        if needs_header:
            header = ["timestamp", "pred"]
            with open(self.path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)

        self._initialized = True

    def append(self, ts: pd.Timestamp, pred: float) -> None:
        """
        Append a single prediction row: timestamp, pred.
        """
        if not self._initialized:
            raise RuntimeError("CsvLogger.append() called before init().")

        try:
            with open(self.path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([ts.strftime("%Y-%m-%d"), f"{pred:.10f}"])
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    # If fsync isn't supported, just ignore.
                    pass
        except PermissionError:
            # If file is open in Excel, append will fail—warn and continue.
            print(
                f"[WARN] Could not write to '{self.path}' "
                f"(maybe open in another program). Will keep streaming."
            )
