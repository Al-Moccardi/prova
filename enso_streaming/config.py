# config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    """Application configuration and CLI parameters."""
    start: str = "2007-01-01"
    end: str = "2017-12-31"
    lead: int = 3
    max_lag: int = 15
    sst_path: str = "sst.mon.mean.trefadj.anom.1880to2018.nc"
    enso_path: str = "nino34.long.anom.data.txt"
    model_path: str = "linear_lag.joblib"
    interval: float = 10.0
    out_csv: Optional[str] = None
    show_features: bool = False

    @classmethod
    def from_args(cls, args) -> "AppConfig":
        """Build AppConfig from argparse.Namespace."""
        return cls(
            start=args.start,
            end=args.end,
            lead=args.lead,
            max_lag=args.max_lag,
            sst_path=args.sst_path,
            enso_path=args.enso_path,
            model_path=args.model,
            interval=args.interval,
            out_csv=args.out_csv,
            show_features=args.show_features,
        )

    def validate(self) -> None:
        """Validate configuration (paths, simple ranges)."""
        # Input files
        for path, label in [
            (self.sst_path, "SST file"),
            (self.enso_path, "ENSO file"),
            (self.model_path, "model file"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} not found: {path}")

        # Simple sanity checks
        if self.lead <= 0:
            raise ValueError(f"Lead time must be positive, got {self.lead}")
        if self.max_lag <= 0:
            raise ValueError(f"max_lag must be positive, got {self.max_lag}")
        if self.interval <= 0:
            raise ValueError(f"interval must be positive, got {self.interval}")
