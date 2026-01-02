# main.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Optional

from .config import AppConfig
from .data_pipeline import DataPipeline
from .model import PredictionModel
from .streaming import StreamEngine
from .utils import CsvLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream monthly ENSO predictions every N seconds (inference only)."
    )
    parser.add_argument("--start", type=str, default="2007-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2017-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--lead", type=int, default=3, help="Lead time in months used at training")
    parser.add_argument(
        "--max_lag", type=int, default=15, help="Number of ENSO lag features"
    )
    parser.add_argument(
        "--sst_path",
        type=str,
        default="sst.mon.mean.trefadj.anom.1880to2018.nc",
        help="SST anomalies NetCDF file",
    )
    parser.add_argument(
        "--enso_path",
        type=str,
        default="nino34.long.anom.data.txt",
        help="ENSO text file (nino3.4 anomalies)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear_lag.joblib",
        help="Path to trained model (joblib)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between successive predictions",
    )
    parser.add_argument(
        "--show_features",
        action="store_true",
        help="Print a subset of features alongside predictions",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Path to a CSV file for live predictions logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_args(args)
    config.validate()

    pipeline = DataPipeline(config)
    model = PredictionModel(config)

    logger: Optional[CsvLogger] = None
    if config.out_csv is not None:
        logger = CsvLogger(config.out_csv)

    engine = StreamEngine(config, pipeline, model, logger)
    engine.run()


if __name__ == "__main__":
    main()


# python -m enso_streaming.main `
#   --model .\data\linear_lag.joblib `
#   --sst_path .\data\sst.mon.mean.trefadj.anom.1880to2018.nc `
#   --enso_path .\data\nino34.long.anom.data.txt `
#   --start 2007-01-01 `
#   --end 2017-12-31 `
#   --lead 1 `
#   --max_lag 15 `
#   --interval 10 `
#   --out_csv .\out\live_predictions.csv `
#   --show_features
