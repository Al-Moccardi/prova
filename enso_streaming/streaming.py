# streaming.py
from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from .config import AppConfig
from .data_pipeline import DataPipeline
from .model import PredictionModel
from .utils import CsvLogger


class StreamEngine:
    """
    Orchestrates the streaming inference:
    - Builds features via DataPipeline
    - Uses PredictionModel to predict
    - Optionally logs to CSV via CsvLogger
    - Streams one prediction every N seconds
    """

    def __init__(
        self,
        config: AppConfig,
        pipeline: DataPipeline,
        model: PredictionModel,
        logger: Optional[CsvLogger] = None,
):
        self.config = config
        self.pipeline = pipeline
        self.model = model
        self.logger = logger

    def run(self) -> None:
        """
        Main streaming loop.
        """
        cfg = self.config

        feats_df, feature_names = self.pipeline.build_features()

        if feats_df.empty:
            print(
                "[WARN] No rows to score after feature alignment. "
                "Check date window, lead_time/max_lag, and that input files cover the window."
            )
            return

        # Prepare CSV if requested
        if self.logger is not None:
            self.logger.init()
            print(f"[INFO] Live CSV logging to: {self.logger.path}")

        print(
            f"\n[INFO] Streaming predictions from "
            f"{feats_df.index.min().date()} to {feats_df.index.max().date()}"
        )
        print(
            f"[INFO] Interval: {cfg.interval} sec — Model: {cfg.model_path} "
            f"— Rows: {len(feats_df)}"
        )
        print("[INFO] Press Ctrl+C to stop.\n")

        try:
            for ts, row in feats_df.iterrows():
                y_out = self.model.predict_row(row)

                # Console output
                if cfg.show_features:
                    preview_keys = list(row.index)[:6]  # mean, std, a few lags
                    feat_str = ", ".join(f"{k}={row[k]:.4f}" for k in preview_keys)
                    print(
                        f"{ts.strftime('%Y-%m-%d')}  pred={y_out:.6f}  | {feat_str} ..."
                    )
                else:
                    print(f"{ts.strftime('%Y-%m-%d')}  pred={y_out:.6f}")

                # CSV logging: now only timestamp + prediction
                if self.logger is not None:
                    self.logger.append(ts, y_out)

                time.sleep(cfg.interval)

        except KeyboardInterrupt:
            print("\n[INFO] Streaming interrupted by user.")
