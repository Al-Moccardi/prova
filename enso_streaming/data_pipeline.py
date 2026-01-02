# data_pipeline.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .config import AppConfig
from .utils import load_enso_indices, clean_enso


class DataPipeline:
    """
    Responsible for constructing the feature matrix used for inference.

    - Loads SST NetCDF
    - Computes monthly mean/std of SST
    - Loads & cleans ENSO series
    - Builds ENSO lag features
    """

    def __init__(self, config: AppConfig):
        self.config = config

    def build_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Returns:
            feats_df: DataFrame indexed by monthly dates,
                      columns: ['mean', 'std', 'ENSO_lag1', ..., 'ENSO_lagK']
            feature_names: ordered list of column names.
        """
        cfg = self.config  #  = config = config.py = settato dall'utente

        # --- Load SST and aggregate ---
        ds = xr.open_dataset(cfg.sst_path)
        try:
            sst = ds["sst"].sel(time=slice(cfg.start, cfg.end))

            # Use dataset's own time coordinate, snapped to month START
            idx = pd.to_datetime(sst["time"].values)
            idx = pd.DatetimeIndex(idx).to_period("M").to_timestamp(how="S")

            nT = sst.shape[0]
            sst_vals = np.asarray(sst.values).reshape(nT, -1)
            sst_vals[np.isnan(sst_vals)] = 0

            df_monthly = pd.DataFrame(sst_vals, index=idx)

            # Drop all-zero columns (e.g., land masks)
            nonzero = ~(df_monthly == 0).all(axis=0)
            df_monthly = df_monthly.loc[:, nonzero]

            # Aggregate features
            df_feat = pd.DataFrame(
                {
                    "mean": df_monthly.mean(axis=1),
                    "std": df_monthly.std(axis=1, ddof=0),
                }
            )
        finally:
            ds.close()

        # --- ENSO lags ---
        enso_raw = load_enso_indices(cfg.enso_path)
        enso_clean = clean_enso(enso_raw)

        # Force monthly frequency; no fill so we avoid leakage
        enso_ms = enso_clean.asfreq("MS")

        lag_df = pd.DataFrame(index=df_feat.index)
        for k in range(1, cfg.max_lag + 1):
            offset = cfg.lead - k
            # For month t, lag k = ENSO[t + lead_time - k]
            lag_k = enso_ms.shift(-offset)  # negative brings future into current row
            lag_df[f"ENSO_lag{k}"] = lag_k.reindex(df_feat.index)

        feats = pd.concat([df_feat, lag_df], axis=1).dropna()

        feature_names = ["mean", "std"] + [f"ENSO_lag{k}" for k in range(1, cfg.max_lag + 1)]
        feats = feats[feature_names]

        return feats, feature_names
