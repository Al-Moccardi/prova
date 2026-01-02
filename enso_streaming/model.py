# model.py
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

from .config import AppConfig


class PredictionModel:
    """
    Thin wrapper around a scikit-learn model loaded via joblib.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._model = self._load_model(config.model_path)

    @staticmethod
    def _load_model(model_path: str) -> Any:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return load(model_path)

    def predict_row(self, row: pd.Series) -> float:
        """
        Predict a single row of features.
        """
        x = row.values.reshape(1, -1)
        y_pred = self._model.predict(x)[0]
        return float(np.ravel([y_pred])[0])


# data\linear_lag.joblib