"""
data/data_validator.py
=======================
Validates OHLCV DataFrames produced by DataLoader.

Called from utils/validators.py → validate_ohlcv_dataframe().
Raises ValueError on critical failures; logs warnings for minor issues.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


class DataValidator:
    """
    Validates an OHLC(V) DataFrame for minimum integrity requirements.

    Required columns (lowercase): open, high, low, close.
    Optional columns: volume (may be missing when using Ninja Trader loader).
    """

    REQUIRED_COLUMNS: List[str] = ["open", "high", "low", "close"]
    OPTIONAL_COLUMNS: List[str] = ["volume"]

    def validate(self, df: pd.DataFrame, context: str = "") -> None:
        """
        Run all validation checks.  Raises ValueError on any critical failure.

        Parameters
        ----------
        df : pd.DataFrame
            Must have lowercase OHLCV columns and a DatetimeIndex.
        context : str
            Label used in error messages (e.g. "train_set" or "2024-10-01").
        """
        label = f"[{context}] " if context else ""

        # 1. Required columns present
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{label}Missing required OHLCV columns: {missing}. "
                f"Found: {list(df.columns)}. "
                "Check that DataLoader has renamed columns to lowercase."
            )

        # 2. Non-empty
        if df.empty:
            raise ValueError(f"{label}DataFrame is empty.")

        # 3. No NaN in OHLCV
        nan_counts = df[self.REQUIRED_COLUMNS].isna().sum()
        if nan_counts.any():
            raise ValueError(
                f"{label}NaN values found in OHLCV columns: "
                f"{nan_counts[nan_counts > 0].to_dict()}"
            )

        # 4. All prices positive
        for col in ("open", "high", "low", "close"):
            non_positive = (df[col] <= 0).sum()
            if non_positive:
                raise ValueError(
                    f"{label}{col} has {non_positive} non-positive values. "
                    "Prices must be > 0."
                )

        # 5. OHLC integrity: high >= open, close, low; low <= open, close
        bad_high = ((df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])).sum()
        bad_low = ((df["low"] > df["open"]) | (df["low"] > df["close"])).sum()
        if bad_high:
            raise ValueError(f"{label}{bad_high} bars where High < Open/Close/Low.")
        if bad_low:
            raise ValueError(f"{label}{bad_low} bars where Low > Open/Close.")

        # 6. Volume non-negative (only if volume column present)
        if "volume" in df.columns:
            neg_vol = (df["volume"] < 0).sum()
            if neg_vol:
                raise ValueError(f"{label}{neg_vol} bars with negative volume.")

        # 7. DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"{label}DataFrame index must be a DatetimeIndex. "
                f"Got {type(df.index).__name__}."
            )

        # 8. Sorted chronologically
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"{label}DatetimeIndex is not sorted in ascending order.")
