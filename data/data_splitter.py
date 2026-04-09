"""
data/data_splitter.py
======================
Splits a list of trading-day strings into train / validation / test sets.

Split is purely chronological — no shuffling.  Test set is always the most
recent data and must never be touched during training or hyperparameter tuning.

Usage (percentage-based):
    splitter = DataSplitter(train_pct=0.70, val_pct=0.15)
    split = splitter.split(trading_days)

Usage (fixed count — preferred for reproducibility):
    split = DataSplitter.split_by_counts(trading_days, n_train=252, n_val=26)
    # split.train, split.validation, split.test  →  List[str]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DataSplit:
    """Container for the three non-overlapping day lists."""
    train: List[str]
    validation: List[str]
    test: List[str]

    def summary(self) -> str:
        return (
            f"DataSplit — train: {len(self.train)} days "
            f"({self.train[0]} → {self.train[-1]}), "
            f"val: {len(self.validation)} days "
            f"({self.validation[0]} → {self.validation[-1]}), "
            f"test: {len(self.test)} days "
            f"({self.test[0]} → {self.test[-1]})"
        )


class DataSplitter:
    """
    Chronological train/val/test splitter.

    Parameters
    ----------
    train_pct : float
        Fraction of days allocated to training (e.g. 0.70).
    val_pct : float
        Fraction of days allocated to validation (e.g. 0.15).
        test_pct is inferred as 1 - train_pct - val_pct.
    """

    def __init__(self, train_pct: float = 0.70, val_pct: float = 0.15) -> None:
        if not (0 < train_pct < 1):
            raise ValueError(f"train_pct must be in (0, 1), got {train_pct}")
        if not (0 < val_pct < 1):
            raise ValueError(f"val_pct must be in (0, 1), got {val_pct}")
        if train_pct + val_pct >= 1.0:
            raise ValueError(
                f"train_pct + val_pct must be < 1.0 to leave room for test set. "
                f"Got {train_pct + val_pct:.2f}"
            )
        self.train_pct = train_pct
        self.val_pct = val_pct

    def split(self, trading_days: List[str]) -> DataSplit:
        """
        Split a sorted list of trading-day strings chronologically.

        Parameters
        ----------
        trading_days : List[str]
            Sorted list of YYYY-MM-DD date strings.

        Returns
        -------
        DataSplit with .train, .validation, .test lists.
        """
        n = len(trading_days)
        if n < 10:
            raise ValueError(f"Too few trading days to split: {n}. Need at least 10.")

        train_end = int(n * self.train_pct)
        val_end = train_end + int(n * self.val_pct)

        # Ensure each split has at least one day
        train_end = max(train_end, 1)
        val_end = max(val_end, train_end + 1)
        val_end = min(val_end, n - 1)

        return DataSplit(
            train=trading_days[:train_end],
            validation=trading_days[train_end:val_end],
            test=trading_days[val_end:],
        )

    @classmethod
    def split_by_counts(
        cls,
        trading_days: List[str],
        n_train: int,
        n_val: int,
    ) -> DataSplit:
        """
        Split using fixed day counts instead of percentages.

        The most recent ``n_val`` days after the training block become
        validation.  Everything after that becomes the untouched test set.
        Chronological order is always preserved — no shuffling.

        Parameters
        ----------
        trading_days : List[str]
            Sorted list of YYYY-MM-DD date strings.
        n_train : int
            Number of days to allocate to training  (e.g. 252 ≈ 12 months).
        n_val : int
            Number of days to allocate to validation (e.g. 26 ≈ 5 weeks).

        Returns
        -------
        DataSplit with .train, .validation, .test lists.
        """
        days = sorted(trading_days)
        n = len(days)
        if n < n_train + n_val + 1:
            raise ValueError(
                f"Not enough trading days ({n}) for n_train={n_train} + "
                f"n_val={n_val} + at least 1 test day."
            )
        train = days[:n_train]
        val   = days[n_train:n_train + n_val]
        test  = days[n_train + n_val:]
        return DataSplit(train=train, validation=val, test=test)

    @staticmethod
    def walk_forward_splits(
        trading_days: List[str],
        n_train_days: int = 252,
        n_val_days: int = 25,
        n_folds: int = 1,
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Build rolling walk-forward (train, val) pairs.

        Each fold shifts forward by ``n_val_days`` so validation windows
        are non-overlapping and strictly out-of-sample.

        Fold layout (k = 0, 1, 2, …):
            train : days[k * n_val_days : k * n_val_days + n_train_days]
            val   : days[k * n_val_days + n_train_days :
                         k * n_val_days + n_train_days + n_val_days]

        Parameters
        ----------
        trading_days : List[str]
            Sorted list of YYYY-MM-DD strings (weekdays, ATR-complete).
        n_train_days : int
            Training window in trading days (default 252 ≈ 12 months).
        n_val_days : int
            Validation window in trading days (default 25 ≈ 5 weeks).
        n_folds : int
            Number of folds to generate.  -1 means all possible folds
            given the available data.  Default 1.

        Returns
        -------
        List of (train_days, val_days) tuples, one entry per fold.

        Raises
        ------
        ValueError
            If n_train_days + n_val_days exceeds the number of available
            trading days, making it impossible to produce even one fold.
        """
        days = sorted(trading_days)
        n = len(days)
        min_required = n_train_days + n_val_days
        if n < min_required:
            raise ValueError(
                f"Not enough trading days ({n}) to create a single fold "
                f"(need n_train_days={n_train_days} + n_val_days={n_val_days} = {min_required})."
            )

        # Maximum number of non-overlapping val folds that fit in the data
        max_folds = (n - n_train_days) // n_val_days
        if max_folds < 1:
            raise ValueError(
                f"Data ({n} days) is too short for even one fold with "
                f"n_train_days={n_train_days} and n_val_days={n_val_days}."
            )

        if n_folds == -1:
            target_folds = max_folds
        else:
            target_folds = min(n_folds, max_folds)

        folds: List[Tuple[List[str], List[str]]] = []
        for k in range(target_folds):
            start       = k * n_val_days
            train_end   = start + n_train_days
            val_end     = train_end + n_val_days
            if val_end > n:
                break
            folds.append((days[start:train_end], days[train_end:val_end]))

        return folds
