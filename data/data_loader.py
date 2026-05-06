"""
data/data_loader.py
====================
Loads intraday OHLC data for ES/NQ/GC futures in any supported timeframe.

Supports Ninja Trader CSV files (data/Ninja/{INSTRUMENT} {TF}.csv) and
auto-converts to Feather cache on first load (~10× faster than CSV re-parsing).

CSV format (NinjaTrader export):
  Date, Time, Open, High, Low, Close, [Volume columns — dropped automatically]
  Date format: D/M/YYYY  (e.g. 1/10/2024 = 1 October 2024)
"""

from __future__ import annotations

import bisect
import glob
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Required CSV columns
_REQUIRED_COLS = {"Date", "Time", "Open", "High", "Low", "Close"}

# Full column rename map — optional columns are silently skipped if absent
_COLUMN_MAP = {
    "Date": "date",
    "Time": "time",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "NumberOfTrades": "num_trades",
    "BidVolume": "bid_volume",
    "AskVolume": "ask_volume",
}


class DataLoader:
    """
    Loads intraday and daily OHLCV data for one instrument.

    Supports multiple timeframes (5min, 1min) and multiple file formats
    (feather, pkl, csv). Automatically detects and uses fastest available format.

    Parameters
    ----------
    data_dir : str
        Directory containing the CSV file(s). Ninja Trader files expected in data_dir/Ninja/.
    instrument : str
        Instrument ticker, e.g. "ES", "NQ", or "GC". Used to match the CSV filename.
    timeframe : str
        Intraday bar timeframe string, e.g. "5min" or "1min". Used for cache file naming.
    daily_tf : str
        Daily timeframe string, e.g. "1D".  Only used for labelling.
    file_format : str
        File format preference: "feather" (fastest), "pkl", or "csv" (slowest).
        Cache priority order: feather → pkl → csv.
    tz : str
        Target timezone for the DatetimeIndex, e.g. "America/New_York".
        The CSV timestamps are assumed to be in exchange local time (CT/CST/CDT).
    """

    def __init__(
        self,
        data_dir: str,
        instrument: str = "ES",
        timeframe: str = "5min",
        daily_tf: str = "1D",
        file_format: str = "feather",
        tz: str = "America/New_York",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.instrument = instrument.upper()
        self.timeframe = timeframe
        self.daily_tf = daily_tf
        self.file_format = file_format
        self.tz = tz

        self._intraday: Optional[pd.DataFrame] = None
        self._daily: Optional[pd.DataFrame] = None
        self._day_index: Dict[str, pd.DataFrame] = {}

    # ── Public API ────────────────────────────────────────────

    def load(self) -> None:
        """
        Locate and load data in the fastest available format.

        Priority: feather (fastest) → npy → pkl → csv (slowest).

        On first run the CSV is parsed and a Feather cache is written (~10× faster).
        Subsequent runs load from the cache if it's newer than the CSV.
        Delete cached files to force a full re-parse.
        """
        # Try to load from cached formats (feather → npy → pkl)
        self._intraday = self._try_load_cached()

        # If no cache, load from CSV and convert to Feather
        if self._intraday is None:
            csv_path = self._find_csv()
            raw = self._read_csv(csv_path)
            self._intraday = self._process(raw)

            # Cache as Feather (fastest format)
            try:
                self._cache_as_feather(self._intraday)
            except Exception:
                pass  # non-fatal — next run will just re-parse

        self._daily = self._build_daily(self._intraday)
        self._day_index = self._index_by_day(self._intraday)

    def get_trading_days(self) -> List[str]:
        """
        Return sorted list of trading day strings (YYYY-MM-DD) present in
        the intraday data.
        """
        self._assert_loaded()
        return sorted(self._day_index.keys())

    def get_day_bars(self, date: str) -> pd.DataFrame:
        """
        Return all intraday bars for a single trading day.

        Parameters
        ----------
        date : str
            Date string in YYYY-MM-DD format, e.g. "2024-10-01".

        Returns
        -------
        pd.DataFrame with columns open/high/low/close/volume and a
        tz-aware DatetimeIndex.  Empty DataFrame if date not found.
        """
        self._assert_loaded()
        return self._day_index.get(date, pd.DataFrame())

    def get_bars_before(self, date: str, n_bars: int) -> pd.DataFrame:
        """
        Return the last ``n_bars`` intraday bars that are strictly before
        ``date`` (i.e. from prior trading sessions).

        Uses bisect on the sorted trading-day list and the pre-built day index
        so only the required prior days are touched — O(n_prior_days) instead
        of O(n_total_bars).

        Parameters
        ----------
        date : str
            Cutoff date in YYYY-MM-DD format.  All returned bars are from
            days *earlier* than this date.
        n_bars : int
            Maximum number of bars to return.  Fewer bars are returned when
            there is not enough prior history in the dataset.

        Returns
        -------
        pd.DataFrame with the same columns as the intraday frame, sorted
        chronologically.  Empty DataFrame when no prior data exists.
        """
        self._assert_loaded()
        days = self.get_trading_days()          # sorted list, O(1) cached
        cutoff_idx = bisect.bisect_left(days, date)
        if cutoff_idx == 0:
            return pd.DataFrame(columns=self._intraday.columns)

        # Walk backwards through prior days collecting bars until we have n_bars
        collected: List[pd.DataFrame] = []
        total = 0
        for i in range(cutoff_idx - 1, -1, -1):
            day_bars = self._day_index[days[i]]
            collected.append(day_bars)
            total += len(day_bars)
            if total >= n_bars:
                break

        if not collected:
            return pd.DataFrame(columns=self._intraday.columns)

        combined = pd.concat(reversed(collected))
        return combined.iloc[-n_bars:]

    def get_daily_bar(self, date: str) -> Optional[pd.Series]:
        """
        Return the single daily OHLCV bar for a given date.

        Parameters
        ----------
        date : str
            Date string in YYYY-MM-DD format.

        Returns
        -------
        pd.Series or None if date is not in the daily index.
        """
        self._assert_loaded()
        if self._daily is None or self._daily.empty:
            return None
        try:
            return self._daily.loc[date]
        except KeyError:
            return None

    @property
    def intraday(self) -> pd.DataFrame:
        """Full intraday DataFrame (all days)."""
        self._assert_loaded()
        return self._intraday  # type: ignore[return-value]

    @property
    def daily(self) -> pd.DataFrame:
        """Daily OHLCV DataFrame (one row per trading day)."""
        self._assert_loaded()
        return self._daily  # type: ignore[return-value]

    # ── Private helpers ───────────────────────────────────────

    def _try_load_cached(self) -> Optional[pd.DataFrame]:
        """
        Try to load data from cached formats in priority order: feather → pkl.

        Skips cache if the source CSV is newer (staleness check).

        Returns pd.DataFrame with tz-aware DatetimeIndex, or None.
        """
        cache_dir = self.data_dir / "cached"
        if not cache_dir.exists():
            return None

        # Determine source CSV mtime for staleness check (best-effort; ignore if not found)
        csv_mtime: float = 0.0
        try:
            csv_mtime = self._find_csv().stat().st_mtime
        except (FileNotFoundError, OSError):
            pass

        # Try Feather (fastest, preferred)
        feather_path = cache_dir / f"{self.instrument}_{self.timeframe}.feather"
        if feather_path.exists() and feather_path.stat().st_mtime >= csv_mtime:
            try:
                return pd.read_feather(feather_path)
            except Exception:
                pass

        # Try Pickle (legacy fallback — preserves DatetimeIndex)
        pkl_path = self.data_dir / f"{self.instrument}_{self.timeframe}.pkl"
        if pkl_path.exists() and pkl_path.stat().st_mtime >= csv_mtime:
            try:
                return pd.read_pickle(pkl_path)
            except Exception:
                pass

        return None

    def _cache_as_feather(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame as Feather format in the cache directory.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to cache.
        """
        cache_dir = self.data_dir / "cached"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{self.instrument}_{self.timeframe}.feather"
        df.to_feather(cache_path)

    def _find_csv(self) -> Path:
        """Locate a CSV file in data_dir or data_dir/Ninja/ whose name contains the instrument ticker."""
        # Convert timeframe to Ninja Trader naming: "5min" → "5 min", "1min" → "1 min"
        tf_display = self.timeframe.replace("min", " min").rstrip()

        # Search patterns in priority order
        patterns = [
            # Ninja Trader directory (preferred) — handles both underscore and space naming
            str(self.data_dir / "Ninja" / f"{self.instrument} {tf_display}.csv"),
            str(self.data_dir / "Ninja" / f"{self.instrument}_{self.timeframe}*.csv"),
            str(self.data_dir / "Ninja" / f"{self.instrument}_*.csv"),
            str(self.data_dir / "Ninja" / f"*{self.instrument}*.csv"),
            # Root data directory
            str(self.data_dir / f"{self.instrument} {tf_display}.csv"),
            str(self.data_dir / f"{self.instrument}_{self.timeframe}*.csv"),
            str(self.data_dir / f"{self.instrument}_*.csv"),
            str(self.data_dir / f"*{self.instrument}*.csv"),
            str(self.data_dir / f"*{self.instrument.lower()}*.csv"),
            # Fallback: any CSV in data_dir
            str(self.data_dir / "*.csv"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                return Path(matches[0])
        raise FileNotFoundError(
            f"No CSV file found for instrument '{self.instrument}' (timeframe '{self.timeframe}') "
            f"in {self.data_dir} or {self.data_dir}/Ninja/. "
            f"Expected a file matching patterns like: {self.instrument} {tf_display}.csv or *{self.instrument}*.csv"
        )

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read raw CSV — no date parsing yet, just load strings."""
        df = pd.read_csv(path, dtype=str)
        df.columns = df.columns.str.strip()  # remove any accidental whitespace
        # NinjaTrader exports 'Last' instead of 'Close' — normalise before column check
        if "Last" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Last": "Close"})
        return df

    def _process(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardise the raw DataFrame.

        Steps:
          1. Rename Title-Case columns to lowercase internal names.
          2. Combine 'date' + 'time' strings into a tz-aware DatetimeIndex.
             Date format is D/M/YYYY (dayfirst=True).
          3. Cast OHLCV columns to float/int.
          4. Drop the original string date/time columns.
          5. Sort by index.
        """
        # 1. Rename columns — required cols must be present; optional cols skipped if absent
        missing_required = _REQUIRED_COLS - set(raw.columns)
        if missing_required:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing_required)}. "
                f"Found: {list(raw.columns)}"
            )
        present_map = {k: v for k, v in _COLUMN_MAP.items() if k in raw.columns}
        df = raw.rename(columns=present_map)

        # 2. Build datetime index — D/M/YYYY H:MM:SS
        dt_strings = df["date"].str.strip() + " " + df["time"].str.strip()
        dt_index = pd.to_datetime(dt_strings, dayfirst=True)
        dt_index = dt_index.dt.tz_localize(self.tz, ambiguous="infer", nonexistent="shift_forward")
        df.index = dt_index
        df.index.name = "datetime"

        # 3. Cast numeric columns
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float)
        for col in ("volume", "num_trades", "bid_volume", "ask_volume"):
            if col in df.columns:
                df[col] = df[col].astype(int)

        # 4. Drop redundant string columns
        df.drop(columns=["date", "time"], inplace=True)

        # 5. Sort chronologically (CSV should already be sorted, but be safe)
        df.sort_index(inplace=True)

        return df

    def _build_daily(self, intraday: pd.DataFrame) -> pd.DataFrame:
        """
        Resample intraday bars to daily OHLCV.

        The index of the resulting DataFrame is a DatetimeIndex with
        date-only labels (YYYY-MM-DD strings) for easy lookup.
        """
        agg_cols = {"open": "first", "high": "max", "low": "min", "close": "last"}
        cols = ["open", "high", "low", "close"]
        if "volume" in intraday.columns:
            agg_cols["volume"] = "sum"
            cols.append("volume")
        daily = intraday[cols].resample("1D").agg(agg_cols).dropna()

        # Use YYYY-MM-DD string index for consistent key lookups
        daily.index = daily.index.strftime("%Y-%m-%d")
        return daily

    def _index_by_day(self, intraday: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Build a dict mapping YYYY-MM-DD string → DataFrame of that day's bars.
        """
        index: Dict[str, pd.DataFrame] = {}
        for ts, group in intraday.groupby(intraday.index.date):
            key = ts.strftime("%Y-%m-%d")
            index[key] = group
        return index

    def _assert_loaded(self) -> None:
        if self._intraday is None:
            raise RuntimeError("DataLoader.load() must be called before accessing data.")
