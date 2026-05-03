from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
CLASS_NAMES = ("Downtrend", "Neutral", "Uptrend")
DEFAULT_HORIZONS = (1, 3, 5)
DEFAULT_WINDOW_SIZE = 60
DEFAULT_TREND_THRESHOLD = 0.004
DEFAULT_TRAIN_FRACTION = 0.8


@dataclass(slots=True)
class ProcessedBundle:
    symbol: str
    source_path: Path
    feature_names: tuple[str, ...]
    window_size: int
    horizons: tuple[int, ...]
    train_fraction: float
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    all_x: np.ndarray
    all_y: np.ndarray
    raw_windows: np.ndarray
    dates: list[str]
    closes: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def list_symbols(dataset_dir: Path) -> list[dict[str, str]]:
    symbols: list[dict[str, str]] = []
    for path in sorted(dataset_dir.glob("*.csv")):
        symbols.append({"symbol": path.stem, "file": path.name})
    return symbols


def resolve_csv_path(dataset_dir: Path, symbol: str) -> Path:
    candidate = dataset_dir / f"{symbol}.csv"
    if candidate.exists():
        return candidate
    matches = list(dataset_dir.glob(f"{symbol}*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No CSV found for symbol '{symbol}'")


def read_stock_frame(source: Path | str | bytes | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    elif isinstance(source, (str, Path)):
        try:
            frame = pd.read_csv(source, encoding="utf-8")
        except UnicodeDecodeError:
            frame = pd.read_csv(source, encoding="utf-8-sig")
    else:
        try:
            frame = pd.read_csv(source, encoding="utf-8")
        except UnicodeDecodeError:
            frame = pd.read_csv(source, encoding="utf-8-sig")
    if "Date" not in frame.columns:
        raise ValueError("CSV must contain a Date column")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    numeric = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    frame.loc[:, FEATURE_COLUMNS] = numeric
    return frame


def _rolling_returns(close: np.ndarray) -> np.ndarray:
    returns = np.zeros_like(close, dtype=np.float32)
    returns[1:] = np.diff(close) / np.clip(close[:-1], 1e-8, None)
    return returns


def _trend_label(future_return: float, threshold: float) -> int:
    if future_return > threshold:
        return 2
    if future_return < -threshold:
        return 0
    return 1


def create_sliding_windows(
    frame: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    horizon_tuple = tuple(sorted({int(h) for h in horizons}))
    if not horizon_tuple:
        raise ValueError("At least one horizon is required")

    values = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    closes = values[:, 3]
    max_horizon = max(horizon_tuple)
    samples: list[np.ndarray] = []
    labels: list[list[int]] = []
    dates: list[str] = []

    last_start = len(frame) - max_horizon
    for end_index in range(window_size - 1, last_start):
        window = values[end_index - window_size + 1 : end_index + 1]
        current_close = closes[end_index]
        horizon_labels: list[int] = []
        for horizon in horizon_tuple:
            future_close = closes[end_index + horizon]
            future_return = (future_close - current_close) / max(current_close, 1e-8)
            horizon_labels.append(_trend_label(float(future_return), trend_threshold))
        samples.append(window)
        labels.append(horizon_labels)
        dates.append(frame.loc[end_index, "Date"].strftime("%Y-%m-%d"))

    if not samples:
        raise ValueError("Not enough rows to create sliding windows")

    return np.stack(samples), np.asarray(labels, dtype=np.int64), dates, closes


def fit_zscore_scaler(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flattened = windows.reshape(-1, windows.shape[-1])
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((windows - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def build_processed_bundle(
    symbol: str,
    source_path: Path,
    window_size: int = DEFAULT_WINDOW_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> ProcessedBundle:
    frame = read_stock_frame(source_path)
    windows, labels, dates, closes = create_sliding_windows(
        frame,
        window_size=window_size,
        horizons=horizons,
        trend_threshold=trend_threshold,
    )

    split_index = max(1, int(len(windows) * train_fraction))
    split_index = min(split_index, len(windows) - 1)
    train_windows = windows[:split_index]
    mean, std = fit_zscore_scaler(train_windows)
    normalized_windows = apply_zscore(windows, mean, std)

    train_x = normalized_windows[:split_index]
    train_y = labels[:split_index]
    val_x = normalized_windows[split_index:]
    val_y = labels[split_index:]

    return ProcessedBundle(
        symbol=symbol,
        source_path=source_path,
        feature_names=FEATURE_COLUMNS,
        window_size=window_size,
        horizons=tuple(sorted({int(h) for h in horizons})),
        train_fraction=train_fraction,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        all_x=normalized_windows,
        all_y=labels,
        raw_windows=windows,
        dates=dates,
        closes=closes,
        mean=mean,
        std=std,
    )


def to_processed_payload(bundle: ProcessedBundle, limit: int = 120) -> dict[str, object]:
    tail_close = bundle.closes[-limit:].tolist()
    tail_dates = bundle.dates[-limit:]
    latest_raw_window = bundle.raw_windows[-1].tolist() if len(bundle.raw_windows) else []
    latest_normalized_window = bundle.all_x[-1].tolist() if len(bundle.all_x) else []
    latest_open = float(bundle.raw_windows[-1, -1, 0]) if len(bundle.raw_windows) else 0.0
    latest_high = float(bundle.raw_windows[-1, -1, 1]) if len(bundle.raw_windows) else 0.0
    latest_low = float(bundle.raw_windows[-1, -1, 2]) if len(bundle.raw_windows) else 0.0
    latest_close = float(bundle.raw_windows[-1, -1, 3]) if len(bundle.raw_windows) else 0.0
    latest_volume = float(bundle.raw_windows[-1, -1, 4]) if len(bundle.raw_windows) else 0.0
    return {
        "symbol": bundle.symbol,
        "window_size": bundle.window_size,
        "horizons": list(bundle.horizons),
        "feature_names": list(bundle.feature_names),
        "mean": bundle.mean.tolist(),
        "std": bundle.std.tolist(),
        "sample_count": int(bundle.all_x.shape[0]),
        "train_size": int(bundle.train_x.shape[0]),
        "val_size": int(bundle.val_x.shape[0]),
        "recent_dates": tail_dates,
        "recent_close": tail_close,
        "recent_windows": latest_normalized_window,
        "recent_raw_window": latest_raw_window,
        "recent_window_labels": [f"T{i + 1}" for i in range(len(latest_raw_window))],
        "recent_open": latest_open,
        "recent_high": latest_high,
        "recent_low": latest_low,
        "recent_close_value": latest_close,
        "recent_volume": latest_volume,
        "recent_labels": bundle.all_y[-1].tolist() if len(bundle.all_y) else [],
    }
