from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .attention import extract_asta_focus_indices
from .model import MultiHorizonTrendTransformer
from .predictor import TREND_CLASSES
from .preprocessing import ProcessedBundle


@dataclass(slots=True)
class ForecastPoint:
    step: int
    label: str
    confidence: float
    close: float
    low: float
    high: float


@dataclass(slots=True)
class ForecastResult:
    target_date: str
    step_offset: int
    label: str
    confidence: float
    predicted_close: float
    low: float
    high: float
    explanation: str
    forecast_curve: list[dict[str, float | int | str]]
    focus_timesteps: list[int]
    volatility_scores: list[float]
    market_regime: str


def parse_future_date(value: str | date | datetime) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value.date())
    if isinstance(value, date):
        return pd.Timestamp(value)
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid future date: {value}")
    return pd.Timestamp(parsed).normalize()


def business_day_offset(last_date: str | pd.Timestamp, future_date: str | date | datetime) -> int:
    last_ts = pd.Timestamp(last_date).normalize()
    future_ts = parse_future_date(future_date)
    if future_ts <= last_ts:
        return 1
    trading_days = pd.bdate_range(last_ts + pd.offsets.BDay(1), future_ts)
    return max(int(len(trading_days)), 1)


def market_regime_from_series(closes: Sequence[float], volatility_scores: Sequence[float] | None = None) -> str:
    close_array = np.asarray(closes, dtype=np.float32)
    if len(close_array) < 3:
        return "Sideways"
    x = np.arange(len(close_array), dtype=np.float32)
    slope = np.polyfit(x, close_array, 1)[0]
    returns = np.diff(close_array) / np.clip(close_array[:-1], 1e-8, None)
    volatility = float(np.std(returns))
    if volatility_scores is not None and len(volatility_scores):
        volatility = max(volatility, float(np.mean(volatility_scores)))
    if slope > 0 and volatility < 0.025:
        return "Bull market"
    if slope < 0 and volatility > 0.02:
        return "Bear market"
    return "Sideways market"


def regime_badge_color(regime: str) -> str:
    if "Bull" in regime:
        return "good"
    if "Bear" in regime:
        return "danger"
    return "warning"


def _smooth_values(values: Sequence[float], alpha: float = 0.35) -> list[float]:
    if not values:
        return []
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return smoothed


def _forecast_step(
    model: MultiHorizonTrendTransformer,
    current_window: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    prediction = model.predict(current_window)
    horizon_labels = prediction.horizon_labels
    horizon_confidences = prediction.horizon_confidences
    trend_score = prediction.probabilities[2] - prediction.probabilities[0]

    base_close = float(current_window[0, -1, 3].item())
    base_high = float(current_window[0, -1, 1].item())
    base_low = float(current_window[0, -1, 2].item())
    base_volume = float(current_window[0, -1, 4].item())

    direction_scale = 1.0 + trend_score * 0.03
    volatility_scale = 1.0 + max(0.01, float(prediction.confidence)) * 0.015
    next_close = max(base_close * direction_scale, 1e-6)
    next_high = max(base_high * volatility_scale, next_close)
    next_low = max(min(base_low * (2.0 - volatility_scale), next_close), 1e-6)
    next_volume = max(base_volume * (1.0 + abs(trend_score) * 0.05), 0.0)

    next_row = current_window.clone()
    shifted = torch.roll(next_row, shifts=-1, dims=1)
    shifted[0, -1, 0] = next_close * 0.995
    shifted[0, -1, 1] = next_high
    shifted[0, -1, 2] = next_low
    shifted[0, -1, 3] = next_close
    shifted[0, -1, 4] = next_volume

    result = {
        "label": prediction.label,
        "confidence": float(prediction.confidence),
        "horizon_labels": horizon_labels,
        "horizon_confidences": horizon_confidences,
        "close": next_close,
        "low": next_low,
        "high": next_high,
        "trend_score": trend_score,
    }
    return shifted, result


def recursive_forecast(
    model: MultiHorizonTrendTransformer,
    bundle: ProcessedBundle,
    steps: int,
    device: str | torch.device,
) -> dict[str, object]:
    steps = max(int(steps), 1)
    current_window = torch.from_numpy(bundle.all_x[-1:]).float().to(device)
    forecast_rows: list[dict[str, float | int | str]] = []
    predicted_closes: list[float] = []
    predictions: list[ForecastPoint] = []

    for step in range(1, steps + 1):
        current_window, step_result = _forecast_step(model, current_window)
        predicted_closes.append(float(step_result["close"]))
        predictions.append(
            ForecastPoint(
                step=step,
                label=str(step_result["label"]),
                confidence=float(step_result["confidence"]),
                close=float(step_result["close"]),
                low=float(step_result["low"]),
                high=float(step_result["high"]),
            )
        )
        forecast_rows.append(
            {
                "step": step,
                "label": step_result["label"],
                "confidence": float(step_result["confidence"]),
                "close": float(step_result["close"]),
                "low": float(step_result["low"]),
                "high": float(step_result["high"]),
                "trend_score": float(step_result["trend_score"]),
            }
        )

    smoothed_closes = _smooth_values(predicted_closes)
    focus_timesteps = extract_asta_focus_indices(torch.from_numpy(bundle.raw_windows[-1]).float()) if len(bundle.raw_windows) else []
    volatility_scores = bundle.raw_windows[-1].var(axis=1).tolist() if len(bundle.raw_windows) else []
    last_prediction = predictions[-1]
    market_regime = market_regime_from_series(bundle.closes, volatility_scores)
    explanation = (
        f"The model leans {last_prediction.label.lower()} because the recent window shows "
        f"{market_regime.lower()} behavior, with the strongest ASTA focus around the last "
        f"{len(focus_timesteps)} timesteps and volatility concentrated in the latest window."
    )
    return {
        "label": last_prediction.label,
        "confidence": last_prediction.confidence,
        "predicted_close": smoothed_closes[-1] if smoothed_closes else last_prediction.close,
        "low": min(row["low"] for row in forecast_rows),
        "high": max(row["high"] for row in forecast_rows),
        "forecast_curve": forecast_rows,
        "focus_timesteps": focus_timesteps,
        "volatility_scores": volatility_scores,
        "market_regime": market_regime,
        "explanation": explanation,
    }


def forecast_for_date(
    model: MultiHorizonTrendTransformer,
    bundle: ProcessedBundle,
    future_date: str | date | datetime,
    device: str | torch.device,
) -> ForecastResult:
    step_offset = business_day_offset(bundle.dates[-1], future_date)
    forecast = recursive_forecast(model=model, bundle=bundle, steps=step_offset, device=device)
    last_point = forecast["forecast_curve"][-1]
    return ForecastResult(
        target_date=str(parse_future_date(future_date).date()),
        step_offset=step_offset,
        label=str(forecast["label"]),
        confidence=float(forecast["confidence"]),
        predicted_close=float(last_point["close"]),
        low=float(forecast["low"]),
        high=float(forecast["high"]),
        explanation=str(forecast["explanation"]),
        forecast_curve=list(forecast["forecast_curve"]),
        focus_timesteps=list(forecast["focus_timesteps"]),
        volatility_scores=list(forecast["volatility_scores"]),
        market_regime=str(forecast["market_regime"]),
    )