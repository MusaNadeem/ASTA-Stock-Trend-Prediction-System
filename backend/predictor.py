from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import torch

TREND_CLASSES = ("Downtrend", "Neutral", "Uptrend")


@dataclass(slots=True)
class TrendPrediction:
    """Final consensus and horizon-level predictions from MHVP."""

    label: str
    confidence: float
    probabilities: list[float]
    horizon_probabilities: dict[str, list[float]]
    horizon_labels: dict[str, str]
    horizon_confidences: dict[str, float]
    horizon_predictions: dict[str, dict[str, object]]


class MultiHorizonVotingPredictor:
    """Multi-Horizon Voting Predictor (MHVP).

    Each horizon produces an independent trend distribution. The final trend is
    obtained through weighted voting over the horizon probabilities. This keeps
    the decision layer simple, interpretable, and easy to analyze in a DAA course
    setting while still exploiting short-, mid-, and long-range signals.
    """

    def __init__(self, horizons: Sequence[int], vote_weights: Sequence[float] | None = None) -> None:
        self.horizons = tuple(int(h) for h in horizons)
        if vote_weights is None:
            vote_weights = (0.5, 0.3, 0.2)
        weights = torch.tensor(list(vote_weights), dtype=torch.float32)
        if len(weights) != len(self.horizons):
            weights = torch.ones(len(self.horizons), dtype=torch.float32)
        self.vote_weights = weights / weights.sum().clamp_min(1e-8)

    def _horizon_name(self, index: int) -> str:
        names = ("short_term", "mid_term", "long_term")
        if index < len(names):
            return names[index]
        return f"horizon_{self.horizons[index]}"

    def predict(self, horizon_logits: torch.Tensor) -> TrendPrediction:
        horizon_probs = torch.softmax(horizon_logits, dim=-1)
        vote_weights = self.vote_weights.to(horizon_probs.device)
        combined_probs = (horizon_probs * vote_weights.view(1, -1, 1)).sum(dim=1)
        confidence, predicted_index = combined_probs.max(dim=-1)

        horizon_probabilities: dict[str, list[float]] = {}
        horizon_labels: dict[str, str] = {}
        horizon_confidences: dict[str, float] = {}
        horizon_predictions: dict[str, dict[str, object]] = {}

        for index, horizon in enumerate(self.horizons):
            horizon_key = self._horizon_name(index)
            horizon_distribution = horizon_probs[0, index]
            horizon_confidence, horizon_prediction = horizon_distribution.max(dim=-1)
            horizon_probabilities[horizon_key] = horizon_distribution.detach().cpu().tolist()
            horizon_label = TREND_CLASSES[int(horizon_prediction.item())]
            horizon_labels[horizon_key] = horizon_label
            horizon_confidences[horizon_key] = float(horizon_confidence.item())
            horizon_predictions[horizon_key] = {
                "horizon": horizon,
                "label": horizon_label,
                "confidence": float(horizon_confidence.item()),
                "probabilities": horizon_distribution.detach().cpu().tolist(),
            }

        return TrendPrediction(
            label=TREND_CLASSES[int(predicted_index.item())],
            confidence=float(confidence.item()),
            probabilities=combined_probs[0].detach().cpu().tolist(),
            horizon_probabilities=horizon_probabilities,
            horizon_labels=horizon_labels,
            horizon_confidences=horizon_confidences,
            horizon_predictions=horizon_predictions,
        )

    @staticmethod
    def weighted_vote_targets(labels: torch.Tensor, vote_weights: torch.Tensor | None = None) -> torch.Tensor:
        if vote_weights is None:
            vote_weights = torch.tensor([0.5, 0.3, 0.2], device=labels.device, dtype=torch.float32)
        vote_scores = torch.zeros(labels.shape[0], 3, device=labels.device, dtype=torch.float32)
        for horizon_index in range(labels.shape[1]):
            vote_scores.scatter_add_(1, labels[:, horizon_index : horizon_index + 1], vote_weights[horizon_index].expand(labels.shape[0], 1))
        return vote_scores.argmax(dim=1)


def future_date_offset(last_date: str | date | datetime, future_date: str | date | datetime) -> int:
    from .forecasting import business_day_offset

    return business_day_offset(last_date, future_date)


def predict_for_date(model, bundle, future_date: str | date | datetime, device: str | torch.device):
    from .forecasting import forecast_for_date

    return forecast_for_date(model=model, bundle=bundle, future_date=future_date, device=device)
