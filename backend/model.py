from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .attention import ASTATransformerBlock
from .encoding import TrendAwarePositionalEncoding
from .predictor import MultiHorizonVotingPredictor, TrendPrediction, TREND_CLASSES


class StandardTransformerBlock(nn.Module):
    """Dense transformer block used when ASTA is disabled.

    This block keeps the repository backward compatible by providing the classic
    O(T^2 × d) attention path alongside ASTA. The model can toggle between the
    two mechanisms through the `use_standard_attention` flag.
    """

    def __init__(self, hidden_size: int, num_heads: int = 4, ff_multiplier: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * ff_multiplier, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


@dataclass(slots=True)
class TrendPrediction:
    label: str
    confidence: float
    probabilities: list[float]
    horizon_probabilities: dict[str, list[float]]
    horizon_labels: dict[str, str]
    horizon_confidences: dict[str, float]
    horizon_predictions: dict[str, dict[str, object]]


class MultiHorizonTrendTransformer(nn.Module):
    def __init__(
        self,
        feature_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        horizons: Sequence[int] = (1, 3, 5),
        num_classes: int = 3,
        use_standard_attention: bool = False,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.horizons = tuple(int(h) for h in horizons)
        self.num_classes = num_classes
        self.use_standard_attention = use_standard_attention
        self.attention_mode = "standard" if use_standard_attention else "asta"
        self.input_projection = nn.Linear(feature_size, hidden_size)
        self.positional_encoding = TrendAwarePositionalEncoding(hidden_size=hidden_size)
        if use_standard_attention:
            self.blocks = nn.ModuleList(
                [StandardTransformerBlock(hidden_size=hidden_size, num_heads=num_heads) for _ in range(num_layers)]
            )
        else:
            self.blocks = nn.ModuleList(
                [ASTATransformerBlock(hidden_size=hidden_size, num_heads=num_heads) for _ in range(num_layers)]
            )
        self.norm = nn.LayerNorm(hidden_size)
        self.pool = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, num_classes) for _ in self.horizons])
        self.vote_predictor = MultiHorizonVotingPredictor(horizons=self.horizons)
        self.vote_weights = self.vote_predictor.vote_weights.clone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(x)
        pooled_mean = encoded.mean(dim=1)
        pooled_last = encoded[:, -1, :]
        features = self.pool(torch.cat([pooled_mean, pooled_last], dim=-1))
        logits = [head(features) for head in self.heads]
        return {"horizon_logits": torch.stack(logits, dim=1), "features": features}

    @staticmethod
    def weighted_vote_targets(labels: torch.Tensor, vote_weights: torch.Tensor | None = None) -> torch.Tensor:
        if vote_weights is None:
            vote_weights = torch.tensor([0.5, 0.3, 0.2], device=labels.device, dtype=torch.float32)
        return MultiHorizonVotingPredictor.weighted_vote_targets(labels, vote_weights=vote_weights)

    def predict(self, x: torch.Tensor) -> TrendPrediction:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            return self.vote_predictor.predict(outputs["horizon_logits"])

    def forecast_steps(self, x: torch.Tensor, steps: int = 1, device: torch.device | None = None) -> dict[str, object]:
        """Perform recursive extrapolation on top of the current sequence.

        The model still produces the same horizon logits, but the forecasting
        engine reuses the latest synthetic window at each step. This preserves the
        existing training path while enabling multi-step forward simulation.
        """

        from .forecasting import recursive_forecast

        if device is None:
            device = x.device
        bundle_stub = type(
            "BundleStub",
            (),
            {
                "all_x": x.detach().cpu().numpy(),
                "raw_windows": x.detach().cpu().numpy(),
                "closes": x[0, :, 3].detach().cpu().numpy(),
                "dates": [f"T{i + 1}" for i in range(x.shape[1])],
            },
        )
        return recursive_forecast(model=self, bundle=bundle_stub, steps=steps, device=device)
