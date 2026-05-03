from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .ann_index import ANNPrediction
from .search import HybridSearcher
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
        # Step 2: Hybrid ANN index (built after training via build_ann_index)
        self.ann_index: HybridSearcher | None = None

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

    # ------------------------------------------------------------------
    # Step 1: Embedding extractor
    # ------------------------------------------------------------------

    def embed(self, x: torch.Tensor) -> np.ndarray:
        """Return a 1-D NumPy embedding vector for a single window batch.

        Parameters
        ----------
        x : Tensor of shape (1, T, feature_size)

        Returns
        -------
        np.ndarray of shape (hidden_size,)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            # "features" is the pooled representation produced in forward()
            return outputs["features"][0].detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Step 2: Build ANN index from training embeddings
    # ------------------------------------------------------------------

    def build_ann_index(
        self,
        all_x: np.ndarray,
        all_y: np.ndarray,
        *,
        device: torch.device | None = None,
        k_vote: int = 7,
        use_hnsw: bool = True,
        batch_size: int = 256,
    ) -> None:
        """Encode every sample in *all_x* and build the ANN index.

        Parameters
        ----------
        all_x   : (N, T, feature_size) normalised feature windows.
        all_y   : (N, H) integer label array (multi-horizon).  The majority-
                  voted label across horizons is used as the ground-truth label
                  stored in the index.
        device  : torch device for encoding.
        k_vote  : default k used later in predict(); stored for reference only.
        use_hnsw: prefer HNSW when hnswlib is available.
        batch_size: number of windows encoded per forward pass.
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()

        # --- derive a single integer label per sample (majority vote across
        #     horizons) so the ANN index stores one label per embedding.
        vote_weights = self.vote_weights.to(device)
        labels_tensor = torch.from_numpy(all_y).long().to(device)
        with torch.no_grad():
            flat_labels = MultiHorizonVotingPredictor.weighted_vote_targets(
                labels_tensor, vote_weights=vote_weights
            ).cpu().numpy()  # (N,)

        # --- encode in batches
        all_embeddings: list[np.ndarray] = []
        n = len(all_x)
        x_tensor = torch.from_numpy(all_x).float()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = x_tensor[start : start + batch_size].to(device)
                outputs = self.forward(batch)
                emb = outputs["features"].detach().cpu().numpy()  # (B, hidden_size)
                all_embeddings.append(emb)
        embeddings = np.concatenate(all_embeddings, axis=0)  # (N, hidden_size)

        # --- build & fit the hybrid index
        self.ann_index = HybridSearcher(dim=self.hidden_size, use_hnsw=use_hnsw, n_clusters=5)
        # Using chronological order for time-aware scoring
        time_indices = np.arange(len(embeddings))
        self.ann_index.build(embeddings, flat_labels, time_indices=time_indices)

    # ------------------------------------------------------------------
    # Step 3: Predict using ANN + majority voting (with MHVP fallback)
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor, k: int = 7) -> TrendPrediction:
        """Predict the trend for window *x*.

        When an ANN index has been built (Step 2), the prediction flows through
        the full pipeline::

            Transformer → embedding → ANN k-NN → majority vote → label

        Otherwise it falls back to the original MHVP path.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            # Always run MHVP so we can return the full TrendPrediction struct
            mhvp_result: TrendPrediction = self.vote_predictor.predict(outputs["horizon_logits"])

            if self.ann_index is not None and self.ann_index.is_built():
                # Step 1 output is already in outputs["features"]
                query_emb = outputs["features"][0].detach().cpu().numpy()
                # Step 2+3: ANN search + majority vote
                ann_pred: ANNPrediction = self.ann_index.predict(query_emb, k=k)
                # Merge ANN result into the existing TrendPrediction dataclass
                return TrendPrediction(
                    label=ann_pred.label,
                    confidence=ann_pred.confidence,
                    probabilities=ann_pred.probabilities,
                    horizon_probabilities=mhvp_result.horizon_probabilities,
                    horizon_labels=mhvp_result.horizon_labels,
                    horizon_confidences=mhvp_result.horizon_confidences,
                    horizon_predictions=mhvp_result.horizon_predictions,
                )
            return mhvp_result

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
