from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class AttentionBenchmark:
    sequence_length: int
    hidden_size: int
    batch_size: int
    standard_ms: float
    asta_ms: float
    speedup: float


def standard_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Dense scaled dot-product attention.

    Complexity: O(T^2 * d) because every query interacts with every key.
    """
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value)


class AdaptiveSparseTemporalAttention(nn.Module):
    """ASTA keeps only local, logarithmic, and volatility-driven key positions.

    Complexity: approximately O(T * (k + log T + v) * d), where:
    - k is the local window size
    - log T is the number of logarithmic samples
    - v is the volatility-selected set size
    """

    def __init__(self, hidden_size: int, num_heads: int = 4, local_k: int = 8, volatility_k: int = 6) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.local_k = local_k
        self.volatility_k = volatility_k
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _build_candidate_matrix(self, values: torch.Tensor) -> torch.Tensor:
        total_steps = values.shape[0]
        # Volatility is measured per timestep so we can prioritize periods where
        # the market is less stable and long-range context is more informative.
        volatility_scores = values.var(dim=-1)
        topk = min(self.volatility_k, total_steps)
        volatility_indices = torch.topk(volatility_scores, k=topk).indices.tolist()
        candidate_rows: list[list[int]] = []
        max_candidates = 0
        for time_index in range(total_steps):
            # Local attention keeps the last k steps, while logarithmic sampling
            # hops backward exponentially to preserve long-range dependencies.
            local_start = max(0, time_index - self.local_k + 1)
            candidates = set(range(local_start, time_index + 1))
            hop = 1
            while hop <= total_steps and time_index - hop >= 0:
                candidates.add(time_index - hop)
                hop *= 2
            candidates.update(index for index in volatility_indices if index <= time_index)
            ordered = sorted(index for index in candidates if index <= time_index) or [time_index]
            candidate_rows.append(ordered)
            max_candidates = max(max_candidates, len(ordered))

        candidate_matrix = torch.empty(total_steps, max_candidates, dtype=torch.long, device=values.device)
        for time_index, ordered in enumerate(candidate_rows):
            padded = ordered + [ordered[-1]] * (max_candidates - len(ordered))
            candidate_matrix[time_index] = torch.tensor(padded, device=values.device)
        return candidate_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        q = self.q_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        outputs = []
        for batch_index in range(batch_size):
            candidate_matrix = self._build_candidate_matrix(x[batch_index])
            candidate_count = candidate_matrix.shape[1]
            expanded_k = k[batch_index].unsqueeze(1).expand(self.num_heads, sequence_length, sequence_length, self.head_dim)
            expanded_v = v[batch_index].unsqueeze(1).expand(self.num_heads, sequence_length, sequence_length, self.head_dim)
            gather_index = candidate_matrix.view(1, sequence_length, candidate_count, 1).expand(self.num_heads, sequence_length, candidate_count, self.head_dim)
            selected_k = torch.gather(expanded_k, 2, gather_index)
            selected_v = torch.gather(expanded_v, 2, gather_index)
            query = q[batch_index].unsqueeze(-2)
            scores = (query * selected_k).sum(dim=-1) / math.sqrt(self.head_dim)
            weights = torch.softmax(scores, dim=-1)
            attended = (weights.unsqueeze(-1) * selected_v).sum(dim=2)
            outputs.append(attended.permute(1, 0, 2))
        merged = torch.stack(outputs, dim=0).contiguous().view(batch_size, sequence_length, self.hidden_size)
        return self.out_proj(merged)


def extract_asta_focus_indices(values: torch.Tensor, local_k: int = 8, volatility_k: int = 6) -> list[int]:
    """Return the timesteps ASTA would likely focus on for a single window.

    The returned set mirrors the ASTA candidate construction: recent local points,
    logarithmically spaced past points, and the most volatile timesteps.
    """

    if values.ndim != 2:
        raise ValueError("values must have shape [T, d]")
    total_steps = values.shape[0]
    volatility_scores = values.var(dim=-1)
    topk = min(volatility_k, total_steps)
    volatility_indices = torch.topk(volatility_scores, k=topk).indices.tolist()
    candidates: set[int] = set(volatility_indices)
    candidates.update(range(max(0, total_steps - local_k), total_steps))

    step = 1
    while step <= total_steps:
        candidates.add(max(0, total_steps - step))
        step *= 2

    return sorted(candidates)


class ASTATransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, ff_multiplier: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = AdaptiveSparseTemporalAttention(hidden_size=hidden_size, num_heads=num_heads)
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
        x = self.norm1(x + self.dropout(self.attention(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


@torch.no_grad()
def compare_runtime_standard_vs_asta(
    sequence_length: int = 60,
    hidden_size: int = 64,
    batch_size: int = 8,
    runs: int = 20,
    device: str | torch.device = "cpu",
) -> AttentionBenchmark:
    generator = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(batch_size, sequence_length, hidden_size, generator=generator, device=device)
    q = x.clone()
    k = x.clone()
    v = x.clone()
    asta = AdaptiveSparseTemporalAttention(hidden_size=hidden_size).to(device).eval()
    warmup = 3
    for _ in range(warmup):
        _ = standard_attention(q, k, v)
        _ = asta(x)
    start = time.perf_counter()
    for _ in range(runs):
        _ = standard_attention(q, k, v)
    standard_elapsed = (time.perf_counter() - start) / runs * 1000.0

    start = time.perf_counter()
    for _ in range(runs):
        _ = asta(x)
    asta_elapsed = (time.perf_counter() - start) / runs * 1000.0

    speedup = standard_elapsed / max(asta_elapsed, 1e-8)
    return AttentionBenchmark(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        batch_size=batch_size,
        standard_ms=standard_elapsed,
        asta_ms=asta_elapsed,
        speedup=speedup,
    )
