from __future__ import annotations

import math

import torch
from torch import nn


class TrendAwarePositionalEncoding(nn.Module):
    """TAPE: Trend-Aware Positional Encoding.

    Standard sinusoidal encodings provide order information, but stock series also
    benefit from market context. TAPE combines:
    - sinusoidal position vectors for absolute order
    - momentum modulation to emphasize directional price changes
    - volatility modulation to highlight uncertain periods

    The operation stays O(T * d): we only scale a precomputed sinusoidal tensor
    with batch-wise market factors instead of introducing extra quadratic attention.
    """

    def __init__(self, hidden_size: int, max_length: int = 512) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(1, max_length, hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("base_encoding", pe)
        self.momentum_scale = nn.Parameter(torch.tensor(0.35))
        self.volatility_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        if sequence_length > self.max_length:
            raise ValueError(f"Sequence length {sequence_length} exceeds max_length {self.max_length}")

        close_values = x[:, :, 3]
        momentum = close_values[:, -1] - close_values[:, 0]
        volatility = close_values.std(dim=1)
        momentum_factor = torch.tanh(momentum).view(batch_size, 1, 1)
        volatility_factor = torch.tanh(volatility).view(batch_size, 1, 1)

        sinusoidal = self.base_encoding[:, :sequence_length, :]
        modulation = 1.0 + self.momentum_scale * momentum_factor + self.volatility_scale * volatility_factor
        bias = 0.1 * (momentum_factor - volatility_factor)
        return x + sinusoidal * modulation + bias
