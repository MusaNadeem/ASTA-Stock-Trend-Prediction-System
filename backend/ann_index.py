"""ann_index.py – Approximate Nearest-Neighbour (ANN) index for embedding-based prediction.

Pipeline (Step 2 + 3):
  Step 2 – Build an ANN index (HNSW via *hnswlib* when available, otherwise a
            pure-NumPy LSH-style random-projection index) over transformer
            embeddings extracted from all training windows.
  Step 3 – At query time, retrieve the k nearest neighbours and return the
            majority-voted label across their ground-truth annotations.

The index is kept in RAM and rebuilt whenever the model is retrained.  It is
also serialised alongside the .pt checkpoint so it can be reloaded instantly
without re-embedding the entire corpus.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import hnswlib (HNSW).  Fall back silently to LSH.
# ---------------------------------------------------------------------------
try:
    import hnswlib  # type: ignore

    _HNSW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HNSW_AVAILABLE = False
    logger.info("hnswlib not installed – using pure-NumPy LSH fallback.")

TREND_CLASSES = ("Downtrend", "Neutral", "Uptrend")


# ---------------------------------------------------------------------------
# Data container returned by the ANN predictor
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ANNPrediction:
    """Prediction produced by the ANN + majority-voting step."""

    label: str                          # majority-voted class name
    confidence: float                   # fraction of neighbours that agree
    probabilities: list[float]          # [P(Down), P(Neutral), P(Up)]
    neighbour_labels: list[str]         # raw labels of the k neighbours
    neighbour_distances: list[float]    # L2 distances to the k neighbours
    index_type: Literal["hnsw", "lsh"]  # which backend was used
    k: int                              # number of neighbours queried


# ---------------------------------------------------------------------------
# LSH (random-projection) fallback index
# ---------------------------------------------------------------------------

class _LSHIndex:
    """Locality-Sensitive Hashing using random projections (no dependencies).

    Each embedding is projected onto *n_planes* random hyperplanes, yielding a
    binary hash.  At query time we scan the candidates whose hash is within a
    Hamming distance of 1 bit (± 1 bucket) from the query hash, then rank them
    by true L2 distance.  This gives a sub-linear average search complexity of
    O(dim × candidates) where candidates ≪ N for large corpora.
    """

    def __init__(self, dim: int, n_planes: int = 12, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # Projection matrix: shape (dim, n_planes)
        self.planes: np.ndarray = rng.standard_normal((dim, n_planes)).astype(np.float32)
        self.dim = dim
        self.n_planes = n_planes
        # Storage
        self._embeddings: np.ndarray | None = None   # (N, dim)
        self._labels: np.ndarray | None = None       # (N,) int
        self._hashes: np.ndarray | None = None       # (N, n_planes) bool

    def _hash(self, embeddings: np.ndarray) -> np.ndarray:
        """Return a boolean hash matrix (N, n_planes)."""
        projected = embeddings @ self.planes        # (N, n_planes)
        return projected >= 0                       # bit per plane

    def add_items(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Index *embeddings* (N, dim) with corresponding integer *labels*."""
        self._embeddings = embeddings.astype(np.float32)
        self._labels = labels.astype(np.int64)
        self._hashes = self._hash(self._embeddings)

    def knn_query(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, distances) for the *k* nearest neighbours.

        Parameters
        ----------
        query:
            Shape (1, dim) or (dim,).
        k:
            Number of neighbours.

        Returns
        -------
        labels : np.ndarray shape (k,)
        distances : np.ndarray shape (k,)
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            raise RuntimeError("LSH index is empty – call add_items first.")
        q = query.reshape(1, -1).astype(np.float32)
        q_hash = self._hash(q)  # (1, n_planes)
        # Hamming distance between query hash and all stored hashes
        hamming = np.sum(self._hashes ^ q_hash, axis=1)  # (N,)
        # Retrieve candidates within hamming distance ≤ 1 (at least k items)
        threshold = 0
        candidate_mask = hamming <= threshold
        while candidate_mask.sum() < min(k, len(self._embeddings)):
            threshold += 1
            candidate_mask = hamming <= threshold
            if threshold >= self.n_planes:
                candidate_mask = np.ones(len(self._embeddings), dtype=bool)
                break
        candidate_idx = np.where(candidate_mask)[0]
        # Rank candidates by true L2 distance
        diffs = self._embeddings[candidate_idx] - q  # (C, dim)
        dists = np.sqrt((diffs ** 2).sum(axis=1))   # (C,)
        order = np.argsort(dists)[:k]
        top_idx = candidate_idx[order]
        top_dists = dists[order]
        return top_idx, top_dists

    @property
    def element_count(self) -> int:
        return len(self._embeddings) if self._embeddings is not None else 0


# ---------------------------------------------------------------------------
# Unified ANN index facade
# ---------------------------------------------------------------------------

class ANNIndex:
    """Wraps either an HNSW index (hnswlib) or an LSH index.

    Usage
    -----
    >>> idx = ANNIndex(dim=64)
    >>> idx.build(embeddings, labels)          # embeddings: (N, 64)
    >>> pred = idx.predict(query_embedding, k=7)
    """

    def __init__(self, dim: int, *, use_hnsw: bool = True) -> None:
        self.dim = dim
        self.backend: Literal["hnsw", "lsh"] = "hnsw" if (_HNSW_AVAILABLE and use_hnsw) else "lsh"
        self._labels: np.ndarray | None = None  # (N,) ground truth int labels
        self._hnsw_index: "hnswlib.Index | None" = None  # type: ignore[name-defined]
        self._lsh_index: _LSHIndex | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Index *embeddings* (N, dim) with int *labels* (N,).

        Supports incremental rebuilds: calling build() again replaces the
        existing index with fresh data.
        """
        n = len(embeddings)
        if n == 0:
            raise ValueError("Cannot build ANN index on an empty embedding set.")
        self._labels = labels.astype(np.int64)

        if self.backend == "hnsw":
            self._build_hnsw(embeddings)
        else:
            self._build_lsh(embeddings)

        logger.info(
            "ANNIndex built: backend=%s  n=%d  dim=%d",
            self.backend, n, self.dim,
        )

    def _build_hnsw(self, embeddings: np.ndarray) -> None:
        import hnswlib  # local to avoid global-scope import issues
        idx = hnswlib.Index(space="cosine", dim=self.dim)
        idx.init_index(max_elements=len(embeddings), ef_construction=400, M=32)
        idx.add_items(embeddings.astype(np.float32), np.arange(len(embeddings)))
        idx.set_ef(100)
        self._hnsw_index = idx

    def _build_lsh(self, embeddings: np.ndarray) -> None:
        lsh = _LSHIndex(dim=self.dim, n_planes=min(16, self.dim))
        lsh.add_items(embeddings.astype(np.float32), self._labels)
        self._lsh_index = lsh

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, embedding: np.ndarray, k: int = 7) -> tuple[np.ndarray, np.ndarray]:
        """Return (label_array, distance_array) for the top-k neighbours."""
        k_actual = min(k, self.element_count)
        if k_actual == 0:
            raise RuntimeError("ANN index is empty.")
        q = embedding.reshape(1, -1).astype(np.float32)
        if self.backend == "hnsw" and self._hnsw_index is not None:
            indices, distances = self._hnsw_index.knn_query(q, k=k_actual)
            indices = indices[0]
            distances = distances[0]
            neighbour_labels = self._labels[indices]
        else:
            assert self._lsh_index is not None
            indices, distances = self._lsh_index.knn_query(q, k=k_actual)
            neighbour_labels = self._labels[indices]
        return neighbour_labels, distances

    # ------------------------------------------------------------------
    # Step 3 – Majority voting
    # ------------------------------------------------------------------

    def predict(self, embedding: np.ndarray, k: int = 7) -> ANNPrediction:
        """ANN retrieval → weighted majority vote → ANNPrediction."""
        neighbour_labels, distances = self.query(embedding, k=k)

        # Exponential decay distance weighting: much more robust for cosine distances
        # than simple inverse distance, providing smoother neighbor influence.
        temperature = 0.1
        weights = np.exp(-distances / temperature)
        weights = weights / weights.sum()

        vote_scores = np.zeros(3, dtype=np.float64)
        for label_int, w in zip(neighbour_labels, weights):
            vote_scores[int(label_int)] += w

        predicted_idx = int(np.argmax(vote_scores))
        probabilities = (vote_scores / vote_scores.sum()).tolist()
        label = TREND_CLASSES[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        return ANNPrediction(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            neighbour_labels=[TREND_CLASSES[int(lbl)] for lbl in neighbour_labels],
            neighbour_distances=distances.tolist(),
            index_type=self.backend,
            k=len(neighbour_labels),
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @property
    def element_count(self) -> int:
        if self.backend == "hnsw" and self._hnsw_index is not None:
            return self._hnsw_index.element_count
        if self._lsh_index is not None:
            return self._lsh_index.element_count
        return 0

    def is_built(self) -> bool:
        return self.element_count > 0

    def to_bytes(self) -> bytes:
        """Serialise the index to a bytes object (for checkpoint embedding)."""
        state = {
            "dim": self.dim,
            "backend": self.backend,
            "labels": self._labels,
        }
        if self.backend == "hnsw" and self._hnsw_index is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
                tmp_path = tmp.name
            try:
                self._hnsw_index.save_index(tmp_path)
                with open(tmp_path, "rb") as fh:
                    state["hnsw_bytes"] = fh.read()
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            state["lsh"] = self._lsh_index
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ANNIndex":
        """Restore an ANNIndex from bytes produced by to_bytes()."""
        state = pickle.loads(data)
        idx = cls(dim=state["dim"])
        idx.backend = state["backend"]
        idx._labels = state["labels"]
        if state["backend"] == "hnsw" and "hnsw_bytes" in state:
            import hnswlib, tempfile
            tmp_path = tempfile.mktemp(suffix=".bin")
            try:
                with open(tmp_path, "wb") as fh:
                    fh.write(state["hnsw_bytes"])
                hnsw_idx = hnswlib.Index(space="cosine", dim=state["dim"])
                hnsw_idx.load_index(tmp_path)
                hnsw_idx.set_ef(100)
                idx._hnsw_index = hnsw_idx
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            idx._lsh_index = state.get("lsh")
        return idx
