from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans

from .ann_index import ANNIndex, ANNPrediction, TREND_CLASSES

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Hybrid Search Optimization Pipeline.

    1. KMeans Clustering to partition the embedding space.
    2. HNSW ANN indices per cluster for fast similarity search.
    3. Exact Cosine Re-Ranking of top candidates.
    4. Time-Aware Scoring: Score = Similarity * exp(-λ * time_difference).
    """

    def __init__(self, dim: int, n_clusters: int = 5, use_hnsw: bool = True):
        self.dim = dim
        self.n_clusters = n_clusters
        self.use_hnsw = use_hnsw
        self.kmeans: KMeans | None = None
        self.cluster_indices: dict[int, ANNIndex] = {}

        # Store exact embeddings and metadata for re-ranking
        self._exact_embeddings: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._time_indices: np.ndarray | None = None
        self._global_count: int = 0
        self.backend = "hybrid-hnsw" if use_hnsw else "hybrid-lsh"

    def build(self, embeddings: np.ndarray, labels: np.ndarray, time_indices: np.ndarray | None = None) -> None:
        n = len(embeddings)
        if n == 0:
            raise ValueError("Cannot build index on empty embeddings.")

        self._exact_embeddings = embeddings.astype(np.float32)
        self._labels = labels.astype(np.int64)
        if time_indices is None:
            # Assume chronological order
            self._time_indices = np.arange(n)
        else:
            self._time_indices = time_indices

        self._global_count = n

        # 1. K-Means clustering
        actual_clusters = min(self.n_clusters, max(1, n // 10))
        self.kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init="auto")
        cluster_labels = self.kmeans.fit_predict(self._exact_embeddings)

        # 2. Build ANN Index per cluster
        for c in range(actual_clusters):
            mask = cluster_labels == c
            if not mask.any():
                continue

            cluster_embs = self._exact_embeddings[mask]
            # Store the original indices (0 to N-1) as the 'label' in the ANN
            # so we can map back to exact embeddings during re-ranking.
            original_indices = np.where(mask)[0]

            idx = ANNIndex(dim=self.dim, use_hnsw=self.use_hnsw)
            idx.build(cluster_embs, original_indices)
            self.cluster_indices[c] = idx

        logger.info("HybridSearcher built: %d clusters, %d total elements", actual_clusters, n)

    def predict(
        self,
        query: np.ndarray,
        query_time: int | None = None,
        k: int = 7,
        candidates_m: int = 50,
        time_lambda: float = 0.01
    ) -> ANNPrediction:
        """Query the hybrid searcher with time-aware scoring and re-ranking."""
        if self.kmeans is None or self._exact_embeddings is None:
            raise RuntimeError("Index not built.")

        q = query.reshape(1, -1).astype(np.float32)
        if query_time is None:
            query_time = self._global_count  # Assume query is the newest point

        # 1. Find nearest cluster
        c = int(self.kmeans.predict(q)[0])
        idx = self.cluster_indices.get(c)
        if idx is None:
            # Fallback to linear search if cluster is empty (rare)
            idx = next(iter(self.cluster_indices.values()))

        # 2. ANN retrieval -> get top M candidates
        # The query returns 'original_indices' because we mapped them during build
        retrieved_indices, _ = idx.query(q, k=min(candidates_m, idx.element_count))
        # Ensure indices are integers
        retrieved_indices = retrieved_indices.astype(np.int64)

        # 3. Exact Cosine Re-ranking
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        candidate_embs = self._exact_embeddings[retrieved_indices]
        candidate_embs_norm = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-9)

        # Cosine similarity is dot product of normalized vectors
        similarities = (candidate_embs_norm @ q_norm.T).flatten()

        # 4. Time-Aware Scoring
        cand_times = self._time_indices[retrieved_indices]
        time_diffs = np.maximum(0, query_time - cand_times)

        # Score = Similarity * exp(-lambda * time_difference)
        # Shift similarities to be positive (0 to 2) since cosine is -1 to 1
        pos_similarities = similarities + 1.0
        scores = pos_similarities * np.exp(-time_lambda * time_diffs)

        # 5. Select top k based on Score
        top_k_rel_idx = np.argsort(scores)[::-1][:min(k, len(scores))]
        top_k_indices = retrieved_indices[top_k_rel_idx]
        top_k_scores = scores[top_k_rel_idx]

        # 6. Majority Voting
        top_k_labels = self._labels[top_k_indices]

        # Weigh votes by the time-aware score
        weights = top_k_scores / (top_k_scores.sum() + 1e-9)

        vote_scores = np.zeros(3, dtype=np.float64)
        for label_int, w in zip(top_k_labels, weights):
            vote_scores[int(label_int)] += w

        predicted_idx = int(np.argmax(vote_scores))
        probabilities = (vote_scores / (vote_scores.sum() + 1e-9)).tolist()

        return ANNPrediction(
            label=TREND_CLASSES[predicted_idx],
            confidence=float(probabilities[predicted_idx]),
            probabilities=probabilities,
            neighbour_labels=[TREND_CLASSES[int(lbl)] for lbl in top_k_labels],
            neighbour_distances=(2.0 - pos_similarities[top_k_rel_idx]).tolist(),  # Convert score back to distance metric format
            index_type=self.backend,
            k=len(top_k_labels)
        )

    @property
    def element_count(self) -> int:
        return self._global_count

    def is_built(self) -> bool:
        return self._global_count > 0

    def to_bytes(self) -> bytes:
        import pickle
        state = {
            "dim": self.dim,
            "n_clusters": self.n_clusters,
            "use_hnsw": self.use_hnsw,
            "kmeans": self.kmeans,
            "exact_embeddings": self._exact_embeddings,
            "labels": self._labels,
            "time_indices": self._time_indices,
            "global_count": self._global_count,
            "cluster_indices": {k: v.to_bytes() for k, v in self.cluster_indices.items()}
        }
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, data: bytes) -> "HybridSearcher":
        import pickle
        state = pickle.loads(data)
        searcher = cls(dim=state["dim"], n_clusters=state["n_clusters"], use_hnsw=state["use_hnsw"])
        searcher.kmeans = state["kmeans"]
        searcher._exact_embeddings = state["exact_embeddings"]
        searcher._labels = state["labels"]
        searcher._time_indices = state["time_indices"]
        searcher._global_count = state["global_count"]
        for k, v in state["cluster_indices"].items():
            searcher.cluster_indices[k] = ANNIndex.from_bytes(v)
        return searcher
