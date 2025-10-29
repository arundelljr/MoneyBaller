from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .preprocessor import preprocess_csv, MoneyballPreprocessor


@dataclass
class PCAKNN:
    """
    A small utility that performs:
    - PCA dimensionality reduction on preprocessed player feature matrix X
    - Nearest-neighbor search on the PCA space

    Typical workflow:
      pre = MoneyballPreprocessor().fit(raw_df)
      X, info = pre.transform(raw_df)
      model = PCAKNN(n_components=20, n_neighbors=15).fit(X)
      sim = model.find_similar(player_id=12345, top_k=10, info=info)
    """

    n_components: Union[int, float] = 20
    n_neighbors: int = 15
    metric: str = "euclidean"
    algorithm: str = "auto"
    pca_random_state: Optional[int] = 42

    pca_: PCA = field(init=False)
    knn_: NearestNeighbors = field(init=False)
    X_pca_: Optional[np.ndarray] = field(default=None, init=False)
    index_: Optional[pd.Index] = field(default=None, init=False)

    def fit(self, X: pd.DataFrame) -> "PCAKNN":
        """Fit PCA on X and then fit NearestNeighbors on the reduced matrix.

        X must be a numeric DataFrame indexed by player_id (as produced by MoneyballPreprocessor).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.isnull().any().any():
            # Safety: PCA/NN don't like NaNs
            X = X.fillna(0.0)

        self.index_ = X.index.copy()

        self.pca_ = PCA(n_components=self.n_components, random_state=self.pca_random_state)
        self.X_pca_ = self.pca_.fit_transform(X.values)

        self.knn_ = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric, algorithm=self.algorithm)
        self.knn_.fit(self.X_pca_)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform X into PCA space using the fitted PCA."""
        if not hasattr(self, "pca_"):
            raise RuntimeError("PCAKNN is not fitted. Call fit() first.")
        X2 = X.fillna(0.0).values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.pca_.transform(X2)

    def _kneighbors_on_pca(self, Z: np.ndarray, n_neighbors: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "knn_"):
            raise RuntimeError("PCAKNN is not fitted. Call fit() first.")
        k = int(n_neighbors) if n_neighbors is not None else self.n_neighbors
        dists, idxs = self.knn_.kneighbors(Z, n_neighbors=k)
        return dists, idxs

    def kneighbors_for_ids(self, player_ids: Union[int, Iterable[int]], top_k: Optional[int] = None,
                           include_self: bool = False) -> pd.DataFrame:
        """Return nearest neighbors for one or multiple player_ids.

        Returns a DataFrame with columns: query_id, neighbor_id, rank, distance
        """
        if self.index_ is None or self.X_pca_ is None:
            raise RuntimeError("PCAKNN is not fitted. Call fit() first.")

        if isinstance(player_ids, int):
            ids = [player_ids]
        else:
            ids = list(player_ids)

        # Map IDs to row indices
        try:
            row_idx = [self.index_.get_loc(i) for i in ids]
        except KeyError as e:
            raise KeyError(f"Player id not found: {e}")

        Zq = self.X_pca_[row_idx]
        dists, idxs = self._kneighbors_on_pca(Zq, n_neighbors=top_k)

        records = []
        for q_i, q_id in enumerate(ids):
            for rank, (dist, idx) in enumerate(zip(dists[q_i], idxs[q_i]), start=1):
                nid = self.index_[idx]
                if not include_self and nid == q_id:
                    # Skip exact self; continue ranks without gap
                    continue
                records.append({
                    "query_id": q_id,
                    "neighbor_id": nid,
                    "rank": rank,
                    "distance": float(dist),
                })
        return pd.DataFrame.from_records(records)

    def kneighbors_for_vectors(self, X: Union[np.ndarray, pd.DataFrame, List[float]],
                               top_k: Optional[int] = None) -> pd.DataFrame:
        """Query neighbors for arbitrary input vectors in original feature space.

        Returns a DataFrame with columns: query_index, neighbor_id, rank, distance
        """
        Z = self.transform(X)
        dists, idxs = self._kneighbors_on_pca(Z, n_neighbors=top_k)
        records = []
        for q_i in range(Z.shape[0] if Z.ndim == 2 else 1):
            for rank, (dist, idx) in enumerate(zip(dists[q_i], idxs[q_i]), start=1):
                nid = self.index_[idx]
                records.append({
                    "query_index": q_i,
                    "neighbor_id": nid,
                    "rank": rank,
                    "distance": float(dist),
                })
        return pd.DataFrame.from_records(records)

    def find_similar(self, player_id: int, top_k: int = 10, info: Optional[pd.DataFrame] = None,
                      include_self: bool = False) -> pd.DataFrame:
        """Convenience wrapper to get top_k similar players for a given player_id.

        If info is provided, merges neighbor details on 'player_id'.
        """
        df = self.kneighbors_for_ids(player_id, top_k=top_k, include_self=include_self)
        if info is not None and len(df) > 0:
            info_cols = [c for c in info.columns if c != "player_id"]
            df = df.merge(info[["player_id", *info_cols]], left_on="neighbor_id", right_on="player_id", how="left")
            df.drop(columns=["player_id"], inplace=True)
        return df

    @property
    def explained_variance_ratio_(self) -> Optional[np.ndarray]:
        return getattr(self, "pca_", None).explained_variance_ratio_ if hasattr(self, "pca_") else None


def pca_knn_from_csv(csv_path: str,
                     n_components: Union[int, float] = 20,
                     n_neighbors: int = 15,
                     metric: str = "euclidean",
                     algorithm: str = "auto",
                     preprocessor: Optional[MoneyballPreprocessor] = None
                     ) -> Tuple[PCAKNN, pd.DataFrame, pd.DataFrame]:
    """
    Convenience helper: preprocess CSV, fit PCAKNN, and return (model, X, info).
    """
    X: pd.DataFrame
    info: pd.DataFrame
    if preprocessor is None:
        X, info, _ = preprocess_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, low_memory=False)
        X, info = preprocessor.fit_transform(df)

    model = PCAKNN(n_components=n_components, n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
    model.fit(X)
    return model, X, info


__all__ = [
    "PCAKNN",
    "pca_knn_from_csv",
]