"""Behavioural segmentation.

This is the hinge between the dataset and the method. Per-client adapters do
not scale; a single adapter is what the previous paper did and it cannot
personalise. Segments are the middle ground: a small library of group-level
LoRA adapters, one per behavioural cluster, selected at inference time.

Design decisions that a reviewer will check, so they are made explicit here:

* **Fit on train only.** Cluster centroids are learned from the training split
  and *applied* to val/test. Fitting on the full data would leak test-set
  structure into the segment definition and inflate every downstream number.
* **Rank-based scaling, not z-scores.** Several features are heavily skewed
  (inflow skew 4.0, fx_share 2.8). Standardising a skewed variable leaves the
  tail dominating the distance metric. Quantile transform to a normal fixes
  this without discarding the ordering.
* **Redundant features dropped.** inflow and outflow correlate at r=0.992;
  keeping both double-weights transaction volume in the distance. Only the
  derived ratios are kept.
* **k chosen by silhouette, reported not assumed.** The selection curve goes in
  the appendix, because "we used k=5" invites the question "why".
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import QuantileTransformer

# inflow/outflow excluded: r=0.992 with each other, and their information is
# already carried by savings_rate and free_cashflow in scale-free form.
FEATURE_ORDER = [
    "free_cashflow",
    "savings_rate",
    "spend_volatility",
    "card_share",
    "fx_share",
    "education_share",
    "transport_share",
    "retail_share",
    "debt_service_ratio",
    "income_trend",
]


def to_matrix(feats: dict[str, dict[str, float]],
              client_ids: list[str]) -> np.ndarray:
    return np.array(
        [[float(feats[c].get(k, 0.0)) for k in FEATURE_ORDER] for c in client_ids],
        dtype=float,
    )


class SegmentModel:
    def __init__(self, k: int, seed: int = 0):
        self.k = k
        self.seed = seed
        self.qt = QuantileTransformer(
            output_distribution="normal", n_quantiles=1000,
            subsample=100_000, random_state=seed,
        )
        self.km = KMeans(n_clusters=k, n_init=10, random_state=seed)

    def fit(self, X: np.ndarray) -> "SegmentModel":
        self.qt.fit(X)
        self.km.fit(self.qt.transform(X))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.km.predict(self.qt.transform(X))

    def silhouette(self, X: np.ndarray) -> float:
        Z = self.qt.transform(X)
        return float(silhouette_score(Z, self.km.predict(Z)))


def select_k(X: np.ndarray, k_range=range(2, 11), seed: int = 0) -> dict:
    """Silhouette sweep. Returns the full curve, not just the argmax.

    Report the curve. If silhouette is flat and low across k, say so plainly:
    that means the behavioural space has no crisp cluster structure, and the
    honest conclusion is that segments are a *pragmatic partition* for adapter
    routing rather than a discovery of natural client types. Overclaiming here
    is an easy way to lose a reviewer.
    """
    curve = {}
    for k in k_range:
        m = SegmentModel(k, seed).fit(X)
        curve[int(k)] = m.silhouette(X)
    best = max(curve, key=curve.get)
    return {"curve": curve, "best_k": int(best), "best_score": curve[best]}


def describe_segments(X: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Human-readable centroid profile in ORIGINAL units.

    Cluster centroids in transformed space are uninterpretable. The paper needs
    a table saying what each segment actually is, in tenge and percentages.
    """
    out = []
    overall = X.mean(axis=0)
    for c in sorted(set(labels.tolist())):
        mask = labels == c
        mean = X[mask].mean(axis=0)
        # feature that deviates most from the population, in robust units
        scale = np.std(X, axis=0) + 1e-9
        dev = (mean - overall) / scale
        order = np.argsort(-np.abs(dev))[:3]
        out.append({
            "segment": int(c),
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "means": {k: float(v) for k, v in zip(FEATURE_ORDER, mean)},
            "distinctive": [
                {"feature": FEATURE_ORDER[i],
                 "value": float(mean[i]),
                 "z_vs_population": float(dev[i])}
                for i in order
            ],
        })
    return out


def save(model: SegmentModel, path: str) -> None:
    import pickle
    with open(path, "wb") as fh:
        pickle.dump(model, fh)


def load(path: str) -> SegmentModel:
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)
