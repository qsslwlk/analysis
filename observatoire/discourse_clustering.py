"""Cluster structured discursive cards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

from observatoire.text_quality import SEMANTIC_STOPWORDS


DISCURSIVE_CLUSTER_COLUMNS = [
    "discursive_cluster",
    "size",
    "share",
    "top_terms",
    "label_auto",
    "mean_confidence",
    "actor_distribution_json",
    "dominant_frames_json",
    "examples_json",
    "representative_quotes_json",
]


def empty_discursive_clusters_df() -> pd.DataFrame:
    return pd.DataFrame(columns=DISCURSIVE_CLUSTER_COLUMNS)


def _top_terms(texts: Sequence[str], n_terms: int = 8) -> List[str]:
    if not texts:
        return []
    try:
        vectorizer = TfidfVectorizer(
            max_features=4000,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=sorted(SEMANTIC_STOPWORDS),
            token_pattern=r"(?u)\b[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ]{2,}\b",
        )
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    top_idx = scores.argsort()[::-1][:n_terms]
    return [str(terms[index]) for index in top_idx if scores[index] > 0]


def _fallback_labels(embeddings: np.ndarray, min_cluster_size: int, distance_threshold: float) -> np.ndarray:
    if len(embeddings) < min_cluster_size:
        return np.full(len(embeddings), -1, dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(embeddings)
    counts = pd.Series(labels).value_counts()
    return np.array([label if counts[label] >= min_cluster_size else -1 for label in labels], dtype=int)


def _loads_list(value: Any) -> List[Any]:
    if not isinstance(value, str) or not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def cluster_discursive_cards(
    cards_df: pd.DataFrame,
    embeddings: np.ndarray,
    min_cluster_size: int = 6,
    distance_threshold: float = 0.35,
    outputs_dir: Path | str = "outputs",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    if cards_df.empty or embeddings.size == 0:
        cards_out = cards_df.copy()
        cards_out["discursive_cluster"] = []
        clusters_out = empty_discursive_clusters_df()
        cards_out.to_csv(outputs_path / "discursive_cards.csv", index=False)
        clusters_out.to_csv(outputs_path / "discursive_clusters.csv", index=False)
        return cards_out, clusters_out

    cards_out = cards_df.copy().reset_index(drop=True)
    try:
        import hdbscan  # type: ignore

        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(embeddings)
    except Exception:
        labels = _fallback_labels(embeddings, min_cluster_size, distance_threshold)

    cards_out["discursive_cluster"] = labels
    cluster_rows: List[Dict[str, Any]] = []
    total = max(len(cards_out), 1)
    for cluster_id in sorted(label for label in set(labels) if label != -1):
        mask = cards_out["discursive_cluster"] == cluster_id
        cluster_cards_df = cards_out.loc[mask].copy()
        cluster_embeddings = embeddings[mask.to_numpy()]
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        distances = pairwise_distances(cluster_embeddings, centroid, metric="cosine").ravel()
        cluster_cards_df["distance_to_centroid"] = distances
        ordered = cluster_cards_df.sort_values("distance_to_centroid")
        examples = ordered["discursive_summary"].dropna().head(10).tolist()
        representative_quotes: List[str] = []
        for value in ordered["representative_quotes_json"].dropna().head(6):
            representative_quotes.extend(str(quote) for quote in _loads_list(value)[:2])
        top_terms = _top_terms(cluster_cards_df["discursive_summary"].dropna().tolist())
        frame_distribution = cluster_cards_df["dominant_frame"].value_counts(normalize=True).round(3).to_dict()
        actor_distribution = cluster_cards_df["actor"].value_counts(normalize=True).round(3).to_dict()
        label_auto = " / ".join(top_terms[:4]) if top_terms else " / ".join(list(frame_distribution)[:3])
        cluster_rows.append(
            {
                "discursive_cluster": int(cluster_id),
                "size": int(mask.sum()),
                "share": float(mask.sum() / total),
                "top_terms": ", ".join(top_terms),
                "label_auto": label_auto or f"discursive_cluster_{cluster_id}",
                "mean_confidence": float(cluster_cards_df["confidence"].mean()),
                "actor_distribution_json": json.dumps(actor_distribution, ensure_ascii=False),
                "dominant_frames_json": json.dumps(frame_distribution, ensure_ascii=False),
                "examples_json": json.dumps(examples, ensure_ascii=False),
                "representative_quotes_json": json.dumps(representative_quotes[:10], ensure_ascii=False),
            }
        )

    clusters_out = pd.DataFrame(cluster_rows, columns=DISCURSIVE_CLUSTER_COLUMNS).sort_values(
        "size",
        ascending=False,
    )
    cards_out.to_csv(outputs_path / "discursive_cards.csv", index=False)
    clusters_out.to_csv(outputs_path / "discursive_clusters.csv", index=False)
    return cards_out, clusters_out
