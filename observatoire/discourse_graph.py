"""Build an interpretable typed discourse graph from discursive cards."""

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


NODE_COLUMNS = [
    "node_id",
    "node_type",
    "label",
    "normalized_label",
    "is_vague",
    "df_comments",
]

EDGE_COLUMNS = [
    "source_node_id",
    "target_node_id",
    "edge_type",
    "comment_id",
    "weight",
    "base_weight",
    "confidence",
    "idf",
    "evidence",
    "raw_value",
]

SIMILARITY_EDGE_COLUMNS = [
    "source_comment_id",
    "target_comment_id",
    "similarity",
]

COMMUNITY_COLUMNS = [
    "discursive_community",
    "size",
    "share",
    "mean_internal_similarity",
    "mean_confidence",
    "top_frames_json",
    "top_claims_json",
    "top_targets_json",
    "top_stance_targets_json",
    "actor_distribution_json",
    "time_distribution_json",
    "examples_json",
    "representative_quotes_json",
]

INCIDENCE_GROUPS = {
    "frame": ["COMMENT_HAS_FRAME"],
    "claim": ["COMMENT_HAS_CANONICAL_CLAIM"],
    "stance": ["COMMENT_EXPRESSES_STANCE_TOWARD_TARGET"],
    "target": ["COMMENT_TARGETS_ACTOR"],
    "argument": ["COMMENT_HAS_ARGUMENT_FAMILY"],
    "tone": ["COMMENT_HAS_TONE"],
}

DEFAULT_RELATION_WEIGHTS = {
    "COMMENT_HAS_THEME": 0.75,
    "COMMENT_HAS_FRAME": 1.40,
    "COMMENT_HAS_CLAIM": 1.00,
    "COMMENT_HAS_CANONICAL_CLAIM": 1.30,
    "COMMENT_HAS_ARGUMENT_FAMILY": 1.10,
    "COMMENT_TARGETS_ACTOR": 0.90,
    "COMMENT_EXPRESSES_STANCE_TOWARD_TARGET": 1.25,
    "COMMENT_HAS_TONE": 0.60,
    "COMMENT_HAS_REGISTER": 0.55,
    "COMMENT_FROM_VIDEO": 0.35,
    "COMMENT_FROM_CHANNEL": 0.30,
    "COMMENT_IN_TIME_BUCKET": 0.25,
    "VIDEO_ASSOCIATED_WITH_SOURCE_ACTOR": 0.35,
    "CLAIM_BELONGS_TO_ARGUMENT_FAMILY": 0.70,
    "FRAME_CO_OCCURS_WITH_CLAIM": 0.55,
}

DEFAULT_SIMILARITY_WEIGHTS = {
    "frame": 0.25,
    "claim": 0.25,
    "stance": 0.20,
    "argument": 0.12,
    "target": 0.08,
    "tone": 0.04,
    "embed": 0.06,
}

VAGUE_NORMALIZED_LABELS = {
    "france",
    "francais",
    "francaises",
    "peuple",
    "gens",
    "pays",
    "politique",
    "politiques",
    "societe",
    "systeme",
    "colere",
    "peur",
    "soutien",
    "rejet",
    "vote",
}

STANCE_NORMALIZATION = {
    "adhesion": "supportive",
    "appui": "supportive",
    "favorable": "supportive",
    "support": "supportive",
    "supportive": "supportive",
    "soutien": "supportive",
    "rejet": "hostile",
    "hostile": "hostile",
    "critique": "hostile",
    "opposition": "hostile",
    "defavorable": "hostile",
    "scepticisme": "ambivalent",
    "sceptique": "ambivalent",
    "ambivalent": "ambivalent",
    "ambigu": "ambivalent",
    "unclear": "unclear",
    "flou": "unclear",
    "incertain": "unclear",
}


@dataclass
class DiscourseGraphResult:
    nodes_df: pd.DataFrame
    edges_df: pd.DataFrame
    similarity_edges_df: pd.DataFrame
    communities_df: pd.DataFrame
    units_df: pd.DataFrame
    profiles_path: Path
    incidence_paths: Dict[str, Path]


def empty_nodes_df() -> pd.DataFrame:
    return pd.DataFrame(columns=NODE_COLUMNS)


def empty_edges_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EDGE_COLUMNS)


def empty_similarity_edges_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SIMILARITY_EDGE_COLUMNS)


def empty_discursive_communities_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COMMUNITY_COLUMNS)


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_label(value: Any) -> str:
    text = _strip_accents(_string(value)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _node_id(node_type: str, label: str) -> str:
    normalized = _normalize_label(label)
    digest = hashlib.sha1(f"{node_type}:{normalized}".encode("utf-8")).hexdigest()[:12]
    slug = normalized.replace(" ", "_")[:42] or "empty"
    return f"{node_type}:{slug}:{digest}"


def _loads_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return [item for item in value if item not in (None, "")]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(payload, list):
            return [item for item in payload if item not in (None, "")]
        if payload:
            return [payload]
    return []


def _split_terms(value: Any) -> List[str]:
    terms: List[str] = []
    for item in _loads_list(value):
        if isinstance(item, str):
            parts = re.split(r"[,;/|]+", item)
            terms.extend(part.strip() for part in parts if part.strip())
        elif item is not None:
            terms.append(str(item).strip())
    return [term for term in terms if term]


def _coerce_float(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return max(0.0, min(1.0, number))


def _time_bucket(row: pd.Series) -> str:
    for column in ["published_at_comment", "published_at_video"]:
        value = _string(row.get(column))
        if not value:
            continue
        timestamp = pd.to_datetime(value, errors="coerce", utc=True)
        if not pd.isna(timestamp):
            return f"{timestamp.isocalendar().year}-W{timestamp.isocalendar().week:02d}"
    return ""


def _normalize_stance(value: Any) -> str:
    normalized = _normalize_label(value)
    return STANCE_NORMALIZATION.get(normalized, normalized or "unclear")


def _is_vague(node_type: str, label: str) -> bool:
    normalized = _normalize_label(label)
    if normalized in VAGUE_NORMALIZED_LABELS:
        return True
    if node_type in {"ThemeNode", "ToneNode", "TargetNode"} and len(normalized) <= 3:
        return True
    return False


def _json_dict(value: Dict[str, float]) -> str:
    return json.dumps({key: round(float(score), 4) for key, score in value.items()}, ensure_ascii=False)


def _json_list(value: Sequence[str]) -> str:
    return json.dumps(list(value), ensure_ascii=False)


def _top_weighted_labels(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    comment_ids: Iterable[str],
    edge_types: Sequence[str],
    limit: int = 8,
) -> Dict[str, float]:
    comment_id_set = set(str(comment_id) for comment_id in comment_ids)
    if edges_df.empty or nodes_df.empty or not comment_id_set:
        return {}
    sub = edges_df[
        edges_df["comment_id"].astype(str).isin(comment_id_set)
        & edges_df["edge_type"].isin(edge_types)
        & (edges_df["weight"] > 0)
    ]
    if sub.empty:
        return {}
    label_by_node = nodes_df.set_index("node_id")["label"].to_dict()
    weighted = sub.groupby("target_node_id")["weight"].sum().sort_values(ascending=False).head(limit)
    return {str(label_by_node.get(node_id, node_id)): float(weight) for node_id, weight in weighted.items()}


def _value_distribution(values: pd.Series, limit: int = 8) -> Dict[str, float]:
    cleaned = values.dropna().astype(str)
    cleaned = cleaned[cleaned.str.len() > 0]
    if cleaned.empty:
        return {}
    return cleaned.value_counts(normalize=True).round(4).head(limit).to_dict()


def _representative_quotes(cards_df: pd.DataFrame, limit: int = 8) -> List[str]:
    quotes: List[str] = []
    if "representative_quotes_json" not in cards_df.columns:
        return quotes
    for value in cards_df["representative_quotes_json"].dropna():
        for quote in _loads_list(value):
            quote_text = _string(quote)
            if quote_text and quote_text not in quotes:
                quotes.append(quote_text)
            if len(quotes) >= limit:
                return quotes
    return quotes


def _write_units_jsonl(units_df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in units_df.to_dict(orient="records"):
            output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_profiles(communities_df: pd.DataFrame, path: Path) -> Path:
    lines = ["# Profils de communautés discursives", ""]
    if communities_df.empty:
        lines.extend(
            [
                "Aucune communauté discursive robuste n'a été détectée avec les seuils actuels.",
                "",
                "Cela peut être normal sur un petit corpus ou lorsque les fiches discursives sont trop hétérogènes.",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    for _, row in communities_df.sort_values(["size", "discursive_community"], ascending=[False, True]).iterrows():
        lines.append(f"## Communauté {int(row['discursive_community'])} — {int(row['size'])} commentaires")
        lines.append("")
        lines.append(f"- Similarité interne moyenne : {row['mean_internal_similarity']:.3f}")
        lines.append(f"- Confiance moyenne : {row['mean_confidence']:.3f}")
        for label, column in [
            ("Frames", "top_frames_json"),
            ("Claims", "top_claims_json"),
            ("Cibles", "top_targets_json"),
            ("Stances × cibles", "top_stance_targets_json"),
            ("Acteurs source", "actor_distribution_json"),
            ("Périodes", "time_distribution_json"),
        ]:
            payload = json.loads(row[column]) if _string(row.get(column)) else {}
            if payload:
                values = ", ".join(f"{key} ({value:.2f})" for key, value in payload.items())
                lines.append(f"- {label} : {values}")
        examples = json.loads(row["examples_json"]) if _string(row.get("examples_json")) else []
        if examples:
            lines.append("- Exemples :")
            for example in examples[:4]:
                lines.append(f"  - {example}")
        quotes = json.loads(row["representative_quotes_json"]) if _string(row.get("representative_quotes_json")) else []
        if quotes:
            lines.append("- Citations représentatives :")
            for quote in quotes[:4]:
                lines.append(f"  - “{quote}”")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _empty_result(outputs_path: Path, units_df: Optional[pd.DataFrame] = None) -> DiscourseGraphResult:
    nodes_df = empty_nodes_df()
    edges_df = empty_edges_df()
    similarity_edges_df = empty_similarity_edges_df()
    communities_df = empty_discursive_communities_df()
    units_out = pd.DataFrame() if units_df is None else units_df.copy()

    nodes_df.to_csv(outputs_path / "discursive_nodes.csv", index=False)
    edges_df.to_csv(outputs_path / "discursive_edges.csv", index=False)
    similarity_edges_df.to_csv(outputs_path / "discursive_similarity_edges.csv", index=False)
    communities_df.to_csv(outputs_path / "discursive_communities.csv", index=False)
    units_out.to_csv(outputs_path / "discursive_units.csv", index=False)
    _write_units_jsonl(units_out, outputs_path / "discursive_units.jsonl")
    profiles_path = _write_profiles(communities_df, outputs_path / "discursive_community_profiles.md")
    return DiscourseGraphResult(
        nodes_df=nodes_df,
        edges_df=edges_df,
        similarity_edges_df=similarity_edges_df,
        communities_df=communities_df,
        units_df=units_out,
        profiles_path=profiles_path,
        incidence_paths={},
    )


def _build_nodes_and_edges(cards_df: pd.DataFrame, comments_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comments_meta = comments_df.copy()
    if not comments_meta.empty and "comment_id" in comments_meta.columns:
        comments_meta = comments_meta.drop_duplicates("comment_id").set_index("comment_id")
    else:
        comments_meta = pd.DataFrame()

    node_records: Dict[str, Dict[str, Any]] = {}
    edge_records: List[Dict[str, Any]] = []
    units_rows: List[Dict[str, Any]] = []

    def add_node(node_type: str, label: Any) -> str:
        clean_label = _string(label)
        if not clean_label:
            return ""
        node_id = _node_id(node_type, clean_label)
        if node_id not in node_records:
            node_records[node_id] = {
                "node_id": node_id,
                "node_type": node_type,
                "label": clean_label,
                "normalized_label": _normalize_label(clean_label),
                "is_vague": _is_vague(node_type, clean_label),
                "df_comments": 0,
            }
        return node_id

    def add_edge(
        source_node_id: str,
        target_node_id: str,
        edge_type: str,
        comment_id: str = "",
        confidence: float = 1.0,
        evidence: Any = "",
        raw_value: Any = "",
    ) -> None:
        if not source_node_id or not target_node_id:
            return
        edge_records.append(
            {
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "edge_type": edge_type,
                "comment_id": _string(comment_id),
                "weight": 0.0,
                "base_weight": float(DEFAULT_RELATION_WEIGHTS.get(edge_type, 0.50)),
                "confidence": _coerce_float(confidence),
                "idf": 1.0,
                "evidence": _string(evidence),
                "raw_value": _string(raw_value),
            }
        )

    for _, card in cards_df.reset_index(drop=True).iterrows():
        comment_id = _string(card.get("comment_id"))
        if not comment_id:
            continue

        meta = comments_meta.loc[comment_id] if not comments_meta.empty and comment_id in comments_meta.index else pd.Series(dtype=object)
        merged = {**meta.to_dict(), **card.to_dict()}
        row = pd.Series(merged)
        time_bucket = _time_bucket(row)
        channel = _string(row.get("channel_title") or row.get("channel_id"))

        comment_node = add_node("CommentNode", comment_id)
        video_node = add_node("VideoNode", row.get("video_title") or row.get("video_id"))
        actor_node = add_node("SourceActorNode", row.get("actor"))
        time_node = add_node("TimeNode", time_bucket)
        channel_node = add_node("ChannelNode", channel)

        add_edge(comment_node, video_node, "COMMENT_FROM_VIDEO", comment_id=comment_id)
        add_edge(video_node, actor_node, "VIDEO_ASSOCIATED_WITH_SOURCE_ACTOR")
        add_edge(comment_node, time_node, "COMMENT_IN_TIME_BUCKET", comment_id=comment_id)
        add_edge(comment_node, channel_node, "COMMENT_FROM_CHANNEL", comment_id=comment_id)

        for theme in [row.get("theme_main"), *_split_terms(row.get("subthemes_json"))]:
            add_edge(comment_node, add_node("ThemeNode", theme), "COMMENT_HAS_THEME", comment_id=comment_id)

        frame_labels = [row.get("dominant_frame"), *_split_terms(row.get("secondary_frames_json"))]
        for frame in frame_labels:
            add_edge(comment_node, add_node("FrameNode", frame), "COMMENT_HAS_FRAME", comment_id=comment_id)

        central_argument = _string(row.get("central_argument"))
        canonical_claim_node = add_node("CanonicalClaimNode", central_argument)
        add_edge(comment_node, canonical_claim_node, "COMMENT_HAS_CANONICAL_CLAIM", comment_id=comment_id)

        attack_or_objection = _string(row.get("attack_or_objection"))
        add_edge(comment_node, add_node("ClaimNode", attack_or_objection), "COMMENT_HAS_CLAIM", comment_id=comment_id)

        argument_type = _string(row.get("argument_type"))
        argument_node = add_node("ArgumentFamilyNode", argument_type)
        add_edge(comment_node, argument_node, "COMMENT_HAS_ARGUMENT_FAMILY", comment_id=comment_id)
        add_edge(canonical_claim_node, argument_node, "CLAIM_BELONGS_TO_ARGUMENT_FAMILY")

        for frame in frame_labels:
            frame_node = add_node("FrameNode", frame)
            add_edge(frame_node, canonical_claim_node, "FRAME_CO_OCCURS_WITH_CLAIM")

        for stance_item in _loads_list(row.get("stance_targets_json")):
            if not isinstance(stance_item, dict):
                continue
            target = _string(stance_item.get("target") or stance_item.get("cible"))
            stance = _normalize_stance(stance_item.get("stance") or stance_item.get("position"))
            evidence = stance_item.get("evidence") or stance_item.get("evidence_quote") or stance_item.get("citation")
            confidence = _coerce_float(stance_item.get("confidence"), default=_coerce_float(row.get("confidence"), 1.0))
            target_node = add_node("TargetNode", target)
            stance_label = f"{stance} → {target}" if target else stance
            stance_target_node = add_node("StanceTargetNode", stance_label)
            add_edge(
                comment_node,
                target_node,
                "COMMENT_TARGETS_ACTOR",
                comment_id=comment_id,
                confidence=confidence,
                evidence=evidence,
                raw_value=target,
            )
            add_edge(
                comment_node,
                stance_target_node,
                "COMMENT_EXPRESSES_STANCE_TOWARD_TARGET",
                comment_id=comment_id,
                confidence=confidence,
                evidence=evidence,
                raw_value=stance_label,
            )

        for tone in _split_terms(row.get("emotion_tone")):
            add_edge(comment_node, add_node("ToneNode", tone), "COMMENT_HAS_TONE", comment_id=comment_id)

        register = _string(row.get("register") or row.get("rhetorical_tone"))
        for item in _split_terms(register):
            add_edge(comment_node, add_node("RegisterNode", item), "COMMENT_HAS_REGISTER", comment_id=comment_id)

        units_rows.append(
            {
                **card.to_dict(),
                "time_bucket": time_bucket,
                "channel": channel,
                "discursive_community": -1,
            }
        )

    nodes_df = pd.DataFrame(node_records.values(), columns=NODE_COLUMNS)
    edges_df = pd.DataFrame(edge_records, columns=EDGE_COLUMNS)
    units_df = pd.DataFrame(units_rows)
    return nodes_df, edges_df, units_df


def _apply_edge_weights(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, vague_penalty: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if edges_df.empty:
        return nodes_df, edges_df

    weighted_edges = edges_df.copy()
    nodes_out = nodes_df.copy()
    comment_edge_mask = weighted_edges["comment_id"].astype(str).str.len() > 0
    n_comments = max(weighted_edges.loc[comment_edge_mask, "comment_id"].nunique(), 1)
    df_by_node = (
        weighted_edges.loc[comment_edge_mask]
        .groupby("target_node_id")["comment_id"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    nodes_out["df_comments"] = nodes_out["node_id"].map(df_by_node).fillna(0).astype(int)
    is_vague_by_node = nodes_out.set_index("node_id")["is_vague"].to_dict()

    def compute_weight(edge: pd.Series) -> float:
        if not _string(edge.get("comment_id")):
            return float(edge["base_weight"]) * float(edge["confidence"])
        df = max(int(df_by_node.get(edge["target_node_id"], 0)), 0)
        idf = math.log((n_comments + 1) / (df + 1))
        if bool(is_vague_by_node.get(edge["target_node_id"], False)):
            idf *= vague_penalty
        quote_multiplier = 1.0
        if edge["edge_type"] in {"COMMENT_EXPRESSES_STANCE_TOWARD_TARGET", "COMMENT_HAS_CLAIM"} and not _string(edge.get("evidence")):
            quote_multiplier = 0.75
        return float(edge["base_weight"]) * float(edge["confidence"]) * idf * quote_multiplier

    weighted_edges["idf"] = weighted_edges.apply(
        lambda edge: math.log(
            (n_comments + 1)
            / (max(int(df_by_node.get(edge["target_node_id"], 0)), 0) + 1)
        )
        if _string(edge.get("comment_id"))
        else 1.0,
        axis=1,
    )
    weighted_edges["weight"] = weighted_edges.apply(compute_weight, axis=1)
    return nodes_out, weighted_edges


def _row_l2_normalize(matrix: Any) -> Any:
    from scipy import sparse

    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms).dot(matrix)


def _build_incidence_and_similarity(
    edges_df: pd.DataFrame,
    units_df: pd.DataFrame,
    outputs_path: Path,
    embeddings: Optional[np.ndarray],
    similarity_weights: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, Path]]:
    from scipy import sparse

    comment_ids = units_df["comment_id"].astype(str).tolist() if "comment_id" in units_df.columns else []
    comment_index = {comment_id: index for index, comment_id in enumerate(comment_ids)}
    incidence_paths: Dict[str, Path] = {}
    similarity = np.zeros((len(comment_ids), len(comment_ids)), dtype=float)

    pd.DataFrame({"comment_id": comment_ids, "row_index": range(len(comment_ids))}).to_csv(
        outputs_path / "discursive_incidence_index.csv",
        index=False,
    )

    for group, edge_types in INCIDENCE_GROUPS.items():
        sub = edges_df[
            edges_df["edge_type"].isin(edge_types)
            & edges_df["comment_id"].astype(str).isin(comment_index)
            & (edges_df["weight"] > 0)
        ]
        columns = sorted(sub["target_node_id"].unique())
        column_index = {node_id: index for index, node_id in enumerate(columns)}
        rows = [comment_index[str(comment_id)] for comment_id in sub["comment_id"]]
        cols = [column_index[node_id] for node_id in sub["target_node_id"]]
        data = sub["weight"].astype(float).tolist()
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(comment_ids), len(columns)))
        matrix_path = outputs_path / f"discursive_incidence_{group}.npz"
        sparse.save_npz(matrix_path, matrix)
        pd.DataFrame({"node_id": columns, "column_index": range(len(columns))}).to_csv(
            outputs_path / f"discursive_incidence_{group}_columns.csv",
            index=False,
        )
        incidence_paths[group] = matrix_path
        if columns and similarity_weights.get(group, 0) > 0:
            normalized = _row_l2_normalize(matrix)
            similarity += float(similarity_weights[group]) * normalized.dot(normalized.T).toarray()

    if embeddings is not None and len(comment_ids) > 0:
        embed_array = np.asarray(embeddings, dtype=float)
        if embed_array.shape[0] == len(comment_ids) and embed_array.ndim == 2 and embed_array.size > 0:
            norms = np.linalg.norm(embed_array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized_embed = embed_array / norms
            similarity += float(similarity_weights.get("embed", 0.0)) * normalized_embed.dot(normalized_embed.T)

    np.fill_diagonal(similarity, 0.0)
    return similarity, incidence_paths


def _similarity_edges(
    similarity: np.ndarray,
    comment_ids: Sequence[str],
    threshold: float,
    top_k: int,
) -> pd.DataFrame:
    records: Dict[Tuple[str, str], float] = {}
    if similarity.size == 0:
        return empty_similarity_edges_df()

    for row_index, source_comment_id in enumerate(comment_ids):
        row = similarity[row_index]
        candidate_indices = np.argsort(row)[::-1]
        kept = 0
        for col_index in candidate_indices:
            if col_index == row_index or row[col_index] < threshold:
                continue
            left, right = sorted([str(source_comment_id), str(comment_ids[col_index])])
            records[(left, right)] = max(float(row[col_index]), records.get((left, right), 0.0))
            kept += 1
            if kept >= top_k:
                break

    rows = [
        {"source_comment_id": source, "target_comment_id": target, "similarity": score}
        for (source, target), score in sorted(records.items(), key=lambda item: item[1], reverse=True)
    ]
    return pd.DataFrame(rows, columns=SIMILARITY_EDGE_COLUMNS)


def _community_labels(
    similarity_edges_df: pd.DataFrame,
    comment_ids: Sequence[str],
    min_community_size: int,
) -> np.ndarray:
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    if not comment_ids:
        return np.array([], dtype=int)
    if similarity_edges_df.empty:
        return np.full(len(comment_ids), -1, dtype=int)

    comment_index = {comment_id: index for index, comment_id in enumerate(comment_ids)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for _, edge in similarity_edges_df.iterrows():
        left = str(edge["source_comment_id"])
        right = str(edge["target_comment_id"])
        if left not in comment_index or right not in comment_index:
            continue
        rows.extend([comment_index[left], comment_index[right]])
        cols.extend([comment_index[right], comment_index[left]])
        data.extend([float(edge["similarity"]), float(edge["similarity"])])

    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(len(comment_ids), len(comment_ids)))
    _, raw_labels = connected_components(adjacency, directed=False, return_labels=True)
    counts = pd.Series(raw_labels).value_counts().to_dict()
    kept_components = sorted(
        [component for component, count in counts.items() if count >= min_community_size],
        key=lambda component: (-counts[component], component),
    )
    remap = {component: index for index, component in enumerate(kept_components)}
    return np.array([remap.get(component, -1) for component in raw_labels], dtype=int)


def _community_profiles(
    units_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    similarity: np.ndarray,
) -> pd.DataFrame:
    if units_df.empty or "discursive_community" not in units_df.columns:
        return empty_discursive_communities_df()

    rows: List[Dict[str, Any]] = []
    total = max(len(units_df), 1)
    for community_id in sorted(label for label in units_df["discursive_community"].unique() if int(label) != -1):
        community_units = units_df[units_df["discursive_community"] == community_id].copy()
        indices = community_units.index.to_numpy()
        if len(indices) > 1:
            sub_similarity = similarity[np.ix_(indices, indices)]
            internal_values = sub_similarity[np.triu_indices_from(sub_similarity, k=1)]
            mean_internal_similarity = float(internal_values.mean()) if internal_values.size else 0.0
        else:
            mean_internal_similarity = 0.0

        comment_ids = community_units["comment_id"].astype(str).tolist()
        summaries = [
            textwrap.shorten(_string(summary), width=180, placeholder="...")
            for summary in community_units.get("discursive_summary", pd.Series(dtype=object)).dropna().head(8)
            if _string(summary)
        ]
        rows.append(
            {
                "discursive_community": int(community_id),
                "size": int(len(community_units)),
                "share": float(len(community_units) / total),
                "mean_internal_similarity": mean_internal_similarity,
                "mean_confidence": float(community_units.get("confidence", pd.Series([0.0])).astype(float).mean()),
                "top_frames_json": _json_dict(
                    _top_weighted_labels(edges_df, nodes_df, comment_ids, ["COMMENT_HAS_FRAME"])
                ),
                "top_claims_json": _json_dict(
                    _top_weighted_labels(edges_df, nodes_df, comment_ids, ["COMMENT_HAS_CANONICAL_CLAIM"])
                ),
                "top_targets_json": _json_dict(
                    _top_weighted_labels(edges_df, nodes_df, comment_ids, ["COMMENT_TARGETS_ACTOR"])
                ),
                "top_stance_targets_json": _json_dict(
                    _top_weighted_labels(edges_df, nodes_df, comment_ids, ["COMMENT_EXPRESSES_STANCE_TOWARD_TARGET"])
                ),
                "actor_distribution_json": _json_dict(_value_distribution(community_units.get("actor", pd.Series(dtype=object)))),
                "time_distribution_json": _json_dict(
                    _value_distribution(community_units.get("time_bucket", pd.Series(dtype=object)))
                ),
                "examples_json": _json_list(summaries),
                "representative_quotes_json": _json_list(_representative_quotes(community_units)),
            }
        )

    return pd.DataFrame(rows, columns=COMMUNITY_COLUMNS).sort_values("size", ascending=False)


def build_discourse_graph(
    cards_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    embeddings: Optional[np.ndarray] = None,
    outputs_dir: Path | str = "outputs",
    min_community_size: int = 4,
    similarity_threshold: float = 0.18,
    top_k: int = 8,
    vague_penalty: float = 0.25,
    similarity_weights: Optional[Dict[str, float]] = None,
) -> DiscourseGraphResult:
    """Build V2.6.3 graph exports and interpretable discursive communities."""
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    if cards_df.empty:
        return _empty_result(outputs_path)

    active_weights = dict(DEFAULT_SIMILARITY_WEIGHTS)
    if similarity_weights:
        active_weights.update(similarity_weights)

    nodes_df, edges_df, units_df = _build_nodes_and_edges(cards_df, comments_df)
    if units_df.empty:
        return _empty_result(outputs_path, units_df)

    nodes_df, edges_df = _apply_edge_weights(nodes_df, edges_df, vague_penalty=vague_penalty)
    similarity, incidence_paths = _build_incidence_and_similarity(
        edges_df,
        units_df,
        outputs_path,
        embeddings=embeddings,
        similarity_weights=active_weights,
    )
    comment_ids = units_df["comment_id"].astype(str).tolist()
    similarity_edges_df = _similarity_edges(
        similarity,
        comment_ids=comment_ids,
        threshold=similarity_threshold,
        top_k=top_k,
    )
    labels = _community_labels(
        similarity_edges_df,
        comment_ids=comment_ids,
        min_community_size=min_community_size,
    )
    units_out = units_df.copy()
    units_out["discursive_community"] = labels
    communities_df = _community_profiles(units_out, nodes_df, edges_df, similarity)

    nodes_df.to_csv(outputs_path / "discursive_nodes.csv", index=False)
    edges_df.to_csv(outputs_path / "discursive_edges.csv", index=False)
    similarity_edges_df.to_csv(outputs_path / "discursive_similarity_edges.csv", index=False)
    communities_df.to_csv(outputs_path / "discursive_communities.csv", index=False)
    units_out.to_csv(outputs_path / "discursive_units.csv", index=False)
    _write_units_jsonl(units_out, outputs_path / "discursive_units.jsonl")
    try:
        units_out.to_parquet(outputs_path / "discursive_units.parquet", index=False)
    except Exception:
        pass
    profiles_path = _write_profiles(communities_df, outputs_path / "discursive_community_profiles.md")

    return DiscourseGraphResult(
        nodes_df=nodes_df,
        edges_df=edges_df,
        similarity_edges_df=similarity_edges_df,
        communities_df=communities_df,
        units_df=units_out,
        profiles_path=profiles_path,
        incidence_paths=incidence_paths,
    )

