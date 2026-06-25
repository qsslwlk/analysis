"""Controlled taxonomy helpers for discursive cards."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_TAXONOMY_PATH = Path("config/discourse_taxonomy.example.json")


def load_discourse_taxonomy(path: Path | str = DEFAULT_TAXONOMY_PATH) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    without_punct = re.sub(r"[^a-z0-9]+", " ", without_accents)
    return re.sub(r"\s+", " ", without_punct).strip()


def loads_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
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


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def split_labels(value: Any) -> List[str]:
    labels: List[str] = []
    for item in loads_list(value):
        if isinstance(item, str):
            labels.extend(part.strip() for part in re.split(r"[,;/|]+", item) if part.strip())
        elif item is not None:
            labels.append(str(item).strip())
    return [label for label in labels if label]


def controlled_value(
    value: Any,
    allowed: List[str],
    aliases: Dict[str, str],
    default: str = "other",
) -> Tuple[str, bool]:
    normalized = normalize_text(value)
    if not normalized:
        return default, False
    normalized_allowed = {normalize_text(item): item for item in allowed}
    if normalized in normalized_allowed:
        return normalized_allowed[normalized], True
    if normalized in aliases:
        return aliases[normalized], True
    return default, False


def frame_lookup(taxonomy: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    frame_to_macro: Dict[str, str] = {}
    for macro_frame, frames in taxonomy.get("macro_frames", {}).items():
        for frame in frames:
            frame_to_macro[frame] = macro_frame
    aliases = {
        normalize_text(alias): target
        for alias, target in taxonomy.get("frame_aliases", {}).items()
    }
    return frame_to_macro, aliases


def normalize_frame(value: Any, taxonomy: Dict[str, Any]) -> Tuple[str, str, bool]:
    frame_to_macro, aliases = frame_lookup(taxonomy)
    normalized = normalize_text(value)
    if not normalized:
        return "other", "other", False
    normalized_frames = {normalize_text(frame): frame for frame in frame_to_macro}
    if normalized in normalized_frames:
        frame = normalized_frames[normalized]
        return frame_to_macro.get(frame, "other"), frame, True
    if normalized in aliases:
        frame = aliases[normalized]
        return frame_to_macro.get(frame, "other"), frame, True
    return "other", "other", False


def normalize_stances(value: Any, taxonomy: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str], float]:
    stance_taxonomy = taxonomy.get("stance", {})
    allowed = stance_taxonomy.get("allowed", ["supportive", "hostile", "ambivalent", "unclear"])
    aliases = {
        normalize_text(alias): target
        for alias, target in stance_taxonomy.get("aliases", {}).items()
    }
    normalized_stances: List[Dict[str, Any]] = []
    targets: List[str] = []
    fit_scores: List[float] = []

    for item in loads_list(value):
        if not isinstance(item, dict):
            continue
        target = str(item.get("target") or item.get("cible") or "").strip()
        stance, fitted = controlled_value(item.get("stance") or item.get("position"), allowed, aliases, default="unclear")
        evidence = str(item.get("evidence") or item.get("evidence_quote") or item.get("citation") or "").strip()
        target_normalized = normalize_text(target)
        if target_normalized:
            targets.append(target_normalized)
        fit_scores.append(1.0 if fitted else 0.0)
        normalized_stances.append(
            {
                **item,
                "target": target,
                "target_normalized": target_normalized,
                "stance": stance,
                "evidence": evidence,
            }
        )

    score = sum(fit_scores) / len(fit_scores) if fit_scores else 0.0
    return normalized_stances, sorted(set(targets)), score


def score_evidence(row: pd.Series, normalized_stances: List[Dict[str, Any]]) -> float:
    quote_count = len([quote for quote in loads_list(row.get("representative_quotes_json")) if str(quote).strip()])
    stance_count = len(normalized_stances)
    stance_evidence_count = len([stance for stance in normalized_stances if str(stance.get("evidence") or "").strip()])
    quote_score = 1.0 if quote_count > 0 else 0.0
    stance_score = stance_evidence_count / stance_count if stance_count else 0.0
    return round((quote_score + stance_score) / 2, 4)


def score_ambiguity(row: pd.Series, normalized_stances: List[Dict[str, Any]], frame_fit: bool) -> float:
    ambiguity_count = len(loads_list(row.get("ambiguities_json")))
    unclear_stances = len([stance for stance in normalized_stances if stance.get("stance") == "unclear"])
    raw_score = 0.0
    raw_score += min(0.5, 0.15 * ambiguity_count)
    raw_score += min(0.3, 0.15 * unclear_stances)
    if not frame_fit:
        raw_score += 0.2
    return round(min(1.0, raw_score), 4)


def normalize_discursive_cards(
    cards_df: pd.DataFrame,
    taxonomy: Optional[Dict[str, Any]] = None,
    taxonomy_path: Path | str = DEFAULT_TAXONOMY_PATH,
) -> pd.DataFrame:
    """Add controlled fields while preserving free LLM labels."""
    if cards_df.empty:
        return cards_df.copy()

    active_taxonomy = taxonomy or load_discourse_taxonomy(taxonomy_path)
    argument_taxonomy = active_taxonomy.get("argument_family", {})
    tone_taxonomy = active_taxonomy.get("tone", {})
    register_taxonomy = active_taxonomy.get("rhetorical_register", {})
    out = cards_df.copy()

    rows: List[Dict[str, Any]] = []
    for _, row in out.iterrows():
        macro_frame, frame_primary, frame_fit = normalize_frame(row.get("dominant_frame"), active_taxonomy)
        argument_family, argument_fit = controlled_value(
            row.get("argument_type"),
            argument_taxonomy.get("allowed", []),
            {normalize_text(alias): target for alias, target in argument_taxonomy.get("aliases", {}).items()},
            default="other",
        )
        tone_candidates = split_labels(row.get("emotion_tone"))
        tone_controlled = "unclear"
        tone_fit = False
        for candidate in tone_candidates or [row.get("emotion_tone")]:
            tone_controlled, tone_fit = controlled_value(
                candidate,
                tone_taxonomy.get("allowed", []),
                {normalize_text(alias): target for alias, target in tone_taxonomy.get("aliases", {}).items()},
                default="other",
            )
            if tone_fit:
                break

        register_controlled, register_fit = controlled_value(
            row.get("attack_or_objection") or row.get("argument_type") or row.get("emotion_tone"),
            register_taxonomy.get("allowed", []),
            {normalize_text(alias): target for alias, target in register_taxonomy.get("aliases", {}).items()},
            default="other",
        )
        normalized_stances, normalized_targets, stance_fit_score = normalize_stances(
            row.get("stance_targets_json"),
            active_taxonomy,
        )
        evidence_score = score_evidence(row, normalized_stances)
        ambiguity_score = score_ambiguity(row, normalized_stances, frame_fit=frame_fit)
        fit_components = [
            1.0 if frame_fit else 0.0,
            1.0 if argument_fit else 0.0,
            1.0 if tone_fit else 0.0,
            1.0 if register_fit else 0.0,
            stance_fit_score,
        ]
        taxonomy_fit_score = round(sum(fit_components) / len(fit_components), 4)
        rows.append(
            {
                "dominant_frame_free": row.get("dominant_frame"),
                "macro_frame": macro_frame,
                "frame_primary": frame_primary,
                "argument_family_controlled": argument_family,
                "tone_controlled": tone_controlled,
                "rhetorical_register": register_controlled,
                "stance_targets_controlled_json": dumps_json(normalized_stances),
                "targets_normalized_json": dumps_json(normalized_targets),
                "evidence_score": evidence_score,
                "ambiguity_score": ambiguity_score,
                "taxonomy_fit_score": taxonomy_fit_score,
            }
        )

    normalized_columns = pd.DataFrame(rows, index=out.index)
    for column in normalized_columns.columns:
        out[column] = normalized_columns[column]
    return out

