"""Structured discursive cards extracted from comments with an LLM."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from observatoire.llm import LLMClient
from observatoire.discourse_taxonomy import DEFAULT_TAXONOMY_PATH, normalize_discursive_cards


DEFAULT_PROMPT_PATH = Path("prompts/extract_discursive_card.md")
DISCURSIVE_CARD_COLUMNS = [
    "card_id",
    "comment_id",
    "video_id",
    "video_title",
    "actor",
    "sequence",
    "theme_main",
    "subthemes_json",
    "dominant_frame",
    "secondary_frames_json",
    "stance_targets_json",
    "central_argument",
    "argument_type",
    "attack_or_objection",
    "emotion_tone",
    "ambiguities_json",
    "representative_quotes_json",
    "confidence",
    "dominant_frame_free",
    "macro_frame",
    "frame_primary",
    "argument_family_controlled",
    "tone_controlled",
    "rhetorical_register",
    "stance_targets_controlled_json",
    "targets_normalized_json",
    "evidence_score",
    "ambiguity_score",
    "taxonomy_fit_score",
    "discursive_summary",
    "raw_comment_preview",
]


def load_prompt(path: Path | str = DEFAULT_PROMPT_PATH) -> str:
    return Path(path).read_text(encoding="utf-8")


def empty_discursive_cards_df() -> pd.DataFrame:
    return pd.DataFrame(columns=DISCURSIVE_CARD_COLUMNS)


def _string(value: Any) -> str:
    return str(value or "").strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item not in (None, "")]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _coerce_stances(value: Any) -> List[Dict[str, Any]]:
    stances = []
    for item in _list(value):
        if isinstance(item, dict):
            target = _string(item.get("target") or item.get("cible"))
            stance = _string(item.get("stance") or item.get("position"))
            evidence = _string(item.get("evidence") or item.get("citation"))
            confidence = _float(item.get("confidence"), default=0.0)
            if target or stance or evidence:
                stances.append(
                    {
                        "target": target,
                        "stance": stance,
                        "evidence": evidence,
                        "confidence": confidence,
                    }
                )
        elif isinstance(item, str):
            stances.append({"target": "", "stance": item.strip(), "evidence": "", "confidence": 0.0})
    return stances


def _build_summary(card: Dict[str, Any]) -> str:
    explicit = _string(card.get("discursive_summary") or card.get("summary"))
    if explicit:
        return explicit

    parts = [
        _string(card.get("theme_main") or card.get("main_theme") or card.get("theme")),
        _string(card.get("dominant_frame") or card.get("frame")),
        _string(card.get("central_argument") or card.get("argument")),
        _string(card.get("emotion_tone") or card.get("tone")),
    ]
    return " | ".join(part for part in parts if part)


def _coerce_card(payload: Dict[str, Any], min_confidence: float) -> Optional[Dict[str, Any]]:
    card = payload.get("discursive_card") or payload.get("card") or payload
    if card is None or not isinstance(card, dict):
        return None
    if card.get("has_discursive_content") is False:
        return None

    confidence = _float(card.get("confidence"), default=0.0)
    if confidence < min_confidence:
        return None

    representative_quotes = _list(
        card.get("representative_quotes")
        or card.get("quotes")
        or card.get("citations")
        or card.get("representative_quote")
    )
    representative_quotes = [_string(quote) for quote in representative_quotes if _string(quote)]
    if not representative_quotes:
        return None

    theme_main = _string(card.get("theme_main") or card.get("main_theme") or card.get("theme"))
    dominant_frame = _string(card.get("dominant_frame") or card.get("frame") or card.get("cadrage_dominant"))
    central_argument = _string(card.get("central_argument") or card.get("argument") or card.get("argument_central"))
    if not any([theme_main, dominant_frame, central_argument]):
        return None

    return {
        "theme_main": theme_main,
        "subthemes_json": _json([_string(item) for item in _list(card.get("subthemes") or card.get("sous_themes"))]),
        "dominant_frame": dominant_frame,
        "secondary_frames_json": _json(
            [_string(item) for item in _list(card.get("secondary_frames") or card.get("cadrages_secondaires"))]
        ),
        "stance_targets_json": _json(_coerce_stances(card.get("stance_targets") or card.get("stances"))),
        "central_argument": central_argument,
        "argument_type": _string(card.get("argument_type") or card.get("type_argument")),
        "attack_or_objection": _string(card.get("attack_or_objection") or card.get("objection") or card.get("attaque")),
        "emotion_tone": _string(card.get("emotion_tone") or card.get("tone") or card.get("tonalite")),
        "ambiguities_json": _json([_string(item) for item in _list(card.get("ambiguities") or card.get("ambiguites"))]),
        "representative_quotes_json": _json(representative_quotes),
        "confidence": confidence,
        "discursive_summary": _build_summary(card),
    }


def build_discursive_card_user_prompt(row: pd.Series) -> str:
    metadata = {
        "comment_id": row.get("comment_id"),
        "actor": row.get("actor"),
        "video_title": row.get("video_title"),
        "sequence": row.get("sequence"),
    }
    return (
        f"Metadata:\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        f"Comment:\n{textwrap.shorten(str(row.get('text_clean') or ''), width=1800, placeholder='...')}"
    )


def extract_discursive_cards_from_comments(
    comments_df: pd.DataFrame,
    client: LLMClient,
    prompt_path: Path | str = DEFAULT_PROMPT_PATH,
    taxonomy_path: Path | str = DEFAULT_TAXONOMY_PATH,
    min_confidence: float = 0.55,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Extract one structured discursive card per semantic candidate comment."""
    if comments_df.empty:
        return empty_discursive_cards_df()

    system_prompt = load_prompt(prompt_path)
    candidates = comments_df.copy()
    if "semantic_candidate" in candidates.columns:
        candidates = candidates[candidates["semantic_candidate"].fillna(False).astype(bool)]
    if limit is not None:
        candidates = candidates.head(limit)

    rows: List[Dict[str, Any]] = []
    for _, comment in candidates.iterrows():
        payload = client.complete_json(
            system_prompt=system_prompt,
            user_prompt=build_discursive_card_user_prompt(comment),
        )
        card = _coerce_card(payload, min_confidence=min_confidence)
        if card is None:
            continue
        rows.append(
            {
                "card_id": f"{comment.get('comment_id')}::discursive_card",
                "comment_id": comment.get("comment_id"),
                "video_id": comment.get("video_id"),
                "video_title": comment.get("video_title"),
                "actor": comment.get("actor"),
                "sequence": comment.get("sequence"),
                "raw_comment_preview": textwrap.shorten(
                    str(comment.get("text_clean") or ""),
                    width=260,
                    placeholder="...",
                ),
                **card,
            }
        )

    cards_df = pd.DataFrame(rows)
    if cards_df.empty:
        return empty_discursive_cards_df()
    normalized_df = normalize_discursive_cards(cards_df, taxonomy_path=taxonomy_path)
    return normalized_df.reindex(columns=DISCURSIVE_CARD_COLUMNS)


def write_discursive_card_coverage(
    comments_df: pd.DataFrame,
    cards_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if comments_df.empty:
        summary = pd.DataFrame(columns=["reason", "n_comments"])
    else:
        candidate_mask = comments_df.get("semantic_candidate", pd.Series(False, index=comments_df.index))
        candidate_ids = set(comments_df.loc[candidate_mask.fillna(False).astype(bool), "comment_id"].dropna())
        card_ids = set(cards_df.get("comment_id", pd.Series(dtype=object)).dropna()) if not cards_df.empty else set()
        summary = pd.DataFrame(
            [
                {"reason": "candidate_with_discursive_card", "n_comments": len(candidate_ids.intersection(card_ids))},
                {"reason": "candidate_without_discursive_card", "n_comments": len(candidate_ids.difference(card_ids))},
            ]
        )
    summary.to_csv(output, index=False)
    return output
