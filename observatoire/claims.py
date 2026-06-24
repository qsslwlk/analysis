"""Inductive claim extraction from comments."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from observatoire.llm import LLMClient


DEFAULT_PROMPT_PATH = Path("prompts/extract_claims.md")
CLAIM_COLUMNS = [
    "claim_id",
    "comment_id",
    "video_id",
    "video_title",
    "actor",
    "sequence",
    "claim_text",
    "evidence",
    "confidence",
    "abstraction_level",
    "raw_comment_preview",
]


def load_prompt(path: Path | str = DEFAULT_PROMPT_PATH) -> str:
    return Path(path).read_text(encoding="utf-8")


def _coerce_claims(payload: Dict[str, Any], max_claims: int, min_confidence: float) -> List[Dict[str, Any]]:
    raw_claims = payload.get("claims", [])
    if not isinstance(raw_claims, list):
        return []

    claims: List[Dict[str, Any]] = []
    for item in raw_claims[:max_claims]:
        if not isinstance(item, dict):
            continue
        claim_text = str(item.get("claim") or item.get("claim_text") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not claim_text or not evidence or confidence < min_confidence:
            continue
        claims.append(
            {
                "claim_text": claim_text,
                "evidence": evidence,
                "confidence": confidence,
                "abstraction_level": str(item.get("abstraction_level") or "low"),
            }
        )
    return claims


def build_claim_user_prompt(row: pd.Series, max_claims: int) -> str:
    metadata = {
        "comment_id": row.get("comment_id"),
        "actor": row.get("actor"),
        "video_title": row.get("video_title"),
        "sequence": row.get("sequence"),
    }
    return (
        f"Metadata:\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        f"Max claims: {max_claims}\n\n"
        f"Comment:\n{textwrap.shorten(str(row.get('text_clean') or ''), width=1600, placeholder='...')}"
    )


def empty_claims_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CLAIM_COLUMNS)


def extract_claims_from_comments(
    comments_df: pd.DataFrame,
    client: LLMClient,
    prompt_path: Path | str = DEFAULT_PROMPT_PATH,
    max_claims_per_comment: int = 3,
    min_confidence: float = 0.65,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Extract free-form claims grounded by evidence spans from semantic candidate comments."""
    if comments_df.empty:
        return empty_claims_df()

    system_prompt = load_prompt(prompt_path)
    rows: List[Dict[str, Any]] = []
    candidates = comments_df.copy()
    if "semantic_candidate" in candidates.columns:
        candidates = candidates[candidates["semantic_candidate"].fillna(False).astype(bool)]
    if limit is not None:
        candidates = candidates.head(limit)

    for _, comment in candidates.iterrows():
        payload = client.complete_json(
            system_prompt=system_prompt,
            user_prompt=build_claim_user_prompt(comment, max_claims_per_comment),
        )
        claims = _coerce_claims(payload, max_claims_per_comment, min_confidence)
        for claim_index, claim in enumerate(claims, start=1):
            rows.append(
                {
                    "claim_id": f"{comment.get('comment_id')}::{claim_index}",
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
                    **claim,
                }
            )

    return pd.DataFrame(rows, columns=CLAIM_COLUMNS)


def write_no_claim_summary(
    comments_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if comments_df.empty:
        summary = pd.DataFrame(columns=["reason", "n_comments"])
    else:
        candidate_ids = set(
            comments_df.loc[
                comments_df.get("semantic_candidate", pd.Series(False, index=comments_df.index)).fillna(False).astype(bool),
                "comment_id",
            ].dropna()
        )
        claimed_ids = set(claims_df.get("comment_id", pd.Series(dtype=object)).dropna()) if not claims_df.empty else set()
        rows = [
            {"reason": "candidate_with_claim", "n_comments": len(candidate_ids.intersection(claimed_ids))},
            {"reason": "candidate_no_claim", "n_comments": len(candidate_ids.difference(claimed_ids))},
        ]
        summary = pd.DataFrame(rows)
    summary.to_csv(output, index=False)
    return output

