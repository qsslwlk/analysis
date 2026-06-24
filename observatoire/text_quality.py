"""Text quality gates for semantic clustering."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List


SEMANTIC_STOPWORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "c",
    "ca",
    "ce",
    "ces",
    "cette",
    "d",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "elles",
    "en",
    "est",
    "et",
    "il",
    "ils",
    "j",
    "je",
    "l",
    "la",
    "le",
    "les",
    "leur",
    "lui",
    "m",
    "ma",
    "mais",
    "me",
    "mes",
    "n",
    "ne",
    "nos",
    "notre",
    "nous",
    "on",
    "ont",
    "ou",
    "par",
    "pas",
    "plus",
    "pour",
    "qu",
    "que",
    "qui",
    "s",
    "sa",
    "se",
    "ses",
    "son",
    "sont",
    "sur",
    "t",
    "ta",
    "te",
    "tes",
    "toi",
    "ton",
    "tous",
    "tout",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
    "y",
}


LOW_INFORMATION_TERMS = {
    "bardella",
    "bravo",
    "force",
    "jlm",
    "jordan",
    "merci",
    "melenchon",
    "melanchon",
    "mélenchon",
    "respect",
    "rn",
    "soutien",
    "vive",
}


TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9'’_-]*")


def fold_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def semantic_tokens(text: str) -> List[str]:
    tokens = []
    for token in TOKEN_RE.findall(text or ""):
        folded = fold_accents(token.lower()).strip("'’_-")
        if folded:
            tokens.append(folded)
    return tokens


def text_quality_metrics(text: str) -> Dict[str, int]:
    tokens = semantic_tokens(text)
    meaningful_tokens = [
        token
        for token in tokens
        if len(token) >= 3 and token not in SEMANTIC_STOPWORDS and not token.isdigit()
    ]
    return {
        "char_count": len((text or "").strip()),
        "token_count": len(tokens),
        "meaningful_token_count": len(meaningful_tokens),
        "low_information_token_count": sum(token in LOW_INFORMATION_TERMS for token in meaningful_tokens),
    }


def semantic_exclusion_reason(text: str, min_chars: int = 80, min_meaningful_tokens: int = 8) -> str:
    metrics = text_quality_metrics(text)
    tokens = semantic_tokens(text)
    meaningful_tokens = [
        token
        for token in tokens
        if len(token) >= 3 and token not in SEMANTIC_STOPWORDS and not token.isdigit()
    ]

    if not tokens:
        return "no_alpha_tokens"
    if metrics["meaningful_token_count"] == 0:
        return "no_meaningful_tokens"
    if metrics["meaningful_token_count"] <= 5 and all(token in LOW_INFORMATION_TERMS for token in meaningful_tokens):
        return "reaction_or_name_only"
    if metrics["char_count"] < min_chars and metrics["meaningful_token_count"] < min_meaningful_tokens:
        return "too_short"
    return "ok"


def is_semantic_candidate(text: str, min_chars: int = 80, min_meaningful_tokens: int = 8) -> bool:
    return semantic_exclusion_reason(text, min_chars, min_meaningful_tokens) == "ok"
