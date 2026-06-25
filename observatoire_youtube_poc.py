"""POC Colab : Observatoire des cadrages politiques à partir de commentaires YouTube.

Le script collecte des commentaires publics via l'API officielle YouTube Data API v3,
les anonymise, les enrichit avec des cadrages lexicaux et des régions sémantiques
émergentes, puis exporte un dashboard HTML autonome.

Usage local :
    export YOUTUBE_API_KEY="..."
    python observatoire_youtube_poc.py --max-comments-per-video 500

Dans Google Colab, stocker la clé dans Secrets sous le nom YOUTUBE_API_KEY.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import textwrap
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances

from observatoire.cache import DatasetCache, build_raw_cache_context
from observatoire.claim_clustering import cluster_claims as cluster_extracted_claims
from observatoire.claim_labeling import label_claim_clusters as label_extracted_claim_clusters
from observatoire.claim_labeling import write_claim_cluster_labels
from observatoire.claims import extract_claims_from_comments, write_no_claim_summary
from observatoire.config import load_video_config
from observatoire.discourse_cards import (
    DISCURSIVE_CARD_COLUMNS,
    extract_discursive_cards_from_comments,
    write_discursive_card_coverage,
)
from observatoire.discourse_clustering import cluster_discursive_cards, empty_discursive_clusters_df
from observatoire.discourse_graph import build_discourse_graph as build_discourse_graph_exports
from observatoire.discourse_graph import empty_discursive_communities_df
from observatoire.llm import make_llm_client
from observatoire.privacy import assert_privacy_safe_columns
from observatoire.text_quality import SEMANTIC_STOPWORDS, semantic_exclusion_reason, text_quality_metrics


VIDEOS = [
    {
        "video_url": "https://www.youtube.com/watch?v=V0xwB3ctfHo",
        "actor": "LFI",
        "sequence": "pouvoir_achat",
        "label": "Pouvoir d'achat : ce que nous allons faire - Clip officiel",
    },
    {
        "video_url": "https://www.youtube.com/watch?v=AS1MFzB0hbw",
        "actor": "LFI",
        "sequence": "pouvoir_achat",
        "label": "Gaz, électricité, alimentation : il faut bloquer les prix !",
    },
    {
        "video_url": "https://www.youtube.com/watch?v=RTvbHZ5dUG8",
        "actor": "LFI",
        "sequence": "pouvoir_achat",
        "label": "Essence : bloquons les prix, Total doit payer !",
    },
    {
        "video_url": "https://www.youtube.com/watch?v=MlTqGnfOnb0",
        "actor": "LFI",
        "sequence": "pouvoir_achat",
        "label": "Pouvoir d'achat : au tour du travail de respirer !",
    },
    {
        "video_url": "https://www.youtube.com/watch?v=wwxGrY7IIVI",
        "actor": "LFI",
        "sequence": "pouvoir_achat",
        "label": "Je veux augmenter les salaires et bloquer les prix",
    },
    {
        "video_url": "https://www.youtube.com/watch?v=_ReVF5CIQ4o",
        "actor": "LFI",
        "sequence": "pouvoir_achat",
        "label": "Contre la vie chère, augmentons les salaires et bloquons les prix",
    },
    {
        "video_url": "https://www.youtube.com/watch?v=A4FMuKKdQto",
        "actor": "RN",
        "sequence": "pouvoir_achat",
        "label": "Immigration, pouvoir d'achat, énergie... Interview de Jordan Bardella",
    },
    {
        "video_url": "https://www.youtube.com/watch?v=3vlREuWT84s",
        "actor": "RN",
        "sequence": "pouvoir_achat",
        "label": "Le pouvoir d'achat et l'urgence de la fin du mois seront mes priorités",
    },
    {
        "video_url": "https://www.youtube.com/watch?v=PD-WorZ1htE",
        "actor": "RN",
        "sequence": "pouvoir_achat",
        "label": "Pouvoir d'achat, éducation, binationaux : le programme du RN dévoilé",
    },
    {
        "video_url": "https://www.youtube.com/watch?v=N7aB5Wn-X7g",
        "actor": "Media",
        "sequence": "pouvoir_achat",
        "label": "Pouvoir d'achat, salaires, impôts : qui propose quoi ? - Législatives 2024",
    },
]


FRAME_LEXICON = {
    "justice_sociale": [
        "injustice",
        "pauvres",
        "riche",
        "riches",
        "salaires",
        "salaire",
        "profits",
        "profit",
        "précarité",
        "classe",
        "travailleurs",
        "travailleuses",
        "smic",
        "fin de mois",
        "pouvoir d'achat",
    ],
    "ordre_chaos": [
        "chaos",
        "désordre",
        "violence",
        "casseurs",
        "émeute",
        "insécurité",
        "délinquance",
        "ensauvagement",
    ],
    "responsabilite_budgetaire": [
        "financer",
        "budget",
        "dette",
        "impôts",
        "impot",
        "taxes",
        "coût",
        "cout",
        "irréaliste",
        "promesses",
        "déficit",
        "payer",
    ],
    "dignite": [
        "dignité",
        "respect",
        "humiliation",
        "mépris",
        "vivre dignement",
        "digne",
        "décence",
    ],
    "ecologie_populaire": [
        "écologie",
        "pollution",
        "agriculture",
        "climat",
        "santé",
        "industrie",
        "énergie",
        "transition",
    ],
    "souverainete": [
        "souveraineté",
        "indépendance",
        "France",
        "Europe",
        "mondialisation",
        "frontières",
        "nation",
        "bruxelles",
    ],
    "libertes_publiques": [
        "liberté",
        "censure",
        "répression",
        "police",
        "droits",
        "manifestation",
        "autoritaire",
        "démocratie",
    ],
    "desaffiliation": [
        "tous les mêmes",
        "ça ne changera rien",
        "ca ne changera rien",
        "politiciens",
        "mensonge",
        "marre",
        "abstention",
        "plus confiance",
        "propagande",
        "blabla",
    ],
    "credibilite_faisabilite": [
        "faisable",
        "impossible",
        "réaliste",
        "irréaliste",
        "crédible",
        "programme",
        "mesure",
        "application",
        "comment",
    ],
    "colere_anti_elite": [
        "élite",
        "elites",
        "macron",
        "oligarchie",
        "système",
        "banques",
        "médias",
        "journalistes",
        "casta",
    ],
}


STANCE_LEXICON = {
    "adhesion": [
        "bravo",
        "merci",
        "soutien",
        "d'accord",
        "exactement",
        "bien dit",
        "je vote",
        "enfin",
        "bonne mesure",
        "totalement",
    ],
    "rejet": [
        "honte",
        "n'importe quoi",
        "mensonge",
        "menteur",
        "nul",
        "jamais",
        "contre",
        "arnaque",
        "propagande",
        "ridicule",
    ],
    "ironie": [
        "lol",
        "mdr",
        "ptdr",
        "ben voyons",
        "bien sûr",
        "😂",
        "🤣",
        "ironie",
        "quelle surprise",
    ],
    "scepticisme": [
        "source",
        "preuve",
        "j'attends de voir",
        "pas sûr",
        "pas sur",
        "comment",
        "ça marche",
        "ca marche",
        "faisable",
        "vraiment",
    ],
    "incomprehension": [
        "je ne comprends pas",
        "je comprends pas",
        "expliquez",
        "explique",
        "quoi",
        "pourquoi",
        "c'est quoi",
        "ça veut dire quoi",
    ],
    "deplacement_du_debat": [
        "et l'immigration",
        "et la sécurité",
        "et gaza",
        "hors sujet",
        "parlez plutôt",
        "parlez plutot",
        "et macron",
        "les retraites",
    ],
}


FINAL_COMMENT_COLUMNS = [
    "video_id",
    "video_title",
    "channel_title",
    "actor",
    "sequence",
    "published_at_video",
    "comment_id",
    "parent_id",
    "is_reply",
    "published_at_comment",
    "like_count_comment",
    "text_original",
    "text_clean",
    "author_hash",
]


FRENCH_STOPWORDS = [
    "alors",
    "au",
    "aucuns",
    "aussi",
    "autre",
    "avant",
    "avec",
    "avoir",
    "bon",
    "car",
    "ce",
    "cela",
    "ces",
    "ceux",
    "chaque",
    "ci",
    "comme",
    "comment",
    "dans",
    "des",
    "du",
    "dedans",
    "dehors",
    "depuis",
    "devrait",
    "doit",
    "donc",
    "dos",
    "droite",
    "début",
    "elle",
    "elles",
    "en",
    "encore",
    "essai",
    "est",
    "et",
    "eu",
    "fait",
    "faites",
    "fois",
    "font",
    "hors",
    "ici",
    "il",
    "ils",
    "je",
    "juste",
    "la",
    "le",
    "les",
    "leur",
    "là",
    "ma",
    "maintenant",
    "mais",
    "mes",
    "mine",
    "moins",
    "mon",
    "mot",
    "même",
    "ni",
    "nommés",
    "notre",
    "nous",
    "ou",
    "où",
    "par",
    "parce",
    "pas",
    "peut",
    "peu",
    "plupart",
    "pour",
    "pourquoi",
    "quand",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "sa",
    "sans",
    "ses",
    "seulement",
    "si",
    "sien",
    "son",
    "sont",
    "sous",
    "soyez",
    "sujet",
    "sur",
    "ta",
    "tandis",
    "tellement",
    "tels",
    "tes",
    "ton",
    "tous",
    "tout",
    "trop",
    "très",
    "tu",
    "voient",
    "vont",
    "votre",
    "vous",
    "vu",
    "ça",
    "étaient",
    "état",
    "étions",
    "été",
    "être",
]


class MissingYouTubeAPIKey(RuntimeError):
    """Raised when no YouTube API key can be found."""


class YouTubeAPIError(RuntimeError):
    """Raised for clear YouTube Data API failures."""


@dataclass
class YouTubeErrorDetail:
    status_code: int
    reason: str
    message: str

    def as_message(self) -> str:
        if self.reason:
            return f"YouTube API error {self.status_code} ({self.reason}) : {self.message}"
        return f"YouTube API error {self.status_code} : {self.message}"


def ensure_dirs(data_dir: Path | str = "data", outputs_dir: Path | str = "outputs") -> Tuple[Path, Path]:
    """Create data and output directories if needed."""
    data_path = Path(data_dir)
    outputs_path = Path(outputs_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    outputs_path.mkdir(parents=True, exist_ok=True)
    return data_path, outputs_path


def resolve_api_key(api_key: Optional[str] = None) -> str:
    """Resolve the API key from an argument, environment variable, or Colab Secrets."""
    if api_key:
        return api_key

    env_key = os.getenv("YOUTUBE_API_KEY")
    if env_key:
        return env_key

    try:
        from google.colab import userdata  # type: ignore

        colab_key = userdata.get("YOUTUBE_API_KEY")
        if colab_key:
            return colab_key
    except Exception:
        pass

    raise MissingYouTubeAPIKey(
        "Clé YouTube API absente. Définis YOUTUBE_API_KEY dans les variables "
        "d'environnement ou dans les Secrets Colab."
    )


def extract_video_id(value: str) -> str:
    """Extract a YouTube video ID from a classic URL, short URL, embed URL, Shorts URL, or raw ID."""
    if not value or not isinstance(value, str):
        raise ValueError("video_url ou video_id vide.")

    value = value.strip()
    raw_id_pattern = r"^[A-Za-z0-9_-]{11}$"
    if re.match(raw_id_pattern, value):
        return value

    parsed = urlparse(value)
    host = parsed.netloc.lower().replace("www.", "")
    path_parts = [part for part in parsed.path.split("/") if part]

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        query_video_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_video_id and re.match(raw_id_pattern, query_video_id):
            return query_video_id
        if len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live"}:
            if re.match(raw_id_pattern, path_parts[1]):
                return path_parts[1]

    if host == "youtu.be" and path_parts:
        if re.match(raw_id_pattern, path_parts[0]):
            return path_parts[0]

    raise ValueError(f"Impossible d'extraire un video_id YouTube depuis : {value}")


def _parse_youtube_error(response: requests.Response) -> YouTubeErrorDetail:
    try:
        payload = response.json()
    except Exception:
        return YouTubeErrorDetail(response.status_code, "", response.text[:500])

    error = payload.get("error", {})
    errors = error.get("errors", [])
    reason = ""
    if errors and isinstance(errors, list):
        reason = errors[0].get("reason", "")
    return YouTubeErrorDetail(
        status_code=response.status_code,
        reason=reason,
        message=error.get("message", response.text[:500]),
    )


def youtube_api_get(endpoint: str, api_key: str, params: Dict[str, Any], retries: int = 2) -> Dict[str, Any]:
    """GET helper for official YouTube Data API v3 endpoints with readable errors."""
    base_url = f"https://www.googleapis.com/youtube/v3/{endpoint}"
    request_params = {**params, "key": api_key}

    for attempt in range(retries + 1):
        response = requests.get(base_url, params=request_params, timeout=30)
        if response.status_code == 200:
            return response.json()

        detail = _parse_youtube_error(response)
        retryable = response.status_code in {429, 500, 502, 503, 504}
        if retryable and attempt < retries:
            time.sleep(2**attempt)
            continue

        if detail.reason in {"commentsDisabled", "quotaExceeded", "dailyLimitExceeded", "keyInvalid"}:
            raise YouTubeAPIError(detail.as_message())
        raise YouTubeAPIError(detail.as_message())

    raise YouTubeAPIError("Erreur inconnue pendant l'appel YouTube Data API.")


def hash_author(author_identifier: Optional[str], salt: Optional[str] = None) -> Optional[str]:
    """Hash an author identifier immediately; never store display names."""
    if not author_identifier:
        return None
    privacy_salt = salt or os.getenv("AUTHOR_HASH_SALT", "observatoire-youtube-poc-local-salt")
    digest = hashlib.sha256(f"{privacy_salt}:{author_identifier}".encode("utf-8")).hexdigest()
    return digest[:24]


def fetch_video_metadata(video_id: str, api_key: str) -> Dict[str, Any]:
    """Fetch public video metadata via videos.list."""
    payload = youtube_api_get(
        "videos",
        api_key,
        {
            "part": "snippet,statistics",
            "id": video_id,
            "maxResults": 1,
        },
    )
    items = payload.get("items", [])
    if not items:
        raise YouTubeAPIError(f"Vidéo introuvable ou non accessible : {video_id}")

    item = items[0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})

    return {
        "video_id": video_id,
        "video_title": snippet.get("title"),
        "video_description": snippet.get("description"),
        "channel_id": snippet.get("channelId"),
        "channel_title": snippet.get("channelTitle"),
        "published_at_video": snippet.get("publishedAt"),
        "view_count": _safe_int(stats.get("viewCount")),
        "like_count_video": _safe_int(stats.get("likeCount")),
        "comment_count_video": _safe_int(stats.get("commentCount")),
    }


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _comment_row_from_snippet(
    video_id: str,
    comment_id: str,
    snippet: Dict[str, Any],
    parent_id: Optional[str],
    is_reply: bool,
) -> Dict[str, Any]:
    author_channel = snippet.get("authorChannelId")
    author_identifier = None
    if isinstance(author_channel, dict):
        author_identifier = author_channel.get("value")

    return {
        "video_id": video_id,
        "comment_id": comment_id,
        "parent_id": parent_id,
        "is_reply": bool(is_reply),
        "published_at_comment": snippet.get("publishedAt"),
        "updated_at_comment": snippet.get("updatedAt"),
        "like_count_comment": _safe_int(snippet.get("likeCount")) or 0,
        "text_original": snippet.get("textDisplay") or snippet.get("textOriginal") or "",
        "author_hash": hash_author(author_identifier),
    }


def fetch_comment_replies(
    parent_id: str,
    video_id: str,
    api_key: str,
    max_replies: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch replies for a top-level comment via comments.list."""
    rows: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    while True:
        payload = youtube_api_get(
            "comments",
            api_key,
            {
                "part": "snippet",
                "parentId": parent_id,
                "maxResults": 100,
                "textFormat": "plainText",
                **({"pageToken": page_token} if page_token else {}),
            },
        )
        for item in payload.get("items", []):
            snippet = item.get("snippet", {})
            rows.append(
                _comment_row_from_snippet(
                    video_id=video_id,
                    comment_id=item.get("id"),
                    snippet=snippet,
                    parent_id=parent_id,
                    is_reply=True,
                )
            )
            if max_replies is not None and len(rows) >= max_replies:
                return rows[:max_replies]

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return rows


def fetch_youtube_comments(
    video_id: str,
    api_key: str,
    max_comments: int = 500,
    include_replies: bool = False,
) -> pd.DataFrame:
    """Fetch top-level comments, and optionally replies, via commentThreads.list.

    Returns an anonymized DataFrame. It never stores author display names.
    """
    if max_comments <= 0:
        return empty_comments_df()

    rows: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    while len(rows) < max_comments:
        payload = youtube_api_get(
            "commentThreads",
            api_key,
            {
                "part": "snippet,replies" if include_replies else "snippet",
                "videoId": video_id,
                "maxResults": min(100, max_comments - len(rows)),
                "textFormat": "plainText",
                "order": "relevance",
                **({"pageToken": page_token} if page_token else {}),
            },
        )

        for item in payload.get("items", []):
            top = item.get("snippet", {}).get("topLevelComment", {})
            top_snippet = top.get("snippet", {})
            top_id = top.get("id")
            if top_id:
                rows.append(
                    _comment_row_from_snippet(
                        video_id=video_id,
                        comment_id=top_id,
                        snippet=top_snippet,
                        parent_id=None,
                        is_reply=False,
                    )
                )

            if include_replies and len(rows) < max_comments:
                total_reply_count = item.get("snippet", {}).get("totalReplyCount", 0) or 0
                if total_reply_count:
                    remaining = max_comments - len(rows)
                    rows.extend(fetch_comment_replies(top_id, video_id, api_key, max_replies=remaining))

            if len(rows) >= max_comments:
                break

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return pd.DataFrame(rows)


def empty_comments_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "video_id",
            "comment_id",
            "parent_id",
            "is_reply",
            "published_at_comment",
            "updated_at_comment",
            "like_count_comment",
            "text_original",
            "author_hash",
        ]
    )


def normalize_videos(videos: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize user-provided video objects and add extracted video IDs."""
    normalized = []
    for item in videos:
        video_value = item.get("video_id") or item.get("video_url")
        video_id = extract_video_id(video_value)
        normalized.append(
            {
                **item,
                "video_id": video_id,
                "video_url": item.get("video_url") or f"https://www.youtube.com/watch?v={video_id}",
                "actor": item.get("actor", "Non renseigné"),
                "sequence": item.get("sequence", "Non renseigné"),
                "label": item.get("label") or item.get("video_url") or video_id,
            }
        )
    return normalized


def build_dataset(
    videos: Sequence[Dict[str, Any]],
    api_key: str,
    max_comments_per_video: int = 500,
    include_replies: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build metadata and comments dataset for all videos."""
    normalized = normalize_videos(videos)
    all_comments: List[pd.DataFrame] = []
    metadata_rows: List[Dict[str, Any]] = []

    for video in normalized:
        video_id = video["video_id"]
        print(f"Collecte vidéo {video_id} — {video.get('label', '')}")
        metadata: Dict[str, Any] = {
            "video_id": video_id,
            "video_url": video.get("video_url"),
            "actor": video.get("actor"),
            "sequence": video.get("sequence"),
            "input_label": video.get("label"),
            "collection_error": None,
        }

        try:
            fetched_metadata = fetch_video_metadata(video_id, api_key)
            metadata.update(fetched_metadata)
        except YouTubeAPIError as error:
            metadata["collection_error"] = str(error)
            print(f"  ⚠️ Métadonnées indisponibles : {error}")
            metadata_rows.append(metadata)
            continue

        try:
            comments = fetch_youtube_comments(
                video_id,
                api_key,
                max_comments=max_comments_per_video,
                include_replies=include_replies,
            )
        except YouTubeAPIError as error:
            comments = empty_comments_df()
            metadata["collection_error"] = str(error)
            print(f"  ⚠️ Commentaires indisponibles : {error}")

        if comments.empty:
            print("  Aucun commentaire collecté pour cette vidéo.")
        else:
            for key, value in metadata.items():
                comments[key] = value
            all_comments.append(comments)
            print(f"  {len(comments)} commentaire(s) collecté(s).")

        metadata_rows.append(metadata)

    metadata_df = pd.DataFrame(metadata_rows)
    comments_df = pd.concat(all_comments, ignore_index=True) if all_comments else empty_comments_df()

    for column in FINAL_COMMENT_COLUMNS:
        if column not in comments_df.columns:
            comments_df[column] = None

    ordered_columns = FINAL_COMMENT_COLUMNS + [
        column for column in comments_df.columns if column not in FINAL_COMMENT_COLUMNS
    ]
    return comments_df[ordered_columns], metadata_df


def clean_comment_text(text: Any, remove_emojis: bool = False) -> str:
    """Normalize spaces, strip URLs, keep French accents and useful punctuation."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    value = html.unescape(str(text))
    value = re.sub(r"https?://\S+|www\.\S+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", value)
    if remove_emojis:
        value = re.sub(
            "["
            "\U0001F300-\U0001FAFF"
            "\U00002700-\U000027BF"
            "\U00002600-\U000026FF"
            "]+",
            " ",
            value,
        )
    value = re.sub(r"\s+", " ", value).strip()
    return value


def detect_language_safe(text: str) -> Optional[str]:
    """Optional lightweight language detection; returns None if unavailable or uncertain."""
    if not text or len(text) < 20:
        return None
    try:
        from langdetect import LangDetectException, detect  # type: ignore

        try:
            return detect(text)
        except LangDetectException:
            return None
    except Exception:
        return None


def _fold_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def assign_frames_lexicon(text: Any, frame_lexicon: Dict[str, List[str]] = FRAME_LEXICON) -> Dict[str, Any]:
    """Assign frames using an interpretable keyword lexicon."""
    clean_text = clean_comment_text(text)
    folded_text = _fold_accents(clean_text.lower())
    scores: Dict[str, int] = {}

    for frame, terms in frame_lexicon.items():
        count = 0
        for term in terms:
            folded_term = _fold_accents(term.lower())
            if " " in folded_term:
                count += folded_text.count(folded_term)
            else:
                count += len(re.findall(rf"(?<!\w){re.escape(folded_term)}s?(?!\w)", folded_text))
        scores[frame] = count

    detected = [frame for frame, score in scores.items() if score > 0]
    dominant_frame = max(scores, key=scores.get) if detected else None
    return {
        "frames_detected": detected,
        "frame_match_count": int(sum(scores.values())),
        "dominant_frame": dominant_frame,
        "frame_scores": scores,
    }


def assign_stance_lexicon(text: Any, stance_lexicon: Dict[str, List[str]] = STANCE_LEXICON) -> Dict[str, Any]:
    """Assign a cautious tonalité/stance baseline using transparent keywords."""
    clean_text = clean_comment_text(text)
    folded_text = _fold_accents(clean_text.lower())
    scores: Dict[str, int] = {}

    for stance, terms in stance_lexicon.items():
        count = 0
        for term in terms:
            folded_term = _fold_accents(term.lower())
            if " " in folded_term:
                count += folded_text.count(folded_term)
            else:
                count += len(re.findall(rf"(?<!\w){re.escape(folded_term)}s?(?!\w)", folded_text))
        scores[stance] = count

    detected = [stance for stance, score in scores.items() if score > 0]
    dominant_stance = max(scores, key=scores.get) if detected else None
    return {
        "stances_detected": detected,
        "stance_match_count": int(sum(scores.values())),
        "dominant_stance": dominant_stance,
        "stance_scores": scores,
    }


def enrich_text_and_frames(comments_df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned text, optional language, lexical frame scores, and stance cues."""
    df = comments_df.copy()
    if df.empty:
        for column in [
            "text_clean",
            "lang",
            "frames_detected",
            "frame_match_count",
            "dominant_frame",
            "stances_detected",
            "stance_match_count",
            "dominant_stance",
        ]:
            df[column] = []
        for frame in FRAME_LEXICON:
            df[f"frame_score_{frame}"] = []
        for stance in STANCE_LEXICON:
            df[f"stance_score_{stance}"] = []
        return df

    df["text_clean"] = df["text_original"].map(clean_comment_text)
    df["lang"] = df["text_clean"].map(detect_language_safe)
    frame_results = df["text_clean"].map(assign_frames_lexicon)
    df["frames_detected"] = frame_results.map(lambda result: json.dumps(result["frames_detected"], ensure_ascii=False))
    df["frame_match_count"] = frame_results.map(lambda result: result["frame_match_count"])
    df["dominant_frame"] = frame_results.map(lambda result: result["dominant_frame"])
    for frame in FRAME_LEXICON:
        df[f"frame_score_{frame}"] = frame_results.map(lambda result, frame=frame: result["frame_scores"][frame])

    stance_results = df["text_clean"].map(assign_stance_lexicon)
    df["stances_detected"] = stance_results.map(lambda result: json.dumps(result["stances_detected"], ensure_ascii=False))
    df["stance_match_count"] = stance_results.map(lambda result: result["stance_match_count"])
    df["dominant_stance"] = stance_results.map(lambda result: result["dominant_stance"])
    for stance in STANCE_LEXICON:
        df[f"stance_score_{stance}"] = stance_results.map(
            lambda result, stance=stance: result["stance_scores"][stance]
        )
    return df


def compute_frame_tables(
    enriched_df: pd.DataFrame,
    outputs_dir: Path | str = "outputs",
) -> Dict[str, pd.DataFrame]:
    """Compute actor x frame, week x frame, video x frame, and sequence x frame tables."""
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)
    frame_columns = [f"frame_score_{frame}" for frame in FRAME_LEXICON]

    if enriched_df.empty:
        empty = pd.DataFrame()
        empty.to_csv(outputs_path / "frame_actor_matrix.csv", index=False)
        empty.to_csv(outputs_path / "frame_time_series.csv", index=False)
        return {
            "actor_frame": empty,
            "time_frame": empty,
            "video_frame": empty,
            "sequence_frame": empty,
        }

    long_df = enriched_df[
        ["comment_id", "actor", "sequence", "video_id", "video_title", "published_at_comment", *frame_columns]
    ].melt(
        id_vars=["comment_id", "actor", "sequence", "video_id", "video_title", "published_at_comment"],
        value_vars=frame_columns,
        var_name="frame",
        value_name="score",
    )
    long_df["frame"] = long_df["frame"].str.replace("frame_score_", "", regex=False)
    long_df["has_frame"] = long_df["score"] > 0

    total_by_actor = enriched_df.groupby("actor")["comment_id"].nunique().rename("total_comments").reset_index()
    actor_frame = (
        long_df.groupby(["actor", "frame"], as_index=False)
        .agg(score=("score", "sum"), comments_with_frame=("has_frame", "sum"))
        .merge(total_by_actor, on="actor", how="left")
    )
    actor_frame["share_comments"] = actor_frame["comments_with_frame"] / actor_frame["total_comments"].replace(0, np.nan)

    dated = long_df.copy()
    dated["published_at_comment"] = pd.to_datetime(dated["published_at_comment"], errors="coerce", utc=True)
    dated["week"] = dated["published_at_comment"].dt.to_period("W").astype(str)
    time_frame = (
        dated.groupby(["week", "actor", "frame"], dropna=False, as_index=False)
        .agg(score=("score", "sum"), comments_with_frame=("has_frame", "sum"))
        .sort_values(["week", "actor", "frame"])
    )

    video_frame = (
        long_df.groupby(["video_id", "video_title", "actor", "frame"], as_index=False)
        .agg(score=("score", "sum"), comments_with_frame=("has_frame", "sum"))
        .sort_values(["actor", "video_id", "frame"])
    )

    sequence_frame = (
        long_df.groupby(["sequence", "actor", "frame"], as_index=False)
        .agg(score=("score", "sum"), comments_with_frame=("has_frame", "sum"))
        .sort_values(["sequence", "actor", "frame"])
    )

    actor_frame.to_csv(outputs_path / "frame_actor_matrix.csv", index=False)
    time_frame.to_csv(outputs_path / "frame_time_series.csv", index=False)
    return {
        "actor_frame": actor_frame,
        "time_frame": time_frame,
        "video_frame": video_frame,
        "sequence_frame": sequence_frame,
    }


def compute_embeddings(
    texts: Sequence[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 64,
) -> np.ndarray:
    """Compute multilingual sentence embeddings. CPU is enough for a small POC."""
    if not texts:
        return np.empty((0, 0))

    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings)


def reduce_embeddings_2d(embeddings: np.ndarray, method: str = "pca", random_state: int = 42) -> np.ndarray:
    """Reduce embeddings to 2D with PCA by default, optional UMAP if installed."""
    if embeddings.size == 0:
        return np.empty((0, 2))
    if len(embeddings) == 1:
        return np.array([[0.0, 0.0]])

    if method == "umap":
        try:
            import umap  # type: ignore

            return umap.UMAP(n_components=2, random_state=random_state).fit_transform(embeddings)
        except Exception as error:
            print(f"UMAP indisponible, fallback PCA : {error}")

    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    reduced = PCA(n_components=n_components, random_state=random_state).fit_transform(embeddings)
    if reduced.shape[1] == 1:
        reduced = np.column_stack([reduced[:, 0], np.zeros(len(reduced))])
    return reduced


def choose_cluster_count(n_comments: int, max_clusters: int = 8) -> int:
    """Small-data heuristic for KMeans cluster count."""
    if n_comments < 6:
        return 1
    return int(min(max_clusters, max(2, round(math.sqrt(n_comments / 2)))))


SEMANTIC_CLUSTER_COLUMNS = [
    "semantic_cluster",
    "size",
    "share",
    "top_terms",
    "label_auto",
    "dominant_actor",
    "actor_distribution_json",
    "top_videos_json",
    "examples_json",
]


def _top_tfidf_terms(texts: Sequence[str], n_terms: int = 8) -> List[str]:
    if not texts:
        return []
    try:
        vectorizer = TfidfVectorizer(
            max_features=4000,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=sorted(set(FRENCH_STOPWORDS).union(SEMANTIC_STOPWORDS)),
            token_pattern=r"(?u)\b[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ]{2,}\b",
        )
        matrix = vectorizer.fit_transform(texts)
        if matrix.shape[1] == 0:
            return []
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        terms = np.array(vectorizer.get_feature_names_out())
        top_idx = scores.argsort()[::-1][:n_terms]
        return [str(terms[i]) for i in top_idx if scores[i] > 0]
    except ValueError:
        return []


def cluster_embeddings(
    enriched_df: pd.DataFrame,
    embeddings: np.ndarray,
    max_clusters: int = 8,
    min_cluster_size: int = 8,
    reduction_method: str = "pca",
    random_state: int = 42,
    outputs_dir: Path | str = "outputs",
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Cluster comments and describe semantic regions with TF-IDF keywords and examples."""
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)
    df = enriched_df.copy().reset_index(drop=True)

    if df.empty or embeddings.size == 0:
        empty = pd.DataFrame(columns=SEMANTIC_CLUSTER_COLUMNS)
        empty.to_csv(outputs_path / "semantic_clusters.csv", index=False)
        return df, empty, np.empty((0, 2))

    reduced = reduce_embeddings_2d(embeddings, method=reduction_method, random_state=random_state)
    df["semantic_x"] = reduced[:, 0]
    df["semantic_y"] = reduced[:, 1]

    n_clusters = choose_cluster_count(len(df), max_clusters=max_clusters)
    if n_clusters <= 1:
        labels = np.zeros(len(df), dtype=int)
    else:
        labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state).fit_predict(embeddings)
    counts = pd.Series(labels).value_counts()
    labels = np.array([label if counts[label] >= min_cluster_size else -1 for label in labels], dtype=int)
    df["semantic_cluster"] = labels

    cluster_rows = []
    for cluster_id in sorted(label for label in df["semantic_cluster"].unique() if label != -1):
        mask = df["semantic_cluster"] == cluster_id
        cluster_df = df.loc[mask].copy()
        cluster_embeddings_matrix = embeddings[mask.to_numpy()]
        centroid = cluster_embeddings_matrix.mean(axis=0, keepdims=True)
        distances = pairwise_distances(cluster_embeddings_matrix, centroid, metric="cosine").ravel()
        cluster_df["distance_to_centroid"] = distances
        examples = (
            cluster_df.sort_values("distance_to_centroid")["text_clean"]
            .dropna()
            .map(lambda value: textwrap.shorten(str(value), width=220, placeholder="…"))
            .head(5)
            .tolist()
        )
        top_terms = _top_tfidf_terms(cluster_df["text_clean"].dropna().tolist(), n_terms=8)
        actor_distribution = cluster_df["actor"].value_counts(normalize=True).round(3).to_dict()
        dominant_actor = cluster_df["actor"].value_counts().index[0] if not cluster_df.empty else None
        video_distribution = (
            cluster_df["video_title"].fillna(cluster_df["video_id"]).value_counts(normalize=True).head(3).round(3).to_dict()
        )
        cluster_rows.append(
            {
                "semantic_cluster": int(cluster_id),
                "size": int(mask.sum()),
                "share": float(mask.mean()),
                "top_terms": ", ".join(top_terms),
                "label_auto": " / ".join(top_terms[:3]) if top_terms else f"cluster_{cluster_id}",
                "dominant_actor": dominant_actor,
                "actor_distribution_json": json.dumps(actor_distribution, ensure_ascii=False),
                "top_videos_json": json.dumps(video_distribution, ensure_ascii=False),
                "examples_json": json.dumps(examples, ensure_ascii=False),
            }
        )

    clusters_df = pd.DataFrame(cluster_rows, columns=SEMANTIC_CLUSTER_COLUMNS).sort_values("size", ascending=False)
    clusters_df.to_csv(outputs_path / "semantic_clusters.csv", index=False)
    return df, clusters_df, reduced


def compute_actor_trajectories(
    enriched_semantic_df: pd.DataFrame,
    outputs_dir: Path | str = "outputs",
) -> pd.DataFrame:
    """Compute actor trajectories in reduced semantic space by week."""
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    required = {"actor", "published_at_comment", "semantic_x", "semantic_y"}
    if enriched_semantic_df.empty or not required.issubset(enriched_semantic_df.columns):
        trajectories = pd.DataFrame()
        trajectories.to_csv(outputs_path / "semantic_actor_trajectories.csv", index=False)
        return trajectories

    df = enriched_semantic_df.copy()
    df["published_at_comment"] = pd.to_datetime(df["published_at_comment"], errors="coerce", utc=True)
    df["week"] = df["published_at_comment"].dt.to_period("W").astype(str)
    trajectories = (
        df.dropna(subset=["week", "semantic_x", "semantic_y"])
        .groupby(["actor", "week"], as_index=False)
        .agg(
            centroid_x=("semantic_x", "mean"),
            centroid_y=("semantic_y", "mean"),
            n_comments=("comment_id", "nunique"),
        )
        .sort_values(["actor", "week"])
    )

    trajectories["prev_x"] = trajectories.groupby("actor")["centroid_x"].shift(1)
    trajectories["prev_y"] = trajectories.groupby("actor")["centroid_y"].shift(1)
    trajectories["velocity_x"] = trajectories["centroid_x"] - trajectories["prev_x"]
    trajectories["velocity_y"] = trajectories["centroid_y"] - trajectories["prev_y"]
    trajectories["semantic_velocity"] = np.sqrt(
        trajectories["velocity_x"].fillna(0) ** 2 + trajectories["velocity_y"].fillna(0) ** 2
    )
    trajectories.to_csv(outputs_path / "semantic_actor_trajectories.csv", index=False)
    return trajectories


def cosine_distance_vector(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two 1D vectors."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(cosine_distances(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def compute_reception_distance(
    enriched_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    comment_embeddings: np.ndarray,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    outputs_dir: Path | str = "outputs",
) -> pd.DataFrame:
    """Compare video-title embeddings with mean comment embeddings by video."""
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    if enriched_df.empty or comment_embeddings.size == 0:
        empty = pd.DataFrame()
        empty.to_csv(outputs_path / "reception_distance_by_video.csv", index=False)
        return empty

    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)
    df = enriched_df.reset_index(drop=True)
    rows = []

    for _, meta in metadata_df.iterrows():
        video_id = meta.get("video_id")
        mask = df["video_id"] == video_id
        if not mask.any():
            continue

        mean_comment_embedding = comment_embeddings[mask.to_numpy()].mean(axis=0)
        title = str(meta.get("video_title") or meta.get("input_label") or video_id)
        title_embedding = np.asarray(model.encode([title], normalize_embeddings=True))[0]
        distance = cosine_distance_vector(mean_comment_embedding, title_embedding)
        cluster_entropy = None
        dominant_cluster_share = None
        if "semantic_cluster" in df.columns:
            distribution = df.loc[mask, "semantic_cluster"].value_counts(normalize=True)
            if not distribution.empty:
                dominant_cluster_share = float(distribution.iloc[0])
                cluster_entropy = float(-(distribution * np.log2(distribution)).sum())

        if np.isnan(distance):
            reception_label = "non calculable"
        elif distance < 0.35:
            reception_label = "plutôt alignée avec le titre"
        elif distance < 0.60:
            reception_label = "mixte ou partiellement déplacée"
        else:
            reception_label = "fortement déplacée ou fragmentée"

        if dominant_cluster_share is not None and dominant_cluster_share < 0.35:
            reception_label = f"{reception_label}; réception dispersée"

        rows.append(
            {
                "video_id": video_id,
                "video_title": title,
                "actor": meta.get("actor"),
                "sequence": meta.get("sequence"),
                "n_comments": int(mask.sum()),
                "reception_distance": distance,
                "dominant_cluster_share": dominant_cluster_share,
                "cluster_entropy": cluster_entropy,
                "reception_label_auto": reception_label,
            }
        )

    reception_df = pd.DataFrame(rows).sort_values("reception_distance", ascending=False)
    reception_df.to_csv(outputs_path / "reception_distance_by_video.csv", index=False)
    return reception_df


def make_collection_figures(
    enriched_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    frame_tables: Dict[str, pd.DataFrame],
    clusters_df: pd.DataFrame,
    trajectories_df: pd.DataFrame,
    reception_df: pd.DataFrame,
) -> Dict[str, go.Figure]:
    """Build Plotly figures for notebook and dashboard."""
    figures: Dict[str, go.Figure] = {}

    if not enriched_df.empty:
        count_by_video = (
            enriched_df.groupby(["video_id", "video_title", "actor"], as_index=False)
            .agg(n_comments=("comment_id", "nunique"))
            .sort_values("n_comments", ascending=True)
        )
        figures["comments_by_video"] = px.bar(
            count_by_video,
            x="n_comments",
            y="video_title",
            color="actor",
            orientation="h",
            title="Nombre de commentaires collectés par vidéo",
            labels={"n_comments": "Commentaires", "video_title": "Vidéo"},
        )

        count_by_actor = enriched_df.groupby("actor", as_index=False).agg(n_comments=("comment_id", "nunique"))
        figures["comments_by_actor"] = px.pie(
            count_by_actor,
            values="n_comments",
            names="actor",
            title="Répartition des commentaires par acteur",
        )

    actor_frame = frame_tables.get("actor_frame", pd.DataFrame())
    if not actor_frame.empty:
        heatmap_data = actor_frame.pivot_table(index="actor", columns="frame", values="comments_with_frame", fill_value=0)
        figures["actor_frame_heatmap"] = px.imshow(
            heatmap_data,
            text_auto=True,
            title="Heatmap acteur × cadrage lexical",
            labels={"x": "Cadrage", "y": "Acteur", "color": "Commentaires"},
            aspect="auto",
        )
        figures["frame_actor_bar"] = px.bar(
            actor_frame,
            x="frame",
            y="comments_with_frame",
            color="actor",
            barmode="group",
            title="Distribution des cadrages lexicaux par acteur",
            labels={"comments_with_frame": "Commentaires avec match lexical", "frame": "Cadrage"},
        )

    time_frame = frame_tables.get("time_frame", pd.DataFrame())
    if not time_frame.empty:
        figures["frame_time_series"] = px.line(
            time_frame,
            x="week",
            y="comments_with_frame",
            color="frame",
            line_dash="actor",
            markers=True,
            title="Évolution temporelle des cadrages lexicaux",
            labels={"comments_with_frame": "Commentaires avec match", "week": "Semaine"},
        )

    if not enriched_df.empty and {"semantic_x", "semantic_y", "semantic_cluster"}.issubset(enriched_df.columns):
        scatter_df = enriched_df.copy()
        scatter_df["marker_size"] = scatter_df["like_count_comment"].fillna(0).clip(lower=0) + 1
        scatter_df["comment_preview"] = scatter_df["text_clean"].map(
            lambda value: textwrap.shorten(str(value), width=180, placeholder="…")
        )
        figures["semantic_scatter"] = px.scatter(
            scatter_df,
            x="semantic_x",
            y="semantic_y",
            color="actor",
            symbol="video_id",
            size="marker_size",
            hover_data={
                "video_title": True,
                "semantic_cluster": True,
                "comment_preview": True,
                "marker_size": False,
                "author_hash": False,
            },
            title="Projection 2D des commentaires — régions sémantiques émergentes",
            labels={"semantic_x": "Axe sémantique 1", "semantic_y": "Axe sémantique 2"},
        )

        figures["clusters"] = px.scatter(
            scatter_df,
            x="semantic_x",
            y="semantic_y",
            color=scatter_df["semantic_cluster"].astype(str),
            hover_data=["actor", "video_title", "comment_preview"],
            title="Clusters sémantiques émergents",
            labels={"color": "Cluster", "semantic_x": "Axe 1", "semantic_y": "Axe 2"},
        )

    if not trajectories_df.empty:
        figure = px.line(
            trajectories_df,
            x="centroid_x",
            y="centroid_y",
            color="actor",
            text="week",
            markers=True,
            title="Trajectoires des acteurs dans l’espace sémantique",
            labels={"centroid_x": "Centroïde axe 1", "centroid_y": "Centroïde axe 2"},
        )
        for _, row in trajectories_df.dropna(subset=["prev_x", "prev_y"]).iterrows():
            figure.add_annotation(
                x=row["centroid_x"],
                y=row["centroid_y"],
                ax=row["prev_x"],
                ay=row["prev_y"],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1,
                opacity=0.5,
            )
        figures["trajectories"] = figure

    if not reception_df.empty:
        figures["reception_distance"] = px.bar(
            reception_df.sort_values("reception_distance", ascending=True),
            x="reception_distance",
            y="video_title",
            color="actor",
            orientation="h",
            title="Distance titre vidéo ↔ commentaires",
            labels={"reception_distance": "Distance cosinus", "video_title": "Vidéo"},
            hover_data=["reception_label_auto", "n_comments"],
        )

    if not clusters_df.empty:
        figures["cluster_sizes"] = px.bar(
            clusters_df.sort_values("size", ascending=True),
            x="size",
            y=clusters_df["semantic_cluster"].astype(str),
            orientation="h",
            color="dominant_actor",
            title="Taille des clusters sémantiques émergents",
            labels={"size": "Commentaires", "y": "Cluster", "dominant_actor": "Acteur dominant"},
            hover_data=["label_auto", "top_terms"],
        )

    return figures


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a safe compact HTML table."""
    if df.empty:
        return "<p>Aucune donnée disponible.</p>"
    return df.head(max_rows).to_html(index=False, escape=True, classes="data-table")


def build_dashboard(
    enriched_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    frame_tables: Dict[str, pd.DataFrame],
    clusters_df: pd.DataFrame,
    trajectories_df: pd.DataFrame,
    reception_df: pd.DataFrame,
    output_path: Path | str = "outputs/observatoire_cadrages_youtube_dashboard.html",
) -> Path:
    """Build a standalone HTML dashboard; no server is required."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figures = make_collection_figures(
        enriched_df,
        metadata_df,
        frame_tables,
        clusters_df,
        trajectories_df,
        reception_df,
    )

    figure_blocks = []
    first = True
    for key, figure in figures.items():
        figure_blocks.append(
            pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs=True if first else False,
                config={"displaylogo": False, "responsive": True},
            )
        )
        first = False

    metadata_columns = [
        column
        for column in [
            "video_id",
            "actor",
            "sequence",
            "video_title",
            "channel_title",
            "published_at_video",
            "view_count",
            "like_count_video",
            "comment_count_video",
            "collection_error",
        ]
        if column in metadata_df.columns
    ]
    cluster_display = clusters_df.copy()
    if not cluster_display.empty and "examples_json" in cluster_display.columns:
        cluster_display["examples"] = cluster_display["examples_json"].map(
            lambda value: " | ".join(json.loads(value)[:3]) if isinstance(value, str) and value else ""
        )
        cluster_display = cluster_display.drop(columns=["examples_json"], errors="ignore")

    summary = {
        "Vidéos configurées": int(len(metadata_df)),
        "Vidéos avec commentaires collectés": int(enriched_df["video_id"].nunique()) if not enriched_df.empty else 0,
        "Commentaires analysés": int(len(enriched_df)),
        "Acteurs": ", ".join(sorted(map(str, enriched_df["actor"].dropna().unique()))) if not enriched_df.empty else "",
        "Généré le": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }

    summary_html = "".join(
        f"<div class='metric'><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>"
        for label, value in summary.items()
    )

    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Observatoire des cadrages politiques à partir de commentaires YouTube</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #38bdf8;
      --card: #1f2937;
      --border: #334155;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #172554 100%);
      color: var(--text);
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 36px 20px 56px; }}
    h1 {{ font-size: clamp(2rem, 5vw, 4rem); line-height: 0.95; margin: 0 0 18px; letter-spacing: -0.04em; }}
    h2 {{ margin-top: 42px; color: #f8fafc; }}
    p, li {{ color: var(--muted); line-height: 1.6; }}
    .lede {{ max-width: 880px; font-size: 1.08rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 28px 0; }}
    .metric {{ background: rgba(31, 41, 55, 0.92); border: 1px solid var(--border); border-radius: 18px; padding: 16px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 0.9rem; }}
    .metric strong {{ display: block; margin-top: 8px; color: white; font-size: 1.15rem; }}
    .section {{ background: rgba(15, 23, 42, 0.76); border: 1px solid var(--border); border-radius: 22px; padding: 18px; margin: 20px 0; }}
    .data-table {{ width: 100%; border-collapse: collapse; background: white; color: #111827; font-size: 0.86rem; border-radius: 12px; overflow: hidden; }}
    .data-table th, .data-table td {{ border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; }}
    .data-table th {{ background: #f1f5f9; }}
    .warning {{ border-left: 4px solid #f59e0b; padding: 12px 16px; background: rgba(245, 158, 11, 0.12); border-radius: 10px; }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
<main>
  <h1>Observatoire des cadrages politiques à partir de commentaires YouTube</h1>
  <p class="lede">Prototype d'analyse agrégée de conversations publiques : thèmes, cadrages lexicaux, régions sémantiques émergentes, trajectoires et distances de réception. Le dashboard ne contient pas de noms d'utilisateurs et ne produit aucun scoring individuel.</p>
  <div class="grid">{summary_html}</div>

  <div class="section warning">
    <strong>Limite méthodologique majeure.</strong>
    <p>Les commentaires YouTube ne sont pas représentatifs de l'opinion générale. Ils reflètent une population auto-sélectionnée, les règles de modération de la plateforme, la visibilité algorithmique et la dynamique propre à chaque chaîne.</p>
  </div>

  <h2>Visualisations</h2>
  {"".join(f"<div class='section'>{block}</div>" for block in figure_blocks) if figure_blocks else "<p>Aucune visualisation disponible.</p>"}

  <h2>Tableau des vidéos</h2>
  <div class="section">{dataframe_to_html_table(metadata_df[metadata_columns] if metadata_columns else metadata_df, max_rows=30)}</div>

  <h2>Clusters émergents et exemples anonymisés</h2>
  <div class="section">{dataframe_to_html_table(cluster_display, max_rows=30)}</div>

  <h2>Limites méthodologiques et juridiques</h2>
  <div class="section">
    <ul>
      <li>Analyse agrégée uniquement : pas de microciblage, pas de score individuel, pas de recommandation de persuasion personnalisée.</li>
      <li>Les identifiants d'auteurs sont hachés ou absents ; les noms publics ne sont pas conservés dans les exports par défaut.</li>
      <li>Les cadrages lexicaux sont interprétables mais incomplets ; les clusters sont des régions sémantiques émergentes, pas des vérités objectives.</li>
      <li>Les distances sémantiques indiquent des déplacements potentiels de discussion ; elles doivent être validées par lecture humaine.</li>
      <li>Respecter les conditions de l'API YouTube, les obligations RGPD applicables et les durées de conservation adaptées au projet.</li>
    </ul>
  </div>
</main>
</body>
</html>"""

    output.write_text(html_doc, encoding="utf-8")
    return output


def generate_interpretation_note(
    enriched_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    frame_tables: Dict[str, pd.DataFrame],
    clusters_df: pd.DataFrame,
    trajectories_df: pd.DataFrame,
    reception_df: pd.DataFrame,
    output_path: Path | str = "outputs/interpretation_note.md",
) -> Path:
    """Generate a cautious Markdown interpretation note."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    videos_lines = []
    if not metadata_df.empty:
        for _, row in metadata_df.iterrows():
            title = row.get("video_title") or row.get("input_label") or row.get("video_id")
            videos_lines.append(f"- `{row.get('video_id')}` — {row.get('actor')} — {title}")

    actor_frame = frame_tables.get("actor_frame", pd.DataFrame())
    if not actor_frame.empty:
        top_frames = (
            actor_frame.sort_values("comments_with_frame", ascending=False)
            .groupby("actor")
            .head(3)
            .loc[:, ["actor", "frame", "comments_with_frame"]]
        )
        frame_lines = [
            f"- {row.actor} : `{row.frame}` ({int(row.comments_with_frame)} commentaires avec match lexical)"
            for _, row in top_frames.iterrows()
        ]
    else:
        frame_lines = ["- Aucun cadrage lexical dominant détecté ou aucun commentaire disponible."]

    if not clusters_df.empty:
        cluster_lines = [
            f"- Cluster {int(row.semantic_cluster)} — {row.size} commentaires — mots-clés : {row.top_terms}"
            for _, row in clusters_df.head(8).iterrows()
        ]
    else:
        cluster_lines = ["- Clusters non calculés, probablement faute de commentaires ou de dépendances embeddings."]

    if not trajectories_df.empty:
        velocity = trajectories_df.dropna(subset=["semantic_velocity"]).sort_values("semantic_velocity", ascending=False)
        trajectory_lines = [
            f"- {row.actor}, semaine {row.week} : vitesse sémantique {row.semantic_velocity:.3f}"
            for _, row in velocity.head(6).iterrows()
        ]
    else:
        trajectory_lines = ["- Trajectoires non calculées ou série temporelle trop courte."]

    if not reception_df.empty:
        reception_lines = [
            f"- {row.actor} — {row.video_title} : distance {row.reception_distance:.3f} ; {row.reception_label_auto}"
            for _, row in reception_df.head(8).iterrows()
        ]
    else:
        reception_lines = ["- Distance de réception non calculée."]

    note = f"""# Note d'interprétation — Observatoire des cadrages YouTube

Générée automatiquement le {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}.

## Données analysées

Dans les commentaires collectés, on observe `{len(enriched_df)}` commentaire(s) provenant de `{metadata_df["video_id"].nunique() if not metadata_df.empty else 0}` vidéo(s) configurée(s).

{chr(10).join(videos_lines) if videos_lines else "- Aucune vidéo disponible."}

## Principaux cadrages détectés

Ces résultats reposent sur un dictionnaire lexical interprétable. Ils signalent des indices de cadrage, pas des intentions individuelles.

{chr(10).join(frame_lines)}

## Différences entre acteurs

Les différences doivent être lues comme des différences dans les commentaires collectés sous les vidéos associées à chaque acteur, et non comme une mesure représentative de l'opinion générale.

## Régions sémantiques émergentes

Les clusters ci-dessous sont des régions sémantiques construites par embeddings et KMeans. Les labels automatiques doivent être validés par lecture humaine.

{chr(10).join(cluster_lines)}

## Déplacements sémantiques observés

{chr(10).join(trajectory_lines)}

## Réception et recodage

Une distance élevée entre le titre de la vidéo et le commentaire moyen peut indiquer un déplacement de discussion, une réception hostile, ironique, dispersée ou centrée sur un autre enjeu. Elle ne suffit pas à conclure sans inspection qualitative.

{chr(10).join(reception_lines)}

## Limites

- Les commentaires YouTube ne sont pas représentatifs de l'opinion générale.
- Les commentaires publics sont produits par des publics auto-sélectionnés et soumis aux dynamiques de plateforme.
- Le lexique de cadrage est incomplet et doit être adapté au terrain.
- Les embeddings et clusters donnent des proximités statistiques, pas des vérités objectives.
- Le POC ne doit pas être utilisé pour du microciblage politique, du scoring individuel ou de la persuasion personnalisée.
- Les identifiants d'auteurs sont hachés ou absents ; les noms d'utilisateurs ne sont pas exportés par défaut.

## Prochaines étapes

- Valider humainement les labels de clusters.
- Ajouter une annotation qualitative sur un échantillon stratifié.
- Comparer plusieurs séquences temporelles ou plusieurs types de sources.
- Tester BERTopic/HDBSCAN si le volume de commentaires devient suffisant.
"""
    output.write_text(note, encoding="utf-8")
    return output


def read_cached_dataset(
    data_dir: Path,
    requested_video_ids: Sequence[str],
    cache_context: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    comments_path = data_dir / "youtube_comments_raw_anonymized.csv"
    metadata_path = data_dir / "video_metadata.csv"
    if not comments_path.exists() or not metadata_path.exists():
        return None

    if cache_context is not None:
        cached = DatasetCache(data_dir).read(cache_context)
        if cached is not None:
            print("Cache manifesté détecté : réutilisation des fichiers data/*.csv.")
            return cached
        print("Cache présent mais paramètres différents ou manifeste absent : nouvelle collecte.")
        return None

    comments_df = pd.read_csv(comments_path)
    metadata_df = pd.read_csv(metadata_path)
    cached_ids = set(metadata_df.get("video_id", pd.Series(dtype=str)).dropna().astype(str))
    if set(requested_video_ids).issubset(cached_ids):
        print("Cache legacy détecté : réutilisation des fichiers data/*.csv.")
        return comments_df, metadata_df
    print("Cache présent mais incomplet pour la liste de vidéos demandée : nouvelle collecte.")
    return None


def run_pipeline(
    videos: Sequence[Dict[str, Any]] = VIDEOS,
    api_key: Optional[str] = None,
    max_comments_per_video: int = 500,
    include_replies: bool = False,
    force_refresh: bool = False,
    data_dir: Path | str = "data",
    outputs_dir: Path | str = "outputs",
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    skip_embeddings: bool = False,
    min_cluster_chars: int = 80,
    min_cluster_meaningful_tokens: int = 8,
    semantic_cluster_min_size: int = 8,
    extract_discourse_cards: bool = False,
    cluster_discourse_cards: bool = False,
    build_discourse_graph: bool = False,
    extract_claims: bool = False,
    cluster_claims: bool = False,
    label_claim_clusters: bool = False,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4.1-mini",
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    max_claims_per_comment: int = 1,
    claim_min_confidence: float = 0.65,
    claim_extraction_limit: Optional[int] = None,
    claim_cluster_min_size: int = 8,
    claim_cluster_distance_threshold: float = 0.35,
    discursive_card_min_confidence: float = 0.55,
    discursive_card_limit: Optional[int] = None,
    discursive_card_prompt: Path | str = "prompts/extract_discursive_card.md",
    discursive_card_cache: Optional[Path | str] = None,
    disable_discursive_card_cache: bool = False,
    discursive_card_workers: int = 1,
    reuse_discourse_cards: bool = False,
    discursive_cluster_min_size: int = 6,
    discursive_cluster_distance_threshold: float = 0.35,
    discourse_graph_min_community_size: int = 4,
    discourse_graph_similarity_threshold: float = 0.18,
    discourse_graph_top_k: int = 8,
) -> Dict[str, Any]:
    """Run the full POC pipeline and export all requested files."""
    data_path, outputs_path = ensure_dirs(data_dir, outputs_dir)
    normalized = normalize_videos(videos)
    requested_ids = [video["video_id"] for video in normalized]
    cache_context = build_raw_cache_context(
        requested_video_ids=requested_ids,
        max_comments_per_video=max_comments_per_video,
        include_replies=include_replies,
    )
    raw_cache = DatasetCache(data_path)

    cached = None if force_refresh else read_cached_dataset(data_path, requested_ids, cache_context)
    if cached is None:
        resolved_key = resolve_api_key(api_key)
        raw_df, metadata_df = build_dataset(
            normalized,
            resolved_key,
            max_comments_per_video=max_comments_per_video,
            include_replies=include_replies,
        )
        assert_privacy_safe_columns(raw_df.columns)
        raw_df.to_csv(data_path / "youtube_comments_raw_anonymized.csv", index=False)
        assert_privacy_safe_columns(metadata_df.columns)
        metadata_df.to_csv(data_path / "video_metadata.csv", index=False)
        raw_cache.write_manifest(
            cache_context,
            stats={
                "comments_rows": int(len(raw_df)),
                "metadata_rows": int(len(metadata_df)),
            },
        )
    else:
        raw_df, metadata_df = cached

    enriched_df = enrich_text_and_frames(raw_df)
    if not enriched_df.empty:
        quality = enriched_df["text_clean"].map(text_quality_metrics)
        enriched_df["semantic_char_count"] = quality.map(lambda metrics: metrics["char_count"])
        enriched_df["semantic_meaningful_token_count"] = quality.map(
            lambda metrics: metrics["meaningful_token_count"]
        )
        enriched_df["semantic_exclusion_reason"] = enriched_df["text_clean"].map(
            lambda text: semantic_exclusion_reason(
                text,
                min_chars=min_cluster_chars,
                min_meaningful_tokens=min_cluster_meaningful_tokens,
            )
        )
        enriched_df["semantic_candidate"] = enriched_df["semantic_exclusion_reason"] == "ok"
        filter_summary = (
            enriched_df.groupby(["semantic_exclusion_reason"], as_index=False)
            .agg(n_comments=("comment_id", "nunique"))
            .sort_values("n_comments", ascending=False)
        )
        filter_summary.to_csv(outputs_path / "semantic_filter_summary.csv", index=False)
    else:
        enriched_df["semantic_candidate"] = []
        enriched_df["semantic_exclusion_reason"] = []
        pd.DataFrame(columns=["semantic_exclusion_reason", "n_comments"]).to_csv(
            outputs_path / "semantic_filter_summary.csv",
            index=False,
        )
    assert_privacy_safe_columns(enriched_df.columns)
    enriched_df.to_csv(data_path / "youtube_comments_enriched.csv", index=False)
    frame_tables = compute_frame_tables(enriched_df, outputs_path)

    embeddings = np.empty((0, 0))
    clusters_df = pd.DataFrame()
    semantic_df = enriched_df.copy()
    trajectories_df = pd.DataFrame()
    reception_df = pd.DataFrame()
    claims_df = pd.DataFrame()
    claim_clusters_df = pd.DataFrame()
    claim_cluster_labels_path: Optional[Path] = None
    discursive_cards_df = pd.DataFrame()
    discursive_clusters_df = empty_discursive_clusters_df()
    discursive_embeddings = np.empty((0, 0))
    discursive_communities_df = empty_discursive_communities_df()
    discourse_graph_result = None

    non_empty_mask = (
        semantic_df["semantic_candidate"].fillna(False).astype(bool)
        if not semantic_df.empty and "semantic_candidate" in semantic_df.columns
        else pd.Series(dtype=bool)
    )
    if skip_embeddings:
        print("Embeddings désactivés (--skip-embeddings).")
    elif not semantic_df.empty and non_empty_mask.any():
        texts = semantic_df.loc[non_empty_mask, "text_clean"].tolist()
        excluded = int((~non_empty_mask).sum())
        print(f"Calcul embeddings pour {len(texts)} commentaire(s) candidat(s). {excluded} exclu(s) du clustering.")
        embeddings_subset = compute_embeddings(texts, model_name=embedding_model)
        semantic_subset, clusters_df, _ = cluster_embeddings(
            semantic_df.loc[non_empty_mask].reset_index(drop=True),
            embeddings_subset,
            min_cluster_size=semantic_cluster_min_size,
            outputs_dir=outputs_path,
        )
        semantic_df = semantic_df.copy()
        for column in ["semantic_x", "semantic_y", "semantic_cluster"]:
            semantic_df[column] = np.nan
        semantic_df.loc[non_empty_mask, ["semantic_x", "semantic_y", "semantic_cluster"]] = semantic_subset[
            ["semantic_x", "semantic_y", "semantic_cluster"]
        ].to_numpy()
        semantic_df.to_csv(data_path / "youtube_comments_enriched.csv", index=False)
        trajectories_df = compute_actor_trajectories(semantic_df, outputs_path)
        reception_df = compute_reception_distance(
            semantic_df.loc[non_empty_mask].reset_index(drop=True),
            metadata_df,
            embeddings_subset,
            model_name=embedding_model,
            outputs_dir=outputs_path,
        )
        embeddings = embeddings_subset
    else:
        print("Aucun texte exploitable pour les embeddings.")
        clusters_df.to_csv(outputs_path / "semantic_clusters.csv", index=False)
        trajectories_df.to_csv(outputs_path / "semantic_actor_trajectories.csv", index=False)
        reception_df.to_csv(outputs_path / "reception_distance_by_video.csv", index=False)

    should_extract_discourse_cards = extract_discourse_cards or cluster_discourse_cards or build_discourse_graph
    if should_extract_discourse_cards:
        existing_cards_path = outputs_path / "discursive_cards.csv"
        should_reuse_existing_cards = existing_cards_path.exists() and (
            reuse_discourse_cards or not extract_discourse_cards
        )
        if should_reuse_existing_cards:
            print(f"Réutilisation des fiches discursives existantes : {existing_cards_path}")
            discursive_cards_df = pd.read_csv(existing_cards_path).reindex(columns=DISCURSIVE_CARD_COLUMNS)
        else:
            client = make_llm_client(llm_provider, llm_model, api_key=llm_api_key, base_url=llm_base_url)
            if client is None:
                raise RuntimeError("Discursive card extraction requires an LLM provider. Use --llm-provider openai or ollama.")
            cache_path = None
            if not disable_discursive_card_cache:
                cache_path = Path(discursive_card_cache) if discursive_card_cache else outputs_path / "discursive_card_llm_cache.jsonl"
            print(
                "Extraction des fiches discursives structurées par LLM "
                f"(workers={max(1, int(discursive_card_workers or 1))}, "
                f"cache={'off' if cache_path is None else cache_path}, "
                f"prompt={discursive_card_prompt})."
            )
            discursive_cards_df = extract_discursive_cards_from_comments(
                semantic_df,
                client=client,
                prompt_path=discursive_card_prompt,
                min_confidence=discursive_card_min_confidence,
                limit=discursive_card_limit,
                llm_cache_path=cache_path,
                cache_namespace=f"{llm_provider}:{llm_model}",
                workers=discursive_card_workers,
            )
            discursive_cards_df.to_csv(existing_cards_path, index=False)
        write_discursive_card_coverage(
            semantic_df,
            discursive_cards_df,
            outputs_path / "discursive_card_coverage.csv",
        )

    if cluster_discourse_cards and not discursive_cards_df.empty:
        if skip_embeddings:
            print("Clustering des fiches discursives ignoré : embeddings désactivés (--skip-embeddings).")
            discursive_clusters_df.to_csv(outputs_path / "discursive_clusters.csv", index=False)
        else:
            print(f"Clustering de {len(discursive_cards_df)} fiche(s) discursive(s).")
            discursive_embeddings = compute_embeddings(
                discursive_cards_df["discursive_summary"].fillna("").tolist(),
                model_name=embedding_model,
            )
            discursive_cards_df, discursive_clusters_df = cluster_discursive_cards(
                discursive_cards_df,
                discursive_embeddings,
                min_cluster_size=discursive_cluster_min_size,
                distance_threshold=discursive_cluster_distance_threshold,
                outputs_dir=outputs_path,
            )
    elif should_extract_discourse_cards:
        discursive_clusters_df.to_csv(outputs_path / "discursive_clusters.csv", index=False)

    if build_discourse_graph:
        graph_embeddings = None
        if not discursive_cards_df.empty and not skip_embeddings:
            if discursive_embeddings.size == 0 or discursive_embeddings.shape[0] != len(discursive_cards_df):
                print(f"Calcul embeddings pour {len(discursive_cards_df)} fiche(s) discursive(s) du graphe.")
                discursive_embeddings = compute_embeddings(
                    discursive_cards_df["discursive_summary"].fillna("").tolist(),
                    model_name=embedding_model,
                )
            graph_embeddings = discursive_embeddings
        if discursive_cards_df.empty:
            print("Graphe discursif ignoré : aucune fiche discursive exploitable.")
        else:
            print(f"Construction du graphe discursif V2.6.3 pour {len(discursive_cards_df)} fiche(s).")
        discourse_graph_result = build_discourse_graph_exports(
            discursive_cards_df,
            semantic_df,
            embeddings=graph_embeddings,
            outputs_dir=outputs_path,
            min_community_size=discourse_graph_min_community_size,
            similarity_threshold=discourse_graph_similarity_threshold,
            top_k=discourse_graph_top_k,
        )
        discursive_communities_df = discourse_graph_result.communities_df

    should_extract_claims = extract_claims or cluster_claims or label_claim_clusters
    if should_extract_claims:
        client = make_llm_client(llm_provider, llm_model, api_key=llm_api_key, base_url=llm_base_url)
        if client is None:
            raise RuntimeError("Claim extraction requires an LLM provider. Use --llm-provider openai or ollama.")
        print("Extraction inductive des claims par LLM.")
        claims_df = extract_claims_from_comments(
            semantic_df,
            client=client,
            max_claims_per_comment=max_claims_per_comment,
            min_confidence=claim_min_confidence,
            limit=claim_extraction_limit,
        )
        claims_df.to_csv(outputs_path / "comment_claims.csv", index=False)
        write_no_claim_summary(semantic_df, claims_df, outputs_path / "no_claim_summary.csv")

    should_cluster_claims = cluster_claims or label_claim_clusters
    if should_cluster_claims and not claims_df.empty:
        if skip_embeddings:
            print("Clustering des claims ignoré : embeddings désactivés (--skip-embeddings).")
        else:
            print(f"Clustering de {len(claims_df)} claim(s) inductif(s).")
            claim_embeddings = compute_embeddings(claims_df["claim_text"].fillna("").tolist(), model_name=embedding_model)
            claims_df, claim_clusters_df = cluster_extracted_claims(
                claims_df,
                claim_embeddings,
                min_cluster_size=claim_cluster_min_size,
                distance_threshold=claim_cluster_distance_threshold,
                outputs_dir=outputs_path,
            )
    elif should_extract_claims:
        claim_clusters_df.to_csv(outputs_path / "claim_clusters.csv", index=False)

    if label_claim_clusters:
        label_client = make_llm_client(llm_provider, llm_model, api_key=llm_api_key, base_url=llm_base_url)
        labels = label_extracted_claim_clusters(claim_clusters_df, client=label_client)
        claim_cluster_labels_path = write_claim_cluster_labels(labels, outputs_path / "claim_cluster_labels.md")

    dashboard_path = build_dashboard(
        semantic_df,
        metadata_df,
        frame_tables,
        clusters_df,
        trajectories_df,
        reception_df,
        output_path=outputs_path / "observatoire_cadrages_youtube_dashboard.html",
    )
    note_path = generate_interpretation_note(
        semantic_df,
        metadata_df,
        frame_tables,
        clusters_df,
        trajectories_df,
        reception_df,
        output_path=outputs_path / "interpretation_note.md",
    )

    return {
        "raw_df": raw_df,
        "enriched_df": semantic_df,
        "metadata_df": metadata_df,
        "frame_tables": frame_tables,
        "clusters_df": clusters_df,
        "trajectories_df": trajectories_df,
        "reception_df": reception_df,
        "claims_df": claims_df,
        "claim_clusters_df": claim_clusters_df,
        "claim_cluster_labels_path": claim_cluster_labels_path,
        "discursive_cards_df": discursive_cards_df,
        "discursive_clusters_df": discursive_clusters_df,
        "discursive_communities_df": discursive_communities_df,
        "discourse_graph_result": discourse_graph_result,
        "embeddings": embeddings,
        "dashboard_path": dashboard_path,
        "interpretation_note_path": note_path,
    }


def load_videos_from_json(path: Path | str) -> List[Dict[str, Any]]:
    """Load a custom VIDEOS list from a JSON file."""
    return load_video_config(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POC observatoire des cadrages YouTube.")
    parser.add_argument("--api-key", default=None, help="Clé YouTube Data API v3. Sinon YOUTUBE_API_KEY.")
    parser.add_argument("--videos-json", default=None, help="Fichier JSON contenant la liste VIDEOS.")
    parser.add_argument("--videos-config", default=None, help="Fichier JSON/YAML contenant une clé videos ou une liste.")
    parser.add_argument("--max-comments-per-video", type=int, default=500)
    parser.add_argument("--include-replies", action="store_true", help="Collecter aussi les réponses aux commentaires.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignorer le cache CSV local.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Modèle SentenceTransformer multilingue léger.",
    )
    parser.add_argument("--skip-embeddings", action="store_true", help="Ne produire que la collecte et le lexique.")
    parser.add_argument(
        "--min-cluster-chars",
        type=int,
        default=80,
        help="Longueur minimale pour inclure un commentaire dans les embeddings/clusters.",
    )
    parser.add_argument(
        "--min-cluster-meaningful-tokens",
        type=int,
        default=8,
        help="Nombre minimal de tokens informatifs pour inclure un commentaire dans les embeddings/clusters.",
    )
    parser.add_argument(
        "--semantic-cluster-min-size",
        type=int,
        default=8,
        help="Taille minimale d'un cluster sémantique conservé ; les groupes plus petits restent en bruit.",
    )
    parser.add_argument("--extract-discourse-cards", action="store_true", help="Extraire des fiches discursives V2.6.1 par LLM.")
    parser.add_argument("--cluster-discourse-cards", action="store_true", help="Clusteriser les fiches discursives V2.6.1.")
    parser.add_argument(
        "--build-discourse-graph",
        "--build-discursive-graph",
        dest="build_discourse_graph",
        action="store_true",
        help="Construire le graphe discursif typé V2.6.3 à partir des fiches LLM.",
    )
    parser.add_argument("--extract-claims", action="store_true", help="Extraire des claims inductifs par LLM.")
    parser.add_argument("--cluster-claims", action="store_true", help="Clusteriser les claims extraits.")
    parser.add_argument("--label-claim-clusters", action="store_true", help="Nommer les clusters de claims.")
    parser.add_argument("--llm-provider", default="openai", help="Provider LLM : openai, ollama ou none.")
    parser.add_argument("--llm-model", default="gpt-4.1-mini", help="Modèle LLM pour extraction/labeling.")
    parser.add_argument("--llm-api-key", default=None, help="Clé API LLM. Sinon OPENAI_API_KEY.")
    parser.add_argument("--llm-base-url", default=None, help="Base URL LLM. Ollama par défaut : http://localhost:11434.")
    parser.add_argument("--max-claims-per-comment", type=int, default=1)
    parser.add_argument("--claim-min-confidence", type=float, default=0.65)
    parser.add_argument("--claim-extraction-limit", type=int, default=None)
    parser.add_argument("--claim-cluster-min-size", type=int, default=8)
    parser.add_argument("--claim-cluster-distance-threshold", type=float, default=0.35)
    parser.add_argument("--discursive-card-min-confidence", type=float, default=0.55)
    parser.add_argument("--discursive-card-limit", type=int, default=None)
    parser.add_argument("--discursive-card-prompt", default="prompts/extract_discursive_card.md")
    parser.add_argument("--discursive-card-cache", default=None, help="Cache JSONL des réponses LLM pour les fiches discursives.")
    parser.add_argument("--disable-discursive-card-cache", action="store_true", help="Désactiver le cache LLM des fiches discursives.")
    parser.add_argument("--discursive-card-workers", type=int, default=1, help="Nombre d'appels LLM concurrents pour les fiches discursives.")
    parser.add_argument("--reuse-discourse-cards", action="store_true", help="Réutiliser outputs/discursive_cards.csv si présent.")
    parser.add_argument("--discursive-cluster-min-size", type=int, default=6)
    parser.add_argument("--discursive-cluster-distance-threshold", type=float, default=0.35)
    parser.add_argument("--discourse-graph-min-community-size", type=int, default=4)
    parser.add_argument("--discourse-graph-similarity-threshold", type=float, default=0.18)
    parser.add_argument("--discourse-graph-top-k", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos_config = args.videos_config or args.videos_json
    videos = load_videos_from_json(videos_config) if videos_config else VIDEOS
    result = run_pipeline(
        videos=videos,
        api_key=args.api_key,
        max_comments_per_video=args.max_comments_per_video,
        include_replies=args.include_replies,
        force_refresh=args.force_refresh,
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        embedding_model=args.embedding_model,
        skip_embeddings=args.skip_embeddings,
        min_cluster_chars=args.min_cluster_chars,
        min_cluster_meaningful_tokens=args.min_cluster_meaningful_tokens,
        semantic_cluster_min_size=args.semantic_cluster_min_size,
        extract_discourse_cards=args.extract_discourse_cards,
        cluster_discourse_cards=args.cluster_discourse_cards,
        build_discourse_graph=args.build_discourse_graph,
        extract_claims=args.extract_claims,
        cluster_claims=args.cluster_claims,
        label_claim_clusters=args.label_claim_clusters,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        max_claims_per_comment=args.max_claims_per_comment,
        claim_min_confidence=args.claim_min_confidence,
        claim_extraction_limit=args.claim_extraction_limit,
        claim_cluster_min_size=args.claim_cluster_min_size,
        claim_cluster_distance_threshold=args.claim_cluster_distance_threshold,
        discursive_card_min_confidence=args.discursive_card_min_confidence,
        discursive_card_limit=args.discursive_card_limit,
        discursive_card_prompt=args.discursive_card_prompt,
        discursive_card_cache=args.discursive_card_cache,
        disable_discursive_card_cache=args.disable_discursive_card_cache,
        discursive_card_workers=args.discursive_card_workers,
        reuse_discourse_cards=args.reuse_discourse_cards,
        discursive_cluster_min_size=args.discursive_cluster_min_size,
        discursive_cluster_distance_threshold=args.discursive_cluster_distance_threshold,
        discourse_graph_min_community_size=args.discourse_graph_min_community_size,
        discourse_graph_similarity_threshold=args.discourse_graph_similarity_threshold,
        discourse_graph_top_k=args.discourse_graph_top_k,
    )
    print("\nExports générés :")
    print(" - data/youtube_comments_raw_anonymized.csv")
    print(" - data/youtube_comments_enriched.csv")
    print(" - data/video_metadata.csv")
    print(" - outputs/frame_actor_matrix.csv")
    print(" - outputs/frame_time_series.csv")
    print(" - outputs/semantic_clusters.csv")
    print(" - outputs/semantic_filter_summary.csv")
    print(" - outputs/reception_distance_by_video.csv")
    if args.extract_discourse_cards or args.cluster_discourse_cards or args.build_discourse_graph:
        print(" - outputs/discursive_cards.csv")
        print(" - outputs/discursive_card_coverage.csv")
        print(" - outputs/discursive_clusters.csv")
    if args.build_discourse_graph:
        print(" - outputs/discursive_nodes.csv")
        print(" - outputs/discursive_edges.csv")
        print(" - outputs/discursive_similarity_edges.csv")
        print(" - outputs/discursive_communities.csv")
        print(" - outputs/discursive_community_profiles.md")
    if args.extract_claims or args.cluster_claims or args.label_claim_clusters:
        print(" - outputs/comment_claims.csv")
        print(" - outputs/no_claim_summary.csv")
        print(" - outputs/claim_clusters.csv")
        print(" - outputs/claim_cluster_labels.md")
    print(f" - {result['dashboard_path']}")
    print(f" - {result['interpretation_note_path']}")


if __name__ == "__main__":
    main()
