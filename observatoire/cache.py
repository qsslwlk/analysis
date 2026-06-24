"""Versioned cache helpers for raw YouTube collection outputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


CACHE_SCHEMA_VERSION = "raw-youtube-cache-v1"


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_raw_cache_context(
    requested_video_ids: Sequence[str],
    max_comments_per_video: int,
    include_replies: bool,
    collector: str = "youtube_data_api_v3",
) -> Dict[str, Any]:
    """Build the stable part of the raw-data cache manifest."""
    context = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "collector": collector,
        "requested_video_ids": sorted(str(video_id) for video_id in requested_video_ids),
        "max_comments_per_video": int(max_comments_per_video),
        "include_replies": bool(include_replies),
    }
    return {**context, "fingerprint": _fingerprint(context)}


class DatasetCache:
    """Read and write cached raw comments with a manifest guard."""

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.comments_path = self.data_dir / "youtube_comments_raw_anonymized.csv"
        self.metadata_path = self.data_dir / "video_metadata.csv"
        self.manifest_path = self.data_dir / "raw_collection_manifest.json"

    def has_files(self) -> bool:
        return self.comments_path.exists() and self.metadata_path.exists()

    def read_manifest(self) -> Optional[Dict[str, Any]]:
        if not self.manifest_path.exists():
            return None
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def is_valid_for(self, expected_context: Dict[str, Any]) -> bool:
        if not self.has_files():
            return False
        manifest = self.read_manifest()
        if not manifest:
            return False
        return manifest.get("fingerprint") == expected_context.get("fingerprint")

    def read(self, expected_context: Dict[str, Any]) -> Optional[Tuple[Any, Any]]:
        if not self.is_valid_for(expected_context):
            return None

        import pandas as pd

        comments_df = pd.read_csv(self.comments_path)
        metadata_df = pd.read_csv(self.metadata_path)
        return comments_df, metadata_df

    def write_manifest(self, context: Dict[str, Any], stats: Optional[Dict[str, Any]] = None) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            **context,
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": {
                "comments": self.comments_path.name,
                "metadata": self.metadata_path.name,
            },
            "stats": stats or {},
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self.manifest_path
