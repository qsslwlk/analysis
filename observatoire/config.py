"""Configuration loaders for V2 corpus and frame files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _read_structured_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("YAML config requires PyYAML. Install requirements.txt first.") from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config format: {path.suffix}. Use .json, .yaml, or .yml.")


def load_video_config(path: Path | str) -> List[Dict[str, Any]]:
    """Load a video corpus from a JSON/YAML list or an object with a videos key."""
    config_path = Path(path)
    payload = _read_structured_file(config_path)
    videos = payload.get("videos") if isinstance(payload, dict) else payload
    if not isinstance(videos, list):
        raise ValueError("Video config must be a list or an object with a 'videos' list.")

    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(videos, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Video entry #{index} must be an object.")
        if not item.get("video_id") and not item.get("video_url"):
            raise ValueError(f"Video entry #{index} needs either video_id or video_url.")
        normalized.append(dict(item))
    return normalized


def load_frame_lexicon(path: Path | str) -> Dict[str, List[str]]:
    """Load a frame lexicon from JSON/YAML."""
    payload = _read_structured_file(Path(path))
    frames = payload.get("frames") if isinstance(payload, dict) and "frames" in payload else payload
    if not isinstance(frames, dict):
        raise ValueError("Frame lexicon must be an object mapping frame names to term lists.")
    for frame, terms in frames.items():
        if not isinstance(frame, str) or not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
            raise ValueError("Frame lexicon values must be lists of strings.")
    return {str(frame): list(terms) for frame, terms in frames.items()}

