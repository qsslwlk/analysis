"""Small typed contracts used by the V2 wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class VideoSpec:
    video_url: Optional[str] = None
    video_id: Optional[str] = None
    actor: str = "Non renseigné"
    sequence: str = "Non renseigné"
    label: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VideoSpec":
        return cls(
            video_url=payload.get("video_url"),
            video_id=payload.get("video_id"),
            actor=payload.get("actor", "Non renseigné"),
            sequence=payload.get("sequence", "Non renseigné"),
            label=payload.get("label"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

