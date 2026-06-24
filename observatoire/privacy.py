"""Privacy checks for public exports."""

from __future__ import annotations

from typing import Iterable, List


FORBIDDEN_EXPORT_COLUMNS = {
    "author_display_name",
    "author_channel_url",
    "author_profile_image_url",
    "author_channel_id",
    "author_external_channel_id",
}


def forbidden_columns(columns: Iterable[str]) -> List[str]:
    """Return forbidden export columns found in a column collection."""
    return sorted(FORBIDDEN_EXPORT_COLUMNS.intersection(str(column) for column in columns))


def assert_privacy_safe_columns(columns: Iterable[str]) -> None:
    """Raise if an export would include directly identifying author fields."""
    unsafe = forbidden_columns(columns)
    if unsafe:
        joined = ", ".join(unsafe)
        raise ValueError(f"Unsafe export columns detected: {joined}")

