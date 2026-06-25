#!/usr/bin/env python3
"""
Controlled taxonomy induction and local remapping for discursive graph outputs.

The script works from existing CSV outputs. It never re-encodes comments and
never overwrites the source taxonomy file. It produces auditable candidates,
a remap table, a candidate taxonomy and before/after diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


NON_SIGNAL_LABELS = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "other",
    "autre",
    "unknown",
    "inconnu",
    "unclear",
    "indetermine",
    "indéterminé",
    "non_classe",
    "non classé",
    "not_applicable",
    "pas clair",
}

ANALYZED_FIELDS = [
    "macro_frame",
    "frame_primary",
    "argument_family_controlled",
    "tone_controlled",
    "rhetorical_register",
]

FIELD_CONFIG = {
    "frame_primary": {
        "free_columns": ["dominant_frame_free", "dominant_frame", "theme_main", "frame_primary"],
        "taxonomy_axis": "frame_aliases",
    },
    "argument_family_controlled": {
        "free_columns": ["argument_type", "argument_family_controlled"],
        "taxonomy_axis": "argument_family",
    },
    "tone_controlled": {
        "free_columns": ["emotion_tone", "tone_controlled"],
        "taxonomy_axis": "tone",
    },
    "rhetorical_register": {
        "free_columns": ["attack_or_objection", "argument_type", "emotion_tone", "rhetorical_register"],
        "taxonomy_axis": "rhetorical_register",
    },
}

REMAP_COLUMNS = [
    "field",
    "raw_label",
    "proposed_canonical_label",
    "parent_category",
    "action",
    "confidence",
    "evidence_count",
    "example_comments",
    "needs_human_validation",
    "rationale",
]


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    without_punct = re.sub(r"[^a-z0-9]+", " ", without_accents)
    return re.sub(r"\s+", " ", without_punct).strip()


def slugify_label(value: Any) -> str:
    normalized = normalize_text(value)
    return normalized.replace(" ", "_")


def is_non_signal(value: Any) -> bool:
    return normalize_text(value) in {normalize_text(label) for label in NON_SIGNAL_LABELS}


def short_text(value: Any, max_chars: int = 240) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def read_csv_optional(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_taxonomy(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frame_to_macro(taxonomy: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for macro, frames in taxonomy.get("macro_frames", {}).items():
        for frame in frames:
            mapping[str(frame)] = str(macro)
    return mapping


def existing_labels_for_field(taxonomy: Dict[str, Any], field: str) -> set[str]:
    if field == "macro_frame":
        return set(taxonomy.get("macro_frames", {}).keys())
    if field == "frame_primary":
        return set(frame_to_macro(taxonomy).keys())
    if field == "argument_family_controlled":
        return set(taxonomy.get("argument_family", {}).get("allowed", []))
    if field == "tone_controlled":
        return set(taxonomy.get("tone", {}).get("allowed", []))
    if field == "rhetorical_register":
        return set(taxonomy.get("rhetorical_register", {}).get("allowed", []))
    return set()


def alias_map_for_field(taxonomy: Dict[str, Any], field: str) -> Dict[str, str]:
    if field == "frame_primary":
        aliases = taxonomy.get("frame_aliases", {})
    elif field == "argument_family_controlled":
        aliases = taxonomy.get("argument_family", {}).get("aliases", {})
    elif field == "tone_controlled":
        aliases = taxonomy.get("tone", {}).get("aliases", {})
    elif field == "rhetorical_register":
        aliases = taxonomy.get("rhetorical_register", {}).get("aliases", {})
    else:
        aliases = {}
    return {normalize_text(alias): str(target) for alias, target in aliases.items()}


def canonical_lookup_for_field(taxonomy: Dict[str, Any], field: str) -> Dict[str, str]:
    lookup = {normalize_text(label): label for label in existing_labels_for_field(taxonomy, field)}
    lookup.update(alias_map_for_field(taxonomy, field))
    return lookup


def parent_for_field_label(taxonomy: Dict[str, Any], field: str, canonical_label: str) -> str:
    if field == "frame_primary":
        return frame_to_macro(taxonomy).get(canonical_label, "other")
    if field == "macro_frame":
        return canonical_label if canonical_label in taxonomy.get("macro_frames", {}) else "other"
    if field == "argument_family_controlled":
        return "argument_family"
    if field == "tone_controlled":
        return "tone"
    if field == "rhetorical_register":
        return "rhetorical_register"
    return ""


def row_examples(group: pd.DataFrame, limit: int = 3) -> List[str]:
    examples = []
    for _, row in group.head(limit).iterrows():
        text = row.get("raw_comment_preview") or row.get("discursive_summary") or row.get("central_argument") or ""
        if text:
            examples.append(short_text(text))
    return examples


def source_distribution(units: pd.DataFrame) -> Dict[str, int]:
    if "actor" not in units.columns:
        return {}
    return units["actor"].fillna("unknown").astype(str).value_counts().to_dict()


def other_rate(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    values = series.fillna("").astype(str).map(normalize_text)
    return round(float(values.isin({normalize_text(x) for x in NON_SIGNAL_LABELS}).mean()), 4)


def top_labels(series: pd.Series, limit: int = 10) -> Dict[str, int]:
    if series is None or series.empty:
        return {}
    return series.fillna("").astype(str).map(lambda value: value.strip() or "other").value_counts().head(limit).to_dict()


def classify_label_heuristic(
    field: str,
    raw_label: str,
    taxonomy: Dict[str, Any],
    evidence_count: int,
) -> Dict[str, Any]:
    norm = normalize_text(raw_label)
    slug = slugify_label(raw_label)
    lookup = canonical_lookup_for_field(taxonomy, field)
    existing = existing_labels_for_field(taxonomy, field)

    if norm in {normalize_text(label) for label in NON_SIGNAL_LABELS}:
        return {
            "field": field,
            "raw_label": raw_label,
            "canonical": "other" if field != "tone_controlled" else "unclear",
            "parent": "other",
            "action": "keep_other",
            "confidence": 0.95,
            "needs_human_validation": False,
            "rationale": "Label non informatif ou explicitement vague ; ne pas remplacer automatiquement.",
        }

    if norm in lookup:
        canonical = lookup[norm]
        return {
            "field": field,
            "raw_label": raw_label,
            "canonical": canonical,
            "parent": parent_for_field_label(taxonomy, field, canonical),
            "action": "alias_to_existing",
            "confidence": 0.92,
            "needs_human_validation": False,
            "rationale": "Correspondance exacte avec un label ou alias existant de la taxonomie.",
        }

    # Cross-axis guardrails from the product spec.
    if field == "frame_primary" and norm in {"journaliste", "journalistes", "macron", "melenchon", "bardella", "lfi", "rn"}:
        return {
            "field": field,
            "raw_label": raw_label,
            "canonical": slug,
            "parent": "target",
            "action": "move_to_other_axis",
            "confidence": 0.84,
            "needs_human_validation": True,
            "rationale": "Le label désigne une cible ou un acteur plutôt qu'un cadrage discursif.",
        }
    if field == "frame_primary" and norm in {"optimiste", "optimisme", "espoir", "enthousiasme", "colere", "indignation"}:
        tone = "hope" if norm in {"optimiste", "optimisme", "espoir", "enthousiasme"} else "anger"
        return {
            "field": field,
            "raw_label": raw_label,
            "canonical": tone,
            "parent": "tone",
            "action": "move_to_other_axis",
            "confidence": 0.86,
            "needs_human_validation": True,
            "rationale": "Le label décrit une tonalité affective, pas une frame.",
        }
    if field == "frame_primary" and norm in {"persuasion", "appel", "appel a convaincre", "convaincre"}:
        return {
            "field": field,
            "raw_label": raw_label,
            "canonical": "call_to_action",
            "parent": "rhetorical_register",
            "action": "move_to_other_axis",
            "confidence": 0.82,
            "needs_human_validation": True,
            "rationale": "Le label décrit une fonction rhétorique d'appel à convaincre.",
        }
    if field == "argument_family_controlled" and norm in {"colere", "indignation", "enthousiasme", "optimisme"}:
        return {
            "field": field,
            "raw_label": raw_label,
            "canonical": "anger" if norm in {"colere", "indignation"} else "hope",
            "parent": "tone",
            "action": "move_to_other_axis",
            "confidence": 0.84,
            "needs_human_validation": True,
            "rationale": "Le label relève d'une tonalité, pas d'une famille argumentative.",
        }
    if field == "argument_family_controlled" and norm in {"objection", "question"}:
        return {
            "field": field,
            "raw_label": raw_label,
            "canonical": "question",
            "parent": "rhetorical_register",
            "action": "move_to_other_axis",
            "confidence": 0.82,
            "needs_human_validation": True,
            "rationale": "Le label relève plutôt du registre rhétorique.",
        }

    # Field-specific keyword mappings.
    if field == "frame_primary":
        keyword_rules = [
            (["media", "medias", "journaliste", "journalistes", "bfm", "interview"], "critique_medias", 0.78),
            (["macron", "macronisme"], "critique_macronisme", 0.76),
            (["pouvoir achat", "smic", "salaire", "salaires", "prix", "inflation", "retraite"], "pouvoir_achat", 0.80),
            (["justice fiscale", "impot", "impots", "taxe"], "justice_fiscale", 0.78),
            (["vote", "voter", "election", "urne"], "appel_vote", 0.76),
            (["mobilisation", "convaincre", "militant", "soutien"], "engagement_militant", 0.76),
            (["victoire", "gagner", "espoir"], "victoire_possible", 0.72),
            (["securite", "ordre"], "securite", 0.72),
            (["immigration", "frontiere", "frontieres"], "priorite_nationale", 0.72),
        ]
        for keywords, canonical, confidence in keyword_rules:
            if any(keyword in norm for keyword in keywords) and canonical in existing:
                return {
                    "field": field,
                    "raw_label": raw_label,
                    "canonical": canonical,
                    "parent": parent_for_field_label(taxonomy, field, canonical),
                    "action": "alias_to_existing",
                    "confidence": confidence,
                    "needs_human_validation": confidence < 0.8,
                    "rationale": "Correspondance heuristique par mots-clés avec une frame existante.",
                }
        if any(keyword in norm for keyword in ["corse", "autonomie", "autodetermination", "referendum"]):
            return {
                "field": field,
                "raw_label": raw_label,
                "canonical": "autonomie_territoriale",
                "parent": "souverainete",
                "action": "new_subframe_candidate",
                "confidence": 0.72,
                "needs_human_validation": True,
                "rationale": "Sous-frame récurrente possible autour de l'autonomie territoriale.",
            }

    if field == "argument_family_controlled":
        rules = [
            (["econom", "prix", "salaire", "smic", "impot", "taxe"], "economique", 0.82),
            (["moral", "justice", "injustice", "dignite"], "moral", 0.78),
            (["media", "journaliste", "bfm"], "mediatique", 0.82),
            (["nation", "france", "identite", "immigration"], "identitaire", 0.76),
            (["institution", "republique", "referendum", "autonomie"], "institutionnel", 0.76),
            (["programme", "mesure", "projet"], "programmatique", 0.76),
            (["mobilisation", "convaincre", "voter", "appel"], "mobilisation", 0.78),
            (["competence", "serieux", "credible"], "competence", 0.76),
        ]
        for keywords, canonical, confidence in rules:
            if any(keyword in norm for keyword in keywords) and canonical in existing:
                return {
                    "field": field,
                    "raw_label": raw_label,
                    "canonical": canonical,
                    "parent": parent_for_field_label(taxonomy, field, canonical),
                    "action": "alias_to_existing",
                    "confidence": confidence,
                    "needs_human_validation": confidence < 0.8,
                    "rationale": "Correspondance heuristique avec une famille argumentative existante.",
                }

    if field == "tone_controlled":
        rules = [
            (["colere", "indignation", "rage", "mecontent"], "anger", 0.86),
            (["espoir", "optimisme", "optimiste", "encouragement"], "hope", 0.86),
            (["enthousiasme", "enthousiaste", "bravo"], "enthusiasm", 0.84),
            (["inquiet", "inquietude", "crainte", "peur"], "concern", 0.84),
            (["ironie", "sarcasme"], "irony", 0.84),
            (["mefiance", "scepticisme", "defiance"], "mistrust", 0.82),
            (["admiration", "respect"], "admiration", 0.82),
            (["neutre", "neutral"], "neutral", 0.90),
        ]
        for keywords, canonical, confidence in rules:
            if any(keyword in norm for keyword in keywords) and canonical in existing:
                return {
                    "field": field,
                    "raw_label": raw_label,
                    "canonical": canonical,
                    "parent": parent_for_field_label(taxonomy, field, canonical),
                    "action": "alias_to_existing",
                    "confidence": confidence,
                    "needs_human_validation": False,
                    "rationale": "Correspondance heuristique avec une tonalité existante.",
                }

    if field == "rhetorical_register":
        rules = [
            (["soutien", "bravo", "merci"], "support", 0.82),
            (["critique", "denonciation", "attaque"], "criticism", 0.82),
            (["voter", "convaincre", "appel", "mobilisation"], "call_to_action", 0.84),
            (["temoignage", "je suis", "j ai"], "testimony", 0.76),
            (["question", "pourquoi", "comment", "objection"], "question", 0.84),
            (["ironie", "sarcasme"], "sarcasm", 0.82),
            (["accusation", "honte", "traitre"], "accusation", 0.78),
            (["programme", "mesure", "projet"], "policy_argument", 0.78),
        ]
        for keywords, canonical, confidence in rules:
            if any(keyword in norm for keyword in keywords) and canonical in existing:
                return {
                    "field": field,
                    "raw_label": raw_label,
                    "canonical": canonical,
                    "parent": parent_for_field_label(taxonomy, field, canonical),
                    "action": "alias_to_existing",
                    "confidence": confidence,
                    "needs_human_validation": confidence < 0.8,
                    "rationale": "Correspondance heuristique avec un registre rhétorique existant.",
                }

    if evidence_count >= 3 and field == "frame_primary":
        return {
            "field": field,
            "raw_label": raw_label,
            "canonical": slug or "other",
            "parent": "other",
            "action": "new_subframe_candidate",
            "confidence": 0.45,
            "needs_human_validation": True,
            "rationale": "Label fréquent mais non couvert ; candidat à examiner humainement.",
        }

    return {
        "field": field,
        "raw_label": raw_label,
        "canonical": "other" if field != "tone_controlled" else "unclear",
        "parent": "other",
        "action": "needs_review" if evidence_count >= 2 else "keep_other",
        "confidence": 0.35,
        "needs_human_validation": evidence_count >= 2,
        "rationale": "Pas de correspondance heuristique fiable.",
    }


def collect_field_candidates(units: pd.DataFrame, field: str, taxonomy: Dict[str, Any], min_count: int) -> List[Dict[str, Any]]:
    config = FIELD_CONFIG.get(field)
    if not config:
        return []
    controlled = units[field] if field in units.columns else pd.Series(["other"] * len(units), index=units.index)
    controlled_norm = controlled.fillna("").astype(str).map(normalize_text)
    needs_mapping = controlled_norm.isin({normalize_text(x) for x in NON_SIGNAL_LABELS})

    rows = []
    for free_col in config["free_columns"]:
        if free_col not in units.columns:
            continue
        work = units.loc[needs_mapping].copy()
        work["_raw_label"] = work[free_col].fillna("").astype(str).map(str.strip)
        work = work[~work["_raw_label"].map(is_non_signal)]
        if work.empty:
            continue
        for raw_label, group in work.groupby("_raw_label"):
            evidence_count = int(len(group))
            if evidence_count < min_count:
                continue
            proposal = classify_label_heuristic(field, raw_label, taxonomy, evidence_count)
            proposal["source_column"] = free_col
            proposal["evidence_count"] = evidence_count
            proposal["examples"] = row_examples(group)
            rows.append(proposal)

    # Deduplicate per field/raw label; keep highest confidence, then highest count.
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["field"], normalize_text(row["raw_label"]))
        current = best.get(key)
        if current is None or (row["confidence"], row["evidence_count"]) > (current["confidence"], current["evidence_count"]):
            best[key] = row
    return sorted(best.values(), key=lambda item: (-item["evidence_count"], item["field"], item["raw_label"]))


def collect_placeholder_keep_other(units: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for field in ANALYZED_FIELDS:
        if field not in units.columns:
            continue
        values = units[field].fillna("").astype(str)
        for raw_label, count in values.value_counts().items():
            if is_non_signal(raw_label):
                rows.append(
                    {
                        "field": field,
                        "raw_label": str(raw_label),
                        "canonical": "other",
                        "parent": "other",
                        "action": "keep_other",
                        "confidence": 0.95,
                        "evidence_count": int(count),
                        "needs_human_validation": False,
                        "rationale": "Placeholder non informatif conservé tel quel.",
                        "examples": [],
                    }
                )
    return rows


def remap_rows_from_proposals(proposals: Sequence[Dict[str, Any]], keep_other_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for proposal in list(proposals) + list(keep_other_rows):
        rows.append(
            {
                "field": proposal.get("field", ""),
                "raw_label": proposal.get("raw_label", ""),
                "proposed_canonical_label": proposal.get("canonical", ""),
                "parent_category": proposal.get("parent", ""),
                "action": proposal.get("action", "needs_review"),
                "confidence": round(float(proposal.get("confidence", 0.0)), 4),
                "evidence_count": int(proposal.get("evidence_count", 0) or 0),
                "example_comments": json.dumps(proposal.get("examples", []), ensure_ascii=False),
                "needs_human_validation": bool(proposal.get("needs_human_validation", True)),
                "rationale": proposal.get("rationale", ""),
            }
        )
    table = pd.DataFrame(rows, columns=REMAP_COLUMNS)
    if table.empty:
        return table
    return table.sort_values(["field", "action", "evidence_count"], ascending=[True, True, False]).reset_index(drop=True)


def apply_remapping(units: pd.DataFrame, remap_table: pd.DataFrame, taxonomy: Dict[str, Any], min_confidence: float = 0.75) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = units.copy()
    stats = {"comments_affected": 0, "field_values_affected": 0}
    applicable = remap_table.copy()
    if applicable.empty:
        for field in ANALYZED_FIELDS:
            if field in out.columns:
                out[f"{field}_raw"] = out[field]
                out[f"{field}_remapped"] = out[field]
        return out, stats

    applicable = applicable[
        applicable["action"].isin(["alias_to_existing", "merge_with_existing"])
        & (~applicable["needs_human_validation"].astype(bool))
        & (pd.to_numeric(applicable["confidence"], errors="coerce").fillna(0.0) >= min_confidence)
    ].copy()

    affected_mask = pd.Series(False, index=out.index)
    for field in ANALYZED_FIELDS:
        if field not in out.columns:
            continue
        out[f"{field}_raw"] = out[field]
        out[f"{field}_remapped"] = out[field]
        field_map = {
            normalize_text(row.raw_label): str(row.proposed_canonical_label)
            for row in applicable[applicable["field"] == field].itertuples(index=False)
            if not is_non_signal(row.raw_label)
        }
        if not field_map:
            continue
        free_cols = FIELD_CONFIG.get(field, {}).get("free_columns", [field])
        for idx, row in out.iterrows():
            current_norm = normalize_text(row.get(field, ""))
            if current_norm not in {normalize_text(x) for x in NON_SIGNAL_LABELS}:
                continue
            replacement = None
            for free_col in free_cols:
                raw_norm = normalize_text(row.get(free_col, ""))
                if raw_norm in field_map:
                    replacement = field_map[raw_norm]
                    break
            if replacement:
                out.at[idx, f"{field}_remapped"] = replacement
                affected_mask.at[idx] = True
                stats["field_values_affected"] += 1

    if "frame_primary_remapped" in out.columns:
        macro_lookup = frame_to_macro(taxonomy)
        if "macro_frame" in out.columns:
            out["macro_frame_raw"] = out.get("macro_frame_raw", out["macro_frame"])
            out["macro_frame_remapped"] = out["frame_primary_remapped"].map(macro_lookup).fillna(out["macro_frame"])
    stats["comments_affected"] = int(affected_mask.sum())
    return out, stats


def remapped_series(units: pd.DataFrame, field: str) -> pd.Series:
    remapped = f"{field}_remapped"
    if remapped in units.columns:
        return units[remapped]
    if field in units.columns:
        return units[field]
    return pd.Series(dtype=str)


def metrics_before_after(units: pd.DataFrame, remapped_units: pd.DataFrame) -> Dict[str, Any]:
    metrics = {"other_rates": {"before": {}, "after": {}}, "unique_labels": {"before": {}, "after": {}}, "top_labels": {"before": {}, "after": {}}}
    for field in ANALYZED_FIELDS:
        before = units[field] if field in units.columns else pd.Series(dtype=str)
        after = remapped_series(remapped_units, field)
        metrics["other_rates"]["before"][field] = other_rate(before)
        metrics["other_rates"]["after"][field] = other_rate(after)
        metrics["unique_labels"]["before"][field] = int(before.fillna("").astype(str).nunique()) if not before.empty else 0
        metrics["unique_labels"]["after"][field] = int(after.fillna("").astype(str).nunique()) if not after.empty else 0
        metrics["top_labels"]["before"][field] = top_labels(before)
        metrics["top_labels"]["after"][field] = top_labels(after)
    return metrics


def build_candidates_json(
    run_id: str,
    taxonomy_path: Path,
    source_files: Sequence[str],
    units: pd.DataFrame,
    metrics: Dict[str, Any],
    proposals: Sequence[Dict[str, Any]],
    keep_other_rows: Sequence[Dict[str, Any]],
    remap_table: pd.DataFrame,
) -> Dict[str, Any]:
    aliases = [
        {
            "field": p["field"],
            "raw_label": p["raw_label"],
            "canonical_label": p["canonical"],
            "parent_category": p["parent"],
            "evidence_count": int(p["evidence_count"]),
            "confidence": round(float(p["confidence"]), 4),
            "rationale": p["rationale"],
            "examples": p.get("examples", []),
        }
        for p in proposals
        if p.get("action") == "alias_to_existing"
    ]
    new_subframes = [
        {
            "macro_frame": p.get("parent", "other"),
            "new_subframe": p.get("canonical", ""),
            "definition": "Candidat induit automatiquement à partir des labels libres observés.",
            "include_when": p.get("examples", []),
            "exclude_when": ["Ne pas utiliser si le label relève plutôt d'une tonalité, cible, stance ou registre."],
            "evidence_count": int(p.get("evidence_count", 0)),
            "examples": p.get("examples", []),
            "needs_human_validation": True,
        }
        for p in proposals
        if p.get("action") == "new_subframe_candidate"
    ]
    moves = [
        {
            "raw_label": p.get("raw_label", ""),
            "from_field": p.get("field", ""),
            "to_field": p.get("parent", ""),
            "canonical_label": p.get("canonical", ""),
            "rationale": p.get("rationale", ""),
        }
        for p in proposals
        if p.get("action") == "move_to_other_axis"
    ]
    keep_other = [
        {
            "field": p.get("field", ""),
            "raw_label": p.get("raw_label", ""),
            "rationale": p.get("rationale", ""),
        }
        for p in list(keep_other_rows) + [p for p in proposals if p.get("action") == "keep_other"]
    ]
    warnings = []
    if remap_table["needs_human_validation"].astype(bool).any() if not remap_table.empty else False:
        warnings.append("Certaines propositions nécessitent une validation humaine avant application.")
    if any(p.get("action") == "move_to_other_axis" for p in proposals):
        warnings.append("Des labels semblent appartenir à un autre axe conceptuel que leur champ actuel.")

    return {
        "run_id": run_id,
        "source_files": list(source_files),
        "taxonomy_base_file": str(taxonomy_path),
        "dataset_summary": {
            "n_comments": int(len(units)),
            "source_distribution": source_distribution(units),
            "fields_analyzed": ANALYZED_FIELDS,
        },
        "other_rates": {
            "before": metrics["other_rates"]["before"],
            "after_candidate_remap_estimate": metrics["other_rates"]["after"],
        },
        "candidate_aliases": aliases,
        "candidate_new_subframes": new_subframes,
        "candidate_merges": [],
        "candidate_moves": moves,
        "keep_other": keep_other,
        "warnings": warnings,
    }


def build_candidate_taxonomy(
    taxonomy: Dict[str, Any],
    taxonomy_path: Path,
    proposals: Sequence[Dict[str, Any]],
    run_id: str,
) -> Dict[str, Any]:
    candidate = {
        "run_id": run_id,
        "status": "candidate_requires_human_validation",
        "base_taxonomy_file": str(taxonomy_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "existing_taxonomy": copy.deepcopy(taxonomy),
        "candidate_additions": {
            "aliases": {
                "frame_aliases": {},
                "argument_family": {"aliases": {}},
                "tone": {"aliases": {}},
                "rhetorical_register": {"aliases": {}},
            },
            "new_subframes": [],
            "moves_to_review": [],
            "keep_other": [],
        },
    }
    for proposal in proposals:
        action = proposal.get("action")
        field = proposal.get("field")
        raw = normalize_text(proposal.get("raw_label", ""))
        canonical = proposal.get("canonical", "")
        if action == "alias_to_existing":
            if field == "frame_primary":
                candidate["candidate_additions"]["aliases"]["frame_aliases"][raw] = canonical
            elif field == "argument_family_controlled":
                candidate["candidate_additions"]["aliases"]["argument_family"]["aliases"][raw] = canonical
            elif field == "tone_controlled":
                candidate["candidate_additions"]["aliases"]["tone"]["aliases"][raw] = canonical
            elif field == "rhetorical_register":
                candidate["candidate_additions"]["aliases"]["rhetorical_register"]["aliases"][raw] = canonical
        elif action == "new_subframe_candidate":
            candidate["candidate_additions"]["new_subframes"].append(
                {
                    "macro_frame": proposal.get("parent", "other"),
                    "subframe": canonical,
                    "raw_label": proposal.get("raw_label", ""),
                    "evidence_count": proposal.get("evidence_count", 0),
                    "examples": proposal.get("examples", []),
                    "needs_human_validation": True,
                }
            )
        elif action == "move_to_other_axis":
            candidate["candidate_additions"]["moves_to_review"].append(
                {
                    "raw_label": proposal.get("raw_label", ""),
                    "from_field": field,
                    "to_axis": proposal.get("parent", ""),
                    "canonical_label": canonical,
                    "rationale": proposal.get("rationale", ""),
                }
            )
        elif action == "keep_other":
            candidate["candidate_additions"]["keep_other"].append(
                {"field": field, "raw_label": proposal.get("raw_label", ""), "rationale": proposal.get("rationale", "")}
            )
    return candidate


def write_report(
    path: Path,
    metrics: Dict[str, Any],
    remap_table: pd.DataFrame,
    remap_stats: Dict[str, int],
    source_files: Sequence[str],
) -> None:
    validation_count = int(remap_table["needs_human_validation"].astype(bool).sum()) if not remap_table.empty else 0
    proposed_aliases = int((remap_table["action"] == "alias_to_existing").sum()) if not remap_table.empty else 0
    new_subframes = int((remap_table["action"] == "new_subframe_candidate").sum()) if not remap_table.empty else 0
    moves = int((remap_table["action"] == "move_to_other_axis").sum()) if not remap_table.empty else 0
    total_rows = len(remap_table)
    validation_share = round(validation_count / total_rows, 4) if total_rows else 0.0

    lines = [
        "# Taxonomy induction and remapping report",
        "",
        "Ce rapport compare la taxonomie contrôlée avant/après remapping candidat. Aucun commentaire n'a été réencodé.",
        "",
        "## Fichiers sources",
        "",
    ]
    lines.extend([f"- `{source}`" for source in source_files])
    lines += [
        "",
        "## Métriques globales",
        "",
        f"- Commentaires affectés par un remapping appliqué : `{remap_stats.get('comments_affected', 0)}`",
        f"- Valeurs de champs affectées : `{remap_stats.get('field_values_affected', 0)}`",
        f"- Alias proposés : `{proposed_aliases}`",
        f"- Nouveaux sous-frames candidats : `{new_subframes}`",
        f"- Déplacements d'axe à examiner : `{moves}`",
        f"- Cas nécessitant validation humaine : `{validation_count}` (`{validation_share:.1%}`)",
        "",
        "## Taux de `other` avant/après",
        "",
        "| Champ | Before | After candidate remap | Labels distincts before | Labels distincts after |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for field in ANALYZED_FIELDS:
        before = metrics["other_rates"]["before"].get(field, 0.0)
        after = metrics["other_rates"]["after"].get(field, 0.0)
        ub = metrics["unique_labels"]["before"].get(field, 0)
        ua = metrics["unique_labels"]["after"].get(field, 0)
        lines.append(f"| `{field}` | {before:.1%} | {after:.1%} | {ub} | {ua} |")

    lines += ["", "## Top labels avant/après", ""]
    for field in ANALYZED_FIELDS:
        lines += [
            f"### {field}",
            "",
            f"- Avant : `{json.dumps(metrics['top_labels']['before'].get(field, {}), ensure_ascii=False)}`",
            f"- Après : `{json.dumps(metrics['top_labels']['after'].get(field, {}), ensure_ascii=False)}`",
            "",
        ]

    lines += [
        "## Lecture méthodologique",
        "",
        "- Une baisse de `other` indique seulement que des labels libres ont été remappés vers des catégories contrôlées.",
        "- Les lignes `needs_human_validation=true` ne doivent pas être appliquées automatiquement.",
        "- Les propositions `move_to_other_axis` signalent une confusion conceptuelle entre frame, tone, register, argument, stance ou target.",
        "- Le fichier de taxonomie source n'est jamais modifié ; `discourse_taxonomy.candidate.json` reste une proposition versionnée.",
        "",
        "## Impact potentiel sur le graphe",
        "",
        "Le remapping peut réduire les hubs `other` et rendre les communautés discursives plus lisibles. Pour vérifier l'effet, relancer ensuite le post-processing graphe sur `discursive_units_remapped.csv` ou intégrer les mappings validés dans une future version contrôlée de la taxonomie.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def induce_taxonomy_with_llm(
    taxonomy: Dict[str, Any],
    observed_labels: Dict[str, Dict[str, int]],
    examples: Dict[str, Dict[str, List[str]]],
    client: Any,
) -> Dict[str, Any]:
    """Optional LLM helper. Returns strict JSON; caller remains responsible for validation."""
    system = (
        "Tu aides à proposer une induction contrôlée de taxonomie discursive. "
        "Tu ne remplaces pas la taxonomie stable : tu proposes des alias, fusions, déplacements d'axe "
        "et candidats à validation humaine. Réponds uniquement en JSON conforme au schéma demandé."
    )
    user = json.dumps(
        {
            "taxonomy": taxonomy,
            "observed_labels": observed_labels,
            "examples": examples,
            "schema": {
                "candidate_aliases": [],
                "candidate_new_subframes": [],
                "candidate_merges": [],
                "candidate_moves": [],
                "keep_other": [],
                "warnings": [],
            },
        },
        ensure_ascii=False,
    )
    return client.complete_json(system, user)


def run_induction(args: argparse.Namespace) -> Dict[str, Path]:
    taxonomy_path = args.taxonomy
    units_path = args.units
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(taxonomy_path)
    units = read_csv_optional(units_path)
    if units.empty:
        raise ValueError(f"Missing or empty units file: {units_path}")

    nodes = read_csv_optional(args.nodes)
    community_summary = read_csv_optional(args.community_summary)
    attribute_contributions = read_csv_optional(args.attribute_contributions)
    _ = (nodes, community_summary, attribute_contributions)  # read for availability/provenance; heuristics use units.

    source_files = [str(units_path)]
    for optional in [args.nodes, args.community_summary, args.attribute_contributions]:
        if optional is not None and optional.exists():
            source_files.append(str(optional))

    proposals: List[Dict[str, Any]] = []
    for field in FIELD_CONFIG:
        proposals.extend(collect_field_candidates(units, field, taxonomy, min_count=args.min_count))
    keep_other_rows = collect_placeholder_keep_other(units)
    remap_table = remap_rows_from_proposals(proposals, keep_other_rows)
    remapped_units, remap_stats = apply_remapping(
        units,
        remap_table,
        taxonomy,
        min_confidence=args.apply_min_confidence,
    )
    metrics = metrics_before_after(units, remapped_units)

    run_id = datetime.now(timezone.utc).strftime("taxonomy-induction-%Y%m%dT%H%M%SZ")

    llm_payload = None
    if args.mode == "llm":
        from observatoire.llm import make_llm_client

        client = make_llm_client(args.llm_provider, args.llm_model, api_key=args.llm_api_key, base_url=args.llm_base_url)
        if client is None:
            raise RuntimeError("LLM mode requires an active LLM provider.")
        observed_labels = {
            field: top_labels(units[field] if field in units.columns else pd.Series(dtype=str), limit=50)
            for field in ANALYZED_FIELDS
        }
        example_map = defaultdict(dict)
        for proposal in proposals:
            example_map[proposal["field"]][proposal["raw_label"]] = proposal.get("examples", [])
        llm_payload = induce_taxonomy_with_llm(taxonomy, observed_labels, example_map, client)

    candidates = build_candidates_json(
        run_id=run_id,
        taxonomy_path=taxonomy_path,
        source_files=source_files,
        units=units,
        metrics=metrics,
        proposals=proposals,
        keep_other_rows=keep_other_rows,
        remap_table=remap_table,
    )
    if llm_payload is not None:
        candidates["llm_suggestions"] = llm_payload
        candidates["warnings"].append("Des suggestions LLM sont incluses séparément et doivent être validées avant usage.")

    candidate_taxonomy = build_candidate_taxonomy(taxonomy, taxonomy_path, proposals, run_id)

    candidates_path = output_dir / "taxonomy_induction_candidates.json"
    remap_path = output_dir / "taxonomy_remap_table.csv"
    candidate_taxonomy_path = output_dir / "discourse_taxonomy.candidate.json"
    remapped_units_path = output_dir / "discursive_units_remapped.csv"
    report_path = output_dir / "taxonomy_remap_report.md"

    candidates_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    remap_table.to_csv(remap_path, index=False)
    candidate_taxonomy_path.write_text(json.dumps(candidate_taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
    remapped_units.to_csv(remapped_units_path, index=False)
    write_report(report_path, metrics, remap_table, remap_stats, source_files)

    return {
        "candidates": candidates_path,
        "remap_table": remap_path,
        "candidate_taxonomy": candidate_taxonomy_path,
        "remapped_units": remapped_units_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Induce controlled taxonomy candidates from existing discursive outputs.")
    parser.add_argument("--taxonomy", type=Path, default=Path("config/discourse_taxonomy.example.json"))
    parser.add_argument("--units", type=Path, default=Path("outputs/discursive_units.csv"))
    parser.add_argument("--nodes", type=Path, default=None)
    parser.add_argument("--community-summary", type=Path, default=None)
    parser.add_argument("--attribute-contributions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/taxonomy_induction"))
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--min-count", type=int, default=1, help="Minimum observed occurrences for a raw label proposal.")
    parser.add_argument("--apply-min-confidence", type=float, default=0.75, help="Minimum confidence for automatic remapped columns.")
    return parser.parse_args()


def main() -> None:
    paths = run_induction(parse_args())
    print("Taxonomy induction complete.")
    for label, path in paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
