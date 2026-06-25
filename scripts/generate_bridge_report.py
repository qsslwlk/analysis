#!/usr/bin/env python3
"""
Generate a readable RN/LFI bridge report from already-computed bridge outputs.

This script does not call an LLM and does not recompute embeddings. It turns
`rn_lfi_attribute_bridges.csv` into an auditable CSV + Markdown report with
scores, representative quotes and a first-pass interpretation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


CONTENT_NODE_TYPES = {
    "CanonicalClaimNode",
    "FrameNode",
    "TargetNode",
    "StanceTargetNode",
    "ArgumentFamilyNode",
}

FORM_NODE_TYPES = {
    "ToneNode",
    "RegisterNode",
    "RhetoricalFunctionNode",
    "ThemeNode",
}

TYPE_WEIGHTS = {
    "StanceTargetNode": 1.40,
    "CanonicalClaimNode": 1.30,
    "FrameNode": 1.20,
    "TargetNode": 0.75,
    "ArgumentFamilyNode": 0.55,
    "ThemeNode": 0.35,
    "RegisterNode": 0.30,
    "RhetoricalFunctionNode": 0.30,
    "ToneNode": 0.25,
}

DEFAULT_AUDIT_LABELS = [
    "vrai_pont_discursif",
    "pont_conflictuel",
    "pont_rhetorique",
    "pont_artefactuel",
    "pont_source",
    "a_verifier",
]

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


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def short_text(value: object, max_chars: int = 260) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def md_escape(value: object) -> str:
    text = short_text(value, max_chars=500)
    return text.replace("|", "\\|")


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_json_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [x.strip() for x in text.split("|") if x.strip()]


def infer_sources(bridges: pd.DataFrame, diagnostics: dict, source_a: Optional[str], source_b: Optional[str]) -> Tuple[str, str]:
    if source_a and source_b:
        return source_a, source_b
    diag_a = diagnostics.get("source_a")
    diag_b = diagnostics.get("source_b")
    if diag_a and diag_b:
        return str(diag_a), str(diag_b)

    suffixes = []
    for col in bridges.columns:
        match = re.match(r"^n_(.+)_comments$", col)
        if match:
            suffixes.append(match.group(1))
    if len(suffixes) >= 2:
        return suffixes[0], suffixes[1]
    return source_a or "RN", source_b or "LFI"


def bridge_category(attribute_type: str) -> str:
    if attribute_type in FORM_NODE_TYPES:
        return "form"
    if attribute_type in CONTENT_NODE_TYPES:
        return "content"
    return "mixed"


def bridge_type_label(attribute_type: str) -> str:
    mapping = {
        "CanonicalClaimNode": "claim",
        "FrameNode": "frame",
        "TargetNode": "target",
        "StanceTargetNode": "stance_target",
        "ArgumentFamilyNode": "argument_family",
        "ToneNode": "tone",
        "RegisterNode": "register",
        "RhetoricalFunctionNode": "rhetorical_function",
        "ThemeNode": "theme",
    }
    return mapping.get(attribute_type, attribute_type.replace("Node", "").lower() or "attribute")


def compute_balance(n_a: int, n_b: int) -> float:
    denom = n_a + n_b
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - abs(n_a - n_b) / denom))


def compute_specificity(total_units: int, df_comments: float) -> float:
    """Positive IDF-style specificity: high when an attribute is not corpus-wide."""
    if total_units <= 0:
        return 0.0
    df = max(float(df_comments or 0.0), 1.0)
    return max(0.0, math.log((total_units + 1.0) / (df + 1.0)))


def compute_report_score(n_a: int, n_b: int, balance: float, specificity: float, attribute_type: str) -> float:
    type_weight = TYPE_WEIGHTS.get(attribute_type, 1.0)
    return math.log1p(max(n_a, 0)) * math.log1p(max(n_b, 0)) * balance * specificity * type_weight


def build_unit_lookup(units: pd.DataFrame) -> Dict[str, dict]:
    if "comment_id" not in units.columns:
        return {}
    work = units.copy()
    work["comment_id"] = work["comment_id"].astype(str)
    return work.drop_duplicates("comment_id").set_index("comment_id", drop=False).to_dict("index")


def quote_for_comment(comment_id: str, unit_lookup: Dict[str, dict]) -> dict:
    row = unit_lookup.get(str(comment_id), {})
    raw = row.get("raw_comment_preview", row.get("raw_text", ""))
    summary = row.get("discursive_summary", row.get("summary", ""))
    return {
        "comment_id": str(comment_id),
        "summary": short_text(summary, 280),
        "quote": short_text(raw, 360),
        "video_title": short_text(row.get("video_title", ""), 160),
        "time_bucket": short_text(row.get("time_bucket", ""), 60),
        "frame": short_text(row.get("frame_primary", ""), 80),
        "argument_family": short_text(row.get("argument_family_controlled", ""), 80),
        "tone": short_text(row.get("tone_controlled", ""), 80),
    }


def quotes_for_ids(comment_ids: Sequence[str], unit_lookup: Dict[str, dict], fallback_summaries: Sequence[str], limit: int) -> List[dict]:
    rows = []
    for cid in list(comment_ids)[:limit]:
        item = quote_for_comment(cid, unit_lookup)
        if item["summary"] or item["quote"]:
            rows.append(item)
    if rows:
        return rows
    for idx, summary in enumerate(list(fallback_summaries)[:limit]):
        rows.append({
            "comment_id": "",
            "summary": short_text(summary, 280),
            "quote": "",
            "video_title": "",
            "time_bucket": "",
            "frame": "",
            "argument_family": "",
            "tone": "",
        })
    return rows


def compact_quote_json(quotes: List[dict]) -> str:
    return json.dumps(quotes, ensure_ascii=False)


def compact_quote_text(quotes: List[dict]) -> str:
    bits = []
    for q in quotes:
        quote = q.get("quote") or q.get("summary")
        if quote:
            bits.append(short_text(quote, 180))
    return " || ".join(bits)


def top_values(quotes: List[dict], key: str) -> List[str]:
    values = []
    for quote in quotes:
        value = normalize_text(quote.get(key, ""))
        if value and value not in NON_SIGNAL_LABELS and value not in values:
            values.append(value)
    return values[:4]


def first_pass_interpretation(
    label: str,
    attribute_type: str,
    category: str,
    source_a: str,
    source_b: str,
    quotes_a: List[dict],
    quotes_b: List[dict],
) -> str:
    type_label = bridge_type_label(attribute_type)
    if category == "form":
        return (
            f"Pont de forme autour de « {label} » ({type_label}) : les deux sources partagent un style, "
            "une tonalité ou un registre. À ne pas interpréter seul comme une convergence discursive."
        )

    frames_a = top_values(quotes_a, "frame")
    frames_b = top_values(quotes_b, "frame")
    args_a = top_values(quotes_a, "argument_family")
    args_b = top_values(quotes_b, "argument_family")
    shared_frames = sorted(set(frames_a) & set(frames_b))
    shared_args = sorted(set(args_a) & set(args_b))

    if attribute_type == "StanceTargetNode":
        return (
            f"Pont conflictuel possible : {source_a} et {source_b} se rejoignent sur une même cible/stance "
            f"autour de « {label} ». Vérifier les citations pour voir si le sens politique est comparable."
        )
    if shared_frames or shared_args:
        shared = ", ".join(shared_frames + shared_args)
        return (
            f"Pont discursif plausible autour de « {label} » ({type_label}) : les citations des deux sources "
            f"partagent aussi {shared}. Vérification humaine recommandée."
        )
    return (
        f"Pont de contenu à vérifier autour de « {label} » ({type_label}) : présent chez {source_a} et {source_b}, "
        "mais les citations peuvent relever de recadrages différents."
    )


def suggested_audit_label(attribute_type: str, label: str, category: str) -> str:
    norm = normalize_text(label)
    if norm in NON_SIGNAL_LABELS:
        return "pont_artefactuel"
    if category == "form":
        return "pont_rhetorique"
    if attribute_type == "StanceTargetNode":
        return "pont_conflictuel"
    return "a_verifier"


def make_report_rows(
    bridges: pd.DataFrame,
    units: pd.DataFrame,
    source_a: str,
    source_b: str,
    max_quotes: int,
) -> pd.DataFrame:
    total_units = len(units)
    unit_lookup = build_unit_lookup(units)
    rows = []

    n_a_col = f"n_{source_a}_comments"
    n_b_col = f"n_{source_b}_comments"
    ids_a_col = f"top_{source_a}_comment_ids"
    ids_b_col = f"top_{source_b}_comment_ids"
    summaries_a_col = f"top_{source_a}_summaries"
    summaries_b_col = f"top_{source_b}_summaries"

    for _, row in bridges.iterrows():
        attribute_type = str(row.get("attribute_type", ""))
        label = str(row.get("attribute_label", row.get("attribute_normalized_label", "")))
        n_a = int(row.get(n_a_col, 0) or 0)
        n_b = int(row.get(n_b_col, 0) or 0)
        df_comments = float(row.get("df_comments", n_a + n_b) or n_a + n_b)

        balance_score = compute_balance(n_a, n_b)
        specificity_score = compute_specificity(total_units, df_comments)
        report_score = compute_report_score(n_a, n_b, balance_score, specificity_score, attribute_type)
        category = bridge_category(attribute_type)

        ids_a = parse_json_list(row.get(ids_a_col, "[]"))
        ids_b = parse_json_list(row.get(ids_b_col, "[]"))
        fallback_a = parse_json_list(row.get(summaries_a_col, "[]"))
        fallback_b = parse_json_list(row.get(summaries_b_col, "[]"))
        quotes_a = quotes_for_ids(ids_a, unit_lookup, fallback_a, max_quotes)
        quotes_b = quotes_for_ids(ids_b, unit_lookup, fallback_b, max_quotes)

        interpretation = first_pass_interpretation(
            label=label,
            attribute_type=attribute_type,
            category=category,
            source_a=source_a,
            source_b=source_b,
            quotes_a=quotes_a,
            quotes_b=quotes_b,
        )
        audit_hint = suggested_audit_label(attribute_type, label, category)

        rows.append({
            "bridge_label": label,
            "bridge_type": bridge_type_label(attribute_type),
            "attribute_type": attribute_type,
            "bridge_category": category,
            f"n_{source_a}": n_a,
            f"n_{source_b}": n_b,
            "df_comments": df_comments,
            "balance_score": round(balance_score, 6),
            "specificity_score": round(specificity_score, 6),
            "type_weight": TYPE_WEIGHTS.get(attribute_type, 1.0),
            "report_score": round(report_score, 6),
            "source_bridge_score": row.get("bridge_score", ""),
            f"top_{source_a}_quotes": compact_quote_json(quotes_a),
            f"top_{source_b}_quotes": compact_quote_json(quotes_b),
            f"top_{source_a}_quotes_short": compact_quote_text(quotes_a),
            f"top_{source_b}_quotes_short": compact_quote_text(quotes_b),
            "interpretation": interpretation,
            "audit_label_suggestion": audit_hint,
            "audit_label": "",
            "audit_notes": "",
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["report_score", "balance_score"], ascending=False).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, columns: Sequence[str], limit: int) -> str:
    if df.empty:
        return "_Aucun pont à afficher._\n"
    work = df.head(limit).copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in work.iterrows():
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            vals.append(md_escape(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def quote_bullets(quotes: List[dict], source: str) -> List[str]:
    lines = []
    for quote in quotes:
        text = quote.get("quote") or quote.get("summary")
        meta_bits = [quote.get("time_bucket", ""), quote.get("video_title", "")]
        meta = " — ".join([short_text(x, 90) for x in meta_bits if x])
        if text:
            suffix = f" ({meta})" if meta else ""
            lines.append(f"- **{source}** : “{md_escape(text)}”{suffix}")
    return lines


def parse_quote_json(value: object) -> List[dict]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    except Exception:
        return []
    return []


def write_markdown_report(
    report: pd.DataFrame,
    output_path: Path,
    source_a: str,
    source_b: str,
    input_dir: Path,
    bridge_dir: Path,
    top_n: int,
) -> None:
    content = report[report["bridge_category"] == "content"].copy()
    form = report[report["bridge_category"] == "form"].copy()
    mixed = report[report["bridge_category"] == "mixed"].copy()

    lines = [
        "# Rapport de ponts discursifs",
        "",
        "Ce rapport transforme les ponts de graphe en objets lisibles et auditables. Il ne rappelle pas de LLM et ne recalcule pas les embeddings.",
        "",
        "## Paramètres",
        "",
        f"- Sources comparées : `{source_a}` / `{source_b}`",
        f"- Dossier discursif : `{input_dir}`",
        f"- Dossier de ponts : `{bridge_dir}`",
        f"- Nombre de ponts attributaires : `{len(report)}`",
        "",
        "## Lecture rapide",
        "",
        "- `report_score` est le score de lecture du rapport : présence bilatérale, équilibre, spécificité et poids du type d'attribut.",
        "- `balance_score` vaut 1 quand le pont est parfaitement équilibré entre les deux sources.",
        "- `specificity_score` est un IDF positif : plus il est élevé, moins l'attribut est générique dans le corpus.",
        "- `audit_label` est volontairement vide dans le CSV : il sert à l'audit humain.",
        "",
        "## Top ponts de contenu",
        "",
        markdown_table(
            content,
            [
                "bridge_label",
                "bridge_type",
                f"n_{source_a}",
                f"n_{source_b}",
                "balance_score",
                "specificity_score",
                "report_score",
                "audit_label_suggestion",
            ],
            top_n,
        ),
        "",
        "## Top ponts de forme",
        "",
        markdown_table(
            form,
            [
                "bridge_label",
                "bridge_type",
                f"n_{source_a}",
                f"n_{source_b}",
                "balance_score",
                "specificity_score",
                "report_score",
                "audit_label_suggestion",
            ],
            top_n,
        ),
    ]

    if not mixed.empty:
        lines += [
            "",
            "## Ponts mixtes ou non classés",
            "",
            markdown_table(
                mixed,
                [
                    "bridge_label",
                    "bridge_type",
                    f"n_{source_a}",
                    f"n_{source_b}",
                    "balance_score",
                    "specificity_score",
                    "report_score",
                    "audit_label_suggestion",
                ],
                top_n,
            ),
        ]

    lines += [
        "",
        "## Fiches détaillées",
        "",
    ]

    for index, row in report.head(top_n).iterrows():
        rank = index + 1
        label = row.get("bridge_label", "")
        bridge_type = row.get("bridge_type", "")
        quotes_a = parse_quote_json(row.get(f"top_{source_a}_quotes"))
        quotes_b = parse_quote_json(row.get(f"top_{source_b}_quotes"))
        lines += [
            f"### {rank}. {label} ({bridge_type})",
            "",
            f"- Catégorie : `{row.get('bridge_category', '')}`",
            f"- Volumes : `{source_a}={row.get(f'n_{source_a}', 0)}` / `{source_b}={row.get(f'n_{source_b}', 0)}` / `df={row.get('df_comments', 0)}`",
            f"- Scores : `balance={row.get('balance_score', 0)}` · `specificity={row.get('specificity_score', 0)}` · `report={row.get('report_score', 0)}`",
            f"- Suggestion d'audit : `{row.get('audit_label_suggestion', 'a_verifier')}`",
            f"- Interprétation automatique : {row.get('interpretation', '')}",
            "",
            "**Citations représentatives**",
            "",
        ]
        bullets = quote_bullets(quotes_a, source_a) + quote_bullets(quotes_b, source_b)
        lines.extend(bullets or ["- _Aucune citation disponible._"])
        lines.append("")

    lines += [
        "## Grille d'audit recommandée",
        "",
        "| Label | Usage |",
        "| --- | --- |",
        "| `vrai_pont_discursif` | Même problème, claim ou frame partagé, avec citations comparables. |",
        "| `pont_conflictuel` | Même cible ou thème, mais sens politique opposé ou conflictuel. |",
        "| `pont_rhetorique` | Même ton ou registre, sans convergence de contenu. |",
        "| `pont_artefactuel` | Catégorie trop vague, erreur LLM ou pont non soutenu par les citations. |",
        "| `pont_source` | Effet de vidéo, média, période, personnalité ou contexte de collecte. |",
        "| `a_verifier` | Signal intéressant mais insuffisant sans lecture humaine. |",
        "",
        "## Critère minimal proposé",
        "",
        "Un pont discursif RN/LFI ne devrait être retenu que si :",
        "",
        f"- `n_{source_a} >= k` et `n_{source_b} >= k` ;",
        "- `balance_score` dépasse un seuil explicite ;",
        "- `specificity_score` montre que l'attribut n'est pas un hub trop générique ;",
        "- les citations des deux sources portent bien sur une structure comparable ;",
        "- l'audit humain ne classe pas le pont comme rhétorique, artefactuel ou source.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a readable bridge report from RN/LFI bridge outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs"), help="Directory containing discursive_units.csv.")
    parser.add_argument("--bridge-dir", type=Path, default=Path("outputs/rn_lfi_bridges"), help="Directory containing rn_lfi_attribute_bridges.csv.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to --bridge-dir.")
    parser.add_argument("--source-a", default=None, help="First source. Inferred from diagnostics or bridge columns when omitted.")
    parser.add_argument("--source-b", default=None, help="Second source. Inferred from diagnostics or bridge columns when omitted.")
    parser.add_argument("--max-quotes", type=int, default=3, help="Representative quotes per source and bridge.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of bridge cards in the Markdown report.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.bridge_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bridges = read_csv_required(args.bridge_dir / "rn_lfi_attribute_bridges.csv")
    units = read_csv_required(args.input_dir / "discursive_units.csv")
    diagnostics = read_json_if_exists(args.bridge_dir / "rn_lfi_bridge_diagnostics.json")
    source_a, source_b = infer_sources(bridges, diagnostics, args.source_a, args.source_b)

    report = make_report_rows(
        bridges=bridges,
        units=units,
        source_a=source_a,
        source_b=source_b,
        max_quotes=args.max_quotes,
    )

    csv_path = output_dir / "bridge_report.csv"
    md_path = output_dir / "bridge_report.md"
    report.to_csv(csv_path, index=False)
    write_markdown_report(
        report=report,
        output_path=md_path,
        source_a=source_a,
        source_b=source_b,
        input_dir=args.input_dir,
        bridge_dir=args.bridge_dir,
        top_n=args.top_n,
    )

    print(f"Wrote {len(report)} bridge rows")
    print(f"- {csv_path}")
    print(f"- {md_path}")
    if not report.empty:
        cols = [
            "bridge_label",
            "bridge_type",
            f"n_{source_a}",
            f"n_{source_b}",
            "balance_score",
            "specificity_score",
            "report_score",
            "audit_label_suggestion",
        ]
        print("\nTop bridges:")
        print(report[cols].head(min(args.top_n, 12)).to_string(index=False))


if __name__ == "__main__":
    main()
