"""Human-readable labels for claim clusters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from observatoire.llm import LLMClient


DEFAULT_PROMPT_PATH = Path("prompts/label_claim_cluster.md")


def load_prompt(path: Path | str = DEFAULT_PROMPT_PATH) -> str:
    return Path(path).read_text(encoding="utf-8")


def _fallback_cluster_block(cluster: pd.Series) -> str:
    examples = json.loads(cluster.get("examples_json") or "[]")
    lines = [
        f"## Cluster {cluster.get('claim_cluster')}: {cluster.get('label_auto')}",
        "",
        f"- Taille : {cluster.get('size')} claims",
        f"- Confiance moyenne : {float(cluster.get('mean_confidence', 0)):.2f}",
        "- Exemples représentatifs :",
    ]
    lines.extend(f"  - {example}" for example in examples[:6])
    return "\n".join(lines)


def label_claim_clusters(
    claim_clusters_df: pd.DataFrame,
    client: Optional[LLMClient],
    prompt_path: Path | str = DEFAULT_PROMPT_PATH,
) -> List[Dict[str, Any]]:
    if claim_clusters_df.empty:
        return []
    if client is None:
        return [
            {
                "claim_cluster": int(row.get("claim_cluster")),
                "cluster_label": row.get("label_auto"),
                "summary": "Label automatique TF-IDF, à valider humainement.",
                "internal_variations": [],
                "representative_claims": json.loads(row.get("examples_json") or "[]")[:5],
                "label_source": "fallback_tfidf",
            }
            for _, row in claim_clusters_df.iterrows()
        ]

    system_prompt = load_prompt(prompt_path)
    labels: List[Dict[str, Any]] = []
    for _, row in claim_clusters_df.iterrows():
        payload = {
            "claim_cluster": int(row.get("claim_cluster")),
            "top_terms": row.get("top_terms"),
            "examples": json.loads(row.get("examples_json") or "[]")[:15],
        }
        result = client.complete_json(system_prompt, json.dumps(payload, ensure_ascii=False, indent=2))
        labels.append({"claim_cluster": payload["claim_cluster"], "label_source": "llm", **result})
    return labels


def write_claim_cluster_labels(
    labels: List[Dict[str, Any]],
    output_path: Path | str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not labels:
        output.write_text("# Labels de clusters de claims\n\nAucun cluster de claims disponible.\n", encoding="utf-8")
        return output

    blocks = ["# Labels de clusters de claims", ""]
    for label in labels:
        blocks.append(f"## Cluster {label.get('claim_cluster')} — {label.get('cluster_label')}")
        blocks.append("")
        blocks.append(f"Source : `{label.get('label_source')}`")
        blocks.append("")
        if label.get("summary"):
            blocks.append(str(label["summary"]))
            blocks.append("")
        variations = label.get("internal_variations") or []
        if variations:
            blocks.append("Variations internes :")
            blocks.extend(f"- {variation}" for variation in variations)
            blocks.append("")
        examples = label.get("representative_claims") or []
        if examples:
            blocks.append("Claims représentatifs :")
            blocks.extend(f"- {example}" for example in examples[:8])
            blocks.append("")
    output.write_text("\n".join(blocks).strip() + "\n", encoding="utf-8")
    return output

