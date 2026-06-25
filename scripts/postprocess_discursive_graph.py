#!/usr/bin/env python3
"""
Post-process discursive graph outputs without re-running LLM extraction or embeddings.

Reads existing CSVs produced by the observatoire pipeline and builds:
- a full audit graph (comment-attribute graph)
- a filtered comment-comment projection for community detection
- post-processed communities and diagnostics
- optional interactive HTML visualizations with pyvis

Typical use:
    python scripts/postprocess_discursive_graph.py \
      --input-dir outputs \
      --output-dir outputs/graph_postprocess \
      --similarity-threshold 0.50 \
      --include-similarity-edges \
      --drop-non-signal-labels

Minimal dependencies:
    pip install pandas networkx
Optional for HTML graph visualization:
    pip install pyvis
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from html import escape as html_escape
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import networkx as nx


NON_SIGNAL_LABELS = {
    "", "nan", "none", "null", "na", "n/a",
    "other", "autre", "unknown", "unk", "unclear", "non classé", "non_classe",
    "non classifie", "non_classifie", "undefined", "pas clair", "indetermine",
}

# Edge types kept for the comment-comment projection.
# We intentionally exclude source/video/channel/time by default because they can glue the graph together.
DEFAULT_RELATION_WEIGHTS = {
    "COMMENT_HAS_FRAME": 1.20,
    "COMMENT_HAS_CANONICAL_CLAIM": 1.40,
    "COMMENT_HAS_ARGUMENT_FAMILY": 1.00,
    "COMMENT_EXPRESSES_STANCE_TOWARD_TARGET": 1.50,
    "COMMENT_TARGETS_ACTOR": 0.80,
    "COMMENT_HAS_TONE": 0.30,
}

STRUCTURAL_EDGE_TYPES = {
    "COMMENT_FROM_VIDEO",
    "VIDEO_ASSOCIATED_WITH_SOURCE_ACTOR",
    "COMMENT_IN_TIME_BUCKET",
    "COMMENT_FROM_CHANNEL",
    "COMMENT_HAS_THEME",  # free themes are usually too sparse/noisy for the first community pass
    "COMMENT_HAS_REGISTER",
    "FRAME_CO_OCCURS_WITH_CLAIM",
    "CLAIM_BELONGS_TO_ARGUMENT_FAMILY",
}

SOURCE_COLORS = {
    "LFI": "#d62728",
    "RN": "#1f77b4",
    "Media": "#7f7f7f",
    "Média": "#7f7f7f",
    "attribute": "#cccccc",
    "mixed": "#9467bd",
    "unknown": "#bbbbbb",
}

NODE_TYPE_SHAPES = {
    "CommentNode": "dot",
    "VideoNode": "box",
    "SourceActorNode": "diamond",
    "FrameNode": "triangle",
    "CanonicalClaimNode": "star",
    "ClaimNode": "star",
    "ArgumentFamilyNode": "hexagon",
    "TargetNode": "square",
    "StanceTargetNode": "triangleDown",
    "ToneNode": "ellipse",
    "TimeNode": "database",
    "ChannelNode": "box",
    "ThemeNode": "ellipse",
    "RegisterNode": "ellipse",
}


PYVIS_STATIC_OPTIONS = (
    'var options = {\n'
    '  "nodes": {"borderWidth": 1, "font": {"size": 11}},\n'
    '  "edges": {"smooth": false, "color": {"opacity": 0.35}},\n'
    '  "interaction": {"hover": true, "navigationButtons": true, "keyboard": true, "dragNodes": true},\n'
    '  "physics": {"enabled": false}\n'
    '}\n'
)


def compute_static_positions(
    H: nx.Graph,
    layout_iterations: int = 300,
    layout_scale: float = 2500.0,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute deterministic 2D positions in Python so PyVis does not need live physics."""
    if H.number_of_nodes() == 0:
        return {}
    raw_pos = nx.spring_layout(
        H,
        seed=seed,
        iterations=max(int(layout_iterations), 1),
        weight="weight",
        k=None,
        scale=float(layout_scale),
    )
    return {str(n): (float(xy[0]), float(xy[1])) for n, xy in raw_pos.items()}


def hard_freeze_pyvis_html(html_path: Path) -> None:
    """Patch generated PyVis HTML as a safety net to stop remaining simulation."""
    if not html_path.exists():
        return
    html = html_path.read_text(encoding="utf-8")
    if "__CHATGPT_STATIC_FREEZE_PATCH__" in html:
        return
    freeze_js = (
        "\n// __CHATGPT_STATIC_FREEZE_PATCH__\n"
        "(function () {\n"
        "  function freezeNetwork() {\n"
        "    if (typeof network !== \"undefined\" && network) {\n"
        "      try { network.stopSimulation(); } catch (e) {}\n"
        "      try { network.setOptions({ physics: { enabled: false }, edges: { smooth: false } }); } catch (e) {}\n"
        "    }\n"
        "  }\n"
        "  if (typeof network !== \"undefined\" && network) {\n"
        "    try { network.once(\"stabilizationIterationsDone\", freezeNetwork); } catch (e) {}\n"
        "    freezeNetwork();\n"
        "    setTimeout(freezeNetwork, 500);\n"
        "    setTimeout(freezeNetwork, 1500);\n"
        "    setTimeout(freezeNetwork, 3000);\n"
        "  }\n"
        "})();\n"
    )
    pattern = r"(network\s*=\s*new\s+vis\.Network\s*\([^;]+;\s*)"
    patched, n = re.subn(pattern, r"\1\n" + freeze_js + "\n", html, count=1)
    if n == 0:
        patched = html.replace("</script>", freeze_js + "\n</script>", 1)
    html_path.write_text(patched, encoding="utf-8")




def clip_text(value, max_chars: int = 2500) -> str:
    """Return a compact, readable string for labels/tooltips."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def tooltip_row(label: str, value, max_chars: int = 2500, html: bool = False) -> str:
    """Build one tooltip row.

    Default is plain text because some PyVis/vis-network builds escape HTML titles,
    which makes <br> and <b> appear literally. Plain text + CSS pre-wrap is robust.
    """
    value = clip_text(value, max_chars=max_chars)
    if not value:
        return ""
    if html:
        return (
            "<tr>"
            f"<th>{html_escape(str(label))}</th>"
            f"<td>{html_escape(value).replace(chr(10), '<br>')}</td>"
            "</tr>"
        )
    return f"{label}: {value}"


def make_node_title(node_id, attrs: dict, max_summary_chars: int = 3000, html: bool = False) -> str:
    """Readable tooltip for a node.

    If html=False, returns plain text with newlines. CSS added later makes it wrap.
    If html=True, returns HTML table. Use html=True only if your PyVis/vis-network
    installation renders HTML in tooltips rather than escaping it.
    """
    node_type = attrs.get("node_type", attrs.get("type", ""))
    label = attrs.get("label", attrs.get("node_label", node_id))

    rows = [
        tooltip_row("label", label, 800, html),
        tooltip_row("node_id", node_id, 500, html),
        tooltip_row("node_type", node_type, 300, html),
        tooltip_row("source", attrs.get("source", attrs.get("actor", "")), 300, html),
        tooltip_row("community", attrs.get("community", attrs.get("post_community", "")), 100, html),
        tooltip_row("degree", attrs.get("degree", ""), 100, html),
        tooltip_row("weighted_degree", attrs.get("weighted_degree", ""), 100, html),
        tooltip_row("video_id", attrs.get("video_id", ""), 300, html),
        tooltip_row("time", attrs.get("time_bucket", ""), 300, html),
        tooltip_row("summary", attrs.get("summary", attrs.get("discursive_summary", "")), max_summary_chars, html),
        tooltip_row("canonical_rewrite", attrs.get("canonical_rewrite", ""), max_summary_chars, html),
        tooltip_row("preview", attrs.get("preview", attrs.get("raw_comment_preview", "")), max_summary_chars, html),
        tooltip_row("raw_text", attrs.get("raw_text", ""), max_summary_chars, html),
        tooltip_row("macro_frame", attrs.get("macro_frame", ""), 600, html),
        tooltip_row("frame", attrs.get("frame_primary", attrs.get("frame", "")), 600, html),
        tooltip_row("claim", attrs.get("canonical_claim", attrs.get("claim", "")), 1200, html),
        tooltip_row("argument_family", attrs.get("argument_family", attrs.get("argument_family_controlled", "")), 600, html),
        tooltip_row("target", attrs.get("target", ""), 600, html),
        tooltip_row("stance", attrs.get("stance", ""), 300, html),
        tooltip_row("tone", attrs.get("tone", attrs.get("tone_controlled", "")), 300, html),
    ]
    rows = [r for r in rows if r]

    if html:
        return (
            '<div class="graph-tooltip">'
            f'<div class="tooltip-title">{html_escape(str(label))}</div>'
            '<table>' + "\n".join(rows) + '</table></div>'
        )
    return "\n".join(rows)


def make_short_node_label(node_id, attrs: dict, max_chars: int = 32) -> str:
    """Keep graph labels short; put details in the tooltip."""
    node_type = attrs.get("node_type", attrs.get("type", ""))
    source = attrs.get("source", attrs.get("actor", ""))
    community = attrs.get("community", attrs.get("post_community", ""))

    if node_type in {"CommentProjectionNode", "CommentNode"}:
        if community not in {"", None, -1, "-1"}:
            base = f"{source} | c{community}"
        else:
            base = f"{source} | comment"
    else:
        base = str(attrs.get("label", attrs.get("node_label", node_id)))

    base = base.strip() or str(node_id)
    if len(base) > max_chars:
        base = base[:max_chars].rstrip() + "…"
    return base


def patch_pyvis_tooltip_css(html_path: Path) -> None:
    """Make vis-network tooltips readable and scrollable.

    Works for both plain-text tooltips (default) and HTML tooltips.
    """
    html_path = Path(html_path)
    if not html_path.exists():
        return
    html = html_path.read_text(encoding="utf-8")
    if "__CHATGPT_TOOLTIP_CSS_PATCH__" in html:
        return

    css = """
<style id="__CHATGPT_TOOLTIP_CSS_PATCH__">
.vis-tooltip {
  max-width: 720px !important;
  min-width: 360px !important;
  max-height: 460px !important;
  overflow-y: auto !important;
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
  word-break: normal !important;
  line-height: 1.38 !important;
  font-size: 13px !important;
  padding: 10px 12px !important;
  z-index: 999999 !important;
}
.graph-tooltip {
  max-width: 690px;
  white-space: normal;
}
.graph-tooltip .tooltip-title {
  font-weight: 700;
  margin-bottom: 8px;
  font-size: 14px;
}
.graph-tooltip table {
  border-collapse: collapse;
  width: 100%;
}
.graph-tooltip th {
  text-align: left;
  vertical-align: top;
  padding: 4px 8px 4px 0;
  color: #555;
  white-space: nowrap;
  font-weight: 600;
}
.graph-tooltip td {
  text-align: left;
  vertical-align: top;
  padding: 4px 0;
  white-space: normal;
  overflow-wrap: anywhere;
}
</style>
"""
    if "</head>" in html:
        html = html.replace("</head>", css + "\n</head>", 1)
    else:
        html = css + "\n" + html
    html_path.write_text(html, encoding="utf-8")

def read_csv_if_exists(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().lower()
    s = " ".join(s.replace("_", " ").split())
    return s


def is_non_signal(label) -> bool:
    return normalize_label(label) in NON_SIGNAL_LABELS


def signal_value_counts(series: pd.Series, limit: int = 8) -> Dict[str, int]:
    """Top counts for readable summaries, excluding vague placeholder labels."""
    if series is None or series.empty:
        return {}
    cleaned = series.dropna().astype(str).map(str.strip)
    cleaned = cleaned[cleaned != ""]
    cleaned = cleaned[~cleaned.map(is_non_signal)]
    return cleaned.value_counts().head(limit).to_dict()


def safe_json_loads(value, default=None):
    if default is None:
        default = []
    if pd.isna(value):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def ensure_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_inputs(input_dir: Path):
    units = read_csv_if_exists(input_dir / "discursive_units.csv")
    nodes = read_csv_if_exists(input_dir / "discursive_nodes.csv")
    edges = read_csv_if_exists(input_dir / "discursive_edges.csv")
    sim_edges = read_csv_if_exists(input_dir / "discursive_similarity_edges.csv", required=False)
    old_comms = read_csv_if_exists(input_dir / "discursive_communities.csv", required=False)

    ensure_columns(units, ["comment_id", "actor"], "discursive_units.csv")
    ensure_columns(nodes, ["node_id", "node_type", "label"], "discursive_nodes.csv")
    ensure_columns(edges, ["source_node_id", "target_node_id", "edge_type"], "discursive_edges.csv")
    if not sim_edges.empty:
        ensure_columns(sim_edges, ["source_comment_id", "target_comment_id", "similarity"], "discursive_similarity_edges.csv")
    return units, nodes, edges, sim_edges, old_comms


def build_comment_maps(units: pd.DataFrame, nodes: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Return comment_id -> node_id, node_id -> comment_id, comment_id -> actor."""
    comment_nodes = nodes[nodes["node_type"] == "CommentNode"].copy()
    # In current outputs, CommentNode.label is the original comment_id.
    comment_id_to_node = dict(zip(comment_nodes["label"].astype(str), comment_nodes["node_id"].astype(str)))
    node_to_comment_id = {v: k for k, v in comment_id_to_node.items()}
    comment_to_actor = dict(zip(units["comment_id"].astype(str), units["actor"].fillna("unknown").astype(str)))
    return comment_id_to_node, node_to_comment_id, comment_to_actor


def aggregate_edges(edges: pd.DataFrame, mode: str = "max") -> pd.DataFrame:
    """Deduplicate edge triples. For graph topology, repeated structural edges should not dominate."""
    work = edges.copy()
    if "weight" not in work.columns:
        work["weight"] = 1.0
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce").fillna(1.0)
    group_cols = ["source_node_id", "target_node_id", "edge_type"]
    if mode == "sum":
        agg = work.groupby(group_cols, as_index=False).agg(
            weight=("weight", "sum"),
            n_edges=("weight", "size"),
        )
    else:
        agg = work.groupby(group_cols, as_index=False).agg(
            weight=("weight", "max"),
            n_edges=("weight", "size"),
        )
    return agg


def build_full_graph(nodes: pd.DataFrame, edges: pd.DataFrame, aggregate_mode: str = "max") -> nx.Graph:
    G = nx.Graph()
    for row in nodes.itertuples(index=False):
        attrs = row._asdict()
        node_id = str(attrs.pop("node_id"))
        G.add_node(node_id, **attrs)

    agg_edges = aggregate_edges(edges, mode=aggregate_mode)
    for row in agg_edges.itertuples(index=False):
        G.add_edge(
            str(row.source_node_id),
            str(row.target_node_id),
            edge_type=str(row.edge_type),
            weight=float(row.weight),
            n_edges=int(row.n_edges),
        )
    return G


def compute_node_sources(
    G: nx.Graph,
    nodes: pd.DataFrame,
    comment_id_to_node: Dict[str, str],
    node_to_comment_id: Dict[str, str],
    comment_to_actor: Dict[str, str],
) -> Dict[str, str]:
    """Infer source for every node from adjacent comments.

    CommentNode: actor from discursive_units.
    SourceActorNode: label.
    Other nodes: actor if all linked comments share one actor, mixed if multiple, attribute if no comment links.
    """
    node_meta = nodes.set_index("node_id").to_dict("index")
    source_by_node: Dict[str, str] = {}

    for node in G.nodes:
        meta = node_meta.get(node, {})
        ntype = meta.get("node_type", "")
        label = meta.get("label", "")
        if ntype == "CommentNode":
            cid = node_to_comment_id.get(node, str(label))
            source_by_node[node] = comment_to_actor.get(cid, "unknown")
        elif ntype == "SourceActorNode":
            source_by_node[node] = str(label) if str(label) else "unknown"
        else:
            actors = Counter()
            for nb in G.neighbors(node):
                nb_type = node_meta.get(nb, {}).get("node_type")
                if nb_type == "CommentNode":
                    cid = node_to_comment_id.get(nb, node_meta.get(nb, {}).get("label", ""))
                    actors[comment_to_actor.get(str(cid), "unknown")] += 1
            if not actors:
                source_by_node[node] = "attribute"
            elif len(actors) == 1:
                source_by_node[node] = next(iter(actors))
            else:
                source_by_node[node] = "mixed"
    return source_by_node


def should_drop_attribute_node(row, drop_non_signal_labels: bool) -> bool:
    if not drop_non_signal_labels:
        return False
    label = row.get("label", "")
    norm = row.get("normalized_label", "")
    if is_non_signal(label) or is_non_signal(norm):
        return True
    if "is_vague" in row and str(row.get("is_vague", "")).lower() == "true":
        return True
    return False


def build_comment_projection(
    units: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    sim_edges: pd.DataFrame,
    relation_weights: Dict[str, float],
    include_similarity_edges: bool = True,
    similarity_threshold: float = 0.50,
    similarity_lambda: float = 1.0,
    drop_non_signal_labels: bool = True,
    min_attribute_df: int = 2,
    max_attribute_df: int = 80,
    min_edge_weight: float = 0.0,
) -> Tuple[nx.Graph, pd.DataFrame]:
    """Build a comment-comment weighted projection from selected attribute relations + optional similarity edges."""
    comment_ids = set(units["comment_id"].astype(str))
    P = nx.Graph()
    for cid in comment_ids:
        row = units.loc[units["comment_id"].astype(str) == cid].iloc[0]
        P.add_node(
            cid,
            node_type="CommentProjectionNode",
            actor=str(row.get("actor", "unknown")),
            video_id=str(row.get("video_id", "")),
            time_bucket=str(row.get("time_bucket", "")),
            discursive_cluster=str(row.get("discursive_cluster", "")),
            summary=str(row.get("discursive_summary", ""))[:400],
            preview=str(row.get("raw_comment_preview", ""))[:400],
        )

    node_meta = nodes.set_index("node_id").to_dict("index")
    comment_node_to_comment_id = {
        row.node_id: str(row.label)
        for row in nodes[nodes["node_type"] == "CommentNode"].itertuples(index=False)
    }

    # Keep only comment -> selected attribute edge types.
    work = edges.copy()
    if "weight" not in work.columns:
        work["weight"] = 1.0
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce").fillna(1.0)
    work = work[work["edge_type"].isin(relation_weights.keys())]
    work = work[work["weight"] >= min_edge_weight]

    attr_to_comments: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
    for row in work.itertuples(index=False):
        src = str(row.source_node_id)
        tgt = str(row.target_node_id)
        etype = str(row.edge_type)
        # In current output, selected edges are CommentNode -> AttributeNode.
        if src in comment_node_to_comment_id:
            cid = comment_node_to_comment_id[src]
            attr_node = tgt
        elif tgt in comment_node_to_comment_id:
            cid = comment_node_to_comment_id[tgt]
            attr_node = src
        else:
            continue
        if cid not in comment_ids:
            continue
        meta = node_meta.get(attr_node, {})
        if should_drop_attribute_node(meta, drop_non_signal_labels):
            continue
        attr_to_comments[attr_node].append((cid, float(row.weight), etype))

    # Projection via shared attributes. Skip too-rare and too-common attributes.
    contribution_rows = []
    for attr_node, items in attr_to_comments.items():
        # Deduplicate same comment-attribute relation, keep max weight per comment.
        best_by_comment: Dict[str, Tuple[float, str]] = {}
        for cid, w, etype in items:
            if cid not in best_by_comment or w > best_by_comment[cid][0]:
                best_by_comment[cid] = (w, etype)
        dedup_items = [(cid, w, etype) for cid, (w, etype) in best_by_comment.items()]
        df = len(dedup_items)
        if df < min_attribute_df or df > max_attribute_df:
            continue
        attr_meta = node_meta.get(attr_node, {})
        attr_label = str(attr_meta.get("label", attr_node))
        attr_type = str(attr_meta.get("node_type", "AttributeNode"))
        # Penalize attributes that connect many comments.
        df_penalty = math.sqrt(df)
        for (c1, w1, etype1), (c2, w2, etype2) in itertools.combinations(dedup_items, 2):
            # If same attr, etype should generally match; use first.
            rw = relation_weights.get(etype1, 1.0)
            contrib = rw * min(w1, w2) / max(df_penalty, 1.0)
            if contrib <= 0:
                continue
            if P.has_edge(c1, c2):
                P[c1][c2]["weight"] += contrib
                P[c1][c2]["shared_attributes"] += 1
                P[c1][c2]["attribute_sources"].append(attr_label)
            else:
                P.add_edge(
                    c1,
                    c2,
                    weight=contrib,
                    shared_attributes=1,
                    similarity=0.0,
                    attribute_sources=[attr_label],
                )
            contribution_rows.append({
                "source_comment_id": c1,
                "target_comment_id": c2,
                "attribute_node_id": attr_node,
                "attribute_label": attr_label,
                "attribute_type": attr_type,
                "edge_type": etype1,
                "contribution": contrib,
                "attribute_df": df,
            })

    if include_similarity_edges and sim_edges is not None and not sim_edges.empty:
        for row in sim_edges.itertuples(index=False):
            c1 = str(row.source_comment_id)
            c2 = str(row.target_comment_id)
            sim = float(row.similarity)
            if c1 not in comment_ids or c2 not in comment_ids or sim < similarity_threshold:
                continue
            contrib = similarity_lambda * sim
            if P.has_edge(c1, c2):
                P[c1][c2]["weight"] += contrib
                P[c1][c2]["similarity"] = max(P[c1][c2].get("similarity", 0.0), sim)
            else:
                P.add_edge(
                    c1,
                    c2,
                    weight=contrib,
                    shared_attributes=0,
                    similarity=sim,
                    attribute_sources=[],
                )

    contributions = pd.DataFrame(contribution_rows)
    return P, contributions


def detect_communities(P: nx.Graph, min_community_size: int = 3, resolution: float = 1.0) -> Dict[str, int]:
    """Detect communities in a weighted comment projection."""
    if P.number_of_edges() == 0:
        return {n: -1 for n in P.nodes}

    try:
        # Available in recent NetworkX versions.
        comms = nx.algorithms.community.louvain_communities(P, weight="weight", resolution=resolution, seed=42)
    except Exception:
        comms = list(nx.algorithms.community.greedy_modularity_communities(P, weight="weight"))

    # Sort by size descending for stable labels.
    comms = sorted([set(c) for c in comms], key=lambda c: (-len(c), sorted(c)[0] if c else ""))
    membership = {}
    label = 0
    for c in comms:
        if len(c) < min_community_size:
            for n in c:
                membership[n] = -1
        else:
            for n in c:
                membership[n] = label
            label += 1
    for n in P.nodes:
        membership.setdefault(n, -1)
    return membership


def summarize_communities(units: pd.DataFrame, membership: Dict[str, int], P: nx.Graph) -> Tuple[pd.DataFrame, pd.DataFrame]:
    units_out = units.copy()
    units_out["post_community"] = units_out["comment_id"].astype(str).map(membership).fillna(-1).astype(int)

    rows = []
    for comm, grp in units_out.groupby("post_community"):
        comment_ids = set(grp["comment_id"].astype(str))
        sub = P.subgraph(comment_ids)
        internal_weights = [d.get("weight", 1.0) for _, _, d in sub.edges(data=True)]
        actor_dist = (grp["actor"].fillna("unknown").value_counts(normalize=True).round(4).to_dict()
                      if "actor" in grp.columns else {})
        time_dist = (grp["time_bucket"].fillna("unknown").value_counts(normalize=True).head(10).round(4).to_dict()
                     if "time_bucket" in grp.columns else {})
        top_frames = signal_value_counts(grp.get("frame_primary", pd.Series(dtype=str)))
        top_macro = signal_value_counts(grp.get("macro_frame", pd.Series(dtype=str)))
        top_args = signal_value_counts(grp.get("argument_family_controlled", pd.Series(dtype=str)))
        top_tones = signal_value_counts(grp.get("tone_controlled", pd.Series(dtype=str)))
        examples = grp.get("discursive_summary", pd.Series(dtype=str)).dropna().astype(str).head(8).tolist()
        quotes = grp.get("raw_comment_preview", pd.Series(dtype=str)).dropna().astype(str).head(8).tolist()
        rows.append({
            "post_community": int(comm),
            "size": int(len(grp)),
            "share": round(len(grp) / max(len(units_out), 1), 4),
            "n_internal_edges": int(sub.number_of_edges()),
            "mean_internal_weight": round(sum(internal_weights) / len(internal_weights), 6) if internal_weights else 0.0,
            "actor_distribution_json": json.dumps(actor_dist, ensure_ascii=False),
            "time_distribution_json": json.dumps(time_dist, ensure_ascii=False),
            "top_macro_frames_json": json.dumps(top_macro, ensure_ascii=False),
            "top_frames_json": json.dumps(top_frames, ensure_ascii=False),
            "top_arguments_json": json.dumps(top_args, ensure_ascii=False),
            "top_tones_json": json.dumps(top_tones, ensure_ascii=False),
            "examples_json": json.dumps(examples, ensure_ascii=False),
            "representative_quotes_json": json.dumps(quotes, ensure_ascii=False),
        })
    summary = pd.DataFrame(rows).sort_values(["post_community"]).reset_index(drop=True)
    return units_out, summary


def graph_diagnostics(G: nx.Graph, P: nx.Graph, nodes: pd.DataFrame, units: pd.DataFrame, membership: Dict[str, int]) -> dict:
    comps = list(nx.connected_components(G)) if G.number_of_nodes() else []
    p_comps = list(nx.connected_components(P)) if P.number_of_nodes() else []
    type_dist = nodes["node_type"].fillna("unknown").value_counts().to_dict() if "node_type" in nodes else {}
    source_dist = units["actor"].fillna("unknown").value_counts().to_dict() if "actor" in units else {}
    comm_dist = Counter(membership.values())

    node_meta = nodes.set_index("node_id").to_dict("index")
    top_full_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)

    def top_degree_rows(drop_non_signal: bool) -> List[dict]:
        rows = []
        for node, deg in top_full_degree:
            meta = node_meta.get(node, {})
            label = meta.get("label", "")
            norm = meta.get("normalized_label", label)
            if drop_non_signal and (is_non_signal(label) or is_non_signal(norm)):
                continue
            rows.append({
                "node_id": node,
                "degree": deg,
                "node_type": meta.get("node_type", ""),
                "label": label,
            })
            if len(rows) >= 25:
                break
        return rows

    non_signal_hubs = []
    for node, deg in top_full_degree:
        meta = node_meta.get(node, {})
        label = meta.get("label", "")
        norm = meta.get("normalized_label", label)
        if is_non_signal(label) or is_non_signal(norm):
            non_signal_hubs.append({
                "node_id": node,
                "degree": deg,
                "node_type": meta.get("node_type", ""),
                "label": label,
            })
        if len(non_signal_hubs) >= 25:
            break

    return {
        "full_graph": {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "n_components": len(comps),
            "largest_component_size": max((len(c) for c in comps), default=0),
            "node_type_distribution": type_dist,
            "source_distribution_units": source_dist,
            "top_degree_nodes": top_degree_rows(drop_non_signal=False),
            "top_degree_signal_nodes": top_degree_rows(drop_non_signal=True),
            "top_non_signal_hubs": non_signal_hubs,
        },
        "comment_projection": {
            "n_nodes": P.number_of_nodes(),
            "n_edges": P.number_of_edges(),
            "n_components": len(p_comps),
            "largest_component_size": max((len(c) for c in p_comps), default=0),
            "community_size_distribution": dict(sorted(comm_dist.items(), key=lambda kv: kv[0])),
        },
    }


def try_write_pyvis_full_graph(
    G: nx.Graph,
    output_path: Path,
    nodes: pd.DataFrame,
    source_by_node: Dict[str, str],
    max_nodes: Optional[int] = None,
    title: str = "Discursive full graph by source",
    static_layout: bool = True,
    layout_iterations: int = 300,
    layout_scale: float = 2500.0,
):
    try:
        from pyvis.network import Network
    except Exception:
        print("pyvis is not installed; skipping HTML full graph. Install with: pip install pyvis")
        nx.write_gexf(sanitize_graph_for_gexf(G), output_path.with_suffix(".gexf"))
        return

    H = G
    if max_nodes is not None and G.number_of_nodes() > max_nodes:
        keep = {n for n, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]}
        H = G.subgraph(keep).copy()

    node_meta = nodes.set_index("node_id").to_dict("index")
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
        cdn_resources="in_line",
    )

    positions: Dict[str, Tuple[float, float]] = {}
    if static_layout:
        print(f"Computing static full-graph layout: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges...")
        positions = compute_static_positions(H, layout_iterations=layout_iterations, layout_scale=layout_scale)
    else:
        net.barnes_hut(gravity=-4500, central_gravity=0.18, spring_length=120, spring_strength=0.02, damping=0.72)

    for n, data in H.nodes(data=True):
        meta = node_meta.get(n, {})
        ntype = meta.get("node_type", data.get("node_type", ""))
        label = str(meta.get("label", data.get("label", n)))
        src = source_by_node.get(n, "unknown")
        color = SOURCE_COLORS.get(src, SOURCE_COLORS.get("unknown"))
        shape = NODE_TYPE_SHAPES.get(ntype, "dot")
        deg = H.degree(n)
        size = 6 + min(24, math.sqrt(max(deg, 1)) * 4)
        tooltip_attrs = dict(data)
        tooltip_attrs.update(meta)
        tooltip_attrs.update({"node_id": n, "label": label, "node_type": ntype, "source": src, "degree": deg})
        title_html = make_node_title(n, tooltip_attrs, html=False)
        short_label = make_short_node_label(n, tooltip_attrs, max_chars=34)
        kwargs = {"label": short_label, "title": title_html, "color": color, "shape": shape, "size": size}
        if static_layout:
            x, y = positions.get(str(n), (0.0, 0.0))
            kwargs.update({"x": x, "y": y, "physics": False})
        net.add_node(n, **kwargs)

    for u, v, data in H.edges(data=True):
        w = float(data.get("weight", 1.0))
        etype = str(data.get("edge_type", ""))
        kwargs = {"value": max(w, 0.1), "title": f"edge_type: {etype}\nweight: {w:.4f}"}
        if static_layout:
            kwargs.update({"physics": False, "smooth": False})
        net.add_edge(u, v, **kwargs)

    if static_layout:
        net.set_options(PYVIS_STATIC_OPTIONS)
    else:
        net.set_options('\n        var options = {\n          "nodes": {"borderWidth": 1, "font": {"size": 12}},\n          "edges": {"smooth": false, "color": {"inherit": true}},\n          "interaction": {"hover": true, "navigationButtons": true, "keyboard": true},\n          "physics": {"stabilization": {"iterations": 500}}\n        }\n        ')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path), notebook=False, open_browser=False)
    patch_pyvis_tooltip_css(output_path)
    if static_layout:
        hard_freeze_pyvis_html(output_path)


def try_write_pyvis_projection(
    P: nx.Graph,
    output_path: Path,
    membership: Dict[str, int],
    max_nodes: Optional[int] = None,
    title: str = "Discursive comment projection by source",
    static_layout: bool = True,
    layout_iterations: int = 300,
    layout_scale: float = 2500.0,
):
    try:
        from pyvis.network import Network
    except Exception:
        print("pyvis is not installed; skipping HTML projection. Install with: pip install pyvis")
        nx.write_gexf(sanitize_graph_for_gexf(P), output_path.with_suffix(".gexf"))
        return

    H = P
    if max_nodes is not None and P.number_of_nodes() > max_nodes:
        keep = {n for n, _ in sorted(P.degree(weight="weight"), key=lambda x: x[1], reverse=True)[:max_nodes]}
        H = P.subgraph(keep).copy()

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
        cdn_resources="in_line",
    )

    positions: Dict[str, Tuple[float, float]] = {}
    if static_layout:
        print(f"Computing static projection layout: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges...")
        positions = compute_static_positions(H, layout_iterations=layout_iterations, layout_scale=layout_scale)
    else:
        net.force_atlas_2based(gravity=-60, central_gravity=0.015, spring_length=130, spring_strength=0.08, damping=0.65)

    for n, data in H.nodes(data=True):
        actor = data.get("actor", "unknown")
        color = SOURCE_COLORS.get(actor, SOURCE_COLORS.get("unknown"))
        comm = membership.get(n, -1)
        strength = sum(float(d.get("weight", 1.0)) for _, _, d in H.edges(n, data=True))
        size = 7 + min(25, math.sqrt(max(strength, 1)) * 3)
        tooltip_attrs = dict(data)
        tooltip_attrs.update({
            "node_id": n,
            "label": n,
            "node_type": "CommentProjectionNode",
            "source": actor,
            "community": comm,
            "degree": H.degree(n),
            "weighted_degree": round(strength, 4),
        })
        title_html = make_node_title(n, tooltip_attrs, html=False)
        label = make_short_node_label(n, tooltip_attrs, max_chars=24)
        kwargs = {"label": label, "title": title_html, "color": color, "shape": "dot", "size": size}
        if static_layout:
            x, y = positions.get(str(n), (0.0, 0.0))
            kwargs.update({"x": x, "y": y, "physics": False})
        net.add_node(n, **kwargs)

    for u, v, data in H.edges(data=True):
        w = float(data.get("weight", 1.0))
        sim = float(data.get("similarity", 0.0))
        title_edge = f"weight: {w:.4f}\nsimilarity: {sim:.4f}\nshared_attributes: {data.get('shared_attributes',0)}"
        kwargs = {"value": max(w, 0.1), "title": title_edge}
        if static_layout:
            kwargs.update({"physics": False, "smooth": False})
        net.add_edge(u, v, **kwargs)

    if static_layout:
        net.set_options(PYVIS_STATIC_OPTIONS)
    else:
        net.set_options('\n        var options = {\n          "nodes": {"borderWidth": 1, "font": {"size": 10}},\n          "edges": {"smooth": false, "color": {"opacity": 0.35}},\n          "interaction": {"hover": true, "navigationButtons": true, "keyboard": true},\n          "physics": {"stabilization": {"iterations": 500}}\n        }\n        ')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path), notebook=False, open_browser=False)
    patch_pyvis_tooltip_css(output_path)
    if static_layout:
        hard_freeze_pyvis_html(output_path)



def sanitize_graph_for_gexf(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    for n, data in G.nodes(data=True):
        clean = {}
        for k, v in data.items():
            if isinstance(v, (list, dict, tuple, set)):
                clean[k] = json.dumps(list(v) if not isinstance(v, dict) else v, ensure_ascii=False)
            elif pd.isna(v) if not isinstance(v, (str, int, float, bool)) else False:
                clean[k] = ""
            else:
                clean[k] = v
        H.add_node(n, **clean)
    for u, v, data in G.edges(data=True):
        clean = {}
        for k, val in data.items():
            if isinstance(val, (list, dict, tuple, set)):
                clean[k] = json.dumps(list(val) if not isinstance(val, dict) else val, ensure_ascii=False)
            else:
                clean[k] = val
        H.add_edge(u, v, **clean)
    return H

def parse_relation_weights(json_str: Optional[str]) -> Dict[str, float]:
    if not json_str:
        return dict(DEFAULT_RELATION_WEIGHTS)
    custom = json.loads(json_str)
    weights = dict(DEFAULT_RELATION_WEIGHTS)
    weights.update({str(k): float(v) for k, v in custom.items()})
    return weights


def main():
    parser = argparse.ArgumentParser(description="Post-process discursive graph outputs without re-encoding comments.")
    parser.add_argument("--input-dir", default="outputs", help="Directory containing discursive_*.csv outputs.")
    parser.add_argument("--output-dir", default="outputs/graph_postprocess", help="Directory for post-processed outputs.")
    parser.add_argument("--include-similarity-edges", action="store_true", help="Include discursive_similarity_edges.csv in comment projection.")
    parser.add_argument("--similarity-threshold", type=float, default=0.50, help="Minimum similarity edge to keep.")
    parser.add_argument("--similarity-lambda", type=float, default=1.0, help="Weight multiplier for similarity edges.")
    parser.add_argument("--drop-non-signal-labels", action="store_true", help="Drop other/unknown/unclear/vague attribute nodes from community graph.")
    parser.add_argument("--min-attribute-df", type=int, default=2, help="Minimum comments connected to an attribute for projection.")
    parser.add_argument("--max-attribute-df", type=int, default=80, help="Maximum comments connected to an attribute before it is treated as a hub.")
    parser.add_argument("--min-edge-weight", type=float, default=0.0, help="Minimum original comment-attribute edge weight to keep.")
    parser.add_argument("--min-community-size", type=int, default=3, help="Communities smaller than this are labeled -1.")
    parser.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution when available.")
    parser.add_argument("--relation-weights-json", default=None, help="Optional JSON object overriding relation weights.")
    parser.add_argument("--full-graph-max-viz-nodes", type=int, default=3000, help="Cap for full graph HTML; 0 means no cap.")
    parser.add_argument("--projection-max-viz-nodes", type=int, default=1000, help="Cap for projection HTML; 0 means no cap.")
    parser.add_argument("--aggregate-edge-mode", choices=["max", "sum"], default="max", help="How to aggregate duplicate full-graph edges.")
    parser.add_argument("--viz-layout", choices=["static", "dynamic"], default="static", help="Static precomputed layout disables PyVis physics; dynamic keeps vis.js physics for small graphs.")
    parser.add_argument("--layout-iterations", type=int, default=300, help="NetworkX spring_layout iterations for static HTML visualizations.")
    parser.add_argument("--layout-scale", type=float, default=2500.0, help="Coordinate scale for static HTML visualizations.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    units, nodes, edges, sim_edges, old_comms = load_inputs(input_dir)
    relation_weights = parse_relation_weights(args.relation_weights_json)

    comment_id_to_node, node_to_comment_id, comment_to_actor = build_comment_maps(units, nodes)
    G = build_full_graph(nodes, edges, aggregate_mode=args.aggregate_edge_mode)
    source_by_node = compute_node_sources(G, nodes, comment_id_to_node, node_to_comment_id, comment_to_actor)

    P, contributions = build_comment_projection(
        units=units,
        nodes=nodes,
        edges=edges,
        sim_edges=sim_edges,
        relation_weights=relation_weights,
        include_similarity_edges=args.include_similarity_edges,
        similarity_threshold=args.similarity_threshold,
        similarity_lambda=args.similarity_lambda,
        drop_non_signal_labels=args.drop_non_signal_labels,
        min_attribute_df=args.min_attribute_df,
        max_attribute_df=args.max_attribute_df,
        min_edge_weight=args.min_edge_weight,
    )

    membership = detect_communities(P, min_community_size=args.min_community_size, resolution=args.resolution)
    units_out, summary = summarize_communities(units, membership, P)
    diagnostics = graph_diagnostics(G, P, nodes, units, membership)
    diagnostics["parameters"] = vars(args)
    diagnostics["relation_weights"] = relation_weights

    # Save outputs.
    units_out.to_csv(output_dir / "postprocessed_comment_communities.csv", index=False)
    summary.to_csv(output_dir / "postprocessed_community_summary.csv", index=False)

    # Save projection edges.
    projection_rows = []
    for u, v, data in P.edges(data=True):
        projection_rows.append({
            "source_comment_id": u,
            "target_comment_id": v,
            "weight": data.get("weight", 1.0),
            "similarity": data.get("similarity", 0.0),
            "shared_attributes": data.get("shared_attributes", 0),
            "attribute_sources_preview": " | ".join(map(str, data.get("attribute_sources", [])[:10])),
        })
    pd.DataFrame(projection_rows).to_csv(output_dir / "postprocessed_comment_projection_edges.csv", index=False)

    if not contributions.empty:
        contributions.to_csv(output_dir / "postprocessed_attribute_contributions.csv", index=False)

    with open(output_dir / "postprocessed_graph_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    # Full graph and projection exports.
    # GEXF does not support list attributes, so write sanitized copies.
    G_gexf = sanitize_graph_for_gexf(G)
    P_gexf = sanitize_graph_for_gexf(P)
    nx.write_gexf(G_gexf, output_dir / "discursive_full_graph.gexf")
    nx.write_gexf(P_gexf, output_dir / "discursive_comment_projection.gexf")

    full_cap = None if args.full_graph_max_viz_nodes == 0 else args.full_graph_max_viz_nodes
    proj_cap = None if args.projection_max_viz_nodes == 0 else args.projection_max_viz_nodes
    static_layout = args.viz_layout == "static"
    try_write_pyvis_full_graph(
        G,
        output_dir / "discursive_full_graph_by_source.html",
        nodes,
        source_by_node,
        max_nodes=full_cap,
        static_layout=static_layout,
        layout_iterations=args.layout_iterations,
        layout_scale=args.layout_scale,
    )
    try_write_pyvis_projection(
        P,
        output_dir / "discursive_comment_projection_by_source.html",
        membership,
        max_nodes=proj_cap,
        static_layout=static_layout,
        layout_iterations=args.layout_iterations,
        layout_scale=args.layout_scale,
    )

    print("\n=== Post-processing complete ===")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Comment projection: {P.number_of_nodes()} nodes, {P.number_of_edges()} edges")
    print("Community sizes:", dict(sorted(Counter(membership.values()).items(), key=lambda kv: kv[0])))
    print("\nTop community summary:")
    if not summary.empty:
        cols = ["post_community", "size", "share", "mean_internal_weight", "actor_distribution_json", "top_frames_json"]
        print(summary[cols].head(15).to_string(index=False))
    print("\nFiles written:")
    for p in sorted(output_dir.iterdir()):
        print(f"- {p}")


if __name__ == "__main__":
    main()
