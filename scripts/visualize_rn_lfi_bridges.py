#!/usr/bin/env python3
"""
Visualize the main discursive bridges between RN and LFI from already-encoded outputs.

This script does NOT call an LLM and does NOT recompute embeddings. It reuses:
  - discursive_units.csv
  - discursive_nodes.csv
  - discursive_edges.csv
  - discursive_similarity_edges.csv

It produces:
  - rn_lfi_attribute_bridges.csv
  - rn_lfi_direct_similarity_bridges.csv
  - rn_lfi_bridge_subgraph_nodes.csv
  - rn_lfi_bridge_subgraph_edges.csv
  - rn_lfi_bridge_diagnostics.json
  - rn_lfi_bridge_graph.gexf
  - rn_lfi_bridge_graph.html (if pyvis is installed)

The visualization focuses on two kinds of bridges:
  1) direct RN-LFI comment similarity edges;
  2) shared attribute nodes connected to both RN and LFI comments
     (frames, claims, targets and stance-targets by default).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import networkx as nx

try:
    from pyvis.network import Network  # type: ignore
except Exception:  # pragma: no cover
    Network = None


NON_SIGNAL_LABELS_DEFAULT = {
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

DEFAULT_ATTRIBUTE_TYPES = {
    "FrameNode",
    "CanonicalClaimNode",
    "TargetNode",
    "StanceTargetNode",
}

# Downweight broad affect/argument types and upweight more specific discursive structures.
DEFAULT_TYPE_WEIGHTS = {
    "StanceTargetNode": 1.40,
    "CanonicalClaimNode": 1.30,
    "FrameNode": 1.20,
    "TargetNode": 0.75,
    "ArgumentFamilyNode": 0.55,
    "ThemeNode": 0.35,
    "ToneNode": 0.25,
}

SOURCE_COLORS = {
    "RN": "#1f77b4",
    "LFI": "#d62728",
    "Media": "#7f7f7f",
    "attribute": "#d9d9d9",
    "mixed": "#9467bd",
    "unknown": "#bdbdbd",
}

NODE_SHAPES = {
    "CommentNode": "dot",
    "FrameNode": "box",
    "CanonicalClaimNode": "diamond",
    "ArgumentFamilyNode": "triangle",
    "TargetNode": "ellipse",
    "StanceTargetNode": "star",
    "ToneNode": "hexagon",
    "ThemeNode": "box",
    "VideoNode": "database",
    "SourceActorNode": "square",
    "TimeNode": "triangleDown",
    "ChannelNode": "database",
}


@dataclass
class BridgeConfig:
    source_a: str = "RN"
    source_b: str = "LFI"
    similarity_threshold: float = 0.50
    top_direct_edges: int = 80
    top_attribute_bridges: int = 40
    comments_per_bridge: int = 3
    min_comments_per_side: int = 2
    max_attribute_df: Optional[int] = 250
    include_node_types: Set[str] = None  # type: ignore
    drop_non_signal_labels: bool = True
    layout_iterations: int = 400
    layout_scale: float = 2800.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.include_node_types is None:
            self.include_node_types = set(DEFAULT_ATTRIBUTE_TYPES)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def short_text(value: object, max_chars: int = 120) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def read_inputs(input_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        "units": read_csv_required(input_dir / "discursive_units.csv"),
        "nodes": read_csv_required(input_dir / "discursive_nodes.csv"),
        "edges": read_csv_required(input_dir / "discursive_edges.csv"),
        "sim": read_csv_required(input_dir / "discursive_similarity_edges.csv"),
    }


def build_comment_node_map(edges: pd.DataFrame) -> Dict[str, str]:
    """Map raw comment_id -> CommentNode node_id using edges with a comment_id column."""
    if "comment_id" not in edges.columns:
        return {}
    rows = edges.dropna(subset=["comment_id"])
    mapping: Dict[str, str] = {}
    for _, r in rows.iterrows():
        cid = str(r["comment_id"])
        s = str(r.get("source_node_id", ""))
        t = str(r.get("target_node_id", ""))
        if s.startswith("CommentNode:"):
            mapping.setdefault(cid, s)
        elif t.startswith("CommentNode:"):
            mapping.setdefault(cid, t)
    return mapping


def build_node_lookup(nodes: pd.DataFrame) -> Dict[str, dict]:
    lookup: Dict[str, dict] = {}
    for _, r in nodes.iterrows():
        node_id = str(r.get("node_id", ""))
        lookup[node_id] = r.to_dict()
    return lookup


def prepare_units(units: pd.DataFrame, comment_node_map: Dict[str, str]) -> pd.DataFrame:
    if "comment_id" not in units.columns:
        raise ValueError("discursive_units.csv must contain a comment_id column")
    if "actor" not in units.columns:
        raise ValueError("discursive_units.csv must contain an actor column")
    out = units.copy()
    out["comment_id"] = out["comment_id"].astype(str)
    out["actor"] = out["actor"].astype(str)
    out["comment_node_id"] = out["comment_id"].map(comment_node_map)
    # Fallback: if node map failed, synthesize an id. Edges will still need the actual mapping, so this is mostly for tables.
    out["comment_node_id"] = out["comment_node_id"].fillna("CommentNode:" + out["comment_id"])
    return out


def is_non_signal_label(label: object, normalized_label: object = None, extra: Optional[Set[str]] = None) -> bool:
    labels = set(NON_SIGNAL_LABELS_DEFAULT)
    if extra:
        labels |= {normalize_text(x) for x in extra}
    for value in [label, normalized_label]:
        txt = normalize_text(value)
        if txt in labels:
            return True
    return False


def comment_attribute_edges(
    edges: pd.DataFrame,
    nodes_lookup: Dict[str, dict],
    comment_node_to_id: Dict[str, str],
    cfg: BridgeConfig,
) -> pd.DataFrame:
    """Return edges between comments and selected attribute node types."""
    rows = []
    for _, r in edges.iterrows():
        s = str(r.get("source_node_id", ""))
        t = str(r.get("target_node_id", ""))
        if s.startswith("CommentNode:") and not t.startswith("CommentNode:"):
            cnode, anode = s, t
        elif t.startswith("CommentNode:") and not s.startswith("CommentNode:"):
            cnode, anode = t, s
        else:
            continue

        cid = comment_node_to_id.get(cnode)
        if not cid:
            # fallback from edge comment_id if present
            cid_val = r.get("comment_id")
            if pd.notna(cid_val):
                cid = str(cid_val)
            else:
                continue

        attrs = nodes_lookup.get(anode, {})
        ntype = str(attrs.get("node_type", ""))
        if ntype not in cfg.include_node_types:
            continue

        label = attrs.get("label", anode)
        norm = attrs.get("normalized_label", label)
        if cfg.drop_non_signal_labels and is_non_signal_label(label, norm):
            continue

        df_comments = attrs.get("df_comments", None)
        if cfg.max_attribute_df is not None and pd.notna(df_comments):
            try:
                if int(float(df_comments)) > cfg.max_attribute_df:
                    continue
            except Exception:
                pass

        rows.append(
            {
                "comment_id": cid,
                "comment_node_id": cnode,
                "attribute_node_id": anode,
                "attribute_type": ntype,
                "attribute_label": label,
                "attribute_normalized_label": norm,
                "edge_type": r.get("edge_type", ""),
                "weight": float(r.get("weight", 1.0) if pd.notna(r.get("weight", 1.0)) else 1.0),
                "confidence": float(r.get("confidence", 1.0) if pd.notna(r.get("confidence", 1.0)) else 1.0),
                "idf": float(r.get("idf", 1.0) if pd.notna(r.get("idf", 1.0)) else 1.0),
                "evidence": r.get("evidence", ""),
                "raw_value": r.get("raw_value", ""),
                "df_comments": df_comments,
            }
        )
    return pd.DataFrame(rows)


def score_attribute_bridges(
    ca_edges: pd.DataFrame,
    units: pd.DataFrame,
    cfg: BridgeConfig,
) -> pd.DataFrame:
    unit_cols = [
        "comment_id",
        "actor",
        "discursive_summary",
        "raw_comment_preview",
        "video_id",
        "video_title",
        "time_bucket",
        "channel",
        "macro_frame",
        "frame_primary",
        "argument_family_controlled",
        "tone_controlled",
    ]
    available = [c for c in unit_cols if c in units.columns]
    merged = ca_edges.merge(units[available], on="comment_id", how="left")
    merged["actor"] = merged["actor"].astype(str)
    merged = merged[merged["actor"].isin([cfg.source_a, cfg.source_b])].copy()

    records = []
    if merged.empty:
        return pd.DataFrame()

    for anode, g in merged.groupby("attribute_node_id"):
        a = g[g["actor"] == cfg.source_a]
        b = g[g["actor"] == cfg.source_b]
        a_comments = set(a["comment_id"].astype(str))
        b_comments = set(b["comment_id"].astype(str))
        cnt_a = len(a_comments)
        cnt_b = len(b_comments)
        if cnt_a < cfg.min_comments_per_side or cnt_b < cfg.min_comments_per_side:
            continue

        label = str(g["attribute_label"].iloc[0])
        norm = str(g["attribute_normalized_label"].iloc[0])
        df_comments = g["df_comments"].dropna()
        df_value = float(df_comments.iloc[0]) if len(df_comments) else len(set(g["comment_id"].astype(str)))

        w_a = float(a["weight"].sum())
        w_b = float(b["weight"].sum())
        total = w_a + w_b
        if total <= 0:
            continue
        balance = 2.0 * min(w_a, w_b) / total
        ntype = str(g["attribute_type"].iloc[0])
        type_weight = DEFAULT_TYPE_WEIGHTS.get(ntype, 1.0)
        hub_penalty = 1.0 / math.sqrt(max(df_value / max(min(cnt_a, cnt_b), 1), 1.0))
        # Prefer balanced, specific bridges and penalize corpus-wide hubs.
        bridge_score = math.sqrt(max(w_a, 0.0) * max(w_b, 0.0)) * balance * type_weight * hub_penalty
        bridge_score *= math.log1p(min(cnt_a, cnt_b))

        top_a = (
            a.sort_values("weight", ascending=False)
            .drop_duplicates("comment_id")
            .head(cfg.comments_per_bridge)
        )
        top_b = (
            b.sort_values("weight", ascending=False)
            .drop_duplicates("comment_id")
            .head(cfg.comments_per_bridge)
        )

        records.append(
            {
                "attribute_node_id": anode,
                "attribute_type": ntype,
                "attribute_label": label,
                "attribute_normalized_label": norm,
                "df_comments": df_value,
                f"n_{cfg.source_a}_comments": cnt_a,
                f"n_{cfg.source_b}_comments": cnt_b,
                f"weight_{cfg.source_a}": round(w_a, 6),
                f"weight_{cfg.source_b}": round(w_b, 6),
                "balance": round(balance, 6),
                "bridge_score": round(bridge_score, 6),
                f"top_{cfg.source_a}_comment_ids": json.dumps(top_a["comment_id"].tolist(), ensure_ascii=False),
                f"top_{cfg.source_b}_comment_ids": json.dumps(top_b["comment_id"].tolist(), ensure_ascii=False),
                f"top_{cfg.source_a}_summaries": json.dumps(top_a.get("discursive_summary", pd.Series()).dropna().astype(str).tolist(), ensure_ascii=False),
                f"top_{cfg.source_b}_summaries": json.dumps(top_b.get("discursive_summary", pd.Series()).dropna().astype(str).tolist(), ensure_ascii=False),
            }
        )

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values("bridge_score", ascending=False).reset_index(drop=True)
    return out


def direct_rn_lfi_similarity_edges(sim: pd.DataFrame, units: pd.DataFrame, cfg: BridgeConfig) -> pd.DataFrame:
    required = {"source_comment_id", "target_comment_id", "similarity"}
    if not required.issubset(sim.columns):
        return pd.DataFrame()

    meta_cols = [
        "comment_id",
        "actor",
        "discursive_summary",
        "raw_comment_preview",
        "video_id",
        "video_title",
        "time_bucket",
        "channel",
        "macro_frame",
        "frame_primary",
        "argument_family_controlled",
        "tone_controlled",
    ]
    meta_cols = [c for c in meta_cols if c in units.columns]
    meta = units[meta_cols].drop_duplicates("comment_id").copy()
    meta["comment_id"] = meta["comment_id"].astype(str)

    out = sim.copy()
    out["source_comment_id"] = out["source_comment_id"].astype(str)
    out["target_comment_id"] = out["target_comment_id"].astype(str)
    out = out.merge(meta.add_prefix("source_"), left_on="source_comment_id", right_on="source_comment_id", how="left")
    out = out.merge(meta.add_prefix("target_"), left_on="target_comment_id", right_on="target_comment_id", how="left")
    out = out[pd.to_numeric(out["similarity"], errors="coerce") >= cfg.similarity_threshold].copy()
    out["source_actor"] = out.get("source_actor", "").astype(str)
    out["target_actor"] = out.get("target_actor", "").astype(str)

    mask = (
        ((out["source_actor"] == cfg.source_a) & (out["target_actor"] == cfg.source_b))
        | ((out["source_actor"] == cfg.source_b) & (out["target_actor"] == cfg.source_a))
    )
    out = out[mask].copy()
    out = out.sort_values("similarity", ascending=False).reset_index(drop=True)
    return out


def make_comment_node_attrs(row: pd.Series, comment_node_id: str, source: str) -> dict:
    return {
        "node_type": "CommentNode",
        "label": f"{source} | {str(row.get('comment_id', 'comment'))[:8]}",
        "source": source,
        "comment_id": row.get("comment_id", ""),
        "video_id": row.get("video_id", ""),
        "video_title": row.get("video_title", ""),
        "time_bucket": row.get("time_bucket", ""),
        "channel": row.get("channel", ""),
        "summary": row.get("discursive_summary", ""),
        "raw_text": row.get("raw_comment_preview", ""),
        "macro_frame": row.get("macro_frame", ""),
        "frame_primary": row.get("frame_primary", ""),
        "argument_family": row.get("argument_family_controlled", ""),
        "tone": row.get("tone_controlled", ""),
    }


def add_comment_node(G: nx.Graph, row: pd.Series, comment_node_id: str, source: str) -> None:
    if comment_node_id not in G:
        G.add_node(comment_node_id, **make_comment_node_attrs(row, comment_node_id, source))


def build_bridge_subgraph(
    units: pd.DataFrame,
    ca_edges: pd.DataFrame,
    attr_bridges: pd.DataFrame,
    direct_bridges: pd.DataFrame,
    comment_id_to_node: Dict[str, str],
    nodes_lookup: Dict[str, dict],
    cfg: BridgeConfig,
) -> nx.Graph:
    G = nx.Graph()
    units_by_id = units.drop_duplicates("comment_id").set_index("comment_id", drop=False)

    selected_attrs = attr_bridges.head(cfg.top_attribute_bridges)
    selected_attr_nodes = set(selected_attrs["attribute_node_id"].astype(str).tolist()) if not selected_attrs.empty else set()

    # Add shared attribute bridge nodes and their top comments on both sides.
    for _, br in selected_attrs.iterrows():
        anode = str(br["attribute_node_id"])
        node_info = nodes_lookup.get(anode, {})
        G.add_node(
            anode,
            node_type=str(br["attribute_type"]),
            label=str(br["attribute_label"]),
            source="bridge_attribute",
            bridge_score=float(br["bridge_score"]),
            n_source_a=int(br[f"n_{cfg.source_a}_comments"]),
            n_source_b=int(br[f"n_{cfg.source_b}_comments"]),
            df_comments=float(br.get("df_comments", 0.0)),
            normalized_label=str(br.get("attribute_normalized_label", "")),
        )

        for side in [cfg.source_a, cfg.source_b]:
            ids_json = br.get(f"top_{side}_comment_ids", "[]")
            try:
                comment_ids = json.loads(ids_json)
            except Exception:
                comment_ids = []
            for cid in comment_ids:
                cid = str(cid)
                if cid not in units_by_id.index:
                    continue
                cnode = comment_id_to_node.get(cid, "CommentNode:" + cid)
                add_comment_node(G, units_by_id.loc[cid], cnode, side)
                # Use the original comment-attribute edge weight if available.
                ce = ca_edges[(ca_edges["comment_id"].astype(str) == cid) & (ca_edges["attribute_node_id"].astype(str) == anode)]
                w = float(ce["weight"].max()) if not ce.empty else 1.0
                G.add_edge(cnode, anode, weight=w, edge_type="SHARED_ATTRIBUTE", bridge_score=float(br["bridge_score"]))

    # Add top direct cross-source similarity edges and their comment nodes.
    for _, r in direct_bridges.head(cfg.top_direct_edges).iterrows():
        s_cid = str(r["source_comment_id"])
        t_cid = str(r["target_comment_id"])
        for cid, actor_prefix in [(s_cid, "source"), (t_cid, "target")]:
            if cid not in units_by_id.index:
                continue
            cnode = comment_id_to_node.get(cid, "CommentNode:" + cid)
            actor = str(units_by_id.loc[cid].get("actor", "unknown"))
            add_comment_node(G, units_by_id.loc[cid], cnode, actor)
        s_node = comment_id_to_node.get(s_cid, "CommentNode:" + s_cid)
        t_node = comment_id_to_node.get(t_cid, "CommentNode:" + t_cid)
        if s_node in G and t_node in G:
            G.add_edge(s_node, t_node, weight=float(r["similarity"]), edge_type="DIRECT_RN_LFI_SIMILARITY")

    # Add degree metadata.
    degrees = dict(G.degree())
    nx.set_node_attributes(G, degrees, "degree")
    return G


def html_escape(value: object) -> str:
    # Small local escape to avoid pulling jinja; PyVis titles are inserted as HTML.
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def tooltip_table(attrs: dict, max_chars: int = 1600) -> str:
    """
    Return a readable plain-text tooltip for PyVis / vis-network.

    Important: do NOT return HTML here. In some PyVis/vis-network versions,
    HTML titles are escaped, which makes tags such as <table>, <tr> and <br>
    appear literally in the tooltip. Plain text + CSS white-space: pre-wrap is
    more robust.
    """
    preferred = [
        "node_type",
        "source",
        "label",
        "bridge_score",
        "n_source_a",
        "n_source_b",
        "df_comments",
        "comment_id",
        "video_id",
        "time_bucket",
        "channel",
        "macro_frame",
        "frame_primary",
        "argument_family",
        "tone",
        "summary",
        "raw_text",
    ]
    lines = []
    for key in preferred:
        if key not in attrs:
            continue
        value = attrs.get(key)
        if value is None or normalize_text(value) in {"", "nan", "none"}:
            continue
        value_text = short_text(value, max_chars=max_chars)
        # Remove accidental HTML from previous runs/fields and normalize whitespace.
        value_text = re.sub(r"<br\s*/?>", "\n", value_text, flags=re.IGNORECASE)
        value_text = re.sub(r"<[^>]+>", "", value_text)
        value_text = value_text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        value_text = re.sub(r"[ \t]+", " ", value_text).strip()

        if key in {"summary", "raw_text"} and len(value_text) > 120:
            lines.append(f"{key}:\n{value_text}")
        else:
            lines.append(f"{key}: {value_text}")
    return "\n".join(lines)


def node_label(node_id: str, attrs: dict) -> str:
    ntype = attrs.get("node_type", "")
    if ntype == "CommentNode":
        return short_text(attrs.get("label", "comment"), 22)
    label = attrs.get("label", node_id)
    return short_text(label, 38)


def node_color(attrs: dict) -> str:
    ntype = attrs.get("node_type", "")
    if ntype == "CommentNode":
        return SOURCE_COLORS.get(str(attrs.get("source", "unknown")), SOURCE_COLORS["unknown"])
    if str(attrs.get("source", "")) == "bridge_attribute":
        # Keep bridge attributes visually distinct but neutral.
        return "#f2c94c"
    return SOURCE_COLORS.get(str(attrs.get("source", "attribute")), SOURCE_COLORS["attribute"])


def node_size(attrs: dict) -> int:
    ntype = attrs.get("node_type", "")
    if ntype == "CommentNode":
        return 10
    score = float(attrs.get("bridge_score", 1.0) or 1.0)
    return max(14, min(42, int(14 + 5 * math.log1p(score))))


def patch_pyvis_html(html_path: Path) -> None:
    if not html_path.exists():
        return
    html = html_path.read_text(encoding="utf-8")
    if "__OBS_BRIDGE_TOOLTIP_CSS__" in html:
        css = ""
    else:
        css = """
<style id="__OBS_BRIDGE_TOOLTIP_CSS__">
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
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif !important;
  padding: 10px 12px !important;
  z-index: 999999 !important;
}
</style>
"""
    if css:
        html = html.replace("</head>", css + "\n</head>")
    # Hard freeze physics if vis-network starts moving.
    marker = "network = new vis.Network(container, data, options);"
    patch = """network = new vis.Network(container, data, options);
network.once('stabilizationIterationsDone', function () {
  network.stopSimulation();
  network.setOptions({ physics: { enabled: false } });
});
setTimeout(function () {
  network.stopSimulation();
  network.setOptions({ physics: { enabled: false } });
}, 2000);
"""
    if marker in html and "network.stopSimulation();" not in html:
        html = html.replace(marker, patch)
    html_path.write_text(html, encoding="utf-8")


def write_pyvis(G: nx.Graph, output_html: Path, cfg: BridgeConfig) -> None:
    if Network is None:
        print("pyvis is not installed; skipping HTML visualization. Install with: pip install pyvis")
        return

    net = Network(
        height="850px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        directed=False,
        cdn_resources="in_line",
    )
    net.toggle_physics(False)

    # Fixed layout computed in Python.
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, seed=cfg.seed, iterations=cfg.layout_iterations, weight="weight", scale=1.0)
    else:
        pos = {}

    for node, attrs in G.nodes(data=True):
        x, y = pos.get(node, (0.0, 0.0))
        ntype = str(attrs.get("node_type", ""))
        net.add_node(
            node,
            label=node_label(node, attrs),
            title=tooltip_table(attrs),
            color=node_color(attrs),
            shape=NODE_SHAPES.get(ntype, "dot"),
            size=node_size(attrs),
            x=float(x * cfg.layout_scale),
            y=float(y * cfg.layout_scale),
            physics=False,
        )

    for u, v, attrs in G.edges(data=True):
        etype = str(attrs.get("edge_type", ""))
        w = float(attrs.get("weight", 1.0) or 1.0)
        if etype == "DIRECT_RN_LFI_SIMILARITY":
            color = "#8e44ad"
            width = 1.5 + 4.0 * min(max(w, 0), 1)
            label = "sim"
        else:
            color = "#999999"
            width = 1.0 + min(3.0, math.log1p(max(w, 0)))
            label = ""
        net.add_edge(
            u,
            v,
            value=w,
            width=width,
            color=color,
            title=f"{etype}\nweight: {w:.4f}",
            label=label,
            physics=False,
            smooth=False,
        )

    net.set_options(
        """
var options = {
  "physics": {"enabled": false},
  "edges": {"smooth": false},
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "keyboard": true,
    "dragNodes": true,
    "tooltipDelay": 120
  },
  "nodes": {
    "borderWidth": 1,
    "font": {"size": 12}
  }
}
"""
    )
    net.write_html(str(output_html), notebook=False, open_browser=False)
    patch_pyvis_html(output_html)


def diagnostics(
    units: pd.DataFrame,
    ca_edges: pd.DataFrame,
    attr_bridges: pd.DataFrame,
    direct_bridges: pd.DataFrame,
    G: nx.Graph,
    cfg: BridgeConfig,
) -> dict:
    return {
        "source_a": cfg.source_a,
        "source_b": cfg.source_b,
        "units_total": int(len(units)),
        "units_by_actor": units.get("actor", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
        "comment_attribute_edges_after_filtering": int(len(ca_edges)),
        "attribute_bridges": int(len(attr_bridges)),
        "direct_similarity_bridges": int(len(direct_bridges)),
        "bridge_graph_nodes": int(G.number_of_nodes()),
        "bridge_graph_edges": int(G.number_of_edges()),
        "bridge_graph_components": int(nx.number_connected_components(G)) if G.number_of_nodes() else 0,
        "type_weights": DEFAULT_TYPE_WEIGHTS,
        "top_attribute_types": attr_bridges.get("attribute_type", pd.Series(dtype=str)).value_counts().head(20).to_dict() if not attr_bridges.empty else {},
        "top_bridge_attributes": attr_bridges.head(20)[
            ["attribute_type", "attribute_label", "bridge_score", f"n_{cfg.source_a}_comments", f"n_{cfg.source_b}_comments"]
        ].to_dict(orient="records") if not attr_bridges.empty else [],
    }


def parse_node_types(value: str) -> Set[str]:
    if not value:
        return set(DEFAULT_ATTRIBUTE_TYPES)
    return {x.strip() for x in value.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize main RN-LFI discursive bridges from already-encoded graph outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs"), help="Directory containing discursive_*.csv files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rn_lfi_bridges"), help="Output directory")
    parser.add_argument("--source-a", default="RN", help="First actor/source, default RN")
    parser.add_argument("--source-b", default="LFI", help="Second actor/source, default LFI")
    parser.add_argument("--similarity-threshold", type=float, default=0.50, help="Minimum direct comment similarity to keep")
    parser.add_argument("--top-direct-edges", type=int, default=80, help="Number of direct RN-LFI similarity edges to visualize")
    parser.add_argument("--top-attribute-bridges", type=int, default=40, help="Number of shared bridge attributes to visualize")
    parser.add_argument("--comments-per-bridge", type=int, default=3, help="Top comments per source connected to each bridge attribute")
    parser.add_argument("--min-comments-per-side", type=int, default=2, help="Minimum comments from each source for an attribute to count as a bridge")
    parser.add_argument("--max-attribute-df", type=int, default=250, help="Drop attribute nodes connected to more than this many comments. Use 0 to disable.")
    parser.add_argument("--include-node-types", default=",".join(sorted(DEFAULT_ATTRIBUTE_TYPES)), help="Comma-separated attribute node types to use")
    parser.add_argument("--keep-non-signal-labels", action="store_true", help="Keep labels like other/unknown/unclear in bridge scoring")
    parser.add_argument("--layout-iterations", type=int, default=400, help="NetworkX spring layout iterations")
    parser.add_argument("--layout-scale", type=float, default=2800.0, help="Scale factor for fixed PyVis layout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for layout")
    args = parser.parse_args()

    cfg = BridgeConfig(
        source_a=args.source_a,
        source_b=args.source_b,
        similarity_threshold=args.similarity_threshold,
        top_direct_edges=args.top_direct_edges,
        top_attribute_bridges=args.top_attribute_bridges,
        comments_per_bridge=args.comments_per_bridge,
        min_comments_per_side=args.min_comments_per_side,
        max_attribute_df=None if args.max_attribute_df == 0 else args.max_attribute_df,
        include_node_types=parse_node_types(args.include_node_types),
        drop_non_signal_labels=not args.keep_non_signal_labels,
        layout_iterations=args.layout_iterations,
        layout_scale=args.layout_scale,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_inputs(args.input_dir)
    units_raw = data["units"]
    nodes = data["nodes"]
    edges = data["edges"]
    sim = data["sim"]

    comment_id_to_node = build_comment_node_map(edges)
    comment_node_to_id = {v: k for k, v in comment_id_to_node.items()}
    nodes_lookup = build_node_lookup(nodes)
    units = prepare_units(units_raw, comment_id_to_node)

    ca_edges = comment_attribute_edges(edges, nodes_lookup, comment_node_to_id, cfg)
    attr_bridges = score_attribute_bridges(ca_edges, units, cfg)
    direct_bridges = direct_rn_lfi_similarity_edges(sim, units, cfg)
    G = build_bridge_subgraph(units, ca_edges, attr_bridges, direct_bridges, comment_id_to_node, nodes_lookup, cfg)

    # Write outputs.
    attr_bridges.to_csv(args.output_dir / "rn_lfi_attribute_bridges.csv", index=False)
    direct_bridges.to_csv(args.output_dir / "rn_lfi_direct_similarity_bridges.csv", index=False)

    node_rows = []
    for n, a in G.nodes(data=True):
        row = {"node_id": n, **a}
        node_rows.append(row)
    pd.DataFrame(node_rows).to_csv(args.output_dir / "rn_lfi_bridge_subgraph_nodes.csv", index=False)

    edge_rows = []
    for u, v, a in G.edges(data=True):
        edge_rows.append({"source": u, "target": v, **a})
    pd.DataFrame(edge_rows).to_csv(args.output_dir / "rn_lfi_bridge_subgraph_edges.csv", index=False)

    diag = diagnostics(units, ca_edges, attr_bridges, direct_bridges, G, cfg)
    (args.output_dir / "rn_lfi_bridge_diagnostics.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")

    if G.number_of_nodes() > 0:
        nx.write_gexf(G, args.output_dir / "rn_lfi_bridge_graph.gexf")
        write_pyvis(G, args.output_dir / "rn_lfi_bridge_graph.html", cfg)

    print(json.dumps(diag, indent=2, ensure_ascii=False))
    print(f"\nWrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
