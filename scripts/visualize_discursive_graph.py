#!/usr/bin/env python3
"""Create interactive HTML visualizations for the V2.6.3 discursive graph.

The script reads graph exports from outputs/ and writes:
- outputs/discursive_full_graph_by_source.html
- outputs/discursive_filtered_graph_by_source.html
- outputs/discursive_comment_projection_by_source.html, when similarity edges exist
"""

from __future__ import annotations

import argparse
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


REQUIRED_FILES = {
    "nodes": "discursive_nodes.csv",
    "edges": "discursive_edges.csv",
}

OPTIONAL_FILES = {
    "units": "discursive_units.csv",
    "similarity_edges": "discursive_similarity_edges.csv",
    "communities": "discursive_communities.csv",
}

SOURCE_PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
    "#7f7f7f",
]

NEUTRAL_ATTRIBUTE_COLOR = "#d9d9d9"
MIXED_COLOR = "#969696"
UNKNOWN_COLOR = "#bdbdbd"

NODE_TYPE_SYMBOLS = {
    "CommentNode": "circle",
    "VideoNode": "square",
    "SourceActorNode": "diamond",
    "ThemeNode": "hexagon",
    "FrameNode": "triangle-up",
    "ClaimNode": "triangle-down",
    "CanonicalClaimNode": "star",
    "ArgumentFamilyNode": "pentagon",
    "TargetNode": "cross",
    "StanceTargetNode": "x",
    "ToneNode": "circle-open",
    "RegisterNode": "square-open",
    "TimeNode": "hourglass",
    "ChannelNode": "bowtie",
}


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, columns: Sequence[str], filename: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{filename} manque les colonnes requises : {', '.join(missing)}")


def load_graph_tables(outputs_dir: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in REQUIRED_FILES.items():
        path = outputs_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Fichier requis absent : {path}")
        tables[key] = pd.read_csv(path)

    for key, filename in OPTIONAL_FILES.items():
        tables[key] = read_csv_if_exists(outputs_dir / filename)

    require_columns(tables["nodes"], ["node_id", "node_type", "label"], REQUIRED_FILES["nodes"])
    require_columns(tables["edges"], ["source_node_id", "target_node_id", "edge_type"], REQUIRED_FILES["edges"])
    return tables


def clean_string(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and math.isnan(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def split_csv_arg(value: Optional[str]) -> Optional[Set[str]]:
    if value is None or not value.strip():
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def build_comment_maps(nodes_df: pd.DataFrame, units_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    comment_node_to_comment_id: Dict[str, str] = {}
    comment_id_to_node: Dict[str, str] = {}

    comment_nodes = nodes_df[nodes_df["node_type"] == "CommentNode"] if "node_type" in nodes_df.columns else pd.DataFrame()
    for _, node in comment_nodes.iterrows():
        node_id = clean_string(node.get("node_id"))
        comment_id = clean_string(node.get("label"))
        if node_id and comment_id:
            comment_node_to_comment_id[node_id] = comment_id
            comment_id_to_node[comment_id] = node_id

    unit_metadata: Dict[str, Any] = {}
    if not units_df.empty and "comment_id" in units_df.columns:
        source_column = first_existing_column(units_df, ["source_actor", "actor"])
        for _, unit in units_df.iterrows():
            comment_id = clean_string(unit.get("comment_id"))
            if not comment_id:
                continue
            unit_metadata[comment_id] = {
                "source": clean_string(unit.get(source_column), "unknown") if source_column else "unknown",
                "community": unit.get("discursive_community", ""),
                "summary": clean_string(unit.get("discursive_summary")),
                "preview": clean_string(unit.get("raw_comment_preview")),
                "video_title": clean_string(unit.get("video_title")),
                "time_bucket": clean_string(unit.get("time_bucket")),
            }
            if comment_id not in comment_id_to_node:
                node_id = f"CommentNode:{comment_id}"
                comment_id_to_node[comment_id] = node_id
                comment_node_to_comment_id[node_id] = comment_id

    return comment_node_to_comment_id, comment_id_to_node, unit_metadata


def collect_comment_sources_by_node(
    edges_df: pd.DataFrame,
    comment_node_to_comment_id: Dict[str, str],
    unit_metadata: Dict[str, Any],
) -> Dict[str, Counter]:
    sources_by_node: Dict[str, Counter] = defaultdict(Counter)

    for node_id, comment_id in comment_node_to_comment_id.items():
        source = unit_metadata.get(comment_id, {}).get("source", "unknown")
        sources_by_node[node_id][source] += 1

    if "comment_id" not in edges_df.columns:
        return sources_by_node

    for _, edge in edges_df.iterrows():
        comment_id = clean_string(edge.get("comment_id"))
        if not comment_id or comment_id not in unit_metadata:
            continue
        source = unit_metadata[comment_id].get("source", "unknown")
        for node_column in ["source_node_id", "target_node_id"]:
            node_id = clean_string(edge.get(node_column))
            if node_id:
                sources_by_node[node_id][source] += 1

    return sources_by_node


def infer_node_source(
    node_id: str,
    node_attrs: Dict[str, Any],
    units_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    comment_node_to_comment_id: Dict[str, str],
    unit_metadata: Dict[str, Any],
    sources_by_node: Dict[str, Counter],
) -> str:
    """Infer the source associated with a graph node.

    Priority:
    1. CommentNode -> source_actor from discursive_units.csv.
    2. SourceActorNode -> own label.
    3. VideoNode -> dominant source among linked comments.
    4. Other attribute nodes -> dominant source when unique, "mixed" when shared, else "attribute".
    """
    node_type = clean_string(node_attrs.get("node_type"))
    label = clean_string(node_attrs.get("label"))

    if node_type == "CommentNode":
        comment_id = comment_node_to_comment_id.get(node_id, label)
        return unit_metadata.get(comment_id, {}).get("source", "unknown")

    if node_type == "SourceActorNode":
        return label or "unknown"

    source_counts = sources_by_node.get(node_id, Counter())
    if not source_counts:
        return "unknown" if node_type == "VideoNode" else "attribute"

    if len(source_counts) == 1:
        return next(iter(source_counts))

    if node_type == "VideoNode":
        return source_counts.most_common(1)[0][0]

    return "mixed"


def node_color_group(node_type: str, inferred_source: str, color_attributes_by_source: bool) -> str:
    if node_type in {"CommentNode", "SourceActorNode", "VideoNode"}:
        return inferred_source or "unknown"
    if color_attributes_by_source and inferred_source not in {"attribute", "unknown"}:
        return inferred_source
    return "attribute" if inferred_source != "mixed" else "mixed"


def source_colors(sources: Iterable[str]) -> Dict[str, str]:
    unique_sources = sorted({source or "unknown" for source in sources})
    colors: Dict[str, str] = {}
    palette_index = 0
    for source in unique_sources:
        if source == "attribute":
            colors[source] = NEUTRAL_ATTRIBUTE_COLOR
        elif source == "mixed":
            colors[source] = MIXED_COLOR
        elif source == "unknown":
            colors[source] = UNKNOWN_COLOR
        else:
            colors[source] = SOURCE_PALETTE[palette_index % len(SOURCE_PALETTE)]
            palette_index += 1
    return colors


def add_nodes_to_graph(
    graph: nx.Graph,
    nodes_df: pd.DataFrame,
    units_df: pd.DataFrame,
    comment_node_to_comment_id: Dict[str, str],
    unit_metadata: Dict[str, Any],
    sources_by_node: Dict[str, Counter],
    color_attributes_by_source: bool,
) -> None:
    for _, node in nodes_df.iterrows():
        node_id = clean_string(node.get("node_id"))
        if not node_id:
            continue
        attrs = node.to_dict()
        inferred_source = infer_node_source(
            node_id=node_id,
            node_attrs=attrs,
            units_df=units_df,
            edges_df=pd.DataFrame(),
            comment_node_to_comment_id=comment_node_to_comment_id,
            unit_metadata=unit_metadata,
            sources_by_node=sources_by_node,
        )
        attrs["source"] = inferred_source
        attrs["color_group"] = node_color_group(clean_string(attrs.get("node_type")), inferred_source, color_attributes_by_source)
        comment_id = comment_node_to_comment_id.get(node_id)
        if comment_id:
            attrs.update(unit_metadata.get(comment_id, {}))
        graph.add_node(node_id, **attrs)

    for node_id, comment_id in comment_node_to_comment_id.items():
        if graph.has_node(node_id):
            continue
        metadata = unit_metadata.get(comment_id, {})
        graph.add_node(
            node_id,
            node_id=node_id,
            node_type="CommentNode",
            label=comment_id,
            source=metadata.get("source", "unknown"),
            color_group=metadata.get("source", "unknown"),
            **metadata,
        )


def edge_weight(edge: pd.Series, fallback: float = 1.0) -> float:
    for column in ["weight", "similarity"]:
        if column in edge.index:
            try:
                value = float(edge.get(column))
            except (TypeError, ValueError):
                continue
            if not math.isnan(value):
                return value
    return fallback


def add_structural_edges(graph: nx.Graph, edges_df: pd.DataFrame, min_edge_weight: float) -> None:
    for _, edge in edges_df.iterrows():
        source = clean_string(edge.get("source_node_id"))
        target = clean_string(edge.get("target_node_id"))
        if not source or not target or source == target:
            continue
        weight = edge_weight(edge)
        if weight < min_edge_weight:
            continue
        if not graph.has_node(source):
            graph.add_node(source, node_id=source, node_type="UnknownNode", label=source, source="unknown", color_group="unknown")
        if not graph.has_node(target):
            graph.add_node(target, node_id=target, node_type="UnknownNode", label=target, source="unknown", color_group="unknown")
        graph.add_edge(
            source,
            target,
            edge_type=clean_string(edge.get("edge_type"), "UNKNOWN_EDGE"),
            weight=weight,
            comment_id=clean_string(edge.get("comment_id")),
            evidence=clean_string(edge.get("evidence")),
            is_similarity=False,
        )


def add_similarity_edges(
    graph: nx.Graph,
    similarity_edges_df: pd.DataFrame,
    comment_id_to_node: Dict[str, str],
    min_similarity: float,
) -> None:
    if similarity_edges_df.empty:
        return
    require_columns(
        similarity_edges_df,
        ["source_comment_id", "target_comment_id", "similarity"],
        "discursive_similarity_edges.csv",
    )
    for _, edge in similarity_edges_df.iterrows():
        source_comment = clean_string(edge.get("source_comment_id"))
        target_comment = clean_string(edge.get("target_comment_id"))
        source = comment_id_to_node.get(source_comment)
        target = comment_id_to_node.get(target_comment)
        if not source or not target or source == target:
            continue
        similarity = edge_weight(edge)
        if similarity < min_similarity:
            continue
        if graph.has_node(source) and graph.has_node(target):
            graph.add_edge(
                source,
                target,
                edge_type="COMMENT_SIMILARITY",
                weight=similarity,
                similarity=similarity,
                is_similarity=True,
            )


def build_graph(
    tables: Dict[str, pd.DataFrame],
    min_edge_weight: float,
    include_similarity_edges: bool,
    min_similarity: float,
    color_attributes_by_source: bool,
) -> Tuple[nx.Graph, Dict[str, str], Dict[str, Any]]:
    nodes_df = tables["nodes"].copy()
    edges_df = tables["edges"].copy()
    units_df = tables.get("units", pd.DataFrame()).copy()
    similarity_edges_df = tables.get("similarity_edges", pd.DataFrame()).copy()

    comment_node_to_comment_id, comment_id_to_node, unit_metadata = build_comment_maps(nodes_df, units_df)
    sources_by_node = collect_comment_sources_by_node(edges_df, comment_node_to_comment_id, unit_metadata)

    graph = nx.Graph()
    add_nodes_to_graph(
        graph,
        nodes_df,
        units_df,
        comment_node_to_comment_id,
        unit_metadata,
        sources_by_node,
        color_attributes_by_source=color_attributes_by_source,
    )
    add_structural_edges(graph, edges_df, min_edge_weight=min_edge_weight)
    if include_similarity_edges:
        add_similarity_edges(
            graph,
            similarity_edges_df,
            comment_id_to_node=comment_id_to_node,
            min_similarity=min_similarity,
        )
    return graph, comment_id_to_node, unit_metadata


def filter_graph(
    graph: nx.Graph,
    min_edge_weight: float = 0.0,
    keep_node_types: Optional[Set[str]] = None,
    drop_high_degree_percentile: Optional[float] = None,
    include_similarity_edges: bool = True,
    max_nodes: Optional[int] = None,
) -> nx.Graph:
    filtered = nx.Graph()

    for node_id, attrs in graph.nodes(data=True):
        node_type = clean_string(attrs.get("node_type"))
        if keep_node_types and node_type not in keep_node_types:
            continue
        filtered.add_node(node_id, **attrs)

    for source, target, attrs in graph.edges(data=True):
        if source not in filtered or target not in filtered:
            continue
        if attrs.get("is_similarity") and not include_similarity_edges:
            continue
        if float(attrs.get("weight", 0.0)) < min_edge_weight:
            continue
        filtered.add_edge(source, target, **attrs)

    if drop_high_degree_percentile is not None and filtered.number_of_nodes() > 0:
        percentile = max(0.0, min(100.0, float(drop_high_degree_percentile)))
        degree_values = [degree for _, degree in filtered.degree()]
        cutoff = pd.Series(degree_values).quantile(percentile / 100.0)
        high_degree_nodes = [
            node_id
            for node_id, degree in filtered.degree()
            if degree > cutoff and clean_string(filtered.nodes[node_id].get("node_type")) != "CommentNode"
        ]
        filtered.remove_nodes_from(high_degree_nodes)

    if max_nodes is not None and filtered.number_of_nodes() > max_nodes:
        ranked = sorted(filtered.degree(), key=lambda item: item[1], reverse=True)
        keep = {node_id for node_id, _ in ranked[:max_nodes]}
        filtered = filtered.subgraph(keep).copy()

    isolates = list(nx.isolates(filtered))
    filtered.remove_nodes_from(isolates)
    return filtered


def graph_analytics(graph: nx.Graph, title: str, max_betweenness_nodes: int = 1200) -> None:
    print(f"\n=== {title} ===")
    print(f"Nœuds : {graph.number_of_nodes()}")
    print(f"Arêtes : {graph.number_of_edges()}")
    components = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
    print(f"Composantes connexes : {components}")

    node_type_counts = Counter(clean_string(attrs.get("node_type"), "unknown") for _, attrs in graph.nodes(data=True))
    source_counts = Counter(clean_string(attrs.get("source"), "unknown") for _, attrs in graph.nodes(data=True))
    community_counts = Counter(
        clean_string(attrs.get("community"), "")
        for _, attrs in graph.nodes(data=True)
        if clean_string(attrs.get("community"), "") != ""
    )

    print("\nDistribution des types de nœuds :")
    for node_type, count in node_type_counts.most_common():
        print(f"  - {node_type}: {count}")

    print("\nDistribution des sources inférées :")
    for source, count in source_counts.most_common():
        print(f"  - {source}: {count}")

    print("\nTop 20 nœuds par degré :")
    for node_id, degree in sorted(graph.degree(), key=lambda item: item[1], reverse=True)[:20]:
        attrs = graph.nodes[node_id]
        label = clean_string(attrs.get("label"), node_id)
        node_type = clean_string(attrs.get("node_type"), "unknown")
        source = clean_string(attrs.get("source"), "unknown")
        print(f"  - {degree:>4} | {node_type:<22} | {source:<12} | {label[:100]}")

    if graph.number_of_nodes() <= max_betweenness_nodes and graph.number_of_edges() > 0:
        print("\nTop 20 nœuds-ponts par betweenness :")
        distance_graph = graph.copy()
        for source, target, attrs in distance_graph.edges(data=True):
            strength = max(float(attrs.get("weight", 0.0)), 1e-6)
            attrs["distance"] = 1.0 / strength
        betweenness = nx.betweenness_centrality(distance_graph, weight="distance", normalized=True)
        for node_id, score in sorted(betweenness.items(), key=lambda item: item[1], reverse=True)[:20]:
            attrs = graph.nodes[node_id]
            label = clean_string(attrs.get("label"), node_id)
            node_type = clean_string(attrs.get("node_type"), "unknown")
            source = clean_string(attrs.get("source"), "unknown")
            print(f"  - {score:0.4f} | {node_type:<22} | {source:<12} | {label[:100]}")
    else:
        print("\nBetweenness ignorée : graphe trop grand ou sans arête.")

    if community_counts:
        print(f"\nNombre de communautés disponibles : {len(community_counts)}")
        for community, count in community_counts.most_common(20):
            print(f"  - communauté {community}: {count} nœud(s)")
    else:
        print("\nCommunautés : non disponibles dans les nœuds/unités chargés.")


def node_hover_text(node_id: str, attrs: Dict[str, Any], degree: int) -> str:
    fields = [
        ("node_id", node_id),
        ("node_type", attrs.get("node_type")),
        ("label", attrs.get("label")),
        ("source", attrs.get("source")),
        ("community", attrs.get("community")),
        ("degree", degree),
        ("df_comments", attrs.get("df_comments")),
        ("time_bucket", attrs.get("time_bucket")),
        ("video_title", attrs.get("video_title")),
        ("summary", attrs.get("summary")),
        ("preview", attrs.get("preview")),
    ]
    lines = []
    for key, value in fields:
        text = clean_string(value)
        if text:
            wrapped = "<br>".join(textwrap.wrap(text, width=90))
            lines.append(f"<b>{key}</b>: {wrapped}")
    return "<br>".join(lines)


def compute_layout(graph: nx.Graph, seed: int, layout_iterations: int) -> Dict[str, Tuple[float, float]]:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_nodes() == 1:
        node_id = next(iter(graph.nodes()))
        return {node_id: (0.0, 0.0)}
    return nx.spring_layout(graph, seed=seed, iterations=layout_iterations, weight="weight")


def make_edge_traces(graph: nx.Graph, positions: Dict[str, Tuple[float, float]]) -> List[go.Scatter]:
    structural_x: List[Optional[float]] = []
    structural_y: List[Optional[float]] = []
    similarity_x: List[Optional[float]] = []
    similarity_y: List[Optional[float]] = []

    for source, target, attrs in graph.edges(data=True):
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        target_x = similarity_x if attrs.get("is_similarity") else structural_x
        target_y = similarity_y if attrs.get("is_similarity") else structural_y
        target_x.extend([x0, x1, None])
        target_y.extend([y0, y1, None])

    traces: List[go.Scatter] = []
    if structural_x:
        traces.append(
            go.Scatter(
                x=structural_x,
                y=structural_y,
                mode="lines",
                line={"width": 0.7, "color": "rgba(120,120,120,0.35)"},
                hoverinfo="skip",
                name="relations discursives",
            )
        )
    if similarity_x:
        traces.append(
            go.Scatter(
                x=similarity_x,
                y=similarity_y,
                mode="lines",
                line={"width": 1.0, "color": "rgba(30,90,180,0.25)", "dash": "dot"},
                hoverinfo="skip",
                name="similarités commentaire-commentaire",
            )
        )
    return traces


def make_node_traces(
    graph: nx.Graph,
    positions: Dict[str, Tuple[float, float]],
    color_attributes_by_source: bool,
    node_size_base: float,
    node_size_scale: float,
    node_size_max_bonus: float,
) -> List[go.Scatter]:
    color_groups = [clean_string(attrs.get("color_group"), "unknown") for _, attrs in graph.nodes(data=True)]
    colors = source_colors(color_groups)
    grouped_nodes: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for node_id, attrs in graph.nodes(data=True):
        node_type = clean_string(attrs.get("node_type"), "UnknownNode")
        color_group = clean_string(attrs.get("color_group"), "unknown")
        if not color_attributes_by_source and node_type not in {"CommentNode", "SourceActorNode", "VideoNode"}:
            color_group = "attribute" if color_group != "mixed" else "mixed"
        grouped_nodes[(color_group, node_type)].append(node_id)

    traces: List[go.Scatter] = []
    for (color_group, node_type), node_ids in sorted(grouped_nodes.items(), key=lambda item: (item[0][0], item[0][1])):
        x_values: List[float] = []
        y_values: List[float] = []
        sizes: List[float] = []
        hovers: List[str] = []
        labels: List[str] = []
        for node_id in node_ids:
            if node_id not in positions:
                continue
            x, y = positions[node_id]
            degree = graph.degree(node_id)
            attrs = graph.nodes[node_id]
            x_values.append(x)
            y_values.append(y)
            sizes.append(node_size_base + min(node_size_max_bonus, node_size_scale * math.sqrt(max(degree, 1))))
            hovers.append(node_hover_text(node_id, attrs, degree))
            label = clean_string(attrs.get("label"), node_id)
            labels.append(label[:28] + "…" if len(label) > 28 else label)
        if not x_values:
            continue
        traces.append(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                marker={
                    "size": sizes,
                    "color": colors.get(color_group, UNKNOWN_COLOR),
                    "symbol": NODE_TYPE_SYMBOLS.get(node_type, "circle"),
                    "line": {"width": 0.7, "color": "#333333"},
                    "opacity": 0.88,
                },
                text=labels,
                hovertext=hovers,
                hoverinfo="text",
                name=f"{color_group} · {node_type}",
            )
        )
    return traces


def write_graph_html(
    graph: nx.Graph,
    output_path: Path,
    title: str,
    seed: int,
    layout_iterations: int,
    color_attributes_by_source: bool,
    node_size_base: float,
    node_size_scale: float,
    node_size_max_bonus: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions = compute_layout(graph, seed=seed, layout_iterations=layout_iterations)
    traces = [
        *make_edge_traces(graph, positions),
        *make_node_traces(
            graph,
            positions,
            color_attributes_by_source,
            node_size_base=node_size_base,
            node_size_scale=node_size_scale,
            node_size_max_bonus=node_size_max_bonus,
        ),
    ]
    figure = go.Figure(data=traces)
    figure.update_layout(
        title={
            "text": f"{title}<br><sup>{graph.number_of_nodes()} nœuds · {graph.number_of_edges()} arêtes</sup>",
            "x": 0.02,
            "xanchor": "left",
        },
        showlegend=True,
        hovermode="closest",
        margin={"b": 20, "l": 10, "r": 10, "t": 70},
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        legend={"itemsizing": "constant"},
        template="plotly_white",
    )
    figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    print(f"\nHTML écrit : {output_path}")


def build_comment_projection(graph: nx.Graph) -> nx.Graph:
    projection = nx.Graph()
    for node_id, attrs in graph.nodes(data=True):
        if clean_string(attrs.get("node_type")) == "CommentNode":
            projection.add_node(node_id, **attrs)
    for source, target, attrs in graph.edges(data=True):
        if not attrs.get("is_similarity"):
            continue
        if source in projection and target in projection:
            projection.add_edge(source, target, **attrs)
    isolates = list(nx.isolates(projection))
    projection.remove_nodes_from(isolates)
    return projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise le graphe discursif V2.6.3 par source.")
    parser.add_argument("--outputs-dir", default="outputs", help="Dossier contenant les exports discursive_*.csv.")
    parser.add_argument("--output-html", default=None, help="Chemin HTML du graphe complet.")
    parser.add_argument("--filtered-output-html", default=None, help="Chemin HTML du graphe filtré.")
    parser.add_argument("--projection-html", default=None, help="Chemin HTML de la projection commentaire-commentaire.")
    parser.add_argument("--min-edge-weight", type=float, default=0.0, help="Seuil de poids pour le graphe complet.")
    parser.add_argument("--filtered-min-edge-weight", type=float, default=0.05, help="Seuil de poids du graphe filtré.")
    parser.add_argument("--min-similarity", type=float, default=0.0, help="Seuil pour les arêtes de similarité.")
    parser.add_argument("--keep-node-types", default=None, help="Liste CSV de node types à conserver dans le graphe filtré.")
    parser.add_argument(
        "--drop-high-degree-percentile",
        type=float,
        default=98.0,
        help="Percentile de degré au-delà duquel retirer les hubs non-commentaires dans le graphe filtré.",
    )
    parser.add_argument("--max-nodes", type=int, default=None, help="Nombre maximal de nœuds dans le graphe complet.")
    parser.add_argument("--filtered-max-nodes", type=int, default=800, help="Nombre maximal de nœuds dans le graphe filtré.")
    parser.add_argument("--include-similarity-edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-comment-projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--color-attributes-by-source", action="store_true", help="Colorer aussi les attributs non mixtes par source.")
    parser.add_argument("--node-size-base", type=float, default=10.0, help="Taille minimale des nœuds.")
    parser.add_argument("--node-size-scale", type=float, default=4.0, help="Facteur appliqué à sqrt(degré) pour grossir les nœuds.")
    parser.add_argument("--node-size-max-bonus", type=float, default=32.0, help="Bonus maximal ajouté à la taille minimale.")
    parser.add_argument("--layout-seed", type=int, default=42)
    parser.add_argument("--layout-iterations", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    tables = load_graph_tables(outputs_dir)

    graph, _, _ = build_graph(
        tables,
        min_edge_weight=args.min_edge_weight,
        include_similarity_edges=args.include_similarity_edges,
        min_similarity=args.min_similarity,
        color_attributes_by_source=args.color_attributes_by_source,
    )
    if args.max_nodes is not None:
        graph = filter_graph(graph, max_nodes=args.max_nodes, include_similarity_edges=args.include_similarity_edges)

    full_output = Path(args.output_html) if args.output_html else outputs_dir / "discursive_full_graph_by_source.html"
    graph_analytics(graph, "Graphe discursif complet")
    write_graph_html(
        graph,
        full_output,
        title="Graphe discursif complet — couleur par source",
        seed=args.layout_seed,
        layout_iterations=args.layout_iterations,
        color_attributes_by_source=args.color_attributes_by_source,
        node_size_base=args.node_size_base,
        node_size_scale=args.node_size_scale,
        node_size_max_bonus=args.node_size_max_bonus,
    )

    filtered = filter_graph(
        graph,
        min_edge_weight=args.filtered_min_edge_weight,
        keep_node_types=split_csv_arg(args.keep_node_types),
        drop_high_degree_percentile=args.drop_high_degree_percentile,
        include_similarity_edges=args.include_similarity_edges,
        max_nodes=args.filtered_max_nodes,
    )
    filtered_output = (
        Path(args.filtered_output_html)
        if args.filtered_output_html
        else outputs_dir / "discursive_filtered_graph_by_source.html"
    )
    graph_analytics(filtered, "Graphe discursif filtré")
    write_graph_html(
        filtered,
        filtered_output,
        title="Graphe discursif filtré — hubs génériques sous-pondérés/retirés",
        seed=args.layout_seed,
        layout_iterations=args.layout_iterations,
        color_attributes_by_source=args.color_attributes_by_source,
        node_size_base=args.node_size_base,
        node_size_scale=args.node_size_scale,
        node_size_max_bonus=args.node_size_max_bonus,
    )

    if args.write_comment_projection:
        projection = build_comment_projection(graph)
        if projection.number_of_nodes() > 0:
            projection_output = (
                Path(args.projection_html)
                if args.projection_html
                else outputs_dir / "discursive_comment_projection_by_source.html"
            )
            graph_analytics(projection, "Projection commentaire-commentaire")
            write_graph_html(
                projection,
                projection_output,
                title="Projection commentaire-commentaire — similarité hybride",
                seed=args.layout_seed,
                layout_iterations=args.layout_iterations,
                color_attributes_by_source=True,
                node_size_base=args.node_size_base,
                node_size_scale=args.node_size_scale,
                node_size_max_bonus=args.node_size_max_bonus,
            )
        else:
            print("\nProjection commentaire-commentaire ignorée : aucune arête de similarité exploitable.")

    print(
        "\nLecture rapide : les zones denses signalent des configurations discursives récurrentes ; "
        "les nœuds-ponts signalent des frames, claims ou cibles transversaux ; les hubs génériques "
        "doivent être lus avec prudence. La couleur par source décrit le corpus collecté, pas une "
        "représentativité électorale."
    )


if __name__ == "__main__":
    main()
