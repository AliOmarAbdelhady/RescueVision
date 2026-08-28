"""Convert road segmentation masks to graph representations for routing."""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


def mask_to_skeleton(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a binary road mask to 1-pixel-wide centerlines.

    Args:
        mask: Binary mask (H, W) with 1=road, 0=background.

    Returns:
        Skeletonized mask (H, W).
    """
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8)
    return skeleton


def skeleton_to_graph(
    skeleton: np.ndarray,
    pixel_scale: float = 1.0,
) -> nx.Graph:
    """Convert a skeletonized mask to a networkx graph.

    Each skeleton pixel becomes a node. Adjacent pixels are connected
    by edges weighted by Euclidean distance.

    Args:
        skeleton: Binary skeleton mask (H, W).
        pixel_scale: Scale factor to convert pixel distances to real-world units.

    Returns:
        NetworkX graph with pixel-coordinate nodes.
    """
    graph = nx.Graph()
    rows, cols = np.where(skeleton > 0)

    # Add all skeleton pixels as nodes
    for r, c in zip(rows, cols):
        graph.add_node((r, c))

    # Connect adjacent pixels (8-connectivity)
    for r, c in zip(rows, cols):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1]:
                    if skeleton[nr, nc] > 0:
                        dist = np.sqrt(dr**2 + dc**2) * pixel_scale
                        graph.add_edge((r, c), (nr, nc), weight=dist)

    return graph


def simplify_graph(graph: nx.Graph, tolerance: int = 2) -> nx.Graph:
    """Simplify graph by removing degree-2 nodes (junction simplification).

    Keeps only junction nodes (degree != 2) and endpoints (degree == 1).
    Intermediate nodes on straight paths are collapsed into single edges.

    Args:
        graph: Input graph.
        tolerance: Minimum edge weight to keep.

    Returns:
        Simplified graph.
    """
    simplified = nx.Graph()

    # Add all nodes with degree != 2
    for node in graph.nodes():
        if graph.degree(node) != 2:
            simplified.add_node(node)

    # For each pair of junction nodes, find shortest path and add edge
    junction_nodes = [n for n in simplified.nodes()]
    for i, start in enumerate(junction_nodes):
        for end in junction_nodes[i + 1:]:
            try:
                path = nx.shortest_path(graph, start, end, weight="weight")
                # Check path only goes through degree-2 intermediate nodes
                intermediate = path[1:-1]
                if all(graph.degree(n) == 2 for n in intermediate):
                    total_weight = sum(
                        graph[path[j]][path[j + 1]]["weight"]
                        for j in range(len(path) - 1)
                    )
                    if total_weight >= tolerance:
                        simplified.add_edge(start, end, weight=total_weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    return simplified


def mask_to_graph(
    road_mask: np.ndarray,
    blocked_mask: np.ndarray | None = None,
    pixel_scale: float = 1.0,
    simplify: bool = True,
) -> nx.Graph:
    """Convert a road segmentation mask to a routable graph.

    Steps:
    1. Skeletonize road mask
    2. Remove blocked/flooded regions
    3. Convert to graph
    4. Optionally simplify

    Args:
        road_mask: Binary road mask (H, W).
        blocked_mask: Optional mask of blocked/flooded regions to remove.
        pixel_scale: Scale factor for edge weights.
        simplify: Whether to simplify the graph.

    Returns:
        NetworkX graph ready for pathfinding.
    """
    # Create safe road mask
    safe_road = road_mask.copy()
    if blocked_mask is not None:
        safe_road[blocked_mask > 0] = 0

    if safe_road.sum() == 0:
        logger.warning("No safe road pixels remaining after blocking")
        return nx.Graph()

    skeleton = mask_to_skeleton(safe_road)
    graph = skeleton_to_graph(skeleton, pixel_scale=pixel_scale)

    if simplify and len(graph.nodes) > 500:
        graph = simplify_graph(graph)

    logger.info(
        "Graph built: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    return graph


def find_nearest_node(
    graph: nx.Graph,
    point: Tuple[int, int],
) -> Tuple[int, int] | None:
    """Find the nearest graph node to a given pixel coordinate.

    Args:
        graph: NetworkX graph.
        point: (row, col) pixel coordinate.

    Returns:
        Nearest node coordinate or None if graph is empty.
    """
    if graph.number_of_nodes() == 0:
        return None

    nodes = np.array(list(graph.nodes()))
    distances = np.sqrt(np.sum((nodes - np.array(point))**2, axis=1))
    nearest_idx = np.argmin(distances)
    return tuple(nodes[nearest_idx])
