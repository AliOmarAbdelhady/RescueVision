"""Emergency route planner using road masks and obstacle detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np

from rescuevision.routing.mask_to_graph import (
    find_nearest_node,
    mask_to_graph,
)

logger = logging.getLogger(__name__)


class RoutePlanner:
    """Plan safe emergency routes using road and obstacle segmentation masks.

    Uses A* or Dijkstra to find shortest safe path between two points,
    avoiding flooded/blocked regions.
    """

    def __init__(self, pixel_scale: float = 1.0):
        self.pixel_scale = pixel_scale
        self.graph: nx.Graph | None = None

    def build_graph(
        self,
        road_mask: np.ndarray,
        blocked_mask: np.ndarray | None = None,
    ) -> None:
        """Build routable graph from masks.

        Args:
            road_mask: Binary road mask (H, W), 1=road.
            blocked_mask: Binary blocked/flooded mask (H, W), 1=blocked.
        """
        self.graph = mask_to_graph(
            road_mask,
            blocked_mask=blocked_mask,
            pixel_scale=self.pixel_scale,
        )

    def plan_route(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        method: str = "astar",
    ) -> dict[str, Any]:
        """Plan a safe route between start and end points.

        Args:
            start: (row, col) start coordinate.
            end: (row, col) end coordinate.
            method: "astar" or "dijkstra".

        Returns:
            Dict with path coordinates, length, and metadata.
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return {
                "path": [],
                "length": float("inf"),
                "found": False,
                "error": "No routable graph available",
            }

        start_node = find_nearest_node(self.graph, start)
        end_node = find_nearest_node(self.graph, end)

        if start_node is None or end_node is None:
            return {
                "path": [],
                "length": float("inf"),
                "found": False,
                "error": "Could not find graph nodes near start/end points",
            }

        try:
            if method == "astar":
                path = nx.astar_path(
                    self.graph,
                    start_node,
                    end_node,
                    heuristic=lambda u, v: np.sqrt((u[0]-v[0])**2 + (u[1]-v[1])**2),
                    weight="weight",
                )
            else:
                path = nx.dijkstra_path(
                    self.graph, start_node, end_node, weight="weight",
                )

            length = nx.path_weight(self.graph, path, weight="weight")

            return {
                "path": path,
                "length": length,
                "found": True,
                "start_node": start_node,
                "end_node": end_node,
                "num_edges": len(path) - 1,
            }

        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            return {
                "path": [],
                "length": float("inf"),
                "found": False,
                "error": str(e),
            }

    def visualize_route(
        self,
        background_image: np.ndarray,
        route_result: dict[str, Any],
        road_mask: np.ndarray | None = None,
        blocked_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Visualize route on background image.

        Args:
            background_image: RGB image (H, W, 3).
            route_result: Output from plan_route().
            road_mask: Optional road mask to overlay.
            blocked_mask: Optional blocked mask to overlay.

        Returns:
            RGB image with route overlay.
        """
        overlay = background_image.copy()

        # Draw road network (light gray)
        if road_mask is not None:
            road_vis = (road_mask > 0)[:, :, np.newaxis] * np.array([180, 180, 180])
            mask = road_mask > 0
            overlay[mask] = (overlay[mask] * 0.5 + road_vis[mask] * 0.5).astype(np.uint8)

        # Draw blocked zones (red tint)
        if blocked_mask is not None:
            blocked_vis = (blocked_mask > 0)[:, :, np.newaxis] * np.array([255, 50, 50])
            mask = blocked_mask > 0
            overlay[mask] = (overlay[mask] * 0.4 + blocked_vis[mask] * 0.6).astype(np.uint8)

        # Draw route (bright green line)
        if route_result["found"] and len(route_result["path"]) > 1:
            path = route_result["path"]
            for i in range(len(path) - 1):
                pt1 = (path[i][1], path[i][0])      # (col, row) for cv2
                pt2 = (path[i + 1][1], path[i + 1][0])
                cv2.line(overlay, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw start (blue circle) and end (red circle)
            start = path[0]
            end = path[-1]
            cv2.circle(overlay, (start[1], start[0]), 8, (0, 0, 255), -1)
            cv2.circle(overlay, (end[1], end[0]), 8, (255, 0, 0), -1)

        return overlay

    def export_route_html(
        self,
        route_result: dict[str, Any],
        road_mask: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        """Export a simple HTML visualization of the route.

        Args:
            route_result: Output from plan_route().
            road_mask: Road mask used for routing.
            output_path: Output HTML path.

        Returns:
            Path to saved HTML file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        h, w = road_mask.shape

        html = f"""<!DOCTYPE html>
<html>
<head><title>RescueVision Route Map</title>
<style>
body {{ margin: 0; font-family: sans-serif; background: #1a1a2e; color: #eee; }}
.container {{ max-width: 800px; margin: 20px auto; padding: 20px; }}
h1 {{ color: #e94560; }}
canvas {{ border: 2px solid #333; background: #000; }}
.info {{ margin-top: 15px; padding: 10px; background: #16213e; border-radius: 5px; }}
</style>
</head>
<body>
<div class="container">
<h1>Emergency Route Map</h1>
<canvas id="routeCanvas" width="{w}" height="{h}"></canvas>
<div class="info">
<p><strong>Route found:</strong> {route_result.get('found', False)}</p>
<p><strong>Route length:</strong> {route_result.get('length', 'N/A')}</p>
<p><strong>Number of segments:</strong> {route_result.get('num_edges', 'N/A')}</p>
</div>
</div>
<script>
const canvas = document.getElementById('routeCanvas');
const ctx = canvas.getContext('2d');
// Draw road mask
const roadData = {json.dumps(road_mask.tolist()) if road_mask.dtype == np.uint8 else '[]'};
// Simplified: just show route info
ctx.fillStyle = '#333';
ctx.fillRect(0, 0, {w}, {h});
</script>
</body>
</html>"""

        # For a simpler approach, just embed route info
        import json as json_mod
        path_coords = route_result.get("path", [])

        html_simple = f"""<!DOCTYPE html>
<html><head><title>RescueVision Route</title>
<style>body{{font-family:sans-serif;max-width:600px;margin:40px auto;padding:20px;background:#1a1a2e;color:#eee;}}
.card{{background:#16213e;padding:20px;border-radius:8px;margin:10px 0;}}
h1{{color:#e94560;}} .label{{color:#888;font-size:0.85em;}}</style>
</head><body>
<h1>Emergency Route Map</h1>
<div class="card">
<p class="label">Status</p>
<p>{"Route found" if route_result.get("found") else "No route found"}</p>
<p class="label">Length</p><p>{route_result.get("length", "N/A"):.1f} pixels</p>
<p class="label">Segments</p><p>{route_result.get("num_edges", "N/A")}</p>
</div>
<div class="card">
<p class="label">Path Coordinates (row, col)</p>
<pre style="font-size:0.8em;max-height:300px;overflow-y:auto;">{json_mod.dumps(path_coords[:50], indent=2)}{f"... ({len(path_coords)} total points)" if len(path_coords) > 50 else ""}</pre>
</div>
</body></html>"""

        output_path.write_text(html_simple)
        return output_path


def plan_route_from_masks(
    road_mask: np.ndarray,
    blocked_mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    output_dir: str | Path,
    background_image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Convenience function: plan and visualize a route from masks.

    Args:
        road_mask: Binary road mask.
        blocked_mask: Binary blocked/flooded mask.
        start: Start point (row, col).
        end: End point (row, col).
        output_dir: Directory to save outputs.
        background_image: Optional background for visualization.

    Returns:
        Route result dict with added file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    planner = RoutePlanner()
    planner.build_graph(road_mask, blocked_mask)
    result = planner.plan_route(start, end)

    # Save route visualization
    if background_image is not None:
        vis = planner.visualize_route(background_image, result, road_mask, blocked_mask)
        cv2.imwrite(str(output_dir / "route_overlay.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        result["route_image"] = str(output_dir / "route_overlay.png")

    # Save HTML route map
    planner.export_route_html(result, road_mask, output_dir / "route_map.html")
    result["route_html"] = str(output_dir / "route_map.html")

    # Save JSON summary
    import json
    serializable = {k: v for k, v in result.items() if k != "path"}
    serializable["path_length"] = len(result.get("path", []))
    with open(output_dir / "route_summary.json", "w") as f:
        json.dump(serializable, f, indent=2)

    return result
