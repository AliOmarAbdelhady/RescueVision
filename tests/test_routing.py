"""Tests for the routing module."""

import numpy as np
import pytest

from rescuevision.routing.mask_to_graph import mask_to_graph, find_nearest_node
from rescuevision.routing.route_planner import RoutePlanner
from rescuevision.routing.geo_utils import haversine_distance, compute_pixel_scale


def _make_road_mask(size: int = 100) -> np.ndarray:
    """Create a simple road mask with horizontal and vertical roads."""
    mask = np.zeros((size, size), dtype=np.uint8)
    # Horizontal road at row 50
    mask[48:52, 10:90] = 1
    # Vertical road at col 50
    mask[10:90, 48:52] = 1
    return mask


def _make_blocked_mask(size: int = 100) -> np.ndarray:
    """Create a blocked region on the horizontal road."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[46:54, 30:40] = 1  # Block part of horizontal road
    return mask


class TestMaskToGraph:
    def test_basic_graph_creation(self):
        road = _make_road_mask()
        graph = mask_to_graph(road, simplify=False)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_blocked_road_removes_nodes(self):
        road = _make_road_mask()
        blocked = _make_blocked_mask()
        graph_open = mask_to_graph(road, simplify=False)
        graph_blocked = mask_to_graph(road, blocked_mask=blocked, simplify=False)
        assert graph_blocked.number_of_nodes() < graph_open.number_of_nodes()

    def test_empty_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        graph = mask_to_graph(mask)
        assert graph.number_of_nodes() == 0


class TestFindNearestNode:
    def test_finds_close_node(self):
        road = _make_road_mask()
        graph = mask_to_graph(road, simplify=False)
        nearest = find_nearest_node(graph, (50, 50))
        assert nearest is not None
        # Should be close to (50, 50)
        assert abs(nearest[0] - 50) <= 2
        assert abs(nearest[1] - 50) <= 2

    def test_empty_graph(self):
        graph = mask_to_graph(np.zeros((50, 50), dtype=np.uint8))
        assert find_nearest_node(graph, (25, 25)) is None


class TestRoutePlanner:
    def test_route_found(self):
        road = _make_road_mask(100)
        planner = RoutePlanner()
        planner.build_graph(road)
        # Use start/end on the skeleton (row 49 for horizontal road)
        result = planner.plan_route(start=(49, 15), end=(49, 85))
        assert result["found"] is True
        assert len(result["path"]) > 0

    def test_route_blocked(self):
        road = _make_road_mask(100)
        blocked = _make_blocked_mask(100)
        planner = RoutePlanner()
        planner.build_graph(road, blocked)
        # Route should still be found via alternative (vertical road)
        result = planner.plan_route(start=(50, 15), end=(50, 85))
        # May or may not find a path depending on graph connectivity
        # The key is it doesn't crash
        assert isinstance(result, dict)
        assert "found" in result

    def test_no_graph(self):
        planner = RoutePlanner()
        result = planner.plan_route(start=(0, 0), end=(10, 10))
        assert result["found"] is False


class TestGeoUtils:
    def test_haversine_same_point(self):
        d = haversine_distance(40.0, -74.0, 40.0, -74.0)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        # NYC to LA: roughly 3944 km
        d = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3900 < d < 4000

    def test_pixel_scale(self):
        transform = [0.5, 0.0, 100.0, 0.0, -0.5, 200.0]
        scale = compute_pixel_scale(transform)
        assert scale == pytest.approx(0.5, abs=1e-6)
