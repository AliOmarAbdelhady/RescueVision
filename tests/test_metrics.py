"""Tests for evaluation metrics."""

import numpy as np
import pytest

from rescuevision.evaluation.metrics_segmentation import compute_iou, compute_dice
from rescuevision.evaluation.metrics_classification import compute_classification_metrics


class TestSegmentationMetrics:
    def test_perfect_iou(self):
        pred = np.ones((10, 10), dtype=np.uint8)
        target = np.ones((10, 10), dtype=np.uint8)
        iou = compute_iou(pred, target)
        assert iou == pytest.approx(1.0, abs=1e-6)

    def test_zero_iou(self):
        pred = np.ones((10, 10), dtype=np.uint8)
        target = np.zeros((10, 10), dtype=np.uint8)
        iou = compute_iou(pred, target)
        assert iou == pytest.approx(0.0, abs=1e-6)

    def test_partial_iou(self):
        pred = np.zeros((10, 10), dtype=np.uint8)
        target = np.zeros((10, 10), dtype=np.uint8)
        pred[:5, :] = 1
        target[:, :5] = 1
        iou = compute_iou(pred, target)
        assert 0.0 < iou < 1.0

    def test_perfect_dice(self):
        pred = np.ones((10, 10), dtype=np.uint8)
        target = np.ones((10, 10), dtype=np.uint8)
        dice = compute_dice(pred, target)
        assert dice == pytest.approx(1.0, abs=1e-6)

    def test_dice_symmetry(self):
        pred = np.zeros((10, 10), dtype=np.uint8)
        target = np.zeros((10, 10), dtype=np.uint8)
        pred[:3, :3] = 1
        target[:5, :5] = 1
        d1 = compute_dice(pred, target)
        d2 = compute_dice(target, pred)
        assert d1 == pytest.approx(d2, abs=1e-6)


class TestClassificationMetrics:
    def test_perfect_accuracy(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_zero_accuracy(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([3, 2, 1, 0])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_partial_accuracy(self):
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 0, 2])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert 0.0 < metrics["accuracy"] < 1.0
