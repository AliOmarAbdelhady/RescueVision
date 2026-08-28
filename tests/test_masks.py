"""Tests for mask generation utilities."""

import numpy as np
import pytest

from rescuevision.constants import DAMAGE_CLASS_TO_ID, DAMAGE_COLOR_MAP, CHANGED_CLASSES


class TestDamageConstants:
    def test_damage_class_mapping(self):
        assert DAMAGE_CLASS_TO_ID["no-damage"] == 1
        assert DAMAGE_CLASS_TO_ID["minor-damage"] == 2
        assert DAMAGE_CLASS_TO_ID["major-damage"] == 3
        assert DAMAGE_CLASS_TO_ID["destroyed"] == 4
        assert DAMAGE_CLASS_TO_ID["un-classified"] == 255

    def test_color_map_completeness(self):
        for cls_id in range(5):
            assert cls_id in DAMAGE_COLOR_MAP
        assert 255 in DAMAGE_COLOR_MAP

    def test_changed_classes(self):
        assert CHANGED_CLASSES == {2, 3, 4}
        assert 1 not in CHANGED_CLASSES  # no-damage is not "changed"


class TestMaskCreation:
    def test_create_binary_mask(self):
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        # Draw a "building" rectangle
        mask[20:40, 30:60] = 1
        assert mask.sum() == 20 * 30
        assert mask[20, 30] == 1
        assert mask[0, 0] == 0

    def test_create_damage_mask(self):
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:20, 10:20] = 1  # no-damage
        mask[30:40, 10:20] = 3  # major-damage
        mask[50:60, 10:20] = 4  # destroyed

        assert np.sum(mask == 1) == 100
        assert np.sum(mask == 3) == 100
        assert np.sum(mask == 4) == 100

    def test_change_mask_from_damage(self):
        damage_mask = np.array([0, 1, 2, 3, 4, 255], dtype=np.uint8)
        change_mask = np.isin(damage_mask, list(CHANGED_CLASSES)).astype(np.uint8)
        assert change_mask[0] == 0  # background
        assert change_mask[1] == 0  # no-damage
        assert change_mask[2] == 1  # minor
        assert change_mask[3] == 1  # major
        assert change_mask[4] == 1  # destroyed
        assert change_mask[5] == 0  # unclassified
