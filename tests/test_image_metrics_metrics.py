# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image Metrics' Mean Red, Mean Green and Mean Blue: each channel's mean over the image's pixels."""

from __future__ import annotations

import pytest
from PIL import Image

from tlc_plugin_image_metrics.metrics import METRIC_BY_ID, compute_metrics_for_image

MEAN_RGB = ["mean_red", "mean_green", "mean_blue"]


@pytest.mark.parametrize("color", [(255, 0, 0), (10, 20, 30)])
def test_a_solid_image_gives_its_color(color: tuple[int, int, int]) -> None:
    results = compute_metrics_for_image(Image.new("RGB", (8, 6), color), MEAN_RGB)

    assert [results[mid] for mid in MEAN_RGB] == [float(c) for c in color]


def test_each_channel_is_averaged_over_every_pixel() -> None:
    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), (0, 100, 255))
    img.putpixel((1, 0), (50, 0, 255))

    results = compute_metrics_for_image(img, MEAN_RGB)

    assert results == {"mean_red": 25.0, "mean_green": 50.0, "mean_blue": 255.0}


def test_a_grayscale_image_gives_three_equal_means() -> None:
    img = Image.new("L", (2, 1))
    img.putpixel((0, 0), 40)
    img.putpixel((1, 0), 80)

    results = compute_metrics_for_image(img, MEAN_RGB)

    assert results == {"mean_red": 60.0, "mean_green": 60.0, "mean_blue": 60.0}


def test_the_means_are_listed_under_color() -> None:
    assert [METRIC_BY_ID[mid].category for mid in MEAN_RGB] == ["Color", "Color", "Color"]
