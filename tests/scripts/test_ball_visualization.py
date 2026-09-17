"""球的可视化：放大窗口、只含球的 PLY、预测/真值标记球。

存在的理由是一个尺度问题。球直径 2.66 px，面积约 5.6 px，占 320x240 画面的
0.007%；整个球只由约 100 个高斯构成，而场景有 138 万个。所以：

  - 全幅视频里，球的几何对不对看起来是一样的 → 需要放大窗口
  - 全场景 PLY 里，球要在 50 万点里找 → 需要只含球的那一份
  - 三维误差只能读数字 → 需要预测和真值两个标记球

三件事各有一个会静默出错的地方，这份测试盯的就是那三个点。

    pytest tests/scripts/test_ball_visualization.py -q
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RENDER_SRC = ROOT / "scripts" / "render_stream25_base.py"
EXPORT_SRC = ROOT / "tools" / "export_gaussian_sequence.py"


def _constant(source: str, name: str):
    tree = ast.parse(source)
    return next(node.value.value for node in tree.body
                if isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == name for t in node.targets))


def _function(source: str, name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == name)


# ---------------------------------------------------------------- 放大窗口

CROP_W = _constant(RENDER_SRC.read_text(), "BALL_ZOOM_CROP_W")
CROP_H = _constant(RENDER_SRC.read_text(), "BALL_ZOOM_CROP_H")


def window(centre, width=320, height=240):
    """The arithmetic ball_zoom performs, without cv2."""
    if centre is None:
        centre = (width / 2.0, height / 2.0)
    half_w, half_h = CROP_W // 2, CROP_H // 2
    x0 = int(round(min(max(centre[0] - half_w, 0), max(0, width - CROP_W))))
    y0 = int(round(min(max(centre[1] - half_h, 0), max(0, height - CROP_H))))
    return x0, y0, x0 + CROP_W, y0 + CROP_H


@pytest.mark.parametrize("centre", [(160, 120), (2, 2), (318, 238), (0, 120), (160, 0), None])
def test_the_crop_never_changes_size(centre):
    """A window that shrank at the edges would change the magnification, and a
    ball that only moved would look like it grew."""
    x0, y0, x1, y1 = window(centre)
    assert (x1 - x0, y1 - y0) == (CROP_W, CROP_H)
    assert 0 <= x0 and x1 <= 320 and 0 <= y0 and y1 <= 240


@pytest.mark.parametrize("centre", [(160, 120), (2, 2), (318, 238), (0, 120), (160, 0)])
def test_the_ball_stays_inside_the_crop(centre):
    x0, y0, x1, y1 = window(centre)
    assert x0 <= centre[0] <= x1 and y0 <= centre[1] <= y1


def test_the_magnification_makes_the_ball_visible():
    """2.66 px is the reason this exists; below about ten pixels it is still a
    smudge and the row would not have earned its space."""
    magnified = 2.66 * 320 / CROP_W
    assert magnified > 20, f"only {magnified:.1f} px after zoom"


def test_zoom_uses_nearest_neighbour():
    """Smoothing would invent detail the 2.66 px never had."""
    body = ast.unparse(_function(RENDER_SRC.read_text(), "ball_zoom"))
    assert "INTER_NEAREST" in body


def test_zoom_centres_on_truth_when_there_is_one():
    """Centring on the prediction would let a model that lost the ball follow
    its own mistake off screen, and the window would look fine."""
    source = RENDER_SRC.read_text()
    assert "if gt_available:" in source and "ball_centre_px(" in source
    index_gt = source.index("centre = None")
    index_pred = source.index("if centre is None and pred_s is not None")
    assert index_gt < index_pred, "truth has to be tried first"


# ---------------------------------------------------------------- 标记球

MARKER_POINTS = _constant(EXPORT_SRC.read_text(), "MARKER_POINTS")


def sphere(centre, radius, count=MARKER_POINTS):
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        rho = math.sqrt(max(0.0, 1.0 - z * z))
        angle = golden * index
        out.append((centre[0] + radius * rho * math.cos(angle),
                    centre[1] + radius * rho * math.sin(angle),
                    centre[2] + radius * z))
    return out


def test_marker_points_lie_on_the_sphere():
    centre, radius = (1.0, 2.0, 3.0), 0.0325
    distances = [math.dist(p, centre) for p in sphere(centre, radius)]
    assert max(distances) == pytest.approx(radius, abs=1e-9)
    assert min(distances) == pytest.approx(radius, abs=1e-9)


def test_marker_points_are_spread_evenly():
    """A lopsided shell would read as an offset that is not there."""
    centre, radius = (0.0, 0.0, 0.0), 0.0325
    points = sphere(centre, radius)
    mean = [sum(p[i] for p in points) / len(points) for i in range(3)]
    assert math.dist(mean, centre) < radius * 1e-3


def test_marker_values_survive_the_ply_transforms():
    """save_ply stores log(scale) and logit(opacity); zero or one is infinite."""
    source = EXPORT_SRC.read_text()
    opacity = 0.99
    scale = _constant(source, "MARKER_POINT_SCALE")
    assert 0.0 < opacity < 1.0 and math.isfinite(math.log(opacity / (1 - opacity)))
    assert scale > 0.0 and math.isfinite(math.log(scale))


def test_markers_are_keyed_by_frame_number_on_both_sides():
    """The exporter looks markers up by frame number while the caller loops
    over target positions; they coincide today and would not say so if that
    changed."""
    assert "(markers or {}).get(frame)" in EXPORT_SRC.read_text()
    assert "ply_markers[int(marker_frames[index])]" in RENDER_SRC.read_text()


# ---------------------------------------------------------------- 只含球的 PLY

def test_ball_only_export_filters_after_the_validity_mask():
    """The labels are per Gaussian before filtering, so they must be indexed by
    the same mask the values were, or the two fall out of step silently."""
    body = ast.unparse(_function(EXPORT_SRC.read_text(), "export_gaussian_sequence"))
    assert "semantic[valid]" in body, "labels must be filtered by the same mask"
    assert "ball_" in body


def test_ball_only_export_checks_the_label_count():
    body = ast.unparse(_function(EXPORT_SRC.read_text(), "export_gaussian_sequence"))
    assert "one label per Gaussian" in body


def test_semantic_comes_from_the_context_pixels_not_the_targets():
    """The Gaussians are one per CONTEXT pixel; target semantics have a
    different length and would raise, or worse, happen to match."""
    source = RENDER_SRC.read_text()
    assert 'input_dict.get("context_task_semantic")' in source
