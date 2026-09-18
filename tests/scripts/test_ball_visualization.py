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


# ---------------------------------------------------------------- 黄色定位框

BOX = _constant(RENDER_SRC.read_text(), "BALL_BOX_PX")


def test_the_box_is_much_larger_than_the_ball():
    """A box the ball's own size would be as invisible as the ball is. It marks
    where to look; it does not pretend to show the ball."""
    assert BOX / 2.66 > 5, "the box has to be findable on a full frame"
    assert BOX / 320 < 0.15, "and it must not swallow the scene around it"


def test_both_rows_get_a_box_from_their_own_source():
    """The GT row marks the recorded ball and the predicted row the rendered
    one. Drawing the GT box on both would hide the position error, which is the
    one thing the two rows side by side are there to reveal.

    Checked by counting which centre each panel is boxed from, rather than by
    matching one spelling of the call: the first version of this test named the
    in-place loop and went red when that loop was replaced, without the
    invariant it guards having changed.
    """
    source = RENDER_SRC.read_text()
    gt_panels = {"gt_img", "gt_dc", "gt_sc"}
    pred_panels = {"pred_img", "pd_dc", "pd_sc"}
    for panel in gt_panels:
        assert f"{panel} = draw_ball_box({panel}, gt_centre)" in source
    for panel in pred_panels:
        assert f"{panel} = draw_ball_box({panel}, pred_centre)" in source
    # And never the other way round, which is the failure that would look fine.
    for panel in gt_panels:
        assert f"draw_ball_box({panel}, pred_centre)" not in source
    for panel in pred_panels:
        assert f"draw_ball_box({panel}, gt_centre)" not in source


def test_a_missing_ball_draws_nothing_rather_than_a_box_at_the_origin():
    body = ast.unparse(_function(RENDER_SRC.read_text(), "draw_ball_box"))
    assert "if centre is None" in body and "return image" in body


def test_panels_are_made_cv2_compatible_before_drawing():
    """OpenCV refuses anything that is not a plain contiguous uint8 buffer, and
    these panels come from permutes, colormap slices and fancy indexing."""
    body = ast.unparse(_function(RENDER_SRC.read_text(), "draw_ball_box"))
    assert "ascontiguousarray" in body


def test_callers_use_the_returned_panel():
    """Making an array contiguous can copy it. Drawing into a copy the caller
    discards is a silent no-op, not an error, so the box would just be absent."""
    source = RENDER_SRC.read_text()
    assert "gt_img = draw_ball_box(gt_img" in source
    assert "pd_sc = draw_ball_box(pd_sc" in source
    assert "for panel in (" not in source, "the in-place loop drew into copies"


def test_cv2_drawing_arguments_stay_positional():
    """cv2.rectangle has a second overload taking a Rect. A keyword argument
    makes the resolver report its mismatch against that overload instead of the
    real problem, which is how the first failure hid behind a wrong message."""
    source = RENDER_SRC.read_text()
    assert "lineType=" not in source


def test_the_zoom_row_is_opt_in():
    """Cropping to 16 px discards the scene to gain detail 2.66 px does not
    carry; the box covers the common case without that trade."""
    source = RENDER_SRC.read_text()
    assert '"--ball-zoom"' in source
    assert "zoom_row = [] if extra.ball_zoom else None" in source
    assert "if zoom_full is not None:" in source, "the frame must assemble without it"


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


def test_the_ball_fills_enough_of_the_panel():
    """The CROP WIDTH sets this, not the magnification: the ball occupies
    ball_px / crop_w of the panel however it is scaled. A 32 px window leaves it
    at 8% and still hard to read, which is what the first version did."""
    fraction = 2.66 / CROP_W
    assert fraction > 0.15, f"ball is only {fraction:.0%} of the panel"
    assert 2.66 * 320 / CROP_W > 40, "and it should be dozens of pixels across"


def test_the_crop_still_holds_the_ball_between_recentrings():
    """Tightening the window trades slack for size; the window re-centres every
    frame and the ball moves about 4.3 px per frame at this range."""
    margin = (CROP_W - 2.66) / 2
    assert margin / 4.3 > 1.0, f"only {margin / 4.3:.1f} frames of slack"


def test_the_truth_ring_lands_on_the_ball():
    """The ring marks where the ball truly is; if the crop offset were dropped
    it would sit at the panel centre always and quietly agree with everything."""
    for ball in ((160, 120), (3, 2), (317, 237)):
        x0, y0, _, _ = window(ball)
        cx = round((ball[0] - x0) * (320 / CROP_W))
        cy = round((ball[1] - y0) * (240 / CROP_H))
        assert 0 <= cx <= 320 and 0 <= cy <= 240
        # Re-projecting back has to recover the ball's own pixel.
        assert abs(x0 + cx / (320 / CROP_W) - ball[0]) < 0.6
        assert abs(y0 + cy / (240 / CROP_H) - ball[1]) < 0.6


def test_the_empty_gt_panel_says_why_it_is_empty():
    """Past the recorded clip there is no truth to show; a black panel reads as
    a rendering failure."""
    assert "no GT past frame 24" in RENDER_SRC.read_text()


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


# ---------------------------------------------------------------- 三维轨迹图

TRACK_SRC = ROOT / "tools" / "export_ball_track.py"


def ring_points(catch, previous, radius, count=80):
    """The ring construction plot_3d performs, without numpy."""
    direction = [a - b for a, b in zip(catch, previous)]
    norm = math.sqrt(sum(x * x for x in direction))
    axis = [x / norm for x in direction]
    seed = [0.0, 1.0, 0.0] if abs(axis[0]) > 0.9 else [1.0, 0.0, 0.0]

    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]]

    u = cross(axis, seed)
    length = math.sqrt(sum(x * x for x in u))
    u = [x / length for x in u]
    w = cross(axis, u)
    points = []
    for index in range(count):
        angle = 2 * math.pi * index / count
        points.append([catch[k] + radius * (math.cos(angle) * u[k]
                                            + math.sin(angle) * w[k])
                       for k in range(3)])
    return points, axis


def test_the_drawn_ring_is_a_circle_of_the_right_size():
    catch, previous, radius = [0.0, 0.0, 0.0], [0.0, 0.070, 0.247], 0.135
    points, _ = ring_points(catch, previous, radius)
    distances = [math.dist(p, catch) for p in points]
    assert max(distances) == pytest.approx(radius, abs=1e-9)
    assert min(distances) == pytest.approx(radius, abs=1e-9)


def test_the_ring_faces_the_arriving_ball():
    """A ring drawn in any other plane would show a wider opening than the ball
    actually has to pass through."""
    catch, previous, radius = [0.0, 0.0, 0.0], [0.0, 0.070, 0.247], 0.135
    points, axis = ring_points(catch, previous, radius)
    for point in points:
        along = sum((point[k] - catch[k]) * axis[k] for k in range(3))
        assert abs(along) < 1e-12, "ring points must lie in the plane across the axis"


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.5, -0.5, 0.7)])
def test_the_ring_survives_an_arrival_along_any_axis(axis):
    """The seed vector for the cross product has to dodge the axis it is
    crossed with, or the ring collapses to a line."""
    catch = [0.0, 0.0, 0.0]
    previous = [-a for a in axis]
    points, _ = ring_points(catch, previous, 0.135)
    distances = [math.dist(p, catch) for p in points]
    assert min(distances) == pytest.approx(0.135, abs=1e-9)


def test_the_closeup_panel_is_scaled_to_the_ring_not_the_flight():
    """The arc is about three metres and the ring is 0.27, so one extent cannot
    serve both; at full extent the aperture is a smudge."""
    source = TRACK_SRC.read_text()
    assert "reach = ring_radius * 2.2" in source
    fraction = 0.27 / (0.135 * 2.2 * 2)
    assert fraction > 0.3, "the ring has to fill a useful part of the close-up"


def test_the_predicted_landing_is_drawn_in_the_closeup():
    """Without it the panel shows the ring and the truth but not the error."""
    source = TRACK_SRC.read_text()
    assert 'rows[catch_frame]["views"][index]' in source


def test_axes_keep_equal_scale():
    """A three metre arc squeezed into a box with unequal axes is a different
    shape, and the fall would not look like a fall."""
    assert "set_box_aspect((1, 1, 1))" in TRACK_SRC.read_text()


def test_missing_matplotlib_does_not_break_the_other_outputs():
    body = ast.unparse(_function(TRACK_SRC.read_text(), "plot_3d"))
    assert "except ImportError" in body and "return None" in body
