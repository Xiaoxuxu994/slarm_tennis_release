"""Export the actual rendered Gaussian snapshots, without target-pixel sky masks."""

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.utils.frame_indices import normalize_frame_indices

#: Points per marker sphere. Sparse enough to see the ball through it.
MARKER_POINTS = 160
#: Each marker point is rendered this big, in metres.
MARKER_POINT_SCALE = 0.003


def marker_sphere(centre: Sequence[float], colour: Sequence[float],
                  radius: float) -> torch.Tensor:
    """A hollow shell of points, in the 14-column layout save_ply expects.

    Markers exist so a predicted and a true ball centre can be compared in
    three dimensions rather than through a number. The shell is hollow and
    sparse on purpose: a solid blob would hide the very Gaussians it is there
    to be compared against.
    """
    if radius <= 0:
        raise ValueError("marker radius must be positive")
    golden = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for index in range(MARKER_POINTS):
        z = 1.0 - 2.0 * (index + 0.5) / MARKER_POINTS
        rho = math.sqrt(max(0.0, 1.0 - z * z))
        angle = golden * index
        points.append([centre[0] + radius * rho * math.cos(angle),
                       centre[1] + radius * rho * math.sin(angle),
                       centre[2] + radius * z])
    values = torch.zeros(MARKER_POINTS, 14, dtype=torch.float32)
    values[:, 0:3] = torch.tensor(points, dtype=torch.float32)
    values[:, 3:6] = torch.tensor(colour, dtype=torch.float32)
    values[:, 6] = 0.99                      # save_ply takes the logit of this
    values[:, 7:10] = MARKER_POINT_SCALE     # and the log of this
    values[:, 10] = 1.0                      # identity rotation
    return values


def export_gaussian_sequence(data: dict, render: dict, directory: Path, *, affine=None,
                             opacity_threshold: float = 0.1,
                             semantic: Optional[torch.Tensor] = None,
                             markers: Optional[Dict[int, List[Tuple[Sequence[float],
                                                                    Sequence[float]]]]] = None,
                             marker_radius: float = 0.0325) -> list:
    """Write one PLY per target frame, plus optional ball-only and marker files.

    ``semantic`` is one class label per Gaussian, in the same (t v h w) order as
    the Gaussians themselves, which is the layout of context_task_semantic. When
    given, each frame also gets a ball_<frame>.ply holding only the class-1
    Gaussians. That file is the point of this parameter: the ball is a hundred
    Gaussians among a million, so in the full export it has to be hunted for,
    and how the ball itself is reconstructed cannot be seen at all.

    ``markers`` maps a frame number to positions to mark with a hollow shell,
    in the same frame as the Gaussians. Pass the predicted and the true ball
    centre to see the error as a distance in space.
    """
    from tools.export_ply import save_ply

    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite Gaussian sequence: {directory}")
    means = render["gs_means"]
    if means.ndim != 4 or means.shape[0] != 1:
        raise ValueError("Gaussian export requires snapshots [1,T,N,3]")
    frames = normalize_frame_indices(
        data["target_frame_idx"], batch_size=1, num_timesteps=means.shape[1],
        num_views=data["target_camtoworlds"].shape[2], name="target_frame_idx")[0].tolist()
    if len(set(frames)) != len(frames):
        raise ValueError("Export target frames must be unique")
    directory.mkdir(parents=True)
    paths = []
    for index, frame in enumerate(frames):
        arrays = [render[key][0, index].detach().float() for key in
                  ("gs_means", "gs_color", "gs_opacities", "gs_scales", "gs_quats")]
        xyz, color, opacity, scales, quats = arrays
        if affine is not None:
            _, t, v, _, h, w = data["context_image"].shape
            if color.shape != (t * v * h * w, 3):
                raise ValueError("Affine export requires unvoxelized context-pixel Gaussians")
            color = color.reshape(1, t, v, h, w, 3)
            color = torch.einsum("btvhwi,bvij->btvhwj", color, affine["linear"].float())
            color = (color + affine["translation"].float()).reshape(-1, 3)
        opacity = opacity.reshape(-1, 1)
        values = torch.cat((xyz, color, opacity, scales, quats), dim=-1)
        valid = (torch.isfinite(values).all(dim=-1) & (opacity[:, 0] > opacity_threshold)
                 & (scales > 0).all(dim=-1) & (quats.norm(dim=-1) > 1e-8))
        values = values[valid].cpu()
        values[:, 3:6].clamp_(0, 1)
        values[:, 6].clamp_(1e-6, 1 - 1e-6)
        values[:, 10:14] = torch.nn.functional.normalize(values[:, 10:14], dim=-1)
        extra = torch.cat([marker_sphere(centre, colour, marker_radius)
                           for centre, colour in (markers or {}).get(frame, [])]
                          ) if (markers or {}).get(frame) else None

        path = directory / f"gs_{frame:04d}.ply"
        save_ply(torch.cat((values, extra))[None] if extra is not None else values[None],
                 str(path))
        paths.append(str(path))

        if semantic is not None:
            if semantic.shape[0] != valid.shape[0]:
                raise ValueError(
                    f"semantic has {semantic.shape[0]} labels for {valid.shape[0]} "
                    f"Gaussians; it must be one label per Gaussian in (t v h w) order"
                )
            ball = values[semantic[valid].reshape(-1).cpu() == 1]
            if extra is not None:
                ball = torch.cat((ball, extra))
            ball_path = directory / f"ball_{frame:04d}.ply"
            save_ply(ball[None], str(ball_path))
            paths.append(str(ball_path))
    return paths
