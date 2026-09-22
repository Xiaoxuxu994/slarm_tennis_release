import sys
from types import ModuleType

import pytest
import torch

from tools.export_gaussian_sequence import export_gaussian_sequence


def test_snapshot_frames_mask_and_no_overwrite(tmp_path, monkeypatch):
    saved = {}
    writer = ModuleType("tools.export_ply")
    writer.save_ply = lambda values, path: saved.update({path: values})
    monkeypatch.setitem(sys.modules, "tools.export_ply", writer)
    data = {"target_frame_idx": torch.tensor([[15, 15, 15, 45, 45, 45]]),
            "target_camtoworlds": torch.zeros(1, 2, 3, 4, 4)}
    render = {"gs_means": torch.zeros(1, 2, 2, 3), "gs_color": torch.ones(1, 2, 2, 3),
              "gs_opacities": torch.tensor([[[[1.], [0.]], [[1.], [0.]]]]),
              "gs_scales": torch.ones(1, 2, 2, 3), "gs_quats": torch.ones(1, 2, 2, 4)}
    folder = tmp_path / "scene_0000"
    paths = export_gaussian_sequence(data, render, folder)
    assert [path.rsplit("/", 1)[-1] for path in paths] == ["gs_0015.ply", "gs_0045.ply"]
    for values in saved.values():
        assert values.shape == (1, 1, 14)
        assert torch.isfinite(values).all() and values[0, 0, 6] < 1
        assert values[0, 0, 10:].norm().item() == pytest.approx(1.)
    with pytest.raises(FileExistsError):
        export_gaussian_sequence(data, render, folder)
