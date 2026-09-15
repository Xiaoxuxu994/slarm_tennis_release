"""Configuration/registration checks without CUDA or dataset dependencies."""

import ast
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


class Pixel10kConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / "configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml").read_text())

    def test_pixel_only_and_initialization(self):
        for key in ("use_ball_token", "use_ball_token_intrunk", "ball_token_freeze_backbone",
                    "ball_velocity_residual", "ball_velocity_only_train", "ball_temporal_refine",
                    "ball_prefix_supervision"):
            self.assertIs(self.config[key], False, key)
        for key in ("ball_pos", "ball_vel", "ball_traj", "landing", "ball_delta_v",
                    "ball_prefix_pos", "ball_prefix_vel", "ball_prefix_landing"):
            self.assertEqual(self.config[f"stream25_{key}_weight"], 0)
        self.assertIn("exp0908_001_", self.config["load_from"])
        self.assertTrue(self.config["load_from"].endswith("ckpt_019999.pth"))

    def test_existing_pixel_loss_contract(self):
        baseline = yaml.safe_load((ROOT / "configs/exp0908_001_slarm_stream25_0903_2k_triview_window6_nolseg_4gpu.yml").read_text())
        for key in baseline:
            if key.startswith("stream25_") and (key.endswith("_weight") or key.endswith("_scale")):
                self.assertEqual(self.config[key], baseline[key], key)

    def test_registration_and_manifests(self):
        tree = ast.parse((ROOT / "src/dataset/constants.py").read_text())
        name = self.config["dataset"][0]
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                found.extend(value for key, value in zip(node.keys, node.values)
                             if isinstance(key, ast.Constant) and key.value == name)
        self.assertEqual(len(found), 2)
        metadata = next(value for value in found
                        if any(isinstance(key, ast.Constant) and key.value == "size" for key in value.keys))
        metadata = ast.literal_eval(metadata)
        self.assertEqual(metadata["size"], self.config["input_size"])
        self.assertEqual(metadata["annotation_txt_file_train"], self.config["train_annotation"])
        self.assertEqual(metadata["annotation_txt_file_val"], self.config["eval_annotation"])
        self.assertEqual(len(metadata["camera_list"][3]), self.config["num_max_cameras"])


if __name__ == "__main__":
    unittest.main()
