"""Dependency-free regression for evaluation's diagnostic imports and unit aliases."""

import ast
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ResidualDiagnosticWiringTest(unittest.TestCase):
    def test_evaluator_imports_exist(self):
        module = ast.parse((ROOT / "src/utils/ball_residual_diagnostics.py").read_text())
        names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
        evaluator = ast.parse((ROOT / "scripts/eval_stream25_base.py").read_text())
        imports = [node for node in ast.walk(evaluator) if isinstance(node, ast.ImportFrom)
                   and node.module == "src.utils.ball_residual_diagnostics"]
        self.assertTrue(imports)
        for node in imports:
            for alias in node.names:
                self.assertIn(alias.name, names)

    def test_unit_conversion_preserves_physical_values(self):
        module = ast.parse((ROOT / "src/utils/ball_residual_diagnostics.py").read_text())
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                        and node.name == "velocity_unit_diagnostics")
        env = {"math": math}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "unit_diagnostics", "exec"), env)
        source = {"base_x": 1.2, "delta_x": .04, "final_frame45_error": .1}
        result = env[function.name](source, .2)
        self.assertEqual(result["base_x_mps"], 1.2)
        self.assertAlmostEqual(result["base_x_normalized"], 6.)
        self.assertAlmostEqual(result["delta_x_normalized"], .2)
        self.assertEqual(source["delta_x"], .04)
        self.assertNotIn("final_frame45_error_normalized", result)


if __name__ == "__main__":
    unittest.main()
