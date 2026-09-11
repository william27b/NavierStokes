"""Analytic metric, coordinate, and frozen-inference checks; no solver runs."""
import unittest
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn

from evaluate_dataset import (SCALE, OFFSET, channel_mse, metric_summary,
                              fit_constant, reconstruct_grid, frozen_predict, evaluate_dataset)
from model import GNN


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_equal_cases_and_physical_units(self):
        # Four small-error nodes versus one large-error node. Case averaging
        # gives (1+9)/2=5, unlike node averaging (4+9)/5=2.6.
        a = channel_mse(np.tile([1., 2., 3.], (4, 1)), np.zeros((4, 3)))
        b = channel_mse(np.array([[3., 6., 9.]]), np.zeros((1, 3)))
        result = metric_summary([a, b])
        np.testing.assert_allclose(result['normalized_mse'], [5., 20., 45.])
        self.assertAlmostEqual(result['loss'], 70/3)
        np.testing.assert_allclose(result['lattice_rmse'], np.sqrt([5., 20., 45.])*SCALE)
        np.testing.assert_array_equal(channel_mse(np.ones((2, 3)), np.ones((2, 3))), np.zeros(3))

    def test_best_constant_respects_case_weights(self):
        cases = [np.zeros((4, 3)), np.array([[2., 4., 6.]])]
        best = fit_constant(cases)
        np.testing.assert_array_equal(best, [1., 2., 3.])
        optimal = metric_summary([channel_mse(np.broadcast_to(best, y.shape), y) for y in cases])['loss']
        for shift in [-.1, .1]:
            other = metric_summary([channel_mse(np.broadcast_to(best+shift, y.shape), y) for y in cases])['loss']
            self.assertGreater(other, optimal)

    def test_grid_mapping_is_coordinate_based_and_masks_solids(self):
        solid = np.array([[False, True], [False, False]])
        positions = np.array([[1, 1], [0, 0], [0, 1]])
        normalized = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        physical = normalized*SCALE+OFFSET
        grid = reconstruct_grid(physical, positions, solid)
        np.testing.assert_array_equal(grid[1, 1], physical[0])
        np.testing.assert_array_equal(grid[0, 0], physical[1])
        self.assertTrue(np.isnan(grid[0, 1]).all())
        np.testing.assert_allclose((grid[positions[:, 1], positions[:, 0]]-OFFSET)/SCALE, normalized)
        with self.assertRaises(ValueError):
            reconstruct_grid(physical, np.array([[0, 0], [0, 0], [0, 1]]), solid)

    def test_nonfinite_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            channel_mse(np.array([[np.nan, 0, 0]]), np.zeros((1, 3)))

    def test_inference_disables_dropout_gradients_and_updates(self):
        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 3)
                self.dropout = nn.Dropout(.9)
                self.grad_enabled = None

            def forward(self, x, edge_index, edge_attr):
                self.grad_enabled = torch.is_grad_enabled()
                return self.dropout(self.linear(x))

        model = Probe().train()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        inputs = {'x': torch.ones(4, 2), 'y': torch.zeros(4, 3),
                  'edge_index': torch.empty((2, 0), dtype=torch.long), 'edge_attr': torch.empty(0, 2)}
        first = frozen_predict(model, inputs)
        second = frozen_predict(model, inputs)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(model.training)
        self.assertFalse(model.grad_enabled)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_end_to_end_archives_checkpoint_baselines_and_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root/'runs'
            expected_mses = []
            for index, speed in enumerate([.01, .02]):
                run = run_root/f'case_{index}'
                run.mkdir(parents=True)
                solid = np.zeros((8, 8), dtype=bool)
                solid[[0, -1], :] = True
                if index:
                    solid[3:5, 3:5] = True
                u = np.zeros((2, 8, 8))
                u[0] = speed
                pressure = np.full((8, 8), 1/3)
                np.savez(run/'fields.npz', f=np.zeros((9, 8, 8)), rho=pressure*3,
                         u=u, pressure=pressure, solid=solid, applied_u=u,
                         solid_name='fixture', nx=8, ny=8, tau=.53, steps=1,
                         max_steps=1, converged=True, stop_reason='converged')
                (run/'metadata.json').write_text(json.dumps({'converged': True,
                    'invalid_fluid_cells': 0, 'generation': {'seed': index}}))
                expected_mses.append([float(np.float32(speed/.05))**2, 0., 0.])
            model = GNN(hidden=64, layers=1, global_context=True)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            checkpoint = root/'model.pt'
            torch.save({'model_state': model.state_dict(), 'hidden': 64,
                        'layers': 1, 'global_context': True}, checkpoint)
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            report = evaluate_dataset(checkpoint, run_root, output_dir=root/'report', count=1, device='cpu')
            np.testing.assert_allclose(report['metrics']['model']['normalized_mse'],
                                       np.mean(expected_mses, axis=0), rtol=1e-7)
            np.testing.assert_allclose(report['metrics']['inlet_constant']['loss'], 0, atol=1e-15)
            self.assertAlmostEqual(report['label_fitted_constant_lattice'][0], .015, places=8)
            self.assertEqual(report['case_count'], 2)
            self.assertEqual(report['checkpoint_sha256'], before)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)
            self.assertEqual(len(list((root/'report').glob('*.png'))), 4)
            stored = json.loads((root/'report'/'metrics.json').read_text())
            self.assertEqual(stored['metrics'], report['metrics'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
