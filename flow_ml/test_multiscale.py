import copy
import unittest
import numpy as np
import torch
from multiscale_graph import build_graph, connected_pool, pool, FEATURES
from multiscale_model import MultiscaleGNN, FlowLoss

class MultiscaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_wall_and_distant_door(self):
        solid = np.zeros((16,16), bool)
        solid[:, 3] = True
        solid[14,3] = False
        graph = build_graph(solid, levels=4)
        ids = np.full(solid.shape, -1)
        ids[~solid] = np.arange((~solid).sum())
        # Same 8x8 pooling region, but only connected through a distant door.
        left, right = int(ids[1,2]), int(ids[1,4])
        for level in graph['levels'][:-1]:
            mapping = level['mapping']
            left, right = int(mapping[left]), int(mapping[right])
            self.assertNotEqual(left, right)
        # Every coarse edge has an actual edge at the preceding finer level.
        for fine, coarse in zip(graph['levels'][:-1], graph['levels'][1:]):
            mapped = fine['mapping'][fine['edges']]
            expected = {tuple(e) for e in mapped.T.tolist() if e[0] != e[1]}
            self.assertEqual(expected, {tuple(e) for e in coarse['edges'].T.tolist()})

    def test_disconnected_regions_never_merge(self):
        solid = np.zeros((16,16), bool)
        solid[:,7] = True
        graph = build_graph(solid, levels=6)
        self.assertEqual(len(graph['levels'][-1]['weights']), 2)
        self.assertEqual(graph['levels'][-1]['edges'].shape[1], 0)

    def test_pooling_conserves_weighted_mean(self):
        solid = np.zeros((17,19), bool)
        solid[3:9,5:8] = True
        graph = build_graph(solid)
        values = torch.randn(len(graph['x']),3)
        original = values.mean(0)
        for fine, coarse in zip(graph['levels'][:-1], graph['levels'][1:]):
            values = pool(values, fine, coarse)
            average = (values*coarse['weights'][:,None]).sum(0)/coarse['weights'].sum()
            torch.testing.assert_close(average, original)

    def test_boundary_features_and_exact_conditions(self):
        solid = np.zeros((16,16), bool)
        solid[[0,-1]] = True
        solid[6,7] = True
        graph = build_graph(solid, inlet_velocity=(.012,.002), levels=3)
        pos = graph['levels'][0]['positions']
        node = torch.where((pos[:,0] == 6) & (pos[:,1] == 6))[0][0]
        self.assertEqual(graph['x'][node, FEATURES.index('solid_right')], 1)
        model = MultiscaleGNN(hidden=16, levels=3, local_steps=1, coarse_steps=2)
        prediction = model(graph)
        torch.testing.assert_close(prediction[graph['bc_mask']], graph['bc_values'][graph['bc_mask']], rtol=0, atol=0)

    def test_permutation_equivariance_all_levels(self):
        solid = np.zeros((16,16), bool)
        solid[4:7,6:9] = True
        graph = build_graph(solid, levels=3)
        shuffled = copy.deepcopy(graph)
        orders = [torch.randperm(len(level['weights'])) for level in graph['levels']]
        inverse = [torch.argsort(order) for order in orders]
        for k, level in enumerate(graph['levels']):
            out = shuffled['levels'][k]
            for name in ('geom', 'weights', 'positions'):
                out[name] = level[name][orders[k]]
            out['edges'] = inverse[k][level['edges']]
            if 'mapping' in level:
                out['mapping'] = inverse[k+1][level['mapping'][orders[k]]]
        for name in ('x', 'bc_mask', 'bc_values', 'far_mask', 'halo_mask'):
            shuffled[name] = graph[name][orders[0]]
        model = MultiscaleGNN(hidden=16, levels=3, local_steps=1, coarse_steps=3).eval()
        with torch.no_grad():
            expected = model(graph)[orders[0]]
            actual = model(shuffled)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_matching_loss_and_constant_pressure_offset(self):
        graph = build_graph(np.zeros((16,16),bool), levels=3)
        graph['y'] = torch.randn(len(graph['x']),3)
        loss = FlowLoss([1,1,1], [1,1,1])
        total, parts = loss(graph['y'], graph)
        self.assertEqual(float(total), 0)
        shifted = graph['y'].clone()
        shifted[:,2] += 2
        total, parts = loss(shifted, graph)
        self.assertGreater(float(total), 0)
        self.assertAlmostEqual(float(parts[1]), 0, places=10)
        self.assertAlmostEqual(float(parts[2]), 4, places=5)

    def test_distant_information_has_gradient_path(self):
        solid = np.zeros((64,64),bool)
        solid[[0,-1]] = True
        graph = build_graph(solid, levels=5)
        graph['x'].requires_grad_()
        model = MultiscaleGNN(hidden=16, levels=5, local_steps=1, coarse_steps=8)
        pos = graph['levels'][0]['positions']
        target = torch.where((pos[:,0] == 8) & (pos[:,1] == 32))[0][0]
        distant = (pos[:,0] > 48) & (pos[:,1] > 16) & (pos[:,1] < 48)
        model(graph)[target,2].backward()
        self.assertGreater(float(graph['x'].grad[distant].abs().sum()), 1e-12)

if __name__ == '__main__':
    unittest.main(verbosity=2)
