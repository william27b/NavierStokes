"""GPU regression checks for open boundaries and low-viscosity channel flow.

Run with: python3 test_solver_stability.py
All solver runs are unsaved; no training archives are created.
"""
from contextlib import redirect_stdout
import io
import unittest

import cupy as cp
import numpy as np

from simulation import D2Q9Solver

C = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
              (1, 1), (-1, 1), (-1, -1), (1, -1)])
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])


class SolverStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = D2Q9Solver()

    def assert_consistent_snapshot(self, result):
        fluid = ~result['solid']
        rho = result['f'].sum(axis=0)
        momentum = np.einsum('qa,qyx->ayx', C, result['f'])
        np.testing.assert_allclose(rho[fluid], result['rho'][fluid], atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(momentum[:, fluid] / rho[fluid], result['u'][:, fluid], atol=1e-13, rtol=1e-13)
        self.assertTrue(np.isfinite(result['f'][:, fluid]).all())
        self.assertTrue((rho[fluid] > 0).all())

    def test_empty_channel_low_tau_converges_with_correct_boundary_labels(self):
        solid = np.zeros((64, 64), dtype=bool)
        solid[[0, -1], :] = True
        with redirect_stdout(io.StringIO()):
            result = self.solver.solve(solid, 'empty_channel_regression', ux=.01, tau=.53, save=False)
        self.assertTrue(result['converged'])
        self.assert_consistent_snapshot(result)
        np.testing.assert_allclose(result['u'][0, 1:-1, 0], .01, atol=1e-13, rtol=0)
        np.testing.assert_allclose(result['u'][1, 1:-1, 0], 0, atol=1e-13, rtol=0)
        np.testing.assert_allclose(result['rho'][1:-1, -1], 1, atol=1e-13, rtol=0)
        fluid = ~solid
        self.assertLess(np.hypot(*result['u'][:, fluid]).max()*np.sqrt(3), .1)
        self.assertLess(np.abs(result['rho'][fluid]-1).max(), .05)
        flux = (result['rho'][1:-1] * result['u'][0, 1:-1]).sum(axis=0)
        imbalance = np.ptp(flux[1:-1]) / np.mean(flux[1:-1])
        self.assertLess(imbalance, .01)
        print(f'Empty channel: {result["steps"]} steps; relative interior flux spread {imbalance:.6g}')

    def test_snapshot_reads_do_not_change_trajectory(self):
        solid = np.zeros((16, 24), dtype=bool)
        solid[[0, -1], :] = True
        results = []
        for check_every in (1, 37):
            with redirect_stdout(io.StringIO()):
                results.append(self.solver.solve(
                    solid, 'snapshot_read_regression', ux=.01, tau=.53,
                    max_steps=500, check_every=check_every, required_stable_checks=1000, save=False))
        np.testing.assert_array_equal(results[0]['f'], results[1]['f'])

    def test_parabolic_channel_matches_profile_and_viscosity(self):
        # Exact incompressible plane-Poiseuille profile between halfway walls.
        # A matching parabolic inlet avoids a developing entrance profile.
        nx, ny, tau, mean_u = 64, 32, .53, .01
        height = ny - 2
        nu = (tau - .5) / 3
        solid = cp.zeros((ny, nx), dtype=cp.bool_)
        solid[[0, -1], :] = True
        y = np.arange(1, ny-1) - .5
        profile = 6*mean_u*y*(height-y)/height**2
        applied = cp.zeros((2, ny, 1), dtype=cp.float64)
        applied[0, 1:-1, 0] = cp.asarray(profile)
        density = cp.broadcast_to(1 + cp.arange(nx-1, -1, -1)[None, :]
                                  * (36*nu*mean_u/height**2), (ny, nx)).copy()
        velocity = cp.zeros((2, ny, nx), dtype=cp.float64)
        velocity[0, 1:-1, :] = cp.asarray(profile)[:, None]
        cu = cp.einsum('qa,ayx->qyx', cp.asarray(C), velocity)
        f = cp.asarray(W)[:, None, None] * density[None] * (1 + 3*cu + 4.5*cu**2 - 1.5*(velocity**2).sum(0)[None])
        f_out, snapshot = cp.empty_like(f), cp.empty_like(f)
        block, grid = (16,16,1), ((nx+15)//16, (ny+15)//16, 1)
        fluid = ~solid
        previous = velocity[:, fluid].copy()
        stable = 0
        for step in range(1, 200001):
            self.solver.fused(grid, block, (f,f_out,density,velocity,applied,solid,np.int32(nx),np.int32(ny),np.float64(tau)))
            f, f_out = f_out, f
            if step % 100:
                continue
            self.solver.reconstruct_state(grid,block,(f,snapshot,density,velocity,applied,solid,np.int32(nx),np.int32(ny)))
            current = velocity[:, fluid]
            change = float(cp.max(cp.abs(current-previous)).get())
            self.assertTrue(np.isfinite(change))
            stable = stable + 1 if change < 1e-10 else 0
            previous = current.copy()
            if stable == 5:
                break
        self.assertEqual(stable, 5)
        rho, ux = density.get(), velocity[0].get()
        profile_error = np.linalg.norm(ux[1:-1, nx//2] - profile) / np.linalg.norm(profile)
        self.assertLess(profile_error, .01)
        columns = np.arange(4, nx-4)
        gradient = np.polyfit(columns, rho[1:-1, columns].mean(0)/3, 1)[0]
        measured_nu = -gradient * height**2 / (12 * rho[1:-1, columns].mean() * ux[1:-1, columns].mean())
        self.assertLess(abs(measured_nu/nu - 1), .02)
        print(f'Poiseuille: profile error {profile_error:.6g}; measured viscosity {measured_nu:.8f}, expected {nu:.8f}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
