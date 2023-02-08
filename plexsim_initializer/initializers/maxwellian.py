import numpy as np
from numba import njit

from .base import BaseInitializer
from ..lib.common import node_to_center_3d


@njit
def distribute_maxwellian(C_idx, U, start_indices, end_indices,
                          gilbert_curve, t_list, n_list, j_array, q, m):
    for i in range(gilbert_curve.shape[0]):
        n = n_list[i]
        if n == 0:
            continue

        start = start_indices[i]
        end = end_indices[i]
        for j in range(gilbert_curve.shape[1]):
            C_idx[start:end+1, j] = gilbert_curve[i, j]

        t = t_list[i]
        sigma = np.sqrt(np.abs(q * t / m))
        for k in range(U.shape[1]):
            current_density = j_array[i, k]
            mu = current_density / (q * n)
            U[start:end+1, k] = np.random.normal(mu, sigma, end - start + 1)


class MaxwellianInitializer(BaseInitializer):
    def load_particles(self, dtype_X, dtype_U, particles, grid_config):
        initial_condition = grid_config.get('initial_condition', {})

        q = particles['q']
        m = particles['m']

        temperature = initial_condition['temperature']
        density = initial_condition['density']
        current_density = initial_condition.get('current_density')

        n_computational_to_physical = int(
                grid_config['n_computational_to_physical'])

        temperature = self.load_relative_npy_file(temperature)
        density = self.load_relative_npy_file(density)

        if current_density is not None:
            current_density = self.load_relative_npy_file(current_density)
        else:
            current_density = np.zeros((*(self.grid_shape+1), 3))

        assert tuple(self.grid_shape + 1) == temperature.shape \
            == density.shape == current_density.shape[:-1]

        temperature_center = np.empty(self.grid_shape, dtype=temperature.dtype)
        density_center = np.empty(self.grid_shape, dtype=density.dtype)
        j_center = np.empty(current_density.shape, dtype=current_density.dtype)

        node_to_center_3d(temperature, temperature_center)
        node_to_center_3d(density, density_center)
        node_to_center_3d(current_density, j_center)

        n_particles_in_cell = density_center * \
            self.cell_size.prod() / n_computational_to_physical
        assert np.all(n_particles_in_cell < np.iinfo(np.int64).max)
        n_particles_in_cell = np.around(n_particles_in_cell).astype(int)

        gilbert_n_particles = np.array([
            n_particles_in_cell[coord] for coord in self.gilbert_curve])
        gilbert_temperature = np.array([
            temperature_center[coord] for coord in self.gilbert_curve])
        gilbert_density = np.array([
            density_center[coord] for coord in self.gilbert_curve])
        gilbert_j = np.array([j_center[coord] for coord in self.gilbert_curve])
        end_indices = gilbert_n_particles.cumsum() - 1
        start_indices = np.empty_like(end_indices).astype(int)
        start_indices[0] = 0
        start_indices[1:] = end_indices[:-1] + 1

        n_particles = gilbert_n_particles.sum()

        X = np.random.random((n_particles, 3))
        if dtype_X != X.dtype:
            X = np.clip(X, 0, 1 - np.finfo(dtype_X).eps).astype(dtype_X)
        C_idx = np.empty((n_particles, 3), dtype=np.int16)
        U = np.empty((n_particles, 3), dtype=dtype_U)

        distribute_maxwellian(C_idx, U, start_indices, end_indices,
                              np.array(self.gilbert_curve),
                              gilbert_temperature, gilbert_density,
                              gilbert_j, q, m)

        particles.update(dict(
            X=X,
            U=U,
            C_idx=C_idx,
            gilbert_n_particles=gilbert_n_particles
        ))
