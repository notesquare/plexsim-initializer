import math

import numpy as np
from numba import njit

from .base import BaseInitializer
from ..lib.common import node_to_center_3d


@njit
def distribute_maxwellian(C_idx, U, start_indices, end_indices, gilbert_curve,
                          vth_list, velocity_list, nvts=3.3):
    def f(v):
        # cumulative distribution function of speed
        return -2 * v * np.exp(-v * v) / np.sqrt(np.pi) + math.erf(v)

    n_velocity_dist = int(1 / (1 - f(nvts)))
    dv = np.sqrt(np.pi) / (4 * n_velocity_dist)
    v_table = np.empty((n_velocity_dist, 3))

    i = 0
    for n in range(n_velocity_dist):
        target_value = n / n_velocity_dist
        while f(i * dv) < target_value:
            i += 1
        vmag = i * dv
        aphi = 2 * np.pi * np.random.random()
        cos_theta = 1 - 2 * np.random.random()
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        v_table[n] = vmag * np.array([sin_theta * np.cos(aphi),
                                      sin_theta * np.sin(aphi),
                                      cos_theta])

    for i in range(gilbert_curve.shape[0]):
        vth = vth_list[i]
        velocity = velocity_list[i]

        start = start_indices[i]
        end = end_indices[i]
        indices = np.random.choice(n_velocity_dist, end - start + 1)

        U[start:end+1] = vth * v_table[indices] + velocity
        C_idx[start:end+1] = gilbert_curve[i]


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

        thermal_velocity = np.sqrt(np.abs(2 * q * temperature) / m)

        vth_center = np.empty(self.grid_shape, dtype=thermal_velocity.dtype)
        density_center = np.empty(self.grid_shape, dtype=density.dtype)
        j_center = np.empty((*self.grid_shape, 3), dtype=current_density.dtype)

        node_to_center_3d(thermal_velocity, vth_center)
        node_to_center_3d(density, density_center)
        node_to_center_3d(current_density, j_center)

        n_particles_in_cell = density_center * \
            self.cell_size.prod() / n_computational_to_physical
        assert np.all(n_particles_in_cell < np.iinfo(np.int64).max)
        n_particles_in_cell = np.around(n_particles_in_cell).astype(int)

        gilbert_n_particles = np.array([
            n_particles_in_cell[coord] for coord in self.gilbert_curve])

        _density_center = np.expand_dims(density_center, axis=-1)
        drifted_velocity = np.divide(j_center, q * _density_center,
                                     out=np.zeros_like(j_center),
                                     where=_density_center != 0)

        gilbert_vth = np.array([
            vth_center[coord] for coord in self.gilbert_curve])
        gilbert_drifted_velocity = np.array([
            drifted_velocity[coord] for coord in self.gilbert_curve])

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
                              np.array(self.gilbert_curve), gilbert_vth,
                              gilbert_drifted_velocity)

        particles.update(dict(
            X=X,
            U=U,
            C_idx=C_idx,
            gilbert_n_particles=gilbert_n_particles
        ))
