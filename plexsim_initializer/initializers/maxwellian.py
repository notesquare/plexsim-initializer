import math

import numpy as np
import h5py

from .base import BaseInitializer
from ..lib.common import node_to_center_3d


def get_v_table(nvts=3.3):
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

    return v_table


def _distribute_maxwellian(start, end, vth, velocity, cell_coords,
                           v_table, dtype_X, dtype_U):
    n_particles = end - start + 1

    X = np.random.random((n_particles, 3))
    if dtype_X != X.dtype:
        X = np.clip(X, 0, 1 - np.finfo(dtype_X).eps).astype(dtype_X)
    U = np.empty((n_particles, 3), dtype=dtype_U)

    n_velocity_dist = v_table.shape[0]
    indices = np.random.choice(n_velocity_dist, n_particles)
    U[:] = vth * v_table[indices] + velocity
    C_idx = np.full((n_particles, 3), cell_coords)

    U2 = (U * U).sum().item()
    return X, U, C_idx, U2


class MaxwellianInitializer(BaseInitializer):
    def load_particles_pre(self, particles, grid_config):
        q = particles['q']
        m = particles['m']

        initial_condition = grid_config.get('initial_condition', {})
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

        particles.update(dict(
            n_particles=gilbert_n_particles.sum(),
            gilbert_n_particles=gilbert_n_particles,
            gilbert_vth=gilbert_vth,
            gilbert_drifted_velocity=gilbert_drifted_velocity
        ))

    def load_particles(self, h5_fp, prefix, dtype_X, dtype_U, particles):
        gilbert_n_particles = particles['gilbert_n_particles']

        end_indices = gilbert_n_particles.cumsum() - 1
        start_indices = np.empty_like(end_indices).astype(int)
        start_indices[0] = 0
        start_indices[1:] = end_indices[:-1] + 1

        self.distribute_maxwellian(
            h5_fp, prefix, start_indices, end_indices,
            np.array(self.gilbert_curve, dtype=np.int16),
            get_v_table(), particles, dtype_X, dtype_U
        )

    def distribute_maxwellian(self, h5_fp, prefix, start_indices, end_indices,
                              gilbert_curve, v_table, particles,
                              dtype_X, dtype_U):
        vth_list = particles['gilbert_vth']
        velocity_list = particles['gilbert_drifted_velocity']
        m = particles['m']
        n_computational_to_physical = particles['n_computational_to_physical']

        axis_labels = ['x', 'y', 'z']
        with h5py.File(h5_fp, 'a') as h5f:
            kinetic_E = 0
            for cell_index, cell_coords in enumerate(gilbert_curve):
                start = start_indices[cell_index]
                end = end_indices[cell_index]
                if start == end + 1:
                    continue
                vth = vth_list[cell_index]
                velocity = velocity_list[cell_index]

                X, U, C_idx, U2 = _distribute_maxwellian(
                    start, end, vth, velocity, cell_coords, v_table,
                    dtype_X, dtype_U)

                # serialize
                X = np.nextafter(X + C_idx, C_idx)
                for i, axis in enumerate(axis_labels):
                    # X
                    _path = f'{prefix}/position/{axis}'
                    h5f[_path][start:end+1] = X[:, i]

                    # U
                    _path = f'{prefix}/momentum/{axis}'
                    h5f[_path][start:end+1] = U[:, i]

                # TODO: compute state

                kinetic_E += 0.5 * m * U2 * n_computational_to_physical
            particles['kinetic_E'] = kinetic_E

            for i, axis in enumerate(axis_labels):
                _path = f'{prefix}/position/{axis}'
                h5f[_path][end+1:] = None

                _path = f'{prefix}/momentum/{axis}'
                h5f[_path][end+1:] = None
