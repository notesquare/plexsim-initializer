from numba import njit
import numpy as np
import h5py

from .base import BaseInitializer


@njit
def distribute_random_normal_vector_length(V, l, s):  # noqa
    for i in range(V.shape[0]):
        target_length = np.random.normal(l, s)

        _sum = 0
        for j in range(V.shape[1]):
            v = (np.random.random() - 0.5) * 2
            V[i, j] = v
            _sum += v ** 2
        _l = np.sqrt(_sum)

        _factor = target_length / _l
        for j in range(V.shape[1]):
            V[i, j] *= _factor


def _distribute_random(start, end, avg_velocity,
                       cell_coords, dtype_X, dtype_U):
    n_particles = end - start + 1

    X = np.random.random((n_particles, 3))
    if dtype_X != X.dtype:
        X = np.clip(X, 0, 1 - np.finfo(dtype_X).eps).astype(dtype_X)

    U = np.empty((n_particles, 3), dtype=dtype_U)
    distribute_random_normal_vector_length(
        U, avg_velocity, avg_velocity * 1e-3)

    C_idx = np.full((n_particles, 3), cell_coords)

    U2 = (U * U).sum().item()
    return X, U, C_idx, U2


class RandomInitializer(BaseInitializer):
    def load_particles_pre(self, particles, grid_config):
        initial_condition = grid_config.get('initial_condition', {})
        n_particles = initial_condition.get('n_particles', 0)
        n_particles = int(n_particles)
        avg_velocity = initial_condition.get('avg_velocity', 0)

        gilbert_n_particles = np.random.random(len(self.gilbert_curve))
        gilbert_n_particles = np.round(gilbert_n_particles * n_particles
                                       / gilbert_n_particles.sum()).astype(int)

        _n = n_particles - gilbert_n_particles.sum()
        _indices = np.random.choice(len(self.gilbert_curve), abs(_n))
        _gilbert_n_particles = np.array([
            (_indices == i).sum()
            for i, _ in enumerate(self.gilbert_curve)])

        gilbert_n_particles += (2 * (_n > 0) - 1) * _gilbert_n_particles

        particles.update(dict(
            n_particles=n_particles,
            gilbert_n_particles=gilbert_n_particles,
            avg_velocity=avg_velocity
        ))

    def load_particles(self, h5_fp, prefix, dtype_X, dtype_U, particles):
        gilbert_n_particles = particles['gilbert_n_particles']

        end_indices = gilbert_n_particles.cumsum() - 1
        start_indices = np.empty_like(end_indices).astype(int)
        start_indices[0] = 0
        start_indices[1:] = end_indices[:-1] + 1

        self.distribute_random(
            h5_fp, prefix, start_indices, end_indices,
            np.array(self.gilbert_curve, dtype=np.int16),
            particles, dtype_X, dtype_U
        )

    def distribute_random(self, h5_fp, prefix, start_indices, end_indices,
                          gilbert_curve, particles, dtype_X, dtype_U):
        avg_velocity = particles['avg_velocity']
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

                X, U, C_idx, U2 = _distribute_random(
                    start, end, avg_velocity, cell_coords, dtype_X, dtype_U
                )

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
