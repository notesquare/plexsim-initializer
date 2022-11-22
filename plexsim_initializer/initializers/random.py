from numba import njit
import numpy as np

from .base import BaseInitializer


@njit
def distribute_random_normal_vector_length(V, l, s):  # noqa
    for i in range(V.shape[0]):
        target_length = np.random.normal(l, s)

        _sum = 0
        for j in range(V.shape[1]):
            v = np.random.random()
            V[i, j] = (v - 0.5) * 2
            _sum += v ** 2
        _l = np.sqrt(_sum)

        _factor = target_length / _l
        for j in range(V.shape[1]):
            V[i, j] *= _factor


@njit
def distribute_random(C_idx, start_indices, end_indices, gilbert_curve):
    for i in range(gilbert_curve.shape[0]):
        start = start_indices[i]
        end = end_indices[i]
        for j in range(gilbert_curve.shape[1]):
            C_idx[start:end+1, j] = gilbert_curve[i, j]


class RandomInitializer(BaseInitializer):
    def load_particles(self, dtype_X, dtype_U, particles, grid_config):
        initial_condition = grid_config.get('initial_condition', {})
        n_particles = initial_condition.get('n_particles', 0)
        n_particles = int(n_particles)

        avg_velocity = initial_condition.get('avg_velocity', 0)
        X = np.random.random((n_particles, 3)).astype(dtype_X)
        U = np.empty((n_particles, 3), dtype=dtype_U)

        distribute_random_normal_vector_length(
            U, avg_velocity, avg_velocity * 1e-3)

        gilbert_n_particles = np.random.random(len(self.gilbert_curve))
        gilbert_n_particles = np.round(gilbert_n_particles * n_particles
                                       / gilbert_n_particles.sum()).astype(int)

        _n = n_particles - gilbert_n_particles.sum()
        _indices = np.random.randint(0, len(self.gilbert_curve), abs(_n))
        _gilbert_n_particles = np.array([
            (_indices == i).sum()
            for i, _ in enumerate(self.gilbert_curve)])

        gilbert_n_particles += (2 * (_n > 0) - 1) * _gilbert_n_particles

        end_indices = gilbert_n_particles.cumsum() - 1
        start_indices = np.empty_like(end_indices).astype(int)
        start_indices[0] = 0
        start_indices[1:] = end_indices[:-1] + 1

        C_idx = np.empty((n_particles, 3), dtype=np.int16)

        distribute_random(C_idx, start_indices, end_indices,
                          np.array(self.gilbert_curve))

        particles.update(dict(
            X=X,
            U=U,
            C_idx=C_idx,
            gilbert_n_particles=gilbert_n_particles
        ))
