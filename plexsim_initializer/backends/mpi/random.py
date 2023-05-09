from itertools import repeat

import h5py
import numpy as np
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from ...initializers import (
    RandomInitializer as _RandomInitializer,
    _distribute_random
)


def pool_initialize(h5_fp, _prefix, _dtype_X, _dtype_U):
    global h5f
    h5f = h5py.File(h5_fp, 'a', driver='mpio', comm=MPI.COMM_WORLD)

    global prefix
    prefix = _prefix
    global dtype_X
    dtype_X = _dtype_X
    global dtype_U
    dtype_U = _dtype_U


def distribute_and_serialize(start, end, cell_coords, avg_velocity):
    global h5f
    global dtype_X
    global dtype_U
    global prefix

    is_exist = (start is not None) and (start <= end)
    if is_exist:
        X, U, C_idx, U2 = _distribute_random(
            start, end, avg_velocity, cell_coords, dtype_X, dtype_U)
        X = np.nextafter(X + C_idx, C_idx)
    else:
        U2 = 0

    axis_labels = ['x', 'y', 'z']
    for i, axis in enumerate(axis_labels):
        # X
        _path = f'{prefix}/position/{axis}'
        X_i = h5f[_path]
        with X_i.collective:
            if is_exist:
                X_i[start:end+1] = X[:, i]
            else:
                X_i[-1:] = None

        # U
        _path = f'{prefix}/momentum/{axis}'
        U_i = h5f[_path]
        with U_i.collective:
            if is_exist:
                U_i[start:end+1] = U[:, i]
            else:
                U_i[-1:] = None

    # TODO: compute state

    return U2


def pool_deinitialize(*args):
    global h5f
    h5f.close()


class RandomInitializer(_RandomInitializer):
    def distribute_random(self, h5_fp, prefix, start_indices, end_indices,
                          gilbert_curve, particles, dtype_X, dtype_U,
                          n_workers=50):
        avg_velocity = particles['avg_velocity']
        m = particles['m']
        n_computational_to_physical = particles['n_computational_to_physical']

        executor = MPIPoolExecutor(
            max_workers=n_workers,
            initializer=pool_initialize,
            initargs=(h5_fp, prefix, dtype_X, dtype_U)
        )
        n_tasks = gilbert_curve.shape[0]
        mod = n_tasks % n_workers
        if mod != 0:
            n_paddings = n_workers - mod

            start_indices = list(start_indices) + [None] * n_paddings
            end_indices = list(end_indices) + [None] * n_paddings
            gilbert_curve = list(gilbert_curve) + [None] * n_paddings

        U2 = executor.map(
            distribute_and_serialize, start_indices, end_indices,
            gilbert_curve, repeat(avg_velocity)
        )
        kinetic_E = 0.5 * m * sum(list(U2)) * n_computational_to_physical

        particles['kinetic_E'] = kinetic_E

        for _ in executor.map(pool_deinitialize, range(n_workers)):
            pass

        axis_labels = ['x', 'y', 'z']
        with h5py.File(h5_fp, 'a') as h5f:
            for i, axis in enumerate(axis_labels):
                _path = f'{prefix}/position/{axis}'
                h5f[_path][particles['n_particles']:] = None

                _path = f'{prefix}/momentum/{axis}'
                h5f[_path][particles['n_particles']:] = None
