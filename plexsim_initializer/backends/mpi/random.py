import h5py
import numpy as np
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from mpi4py.futures._lib import get_max_workers

from .common import (
    create_shm_array,
    gen_arg
)
from ...initializers import (
    RandomInitializer as _RandomInitializer,
    _distribute_random
)
from ...lib.common import compute_grid_velocity_disjunct


def distribute_and_serialize(start, end, cell_coords, h5_fp,
                             _prefix, _avg_velocity, _dtype_X, _dtype_U,
                             grid_shape, _save_state):
    global h5f
    global prefix
    global avg_velocity
    global dtype_X
    global dtype_U
    global grid_n_d
    global grid_U_d
    global grid_U2_d
    global save_state

    if h5_fp is not None:
        # init
        prefix = _prefix
        dtype_X = _dtype_X
        dtype_U = _dtype_U
        avg_velocity = _avg_velocity
        save_state = _save_state

        if MPI.Comm.Get_parent() == MPI.COMM_NULL:
            # static mode
            h5_comm = MPI.COMM_WORLD.Split(0)
        else:
            # dynamic mode
            h5_comm = MPI.COMM_WORLD
        h5f = h5py.File(h5_fp, 'a', driver='mpio', comm=h5_comm)

        if save_state:
            if h5_comm.Get_rank() == 0:
                grid_n_d = create_shm_array(
                    (grid_shape * 2), h5_comm, data=np.float64(0), main=True)
                grid_U_d = create_shm_array(
                    (*(grid_shape * 2), 3), h5_comm,
                    data=np.float64(0), main=True)
                grid_U2_d = create_shm_array(
                    (grid_shape * 2), h5_comm, data=np.float64(0), main=True)
            else:
                grid_n_d = create_shm_array(
                    (grid_shape * 2), h5_comm, dtype=np.float64)
                grid_U_d = create_shm_array(
                    (*(grid_shape * 2), 3), h5_comm, dtype=np.float64)
                grid_U2_d = create_shm_array(
                    (grid_shape * 2), h5_comm, dtype=np.float64)

    is_exist = (start is not None) and (start <= end)
    if is_exist:
        X, U, C_idx, U2 = _distribute_random(
            start, end, avg_velocity, cell_coords, dtype_X, dtype_U)

        if save_state:
            compute_grid_velocity_disjunct(
                X, U, C_idx, grid_n_d, grid_U_d, grid_U2_d)

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

    return U2


def pool_finalize(index):
    global h5f
    global grid_n_d
    global grid_U_d
    global grid_U2_d
    global save_state

    h5f.close()

    if save_state and index == 0:
        return grid_n_d, grid_U_d, grid_U2_d
    else:
        return None, None, None


class RandomInitializer(_RandomInitializer):
    def distribute_random(self, h5_fp, prefix, start_indices, end_indices,
                          gilbert_curve, particles, dtype_X, dtype_U):
        avg_velocity = particles['avg_velocity']
        m = particles['m']
        n_computational_to_physical = particles['n_computational_to_physical']

        if MPI.COMM_WORLD.Get_size() > 1:
            # static mode
            max_workers = MPI.COMM_WORLD.Get_size() - 1
        else:
            # dynamic mode
            max_workers = get_max_workers()
            if max_workers < 1:
                max_workers = 1

        with MPIPoolExecutor(max_workers=max_workers) as executor:
            n_tasks = gilbert_curve.shape[0]
            mod = n_tasks % max_workers
            if mod != 0:
                n_paddings = max_workers - mod

                start_indices = list(start_indices) + [None] * n_paddings
                end_indices = list(end_indices) + [None] * n_paddings
                gilbert_curve = list(gilbert_curve) + [None] * n_paddings

            U2 = executor.map(
                distribute_and_serialize, start_indices, end_indices,
                gilbert_curve, gen_arg(h5_fp, max_workers),
                gen_arg(prefix, max_workers),
                gen_arg(avg_velocity, max_workers),
                gen_arg(dtype_X, max_workers), gen_arg(dtype_U, max_workers),
                gen_arg(self.grid_shape, max_workers),
                gen_arg(self.save_state, max_workers)
            )
            if MPI.COMM_WORLD.Get_size() > 1:
                # static mode
                MPI.COMM_WORLD.Split(MPI.UNDEFINED, 0)
            kinetic_E = 0.5 * m * sum(list(U2)) * n_computational_to_physical

            particles['kinetic_E'] = kinetic_E

            for state in executor.map(pool_finalize, range(max_workers)):
                grid_n_d, grid_U_d, grid_U2_d = state
                if grid_n_d is not None:
                    particles.update(dict(
                        grid_n_d=grid_n_d,
                        grid_U_d=grid_U_d,
                        grid_U2_d=grid_U2_d
                    ))

        axis_labels = ['x', 'y', 'z']
        with h5py.File(h5_fp, 'a') as h5f:
            for i, axis in enumerate(axis_labels):
                _path = f'{prefix}/position/{axis}'
                h5f[_path][particles['n_particles']:] = None

                _path = f'{prefix}/momentum/{axis}'
                h5f[_path][particles['n_particles']:] = None
