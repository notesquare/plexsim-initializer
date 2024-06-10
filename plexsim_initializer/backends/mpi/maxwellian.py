import h5py
import numpy as np
from mpi4py import MPI

from .common import gen_arg, MPIInitializer
from ...initializers import (
    MaxwellianInitializer as _MaxwellianInitializer,
    _distribute_maxwellian
)


def distribute_and_serialize(
        start, end, cell_coords, h5_fp, _prefix,
        _dtype_X, _dtype_U, grid_vertex_shape, _save_state, _axis_labels,
        _c, _coordinate_system, _cylindrical_args, _thermal_velocity,
        _drifted_velocity, _grid_shape, _r0, _dr,
        _halton_sequence, _n_particles_in_grid):
    global h5f
    global prefix
    global dtype_X
    global dtype_U
    global grid_n
    global grid_U
    global grid_U2
    global save_state
    global axis_labels
    global n_particles_in_grid
    global halton_sequence
    global c
    global coordinate_system
    global cylindrical_args
    global thermal_velocity
    global drifted_velocity
    global grid_shape
    global r0
    global dr

    if h5_fp is not None:
        # init
        prefix = _prefix
        dtype_X = _dtype_X
        dtype_U = _dtype_U
        save_state = _save_state
        axis_labels = _axis_labels
        n_particles_in_grid = _n_particles_in_grid
        halton_sequence = _halton_sequence
        c = _c
        coordinate_system = _coordinate_system
        cylindrical_args = _cylindrical_args
        thermal_velocity = _thermal_velocity
        drifted_velocity = _drifted_velocity
        grid_shape = _grid_shape
        r0 = _r0
        dr = _dr
        if MPI.Comm.Get_parent() == MPI.COMM_NULL:
            # static mode
            h5_comm = MPI.COMM_WORLD.Split(0)
        else:
            # dynamic mode
            h5_comm = MPI.COMM_WORLD
        h5f = h5py.File(h5_fp, 'a', driver='mpio', comm=h5_comm)

        if save_state:
            grid_n = np.zeros(grid_vertex_shape, dtype=np.float64)
            grid_U = np.zeros((*grid_vertex_shape, 3), dtype=np.float64)
            grid_U2 = np.zeros((*grid_vertex_shape, 3), dtype=np.float64)

    is_exist = (start is not None) and (start <= end)
    if is_exist:
        X, U, C_idx, U2 = _distribute_maxwellian(
            start, end, thermal_velocity, drifted_velocity, cell_coords,
            grid_shape, r0, dr, halton_sequence, dtype_X, dtype_U,
            n_particles_in_grid, c
        )

        if save_state:
            if coordinate_system == 'cartesian':
                from ...lib.cartesian import compute_grid_velocity
                compute_grid_velocity(
                    X, U, C_idx, grid_n, grid_U, grid_U2)
            elif coordinate_system == 'cylindrical':
                from ...lib.cylindrical import compute_grid_velocity
                dr, r0, nphi = cylindrical_args
                compute_grid_velocity(
                    X, U, C_idx, grid_n, grid_U, grid_U2,
                    dr, r0, nphi)

        X = np.nextafter(X + C_idx, C_idx)
    else:
        U2 = 0

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


def pool_finalize(*args):
    global h5f
    global grid_n
    global grid_U
    global grid_U2
    global save_state

    h5f.close()

    if save_state:
        return grid_n, grid_U, grid_U2
    else:
        return None, None, None


class MaxwellianInitializer(MPIInitializer, _MaxwellianInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def distribute_maxwellian(self, h5_fp, prefix, start_indices, end_indices,
                              gilbert_curve, halton_sequence, particles,
                              dtype_X, dtype_U, _m, _c):
        thermal_velocity = particles['thermal_velocity']
        drifted_velocity = particles['drifted_velocity']
        m = particles['m'] * _m
        n_computational_to_physical = particles['n_computational_to_physical']
        n_particles_in_grid = particles['n_particles_in_grid']

        if self.save_state:
            grid_n = np.zeros(self.grid_vertex_shape, dtype=np.float64)
            grid_U = np.zeros((*self.grid_vertex_shape, 3), dtype=np.float64)
            grid_U2 = np.zeros((*self.grid_vertex_shape, 3), dtype=np.float64)

        max_workers = self.max_workers
        n_tasks = gilbert_curve.shape[0]
        mod = n_tasks % max_workers
        if mod != 0:
            n_paddings = max_workers - mod

            start_indices = list(start_indices) + [None] * n_paddings
            end_indices = list(end_indices) + [None] * n_paddings
            gilbert_curve = list(gilbert_curve) + [None] * n_paddings

        cylindrical_args = None
        if self.coordinate_system == 'cylindrical':
            cylindrical_args = (self.cell_size[1], self.r0, self.grid_shape[2])
        U2 = self.executor.map(
            distribute_and_serialize, start_indices, end_indices,
            gilbert_curve,
            gen_arg(h5_fp, max_workers),
            gen_arg(prefix, max_workers),
            gen_arg(dtype_X, max_workers),
            gen_arg(dtype_U, max_workers),
            gen_arg(self.grid_vertex_shape, max_workers),
            gen_arg(self.save_state, max_workers),
            gen_arg(self.axis_labels, max_workers),
            gen_arg(_c, max_workers),
            gen_arg(self.coordinate_system, max_workers),
            gen_arg(cylindrical_args, max_workers),
            gen_arg(thermal_velocity, max_workers),
            gen_arg(drifted_velocity, max_workers),
            gen_arg(self.grid_shape, max_workers),
            gen_arg(self.r0, max_workers),
            gen_arg(self.cell_size[1], max_workers),
            gen_arg(halton_sequence, max_workers),
            gen_arg(n_particles_in_grid, max_workers),
        )
        if MPI.COMM_WORLD.Get_size() > 1:
            # static mode
            MPI.COMM_WORLD.Split(MPI.UNDEFINED, 0)
        kinetic_E = 0.5 * m * sum(list(U2)) * n_computational_to_physical

        particles['kinetic_E'] = kinetic_E

        for state in self.executor.map(pool_finalize, range(max_workers)):
            if self.save_state:
                _grid_n, _grid_U, _grid_U2 = state
                grid_n += _grid_n
                grid_U += _grid_U
                grid_U2 += _grid_U2

        if self.save_state:
            particles.update(dict(
                grid_n=grid_n,
                grid_U=grid_U,
                grid_U2=grid_U2
            ))

        with h5py.File(h5_fp, 'a') as h5f:
            for i, axis in enumerate(self.axis_labels):
                _path = f'{prefix}/position/{axis}'
                h5f[_path][particles['n_particles']:] = None

                _path = f'{prefix}/momentum/{axis}'
                h5f[_path][particles['n_particles']:] = None
