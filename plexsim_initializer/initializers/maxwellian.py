import math

import numpy as np
import h5py
from numba import njit, prange

from .base import BaseInitializer
from ..lib.common import node_to_center_3d


@njit  # (parallel=True) option is effective only if n_velocity_dist >= 1000
def get_v_table(n_velocity_dist):
    def f(v):
        # cumulative distribution function of speed
        return -2 * v * np.exp(-v * v) / np.sqrt(np.pi) + math.erf(v)

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


@njit
def _halton_sequence(num, base):
    x = 0
    _base = 1 / base
    while num > 0:
        x += (num % base) * _base
        _base /= base
        num //= base
    return x


@njit(parallel=True)
def get_halton_sequence(max_n_particles, bases=(2, 3, 5)):
    halton_sequence = np.empty((max_n_particles, len(bases)))
    for n in prange(max_n_particles):
        for i, base in enumerate(bases):
            halton_sequence[n, i] = _halton_sequence(n + 1, base)
    return halton_sequence


@njit
def rejection_sampling(X, n_grid, halton_sequence, bases=(2, 3, 5)):
    assert X.shape[0] <= halton_sequence.shape[0]
    assert X.shape[1] == halton_sequence.shape[1]

    n_particles = X.shape[0]
    X[:] = halton_sequence[:n_particles]

    c = n_grid.max()
    c1 = (n_grid[1, 1] - n_grid[0, 1] - n_grid[1, 0] + n_grid[0, 0]) / c
    c2 = (n_grid[1, 0] - n_grid[0, 0]) / c
    c3 = (n_grid[0, 1] - n_grid[0, 0]) / c
    c4 = (n_grid[0, 0]) / c

    c = (c1 * X[:, 0] * X[:, 1] + c2 * X[:, 0] + c3 * X[:, 1] + c4)
    mask = np.random.random(n_particles) > c
    n_repos = mask.sum()

    c1 = c1 / 2.0
    c2 = c2 / 2.0
    c3 = c3 / 2.0
    c4 = c4 / 2.0

    z_re = np.zeros(n_repos)
    r_re = np.zeros(n_repos)
    i = n_particles - 1
    for j in range(n_repos):
        while True:
            i = i + 1
            if i < len(halton_sequence):
                z, r, _ = halton_sequence[i]
            else:
                z = _halton_sequence(i + 1, bases[0])
                r = _halton_sequence(i + 1, bases[1])
            c = (c1 * r * z + c2 * z + c3 * r + c4)
            if c > np.random.random():
                break
        r_re[j] = r
        z_re[j] = z

    X[mask, 0] = z_re
    X[mask, 1] = r_re
    return X


def _distribute_maxwellian(
        start, end, thermal_velocity, drifted_velocity, cell_coords,
        grid_shape, r0, dr, halton_sequence, dtype_X, dtype_U,
        n_particles_in_grid, _c):
    n_particles = end - start + 1

    X = np.empty((n_particles, 3), dtype=dtype_X)
    U = np.empty((n_particles, 3), dtype=dtype_U)
    C_idx = np.full((n_particles, 3), cell_coords)

    _X = X[:n_particles // 2]
    ci, cj, ck = cell_coords
    n_grid = n_particles_in_grid[ci:ci+2, cj:cj+2, ck]
    rejection_sampling(_X, n_grid, halton_sequence)

    if dtype_X != X.dtype:
        _X = np.clip(_X, 0, 1 - np.finfo(dtype_X).eps).astype(dtype_X)

    X[_X.shape[0]: _X.shape[0] * 2] = _X[:]

    z, r, phi = [_X[..., i] for i in range(3)]

    xi = (1 - z, z)
    eta = ((1 - r) * (2 * r0 + dr * (2 * cj + r + 1)),
           r * (2 * r0 + dr * (2 * cj + r)))
    eta_sum = (eta[0] + eta[1])
    eta = eta[0] / eta_sum, eta[1] / eta_sum
    zeta = (1 - phi, phi)

    _U = U[:n_particles // 2]
    velocity = np.zeros((_U.shape[0], 3), dtype=dtype_U)
    vth = np.zeros(_U.shape[0], dtype=dtype_U)
    for i in range(2):
        _z = xi[i]
        for j in range(2):
            _r = eta[j]
            for k in range(2):
                _phi = zeta[k]
                vol = _z * _r * _phi

                _ck = ck + k
                if _ck == grid_shape[2]:
                    _ck = 0
                velocity += vol[:, None] * drifted_velocity[ci+i, cj+j, _ck]
                vth += vol * thermal_velocity[ci+i, cj+j, _ck]

    v_table = get_v_table(n_particles // 2)
    np.random.shuffle(v_table)

    _U[:] = (vth[:, None] * v_table + velocity) / _c
    U[_U.shape[0]: _U.shape[0] * 2] = (-vth[:, None] * v_table + velocity) / _c

    if n_particles % 2 != 0:
        # To be fair, particle located on middle of the cell.
        X[-1] = 0.5
        U[-1] = drifted_velocity[ci:ci+2, cj:cj+2, _ck].mean(axis=(0, 1)) / _c

    U2 = (U * U).sum().item() * (_c ** 2)
    return X, U, C_idx, U2


class MaxwellianInitializer(BaseInitializer):
    def load_particles_pre(self, particles, grid_config, _e, _m):
        q = particles['q'] * _e
        m = particles['m'] * _m

        initial_condition = grid_config.get('initial_condition', {})
        temperature = initial_condition['temperature']
        density = initial_condition['density']
        current_density = initial_condition.get('current_density')

        n_computational_to_physical = int(
                grid_config['n_computational_to_physical'])

        temperature = self.load_field_array(
            temperature, self.grid_vertex_shape)
        density = self.load_field_array(density, self.grid_vertex_shape)

        if current_density is None:
            current_density = [0, 0, 0]
        current_density = self.load_field_array(current_density,
                                                (*self.grid_vertex_shape, 3))

        thermal_velocity = np.sqrt(np.abs(2 * q * temperature) / m)

        n_particles_in_cell = np.empty(self.grid_shape, dtype=density.dtype)

        if self.coordinate_system == 'cylindrical':
            cell_volume = self.cell_volume_by_grid * \
                np.power(self.scale_length / (2 * np.pi), 3)
        elif self.coordinate_system == 'cartesian':
            cell_volume = self.cell_volume
        node_to_center_3d(density * cell_volume / n_computational_to_physical,
                          n_particles_in_cell, self.coordinate_system)

        assert np.all(n_particles_in_cell < np.iinfo(np.int64).max)
        n_particles_in_cell = np.around(n_particles_in_cell).astype(int)
        n_particles_in_grid = np.around(
            density * cell_volume / n_computational_to_physical).astype(int)

        gilbert_n_particles = np.array([
            n_particles_in_cell[coord] for coord in self.gilbert_curve])

        _density = np.expand_dims(density, axis=-1)
        drifted_velocity = np.divide(current_density, q * _density,
                                     out=np.zeros_like(current_density),
                                     where=_density != 0)

        particles.update(dict(
            n_particles=gilbert_n_particles.sum(),
            n_particles_in_grid=n_particles_in_grid,
            gilbert_n_particles=gilbert_n_particles,
            thermal_velocity=thermal_velocity,
            drifted_velocity=drifted_velocity
        ))

    def load_particles(self, h5_fp, prefix, dtype_X, dtype_U,
                       particles, _m, _c):
        gilbert_n_particles = particles['gilbert_n_particles']

        end_indices = gilbert_n_particles.cumsum() - 1
        start_indices = np.empty_like(end_indices).astype(int)
        start_indices[0] = 0
        start_indices[1:] = end_indices[:-1] + 1

        halton_sequence = get_halton_sequence(gilbert_n_particles.max())

        self.distribute_maxwellian(
            h5_fp, prefix, start_indices, end_indices,
            np.array(self.gilbert_curve, dtype=np.int16),
            halton_sequence, particles, dtype_X, dtype_U, _m, _c
        )

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

        with h5py.File(h5_fp, 'a') as h5f:
            kinetic_E = 0
            for cell_index, cell_coords in enumerate(gilbert_curve):
                start = start_indices[cell_index]
                end = end_indices[cell_index]
                if start == end + 1:
                    continue

                X, U, C_idx, U2 = _distribute_maxwellian(
                    start, end, thermal_velocity, drifted_velocity,
                    cell_coords, self.grid_shape, self.r0, self.cell_size[1],
                    halton_sequence, dtype_X, dtype_U, n_particles_in_grid, _c)

                if self.save_state:
                    if self.coordinate_system == 'cartesian':
                        from ..lib.cartesian import compute_grid_velocity
                        compute_grid_velocity(
                            X, U, C_idx, grid_n, grid_U, grid_U2)
                    elif self.coordinate_system == 'cylindrical':
                        from ..lib.cylindrical import compute_grid_velocity
                        compute_grid_velocity(
                            X, U, C_idx, grid_n, grid_U, grid_U2,
                            self.cell_size[1], self.r0, self.grid_shape[2])
                # serialize
                X = np.nextafter(X + C_idx, C_idx)
                for i, axis in enumerate(self.axis_labels):
                    # X
                    _path = f'{prefix}/position/{axis}'
                    h5f[_path][start:end+1] = X[:, i]

                    # U
                    _path = f'{prefix}/momentum/{axis}'
                    h5f[_path][start:end+1] = U[:, i]

                kinetic_E += 0.5 * m * U2 * n_computational_to_physical
            particles['kinetic_E'] = kinetic_E
            if self.save_state:
                particles.update(dict(
                    grid_n=grid_n,
                    grid_U=grid_U,
                    grid_U2=grid_U2
                ))

            for i, axis in enumerate(self.axis_labels):
                _path = f'{prefix}/position/{axis}'
                h5f[_path][end+1:] = None

                _path = f'{prefix}/momentum/{axis}'
                h5f[_path][end+1:] = None
