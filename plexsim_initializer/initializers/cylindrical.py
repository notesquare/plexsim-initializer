from pathlib import Path

import numpy as np
import h5py

from ..lib.common import (
    node_to_center_3d,
    SavedFlag
)
from .base import BaseInitializer


class CylindricalInitializer(BaseInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r0 = self.environment_config['r0']
        self.scale_length = self.environment_config['scale_length']

        cell_size_phi = 2 * np.pi / self.grid_shape[2]
        self.cell_size = np.append(self.cell_size, cell_size_phi)

        self.cbc = np.arange(self.grid_shape.prod()) * 3
        nz, nr, nphi = self.grid_shape
        rgn = np.zeros((8, nphi, nr, nz), dtype=int)
        for z in range(nz):
            rgn[:4, :, :, z] += z
        for r in range(nr):
            rgn[:4, :, r] += (nz + 1) * r
        for phi in range(nphi):
            rgn[:4, phi] += (nz + 1) * (nr + 1) * phi
        for i in range(2):
            for j in range(2):
                rgn[i * 2 + j] += (nz + 1) * i + j
        rgn = rgn.reshape(8, -1) * 3
        rgn[4:] = np.roll(rgn[:4], -nz * nr, axis=-1)
        self.rgn = rgn

    @property
    def env_attrs(self):
        return dict(
            grid_shape=self.grid_shape,
            cell_size=self.cell_size,
            permittivity=self.permittivity,
            permeability=self.permeability,
            scale_length=self.scale_length,
            r0=self.r0
        )

    @property
    def grid_vertex_shape(self):
        nz, nr, nphi = self.grid_shape
        return nz + 1, nr + 1, nphi

    @property
    def B_shape(self):
        nz, nr, nphi = self.grid_shape
        return nz, nr, nphi, 3

    @property
    def E_shape(self):
        nz, nr, nphi = self.grid_shape
        return nz, nr, nphi, 3

    @property
    def cell_volume(self):
        # returns cell volume arrays by r-index
        nr = self.grid_shape[1]
        dr = self.cell_size[1]

        cell_volume_by_r = self.cell_size.prod() * (
            (np.arange(nr) + 0.5) * dr + self.r0)
        return cell_volume_by_r.reshape(1, -1, 1)

    @property
    def cell_volume_by_grid(self):
        # returns cell volume defined on grids by r-index
        nr = self.grid_shape[1]
        dr = self.cell_size[1]

        cell_volume_by_r = self.cell_size.prod() * (
            np.arange(nr + 1) * dr + self.r0)
        return cell_volume_by_r.reshape(1, -1, 1)

    @property
    def axis_labels(self):
        dimension = len(self.grid_shape)
        if dimension == 3:
            return ['z', 'r', 't']
        else:
            return NotImplementedError()

    @property
    def grid_global_offset(self):
        dimension = len(self.grid_shape)
        offset = np.zeros(dimension, dtype=np.float64)
        offset[1] = self.r0
        return offset

    @property
    def B_attrs(self):
        return dict(
            geometry=np.string_(self.coordinate_system),
            gridSpacing=self.cell_size,
            gridGlobalOffset=self.grid_global_offset,
            gridUnitSI=np.float64(1),
            dataOrder=np.string_('C'),
            axisLabels=np.array(self.axis_labels).astype(np.string_),
            unitDimension=np.array(
                [0, 1, -2, -1, 0, 0, 0], dtype=np.float64),
            fieldSmoothing=np.string_('none'),
            timeOffset=0.
        )

    @property
    def E_attrs(self):
        return dict(
            geometry=np.string_(self.coordinate_system),
            gridSpacing=self.cell_size,
            gridGlobalOffset=self.grid_global_offset,
            gridUnitSI=np.float64(1),
            dataOrder=np.string_('C'),
            axisLabels=np.array(self.axis_labels).astype(np.string_),
            unitDimension=np.array(
                [1, 1, -3, -1, 0, 0, 0], dtype=np.float64),
            fieldSmoothing=np.string_('none'),
            timeOffset=0.
        )

    @property
    def J_attrs(self):
        return dict(
            geometry=np.string_(self.coordinate_system),
            gridSpacing=self.cell_size,
            gridGlobalOffset=self.grid_global_offset,
            gridUnitSI=np.float64(1),
            dataOrder=np.string_('C'),
            axisLabels=np.array(self.axis_labels).astype(np.string_),
            unitDimension=np.array(
                [-2, 0, 0, 1, 0, 0, 0], dtype=np.float64),
            fieldSmoothing=np.string_('none'),
            timeOffset=0.
        )

    def write_J(self, fields_group, _e=1.602e-19, c=2.99792458e8):
        J_group = fields_group.require_group(np.string_('J_vacuum'))

        J_attrs = self.J_attrs
        self.write_settings(J_group, J_attrs)

        w = 2 * np.pi * c / self.scale_length
        m = self.scale_length / (2 * np.pi)
        J_norm = m ** 2 / (_e * w)
        fJ = 2.224e-13 / self.scale_length
        J_vac = self.J_vac * J_norm * fJ

        axis_labels = [np.string_(v) for v in self.axis_labels]
        dimension = len(self.grid_shape)
        for i, axis in enumerate(axis_labels):
            J_group.create_dataset(axis, data=J_vac[..., i],
                                   **self.create_dataset_kwargs)
            J_group[axis].attrs['position'] = np.zeros(
                dimension, dtype=J_vac.dtype)
            J_group[axis].attrs['unitSI'] = np.float64(1)

    def position_offset_attrs(self, n_particles):
        return dict(
            macroWeighted=np.uint32(1),
            weightingPower=0.,
            timeOffset=0.,
            unitDimension=np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            **{axis: dict(
                value=np.float32(0)+self.r0 if axis == 'r' else np.float32(0),
                shape=np.array([n_particles], dtype=np.uint64),
                unitSI=np.float64(1)
            ) for axis in self.axis_labels}
        )

    def write_particle_patches_offset(self, patches_group, n_splits):
        offset = patches_group.require_group('offset')
        offset.attrs['unitDimension'] = np.array(
            [1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        for i, axis in enumerate(self.axis_labels):
            v = self.r0 if axis == 'r' else 0.
            offset.create_dataset(axis, data=np.full(n_splits, v))
            offset[axis].attrs['unitSI'] = np.float64(self.cell_size[i])

    def yee_to_grid_B(self, B):
        f = np.swapaxes(B, 0, 2).flatten()
        fg = np.zeros(3 * (self.grid_shape + 1).prod(), dtype=B.dtype)

        # B0
        ff = 0.25 * f[self.cbc]
        fg[self.rgn[0, :]] += ff
        fg[self.rgn[2, :]] += ff
        fg[self.rgn[4, :]] += ff
        fg[self.rgn[6, :]] += ff

        # B1
        ff = 0.25 * f[self.cbc + 1]
        fg[self.rgn[0, :] + 1] += ff
        fg[self.rgn[1, :] + 1] += ff
        fg[self.rgn[4, :] + 1] += ff
        fg[self.rgn[5, :] + 1] += ff

        # B2
        ff = 0.25 * f[self.cbc + 2]
        fg[self.rgn[0, :] + 2] += ff
        fg[self.rgn[1, :] + 2] += ff
        fg[self.rgn[2, :] + 2] += ff
        fg[self.rgn[3, :] + 2] += ff

        nz, nr, nphi = self.grid_shape
        return np.swapaxes(fg.reshape(nphi + 1, nr + 1, nz + 1, 3), 0, 2)

    def yee_to_grid_E(self, E):
        f = np.swapaxes(E, 0, 2).flatten()
        fg = np.zeros(3 * (self.grid_shape + 1).prod(), dtype=E.dtype)

        # E0
        ff = 0.5 * f[self.cbc]
        fg[self.rgn[0, :]] += ff
        fg[self.rgn[1, :]] += ff

        # E1
        ff = 0.5 * f[self.cbc + 1]
        fg[self.rgn[0, :] + 1] += ff
        fg[self.rgn[2, :] + 1] += ff

        # E2
        ff = 0.5 * f[self.cbc + 2]
        fg[self.rgn[0, :] + 2] += ff
        fg[self.rgn[4, :] + 2] += ff

        nz, nr, nphi = self.grid_shape
        return np.swapaxes(fg.reshape(nphi + 1, nr + 1, nz + 1, 3), 0, 2)

    def grid_to_yee_E(self, E):
        fg = np.swapaxes(E, 0, 2).flatten()
        f = np.empty(3 * self.grid_shape.prod(), dtype=E.dtype)

        f[self.cbc] = 0.5 * (fg[self.rgn[0, :]] + fg[self.rgn[1, :]])
        f[self.cbc+1] = 0.5 * (fg[self.rgn[0, :] + 1] + fg[self.rgn[2, :] + 1])
        f[self.cbc+2] = 0.5 * (fg[self.rgn[0, :] + 2] + fg[self.rgn[4, :] + 2])

        nz, nr, nphi = self.grid_shape
        return np.swapaxes(f.reshape(nphi, nr, nz, 3), 0, 2)

    @property
    def magnetic_E(self, c=2.99792458e8, _m=9.1093837e-31, _e=1.602e-19):
        cell_volume = np.expand_dims(self.cell_volume, -1)\
            * np.power(self.scale_length / (2 * np.pi), 3)
        factor = 2 * np.pi * c * _m / (_e * self.scale_length)

        B_grid_total = self.yee_to_grid_B(self.B_external + self.B_induced)
        grid_center_shape = np.array((*(self.grid_shape), 3))
        B_center = np.empty(grid_center_shape)

        node_to_center_3d(B_grid_total, B_center, self.coordinate_system)
        magnetic_E = 0.5 * self.permeability * \
            (B_center * B_center * cell_volume).sum() * np.power(factor, 2)

        B_grid_induced = self.yee_to_grid_B(self.B_induced)
        node_to_center_3d(B_grid_induced, B_center, self.coordinate_system)
        induced_magnetic_E = 0.5 * self.permeability * \
            (B_center * B_center * cell_volume).sum() * np.power(factor, 2)

        return magnetic_E, induced_magnetic_E

    @property
    def electric_E(self, c=2.99792458e8, _m=9.1093837e-31, _e=1.602e-19):
        cell_volume = np.expand_dims(self.cell_volume, -1)\
            * np.power(self.scale_length / (2 * np.pi), 3)
        factor = 2 * np.pi * np.power(c, 2) * _m / (_e * self.scale_length)

        E_grid_total = self.yee_to_grid_E(self.E_external + self.E_induced)
        grid_center_shape = np.array((*(self.grid_shape), 3))
        E_center = np.empty(grid_center_shape)

        node_to_center_3d(E_grid_total, E_center, self.coordinate_system)
        electric_E = 0.5 * self.permittivity * \
            (E_center * E_center * cell_volume).sum() * np.power(factor, 2)

        E_grid_induced = self.yee_to_grid_E(self.E_induced)
        node_to_center_3d(E_grid_induced, E_center, self.coordinate_system)
        induced_electric_E = 0.5 * self.permittivity * \
            (E_center * E_center * cell_volume).sum() * np.power(factor, 2)

        return electric_E, induced_electric_E

    def setup_particles(self, h5f, iteration=0):
        flag = SavedFlag.particles

        tracking_start_id = 1
        for grid_index, grid_config in enumerate(self.grids_config):
            # common vars
            species = grid_config['species']
            dtype = grid_config['dtype']
            n_splits = grid_config['n_splits']
            n_computational_to_physical = int(
                grid_config['n_computational_to_physical'])
            initial_condition = grid_config['initial_condition']

            self.particles[grid_index] = dict(
                species=species,
                particle_name=grid_config['name'],
                n_computational_to_physical=n_computational_to_physical
            )

            if species == 'electron':
                q = grid_config.get('q', -1)
                m = grid_config.get('m', 1)
            elif species == 'ion':
                q = grid_config.get('q', 1)
                m = grid_config.get('m', 1833.15)
            else:
                q = grid_config['q']
                m = grid_config['m']
            self.particles[grid_index].update(dict(q=q, m=m))

            if dtype['X'] == 'fp32':
                dtype_X = np.float32
            elif dtype['X'] == 'fp64':
                dtype_X = np.float64
            else:
                raise NotImplementedError()

            if dtype['U'] == 'fp32':
                dtype_U = np.float32
            elif dtype['U'] == 'fp64':
                dtype_U = np.float64
            else:
                raise NotImplementedError()

            self.load_particles_pre(self.particles[grid_index], grid_config,
                                    _e=1.602e-19, _m=9.1093837e-31)

            out_fp = Path(h5f.filename)

            # serialize particles
            grid_fp = out_fp.with_suffix(f'.g{grid_index}.h5')
            particle_name = grid_config['name']

            p_path = f'data/{iteration}/particles/{particle_name}'
            with h5py.File(grid_fp, 'w') as grid_h5f:
                particle_group = grid_h5f.require_group(p_path)
                # custom attribute
                particle_group.attrs['_gridIndex'] = grid_index
                particle_group.attrs['_tracked'] = 0

                particle_data = self.particles[grid_index]
                self.write_particle_attrs(
                    particle_group, particle_data, n_splits,
                    n_computational_to_physical, dtype_X, dtype_U)

            self.load_particles(
                grid_fp, p_path, dtype_X, dtype_U, self.particles[grid_index],
                _m=9.1093837e-31, _c=2.99792458e8)

            with h5py.File(grid_fp, 'a') as grid_h5f:
                # create external link in
                h5f[p_path] = h5py.ExternalLink(grid_fp.name, p_path)

                # serialize tracking particles
                n_track_particles = initial_condition.get('tracking', {}) \
                    .get('n_particles', 0)
                n_particles = particle_data['n_particles']

                if n_track_particles > n_particles:
                    print('Warning: number of tracking particles cannot be'
                          ' greater than number of particles.')
                    n_track_particles = n_particles

                if n_track_particles > 0:
                    tracked_path = f'{p_path}_tracked'
                    tracked_group = grid_h5f.require_group(tracked_path)

                    particle_group = grid_h5f.require_group(p_path)
                    self.serialize_tracked(
                        tracked_group, grid_index, n_track_particles, q, m,
                        n_computational_to_physical, n_particles,
                        tracking_start_id, particle_group
                    )

                    # create external link in
                    h5f[tracked_path] = h5py.ExternalLink(grid_fp.name,
                                                          tracked_path)

                    tracking_start_id += n_track_particles

                    flag |= SavedFlag.tracked

        return flag

    def setup_state(self, h5f, iteration=0, density_threshold=1e-10,
                    _e=1.602e-19, _m=9.1093837e-31, c=2.99792458e8):
        fields_path = self.base_path(h5f, iteration) \
            + np.string_(h5f.attrs['meshesPath'])
        fields_group = h5f.require_group(fields_path)

        for grid_index, grid_values in self.particles.items():
            q = grid_values['q'] * _e
            m = grid_values['m'] * _m
            n_computational_to_physical = \
                grid_values['n_computational_to_physical']
            grid_n = grid_values['grid_n']
            grid_U = grid_values['grid_U'] * c
            grid_U2 = grid_values['grid_U2'] * np.power(c, 2)

            mask = grid_n > density_threshold
            _grid_n = np.expand_dims(grid_n, axis=-1)

            grid_U[mask] = np.divide(grid_U[mask], _grid_n[mask])
            grid_U[~mask].fill(0)

            grid_U2[mask] = np.divide(grid_U2[mask], _grid_n[mask])
            grid_U2[~mask].fill(0)

            grid_U2 -= grid_U * grid_U

            grid_T = grid_U2 * m / abs(q)

            cell_volume = self.cell_volume_by_grid * \
                np.power(self.scale_length / (2 * np.pi), 3)
            grid_n = grid_n * n_computational_to_physical\
                / cell_volume

            if self.constant_external_field_node is not None:
                grid_n[tuple(axis for axis in
                             self.constant_external_field_node.T)] = 0
                grid_U[tuple(axis for axis in
                             self.constant_external_field_node.T)] = 0
                grid_T[tuple(axis for axis in
                             self.constant_external_field_node.T)] = 0

            particle_name = grid_values['particle_name']
            self.write_state(fields_group, particle_name, grid_n,
                             grid_U, grid_T)
        return SavedFlag.state
