from pathlib import Path

import yaml
import h5py
import numpy as np

from ..lib.gilbert3d import gilbert3d
from ..lib.common import (
    is_in_grid,
    node_to_center_3d,
    compute_grid_velocity,
    compute_grid_temperature
)


class BaseInitializer:
    def __init__(self, environment, grids, simulation,
                 input_base_path=Path('.')):
        _environment_config = environment.copy()
        self.cell_size = np.array(_environment_config.pop('cell_size'))
        self.grid_shape = np.array(_environment_config.pop('grid_shape'))
        self.environment_config = _environment_config

        # base path for loading files specified in the yaml config
        self.input_base_path = input_base_path

        self.grids_config = grids.copy()
        self.chunk_size = int(simulation.get('chunk_size', 1e6))
        self.iteration_encoding = simulation.get('iteration_encoding', 'file')

        gilbert_curve = np.array(list(gilbert3d(*self.grid_shape)))
        valid_cells = _environment_config.get('valid_cell_coords')
        if valid_cells is not None:
            valid_cells = self.load_relative_npy_file(valid_cells)
            mask = np.empty(gilbert_curve.shape[0], dtype=np.bool_)
            is_in_grid(gilbert_curve, mask, valid_cells)
            gilbert_curve = gilbert_curve[mask]
        self.gilbert_curve = [tuple(coord) for coord in gilbert_curve]

        self.create_dataset_kwargs = dict(
            chunks=True, shuffle=True,
            compression='gzip', compression_opts=5
        )

    @classmethod
    def load(cls, yaml_fp):
        _fp = Path(yaml_fp)
        with open(_fp, 'r') as f:
            config = yaml.safe_load(f)
        input_base_path = _fp.parent
        environment = config['environment']
        grids = config['grids']
        simulation = config['simulation']

        return cls(environment, grids, simulation, input_base_path)

    def load_relative_npy_file(self, fp):
        return np.load(self.input_base_path / fp)

    def serialize(self, out_fp):
        with h5py.File(out_fp, 'w') as h5f:
            self.setup_root_attr(h5f, out_fp)
            self.setup_base_path(h5f, iteration=0)
            self.setup_fields(h5f, iteration=0)
            self.setup_particles(h5f, out_fp, iteration=0)
            self.setup_stats(h5f, iteration=0)
            self.setup_state(h5f, iteration=0)

    def write_settings(self, h5_group, settings):
        for k, v in settings.items():
            if isinstance(v, dict):
                sub_group = h5_group.require_group(k)
                self.write_settings(sub_group, v)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                for i, _v in enumerate(v):
                    sub_group = h5_group.require_group(k)
                    sub_group = sub_group.require_group(f'{i}')
                    self.write_settings(sub_group, _v)
            else:
                h5_group.attrs[k] = v

    def setup_root_attr(self, h5f, out_fp):
        if self.iteration_encoding == 'file':
            iteration_encoding = 'fileBased'
            iteration_format = str(Path(out_fp.stem).with_suffix('.%T.h5'))
        elif self.iteration_encoding == 'group':
            iteration_encoding = 'groupBased'
            iteration_format = '/cycles/%T/'
        else:
            raise ValueError()

        root_attrs = dict(
            basePath=b'/cycles/%T/',
            iterationEncoding=iteration_encoding,
            iterationFormat=iteration_format,
            meshesPath=b'fields/',
            particlesPath=b'particles/',
            openPMD=b'1.0.0',
            openPMDextension=np.uint32(1),
            software=b'PLEXsim',
        )

        self.write_settings(h5f, root_attrs)

        env_attrs = dict(
            grid_shape=self.grid_shape,
            cell_size=self.cell_size
        )

        settings = h5f.require_group('settings')
        self.write_settings(settings, env_attrs)
        settings.create_dataset('gilbert_curve',
                                data=np.array(self.gilbert_curve),
                                dtype=np.int16)

    def base_path(self, h5f, iteration):
        return np.string_(h5f.attrs['basePath']).replace(
            b'%T', str(iteration).encode('utf-8'))

    def setup_base_path(self, h5f, iteration=0, delta_time=1):
        base_path = self.base_path(h5f, iteration)
        base = h5f.require_group(base_path)

        base_attrs = dict(
            time=np.float64(iteration),
            dt=1.,
            timeUnitSI=np.float64(delta_time)
        )

        self.write_settings(base, base_attrs)

    def load_external_field(self, external_field, field_dtype):
        grid_vertex_shape = (*(self.grid_shape + 1), 3)

        if isinstance(external_field, list):
            return np.full(grid_vertex_shape, external_field,
                           dtype=field_dtype)
        elif isinstance(external_field, str):
            if external_field.endswith('.npy'):
                F = self.load_relative_npy_file(external_field) \
                    .astype(field_dtype)
                assert np.all(F.shape == grid_vertex_shape)
                return F
            else:
                raise NotImplementedError()
        elif isinstance(external_field, np.ndarray):
            return external_field.astype(field_dtype)
        else:
            raise NotImplementedError()

    def setup_fields(self, h5f, iteration=0):
        fields_path = self.base_path(h5f, iteration) \
            + h5f.attrs['meshesPath'].encode('utf-8')
        fields = h5f.require_group(fields_path)

        dimension = len(self.grid_shape)

        fields_attrs = dict(
            fieldSolver=b'none',
            fieldBoundary=np.full(2 * dimension, b"open"),
            particleBoundary=np.full(2 * dimension, b"absorbing"),
            currentSmoothing=b'none',
            chargeCorrection=b'none'
        )

        self.write_settings(fields, fields_attrs)

        field_dtype = self.environment_config.get('field_dtype')
        if field_dtype == 'fp32':
            field_dtype = np.float32
        elif field_dtype == 'fp64':
            field_dtype = np.float64
        else:
            raise NotImplementedError()

        magnetic_field = self.environment_config.get('external_magnetic_field')
        electric_field = self.environment_config.get('external_electric_field')

        B = self.load_external_field(magnetic_field, field_dtype)
        E = self.load_external_field(electric_field, field_dtype)

        self.write_B(fields, B, field_dtype)
        self.write_E(fields, E, field_dtype)

        self.B = B
        self.E = E

    def write_B(self, fields_group, B, field_dtype):
        B_group = fields_group.require_group(b'B')

        dimension = len(self.grid_shape)
        if dimension == 3:
            axis_labels = [b'x', b'y', b'z']
        else:
            raise NotImplementedError()

        B_attrs = dict(
            geometry=b'cartesian',
            gridSpacing=self.cell_size,
            gridGlobalOffset=np.zeros(dimension, dtype=np.float64),
            gridUnitSI=1.,
            dataOrder=b'C',
            axisLabels=np.array(axis_labels),
            unitDimension=np.array(
                [0, 1, -2, -1, 0, 0, 0], dtype=np.float64),
            fieldSmoothing=b'none',
            timeOffset=0.
        )

        self.write_settings(B_group, B_attrs)

        for i, axis in enumerate(axis_labels):
            _B = B[..., i].astype(field_dtype)
            B_group.create_dataset(axis, data=_B,
                                   **self.create_dataset_kwargs)
            B_group[axis].attrs['position'] = np.zeros(
                dimension, dtype=field_dtype)
            B_group[axis].attrs['unitSI'] = 1.

    def write_E(self, fields_group, E, field_dtype):
        E_group = fields_group.require_group(b'E')

        dimension = len(self.grid_shape)
        if dimension == 3:
            axis_labels = [b'x', b'y', b'z']
        else:
            raise NotImplementedError()

        E_attrs = dict(
            geometry=b'cartesian',
            gridSpacing=self.cell_size,
            gridGlobalOffset=np.zeros(dimension, dtype=np.float64),
            gridUnitSI=1.,
            dataOrder=b'C',
            axisLabels=np.array(axis_labels),
            unitDimension=np.array(
                [1, 1, -3, -1, 0, 0, 0], dtype=np.float64),
            fieldSmoothing=b'none',
            timeOffset=0.
        )

        self.write_settings(E_group, E_attrs)

        for i, axis in enumerate(axis_labels):
            _E = E[..., i].astype(field_dtype)
            E_group.create_dataset(axis, data=_E,
                                   **self.create_dataset_kwargs)
            E_group[axis].attrs['position'] = np.zeros(
                dimension, dtype=field_dtype)
            E_group[axis].attrs['unitSI'] = 1.

    def load_particles(self, dtype_X, dtype_U, particles, grid_config):
        raise NotImplementedError()

    def split_in_balance(self, gilbert_n_particles, n_splits):
        n_particles = gilbert_n_particles.sum()
        n_cells = len(self.gilbert_curve)
        n_target = n_particles / n_splits

        # split cells to allocate equal n_particles in each subgrid
        rank = 1
        start_index = 0
        n_current = 0
        n_total = 0
        start_indices = []
        end_indices = []
        for index, n_particles_in_cell in enumerate(gilbert_n_particles):
            if rank == n_splits:
                start_indices.append(start_index)
                end_indices.append(n_cells - 1)
                break

            n_current += n_particles_in_cell
            if n_current >= n_target or n_cells - index <= n_splits - rank\
                    or n_current + n_total >= n_particles\
                    or abs(n_current - n_target) < abs(
                    n_current + gilbert_n_particles[index+1] - n_target):

                start_indices.append(start_index)
                end_indices.append(index)

                n_total += n_current
                n_current = 0
                start_index = index + 1
                rank += 1

        return np.array(start_indices), np.array(end_indices)

    def get_particle_attrs(self, n_particles, q, m, axis_labels):
        charge_attrs = dict(
            value=q,
            shape=np.array([n_particles], dtype=np.uint64),
            macroWeighted=np.uint32(1),
            weightingPower=1.,
            timeOffset=0.,
            unitSI=1.,
            unitDimension=np.array(
                [0, 0, 1, 1, 0, 0, 0], dtype=np.float64)
        )
        mass_attrs = dict(
            value=m,
            shape=np.array([n_particles], dtype=np.uint64),
            macroWeighted=np.uint32(1),
            weightingPower=1.,
            timeOffset=0.,
            unitSI=1.,
            unitDimension=np.array(
                [0, 1, 0, 0, 0, 0, 0], dtype=np.float64)
        )
        position_attrs = dict(
            macroWeighted=np.uint32(1),
            weightingPower=0.,
            timeOffset=0.,
            unitDimension=np.array(
                [1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        )
        offset_attrs = dict(
            macroWeighted=np.uint32(1),
            weightingPower=0.,
            timeOffset=0.,
            unitDimension=np.array(
                [1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            **{axis: dict(value=np.float32(0),
                          shape=np.array([n_particles], dtype=np.uint64),
                          unitSI=1.
                          ) for axis in axis_labels}
        )
        momentum_attrs = dict(
            macroWeighted=np.uint32(1),
            weightingPower=0.,
            timeOffset=0.,
            unitDimension=np.array(
                [1, 1, -1, 0, 0, 0, 0], dtype=np.float64)
        )

        return dict(
            charge=charge_attrs,
            mass=mass_attrs,
            position=position_attrs,
            positionOffset=offset_attrs,
            momentum=momentum_attrs
        )

    def write_particle_attrs(self, grid_index, grid_group, n_splits, q, m,
                             axis_labels, n_computational_to_physical):
        n_particles = self.particles[grid_index]['X'].shape[0]

        particles_attrs = self.get_particle_attrs(n_particles, q, m,
                                                  axis_labels)

        gilbert_n_particles = self.particles[grid_index]['gilbert_n_particles']
        start_indices, end_indices = \
            self.split_in_balance(gilbert_n_particles, n_splits)
        end_p_indices = gilbert_n_particles.cumsum()[end_indices] - 1
        start_p_indices = np.empty_like(end_p_indices).astype(int)
        start_p_indices[0] = 0
        start_p_indices[1:] = end_p_indices[:-1] + 1

        grid_attrs = dict(
            particleShape=3.0,
            currentDeposition=b'none',
            particlePush=b'Boris',
            particleInterpolation=b'uniform',
            particleSmoothing=b'none',
            **particles_attrs,
            start_indices=start_indices,
            end_indices=end_indices,
            start_p_indices=start_p_indices,
            end_p_indices=end_p_indices
        )

        self.write_settings(grid_group, grid_attrs)

        patches = grid_group.require_group('particlePatches')
        patches.create_dataset('numParticles',
                               data=end_indices - start_indices + 1,
                               dtype=np.uint64)
        patches['numParticles'].attrs['unitSI'] = 1.
        patches.create_dataset('numParticlesOffset',
                               data=start_indices,
                               dtype=np.uint64)
        patches['numParticlesOffset'].attrs['unitSI'] = 1.

        patches.require_group('offset')
        patches['offset'].attrs['unitDimension'] = np.array(
            [1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        for i, axis in enumerate(axis_labels):
            patches['offset'].create_dataset(axis, data=np.full(n_splits, 0.))
            patches[f'offset/{axis}'].attrs['unitSI'] = self.cell_size[i]

        patches.require_group('extent')
        patches['extent'].attrs['unitDimension'] = np.array(
            [1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        for i, axis in enumerate(axis_labels):
            patches['extent'].create_dataset(
                axis, data=np.full(n_splits, self.grid_shape[i]))
            patches[f'extent/{axis}'].attrs['unitSI'] = self.cell_size[i]

        weighting_attrs = dict(
            macroWeighted=np.uint32(1),
            weightingPower=1.,
            timeOffset=0.,
            unitSI=1.,
            unitDimension=np.array(
                [0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        )
        _create_dataset_kwargs = self.create_dataset_kwargs.copy()
        if n_particles > self.chunk_size:
            _create_dataset_kwargs['chunks'] = (self.chunk_size,)
        grid_group.create_dataset('weighting', (n_particles,),
                                  dtype=np.float32,
                                  **_create_dataset_kwargs)
        grid_group['weighting'][:] = n_computational_to_physical
        self.write_settings(grid_group['weighting'], weighting_attrs)

    def setup_particles(self, h5f, out_fp, iteration=0):
        particles_path = self.base_path(h5f, iteration) \
            + h5f.attrs['particlesPath'].encode('utf-8')
        particles = h5f.require_group(particles_path)

        dimension = len(self.grid_shape)
        if dimension == 3:
            axis_labels = ['x', 'y', 'z']
        else:
            raise NotImplementedError()

        self.particles = dict()
        tracking_start_id = 1
        for grid_index, grid_config in enumerate(self.grids_config):
            self.particles[grid_index] = dict()
            grid_group = particles.require_group(str(grid_index))

            # common vars
            species = grid_config['species']
            dtype = grid_config['dtype']
            n_splits = grid_config['n_splits']
            n_computational_to_physical = int(
                grid_config['n_computational_to_physical'])
            initial_condition = grid_config['initial_condition']
            self.particles[grid_index]['n_computational_to_physical']\
                = n_computational_to_physical

            if species == 'electron':
                q = -1.602e-19
                m = 9.11e-31
            elif species == 'ion':
                q = 1.602e-19
                m = 1.67e-27
            else:
                raise NotImplementedError()
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

            # load particles
            # sample = initial_condition['sample']

            self.load_particles(dtype_X, dtype_U, self.particles[grid_index],
                                grid_config)

            # if sample == 'random':
            #     self.load_particles_random(dtype_X, dtype_U, grid_index,
            #                                **initial_condition)
            # elif sample == 'maxwellian':
            #     self.load_particles_maxwellian(n_computational_to_physical,
            #                                    dtype_X, dtype_U, grid_index,
            #                                    q, m, **initial_condition)
            # else:
            #     raise NotImplementedError()

            self.write_particle_attrs(grid_index, grid_group, n_splits, q, m,
                                      axis_labels, n_computational_to_physical)

            grid_fp = Path(out_fp).with_suffix(f'.g{grid_index}.h5')
            self.serialize_particles(grid_fp, grid_group, grid_index,
                                     axis_labels)

            # serialize tracking particles
            tracked_group = particles.require_group(f'{grid_index}_tracked')
            n_track_particles = initial_condition['tracking']['n_particles']

            tracked_attrs = self.get_particle_attrs(
                n_track_particles, q, m, axis_labels)
            tracked_attrs.update(
                dict(particleShape=3.0,
                     currentDeposition=b'none',
                     particlePush=b'Boris',
                     particleInterpolation=b'uniform',
                     particleSmoothing=b'none'))

            self.write_settings(tracked_group, tracked_attrs)

            weighting_attrs = dict(
                macroWeighted=np.uint32(1),
                weightingPower=1.,
                timeOffset=0.,
                unitSI=1.,
                unitDimension=np.array(
                    [0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
            )
            tracked_group.create_dataset('weighting', (n_track_particles,),
                                         dtype=np.float32,
                                         **self.create_dataset_kwargs)
            tracked_group['weighting'][:] = n_computational_to_physical
            self.write_settings(tracked_group['weighting'], weighting_attrs)

            self.serialize_tracked(tracked_group, grid_index, axis_labels,
                                   n_track_particles, tracking_start_id)
            tracking_start_id += n_track_particles

            # calculate and store kinetic_E
            U = self.particles[grid_index]['U']
            kinetic_E = 0.5 * m * (U * U).sum().item() \
                * n_computational_to_physical
            self.particles[grid_index]['kinetic_E'] = kinetic_E

    def serialize_particles(self, grid_fp, grid_group, grid_index,
                            axis_labels):
        # serialize X, U
        X = self.particles[grid_index]['X'] +\
            self.particles[grid_index]['C_idx']
        U = self.particles[grid_index]['U']

        _create_dataset_kwargs = self.create_dataset_kwargs.copy()
        if X.shape[0] > self.chunk_size:
            _create_dataset_kwargs['chunks'] = (self.chunk_size,)

        with h5py.File(grid_fp, 'w') as h5f_g:
            base_g = h5f_g.require_group('cycles/0')
            for i, axis in enumerate(axis_labels):
                # X
                _path = f'position/{axis}'
                src = base_g.create_dataset(_path, data=X[:, i],
                                            **_create_dataset_kwargs)
                vs = h5py.VirtualSource(src)
                layout = h5py.VirtualLayout(vs.shape, vs.dtype)
                layout[:] = vs
                grid_group.create_virtual_dataset(_path, layout)
                grid_group[_path].attrs['unitSI'] = \
                    np.float64(self.cell_size[i])

                # U
                _path = f'momentum/{axis}'
                src = base_g.create_dataset(_path, data=U[:, i],
                                            **_create_dataset_kwargs)
                vs = h5py.VirtualSource(src)
                layout = h5py.VirtualLayout(vs.shape, vs.dtype)
                layout[:] = vs
                grid_group.create_virtual_dataset(_path, layout)
                grid_group[_path].attrs['unitSI'] = np.float64(1)

    def serialize_tracked(self, tracked_group, grid_index, axis_labels,
                          n_track_particles, start_id):
        # serialize tracking particles
        X = self.particles[grid_index]['X'] +\
            self.particles[grid_index]['C_idx']
        U = self.particles[grid_index]['U']
        n_particles = X.shape[0]

        particle_indices = np.random.choice(
            n_particles, n_track_particles, False)
        tracking_ids = np.arange(start_id, start_id + n_track_particles,
                                 dtype=np.uint64)

        tracked_group.attrs['p_indices'] = particle_indices
        tracked_group.create_dataset('id', data=tracking_ids,
                                     **self.create_dataset_kwargs)

        for i, axis in enumerate(axis_labels):
            # X
            _path = f'position/{axis}'
            track_X = X[list(particle_indices), i]
            tracked_group.create_dataset(_path, data=track_X,
                                         **self.create_dataset_kwargs)
            tracked_group[_path].attrs['unitSI'] = \
                np.float64(self.cell_size[i])

            # U
            _path = f'momentum/{axis}'
            track_U = U[list(particle_indices), i]
            tracked_group.create_dataset(_path, data=track_U,
                                         **self.create_dataset_kwargs)
            tracked_group[_path].attrs['unitSI'] = np.float64(1)

    @property
    def field_E(self, permeability=1.257e-6, permittivity=8.854e-12):
        B_center = np.empty((*(self.grid_shape), 3))
        E_center = np.empty((*(self.grid_shape), 3))

        node_to_center_3d(self.B, B_center)
        node_to_center_3d(self.E, E_center)

        magnetic_E = 0.5 * self.cell_size.prod() * \
            permeability * (B_center * B_center).sum()
        electric_E = 0.5 * self.cell_size.prod() * \
            permittivity * (E_center * E_center).sum()

        return magnetic_E, electric_E

    def setup_stats(self, h5f, iteration=0):
        magnetic_E, electric_E = self.field_E

        n_particles = []
        kinetic_E = []
        for grid in self.particles.values():
            n_particles.append(grid['X'].shape[0])
            kinetic_E.append(grid['kinetic_E'])

        stats_path = self.base_path(h5f, iteration) + b'stats'
        stats_group = h5f.require_group(stats_path)

        stats_attrs = dict(
            n_particles=np.array(n_particles),
            kinetic_E=np.array(kinetic_E),
            electric_E=electric_E,
            magnetic_E=magnetic_E,
            total_E=sum(kinetic_E) + electric_E + magnetic_E
        )

        self.write_settings(stats_group, stats_attrs)

    def setup_state(self, h5f, iteration=0):
        # TODO: remove GPU dependency
        import cupy as cp

        state_path = self.base_path(h5f, iteration) + b'state'
        state_group = h5f.require_group(state_path)
        for grid_index, grid_values in self.particles.items():
            X = cp.array(grid_values['X'])
            U = cp.array(grid_values['U'])
            C_idx = cp.array(grid_values['C_idx'])
            q = grid_values['q']
            m = grid_values['m']
            n_computational_to_physical = \
                grid_values['n_computational_to_physical']

            grid_n = cp.zeros((self.grid_shape + 1), dtype=np.float32)
            grid_N = cp.zeros((self.grid_shape + 1), dtype=np.int64)
            grid_U = cp.zeros((*(self.grid_shape + 1), 3), dtype=np.float32)
            grid_T = cp.zeros((self.grid_shape + 1), dtype=np.float32)

            compute_grid_velocity(X, U, C_idx, grid_n, grid_U, grid_N)

            cp.divide(grid_n, grid_N, out=grid_n)
            grid_n = cp.nan_to_num(grid_n)
            grid_n = grid_n / self.cell_size.prod()\
                * n_computational_to_physical

            cp.divide(grid_U, cp.expand_dims(grid_N, axis=-1), out=grid_U)
            grid_U = cp.nan_to_num(grid_U)

            compute_grid_temperature(
                X, U, C_idx, grid_T, grid_U, grid_N, q, m
            )

            _state_group = state_group.require_group(str(grid_index))
            _state_group.create_dataset('n', data=cp.asnumpy(grid_n),
                                        **self.create_dataset_kwargs)
            _state_group.create_dataset('U', data=cp.asnumpy(grid_U),
                                        **self.create_dataset_kwargs)
            _state_group.create_dataset('T', data=cp.asnumpy(grid_T),
                                        **self.create_dataset_kwargs)
