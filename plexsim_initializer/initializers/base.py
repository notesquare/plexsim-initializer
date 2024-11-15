from pathlib import Path
from datetime import datetime

import yaml
import h5py
import numpy as np
import numpy.ma as ma

from ..lib.gilbert3d import gilbert3d
from ..lib.common import (
    SavedFlag,
    remove_cycle_pattern_from_filename
)


class BaseInitializer:
    def __init__(self, environment, grids, simulation,
                 input_base_path=Path('.')):
        _environment_config = environment.copy()
        self.cell_size = np.array(_environment_config.pop('cell_size'))
        self.grid_shape = np.array(_environment_config.pop('grid_shape'))
        self.permeability = np.pi * 4.e-7 * \
            _environment_config.pop('relative_permeability', 1)
        self.permittivity = 8.854e-12 * \
            _environment_config.pop('relative_permittivity', 1)
        self.coordinate_system = \
            _environment_config.pop('coordinate_system', 'cartesian')

        self.environment_config = _environment_config

        # base path for loading files specified in the yaml config
        self.input_base_path = input_base_path

        self.grids_config = grids.copy()

        # find appropriate particle name per each particle
        n_grids_per_species = {}
        for grid_config in self.grids_config:
            species = grid_config['species']
            n_grids_per_species.setdefault(species, 0)
            n_grids_per_species[species] += 1
        n_grids_per_species = {
            species: list(reversed(range(n)))
            for species, n in n_grids_per_species.items()
            if n > 1
        }
        for grid_config in self.grids_config:
            species = grid_config['species']
            if len(n_grids_per_species.get(species, [])) > 0:
                v = n_grids_per_species[species].pop()
                grid_config['name'] = f'{species}_{v+1}'
            else:
                grid_config['name'] = species

        self.chunk_size = int(simulation.get('chunk_size', 1e6))
        self.iteration_encoding = simulation.get('iteration_encoding', 'file')
        self.save_state = simulation.get('save_state', False)

        gilbert_curve = np.array(list(gilbert3d(*self.grid_shape)))
        valid_cells = _environment_config.get('valid_cell_coords')
        if valid_cells is not None:
            valid_cells = self.load_relative_npy_file(valid_cells)

            mask = np.zeros(self.grid_shape, dtype=np.bool_)
            mask[tuple(valid_cells.T)] = True

            gilbert_curve_matrix = np.empty(self.grid_shape, dtype=int)
            gilbert_curve_matrix[tuple(gilbert_curve.T)] = \
                np.arange(self.grid_shape.prod())

            idx_i, idx_j, idx_k = np.indices(mask.shape)
            _idx_i = idx_i[ma.masked_where(mask, idx_i).mask]
            _idx_j = idx_j[ma.masked_where(mask, idx_j).mask]
            _idx_k = idx_k[ma.masked_where(mask, idx_k).mask]

            stacked = np.stack([_idx_i, _idx_j, _idx_k,
                                gilbert_curve_matrix[_idx_i, _idx_j, _idx_k]])
            gilbert_curve = stacked[:, np.lexsort(stacked)].T[:, :-1]

        self.gilbert_curve = [tuple(coord) for coord in gilbert_curve]

        constant_external_field_node = _environment_config.get(
            'constant_external_field_node_coords')
        if constant_external_field_node is not None:
            constant_external_field_node = \
                self.load_relative_npy_file(constant_external_field_node)
        self.constant_external_field_node = constant_external_field_node

        constant_induced_field_node = _environment_config.get(
            'constant_induced_field_node_coords')
        if constant_induced_field_node is not None:
            constant_induced_field_node = \
                self.load_relative_npy_file(constant_induced_field_node)
        self.constant_induced_field_node = constant_induced_field_node

        constant_external_field_center = _environment_config.get(
            'constant_external_field_center_coords')
        if constant_external_field_center is not None:
            constant_external_field_center = \
                self.load_relative_npy_file(constant_external_field_center)
        self.constant_external_field_center = constant_external_field_center

        constant_induced_field_center = _environment_config.get(
            'constant_induced_field_center_coords')
        if constant_induced_field_center is not None:
            constant_induced_field_center = \
                self.load_relative_npy_file(constant_induced_field_center)
        self.constant_induced_field_center = constant_induced_field_center

        vacuum_cell_mask = _environment_config.get('vacuum_cell_mask')
        if vacuum_cell_mask is not None:
            vacuum_cell_mask = self.load_relative_npy_file(vacuum_cell_mask)
        self.vacuum_cell_mask = vacuum_cell_mask

        self.create_dataset_kwargs = dict(
            chunks=True, shuffle=True, compression='gzip'
        )

        self.particles = dict()

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

    def serialize(self, out_fp, iteration=0, create_pmd_file=True,
                  author='nobody', t=None):
        out_fp = Path(out_fp)

        if self.iteration_encoding == 'file':
            if '%T' not in out_fp.name:
                _suffix = out_fp.suffix
                out_fp = out_fp.with_suffix(f'.%T{_suffix}')
            init_out_fp = out_fp.parent / \
                out_fp.name.replace('%T', f'{iteration}')

            iteration_encoding = 'fileBased'
            iteration_format = out_fp.name
        elif self.iteration_encoding == 'group':
            if '%T' in out_fp.name:
                # remove the cycle pattern from the file name
                out_fp = remove_cycle_pattern_from_filename(out_fp)
            init_out_fp = out_fp

            iteration_encoding = 'groupBased'
            iteration_format = '/data/%T/'
        else:
            raise ValueError(self.iteration_encoding)

        with h5py.File(init_out_fp, 'w') as h5f:
            flag = SavedFlag.empty

            self.setup_root_attr(h5f, iteration_encoding, iteration_format,
                                 author=author)
            t and t.update()
            self.setup_base_path(h5f, iteration=iteration)
            t and t.update()
            flag |= self.setup_fields(h5f, iteration=iteration)
            t and t.update()
            flag |= self.setup_particles(h5f, iteration=iteration)
            t and t.update()
            flag |= self.setup_stats(h5f, iteration=iteration)
            t and t.update()
            if self.save_state:
                flag |= self.setup_state(h5f, iteration=iteration)
            t and t.update()

            if self.iteration_encoding == 'file':
                h5f.attrs['_saved'] = flag.value
            else:
                h5f[f'data/{iteration}'].attrs['_saved'] = flag.value

        if create_pmd_file:
            _out_fp = remove_cycle_pattern_from_filename(out_fp)
            pmd_fp = _out_fp.with_suffix('.pmd')

            with open(pmd_fp, 'w') as f:
                f.write(f'{out_fp.name}\n')
        t and t.close()

        # print total number of particles
        print('Number of particles generated:')
        for grid_index, data in self.particles.items():
            species = data['species']
            print(f" {grid_index} ({species}) : {data['n_particles']:,}")

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

    @property
    def env_attrs(self):
        raise NotImplementedError()

    def setup_root_attr(self, h5f, iteration_encoding, iteration_format,
                        author='nobody'):
        current_dt = datetime.now().astimezone()
        root_attrs = dict(
            basePath=np.string_('/data/%T/'),
            author=np.string_(author),
            date=np.string_(current_dt.strftime('%Y-%m-%d %H:%M:%S %z')),
            iterationEncoding=np.string_(iteration_encoding),
            iterationFormat=np.string_(iteration_format),
            meshesPath=np.string_('fields/'),
            particlesPath=np.string_('particles/'),
            openPMD=np.string_('1.1.0'),
            openPMDextension=np.uint32(1),
            software=np.string_('PLEXsim Initializer'),
            softwareVersion=np.string_('1'),
            _geometry=np.string_(self.coordinate_system)
        )

        self.write_settings(h5f, root_attrs)

        settings = h5f.require_group('settings')
        self.write_settings(settings, self.env_attrs)
        settings.create_dataset('gilbert_curve',
                                data=np.array(self.gilbert_curve),
                                dtype=np.int16)

        if self.constant_external_field_node is not None:
            settings.create_dataset(
                'constant_external_field_node_coords',
                data=np.array(self.constant_external_field_node),
                dtype=np.int16
            )

        if self.constant_induced_field_node is not None:
            settings.create_dataset(
                'constant_induced_field_node_coords',
                data=np.array(self.constant_induced_field_node),
                dtype=np.int16
            )

        if self.constant_external_field_center is not None:
            settings.create_dataset(
                'constant_external_field_center_coords',
                data=np.array(self.constant_external_field_center),
                dtype=np.int16
            )

        if self.constant_induced_field_center is not None:
            settings.create_dataset(
                'constant_induced_field_center_coords',
                data=np.array(self.constant_induced_field_center),
                dtype=np.int16
            )

        if self.vacuum_cell_mask is not None:
            settings.create_dataset(
                'vacuum_cell_mask', data=np.array(self.vacuum_cell_mask))

    def base_path(self, h5f, iteration):
        return np.string_(h5f.attrs['basePath']).replace(
            b'%T', np.string_(str(iteration)))

    def setup_base_path(self, h5f, iteration=0, delta_time=1):
        base_path = self.base_path(h5f, iteration)
        base = h5f.require_group(base_path)

        base_attrs = dict(
            time=np.float64(iteration),
            dt=1.,
            timeUnitSI=np.float64(delta_time)
        )

        self.write_settings(base, base_attrs)

    @property
    def grid_vertex_shape(self):
        raise NotImplementedError()

    @property
    def B_shape(self):
        raise NotImplementedError()

    @property
    def E_shape(self):
        raise NotImplementedError()

    def load_field_array(self, field, shape, dtype=np.float64):
        if isinstance(field, list):
            return np.full(shape, field, dtype=dtype)
        elif isinstance(field, str):
            if field.endswith('.npy'):
                F = self.load_relative_npy_file(field).astype(dtype)
                assert np.all(F.shape == shape)
                return F
            else:
                raise NotImplementedError()
        elif isinstance(field, np.ndarray):
            assert np.all(field.shape == shape)
            return field.astype(dtype)
        else:
            NotImplementedError()

    def setup_fields(self, h5f, iteration=0):
        fields_path = self.base_path(h5f, iteration) \
            + np.string_(h5f.attrs['meshesPath'])
        fields = h5f.require_group(fields_path)

        dimension = len(self.grid_shape)

        fields_attrs = dict(
            fieldSolver=np.string_('none'),
            fieldBoundary=np.full(2 * dimension, np.string_('open')),
            particleBoundary=np.full(2 * dimension, np.string_('absorbing')),
            currentSmoothing=np.string_('none'),
            chargeCorrection=np.string_('none')
        )

        self.write_settings(fields, fields_attrs)

        field_dtype = self.environment_config.get('field_dtype')
        if field_dtype == 'fp32':
            field_dtype = np.float32
        elif field_dtype == 'fp64':
            field_dtype = np.float64
        else:
            raise NotImplementedError()

        self.B_external = self.load_field_array(
            self.environment_config.get('external_magnetic_field'),
            self.B_shape, field_dtype
        )
        if self.constant_external_field_center is not None:
            self.B_external[tuple(
                axis for axis in self.constant_external_field_center.T)] = 0

        self.B_induced = self.load_field_array(
            self.environment_config.get('induced_magnetic_field', [0, 0, 0]),
            self.B_shape, field_dtype
        )
        if self.constant_induced_field_center is not None:
            self.B_induced[tuple(
                axis for axis in self.constant_induced_field_center.T)] = 0

        settings = h5f.require_group('settings')
        settings.create_dataset(
            'B_fixed', data=np.array(self.B_external + self.B_induced))

        self.E_external = self.load_field_array(
            self.environment_config.get('external_electric_field'),
            self.E_shape, field_dtype
        )
        if self.constant_external_field_node is not None:
            self.E_external[tuple(
                axis for axis in self.constant_external_field_node.T)] = 0

        self.E_induced = self.load_field_array(
            self.environment_config.get('induced_electric_field', [0, 0, 0]),
            self.E_shape, field_dtype)
        if self.constant_induced_field_node is not None:
            self.E_induced[tuple(
                axis for axis in self.constant_induced_field_node.T)] = 0

        self.write_B(fields)
        self.write_E(fields)

        self.J_vac = self.load_field_array(
            self.environment_config.get('vacuum_current_density', [0, 0, 0]),
            self.E_shape, field_dtype)
        self.write_J(fields)

        return SavedFlag.fields

    @property
    def axis_labels(self):
        raise NotImplementedError()

    @property
    def B_attrs(self):
        raise NotImplementedError()

    @property
    def E_attrs(self):
        raise NotImplementedError()

    def write_B(self, fields_group):
        B_group = fields_group.require_group(np.string_('B'))
        B_induced_group = fields_group.require_group(np.string_('B_induced'))

        B_attrs = self.B_attrs

        self.write_settings(B_group, B_attrs)
        self.write_settings(B_induced_group, B_attrs)

        B = self.B_external + self.B_induced
        axis_labels = [np.string_(v) for v in self.axis_labels]
        dimension = len(self.grid_shape)
        for i, axis in enumerate(axis_labels):
            B_group.create_dataset(axis, data=B[..., i],
                                   **self.create_dataset_kwargs)
            B_group[axis].attrs['position'] = np.zeros(
                dimension, dtype=B.dtype)
            B_group[axis].attrs['unitSI'] = np.float64(1)

            B_induced_group.create_dataset(axis, data=self.B_induced[..., i],
                                           **self.create_dataset_kwargs)
            B_induced_group[axis].attrs['position'] = np.zeros(
                dimension, dtype=self.B_induced.dtype)
            B_induced_group[axis].attrs['unitSI'] = np.float64(1)

    def write_E(self, fields_group):
        E_group = fields_group.require_group(np.string_('E'))
        E_induced_group = fields_group.require_group(np.string_('E_induced'))

        E_attrs = self.E_attrs
        self.write_settings(E_group, E_attrs)
        self.write_settings(E_induced_group, E_attrs)

        E = self.E_external + self.E_induced
        axis_labels = [np.string_(v) for v in self.axis_labels]
        dimension = len(self.grid_shape)
        for i, axis in enumerate(axis_labels):
            E_group.create_dataset(axis, data=E[..., i],
                                   **self.create_dataset_kwargs)
            E_group[axis].attrs['position'] = np.zeros(
                dimension, dtype=E.dtype)
            E_group[axis].attrs['unitSI'] = np.float64(1)

            E_induced_group.create_dataset(axis, data=self.E_induced[..., i],
                                           **self.create_dataset_kwargs)
            E_induced_group[axis].attrs['position'] = np.zeros(
                dimension, dtype=self.E_induced.dtype)
            E_induced_group[axis].attrs['unitSI'] = np.float64(1)

    def write_J(self, fields_group):
        raise NotImplementedError()

    def load_particles(self, dtype_X, dtype_U, particles, grid_config):
        raise NotImplementedError()

    def load_particles_pre(self, particles, grid_config):
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

    def position_offset_attrs(self, n_particles):
        raise NotImplementedError()

    def get_particle_attrs(self, n_particles, q, m):
        charge_attrs = dict(
            value=q,
            shape=np.array([n_particles], dtype=np.uint64),
            macroWeighted=np.uint32(1),
            weightingPower=1.,
            timeOffset=0.,
            unitSI=np.float64(1),
            unitDimension=np.array([0, 0, 1, 1, 0, 0, 0], dtype=np.float64)
        )
        mass_attrs = dict(
            value=m,
            shape=np.array([n_particles], dtype=np.uint64),
            macroWeighted=np.uint32(1),
            weightingPower=1.,
            timeOffset=0.,
            unitSI=np.float64(1),
            unitDimension=np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.float64)
        )
        position_attrs = dict(
            macroWeighted=np.uint32(1),
            weightingPower=0.,
            timeOffset=0.,
            unitDimension=np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        )
        momentum_attrs = dict(
            macroWeighted=np.uint32(1),
            weightingPower=0.,
            timeOffset=0.,
            unitDimension=np.array([1, 1, -1, 0, 0, 0, 0], dtype=np.float64),
        )

        return dict(
            charge=charge_attrs,
            mass=mass_attrs,
            position=position_attrs,
            positionOffset=self.position_offset_attrs(n_particles),
            momentum=momentum_attrs
        )

    def setup_particles(self, h5f, iteration=0):
        raise NotImplementedError()

    def write_particle_patches_offset(self, patches, n_splits):
        raise NotImplementedError()

    def write_particle_attrs(self, h5_group, particle_data, n_splits,
                             n_computational_to_physical, dtype_X, dtype_U):
        n_particles = particle_data['n_particles']
        q = particle_data['q']
        m = particle_data['m']

        particles_attrs = self.get_particle_attrs(n_particles, q, m)

        gilbert_n_particles = particle_data['gilbert_n_particles']
        start_indices, end_indices = \
            self.split_in_balance(gilbert_n_particles, n_splits)
        end_p_indices = gilbert_n_particles.cumsum()[end_indices] - 1
        start_p_indices = np.empty_like(end_p_indices).astype(int)
        start_p_indices[0] = 0
        start_p_indices[1:] = end_p_indices[:-1] + 1

        grid_attrs = dict(
            particleShape=3.0,
            currentDeposition=np.string_('none'),
            particlePush=np.string_('Boris'),
            particleInterpolation=np.string_('uniform'),
            particleSmoothing=np.string_('none'),
            **particles_attrs,
            _startIndices=start_indices,
            _endIndices=end_indices,
            _startParticleIndices=start_p_indices,
            _endParticleIndices=end_p_indices,
            _is_cell_sorted=True
        )

        self.write_settings(h5_group, grid_attrs)

        h5_group.create_dataset('_gilbert_n_particles',
                                data=gilbert_n_particles,
                                dtype=np.uint64)

        patches = h5_group.require_group('particlePatches')
        patches.create_dataset('numParticles',
                               data=end_p_indices - start_p_indices + 1,
                               dtype=np.uint64)
        patches['numParticles'].attrs['unitSI'] = np.float64(1)
        patches.create_dataset('numParticlesOffset',
                               data=start_p_indices,
                               dtype=np.uint64)
        patches['numParticlesOffset'].attrs['unitSI'] = np.float64(1)

        self.write_particle_patches_offset(patches, n_splits)

        patches.require_group('extent')
        patches['extent'].attrs['unitDimension'] = np.array(
            [1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        for i, axis in enumerate(self.axis_labels):
            patches['extent'].create_dataset(
                axis, data=np.full(n_splits, self.grid_shape[i]))
            patches[f'extent/{axis}'].attrs['unitSI'] = \
                np.float64(self.cell_size[i])

        weighting_attrs = dict(
            macroWeighted=np.uint32(1),
            weightingPower=1.,
            timeOffset=0.,
            unitSI=np.float64(1),
            unitDimension=np.array(
                [0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        )
        _create_dataset_kwargs = self.create_dataset_kwargs.copy()
        if n_particles > self.chunk_size * 2:
            _create_dataset_kwargs['chunks'] = (self.chunk_size,)
        h5_group.create_dataset('weighting', (n_particles,),
                                dtype=np.uint64,
                                **_create_dataset_kwargs)
        h5_group['weighting'][:] = n_computational_to_physical
        self.write_settings(h5_group['weighting'], weighting_attrs)

        # create dataset with size (n + 1,) for MPI collective serialization
        _create_dataset_kwargs = self.create_dataset_kwargs.copy()
        avg_n_particles = n_particles / len(self.gilbert_curve)
        if avg_n_particles > self.chunk_size * 2:
            _create_dataset_kwargs['chunks'] = (self.chunk_size,)
        for i, axis in enumerate(self.axis_labels):
            # X
            _path = f'position/{axis}'
            h5_group.create_dataset(_path, (n_particles + 1, ),
                                    dtype=dtype_X,
                                    **_create_dataset_kwargs)
            h5_group[_path].attrs['unitSI'] = np.float64(self.cell_size[i])

            # U
            _path = f'momentum/{axis}'
            h5_group.create_dataset(_path, (n_particles + 1, ),
                                    dtype=dtype_U,
                                    **_create_dataset_kwargs)
            h5_group[_path].attrs['unitSI'] = np.float64(m)

    def serialize_tracked(self, tracked_group, grid_index, n_track_particles,
                          q, m, n_computational_to_physical, n_particles,
                          tracking_start_id, particle_group):
        # custom attribute
        tracked_group.attrs['_gridIndex'] = grid_index
        tracked_group.attrs['_tracked'] = 1

        tracked_attrs = self.get_particle_attrs(n_track_particles, q, m)
        tracked_attrs.update(dict(
            particleShape=3.0,
            currentDeposition=np.string_('none'),
            particlePush=np.string_('Boris'),
            particleInterpolation=np.string_('uniform'),
            particleSmoothing=np.string_('none')))

        self.write_settings(tracked_group, tracked_attrs)

        weighting_attrs = dict(
            macroWeighted=np.uint32(1),
            weightingPower=1.,
            timeOffset=0.,
            unitSI=np.float64(1),
            unitDimension=np.array(
                [0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        )
        _create_dataset_kwargs = self.create_dataset_kwargs.copy()
        if n_track_particles > self.chunk_size * 2:
            _create_dataset_kwargs['chunks'] = (self.chunk_size,)
        tracked_group.create_dataset(
            'weighting', (n_track_particles,), dtype=np.uint64,
            **_create_dataset_kwargs)
        tracked_group['weighting'][:] = n_computational_to_physical
        self.write_settings(tracked_group['weighting'],
                            weighting_attrs)

        particle_indices = np.random.choice(
            n_particles, n_track_particles, False)
        particle_indices.sort()
        tracking_ids = np.arange(
            tracking_start_id,
            tracking_start_id + n_track_particles,
            dtype=np.uint64)

        tracked_group.attrs['_particleIndices'] = particle_indices
        tracked_group.create_dataset('id', data=tracking_ids)
        id_attr = dict(
            unitSI=np.float64(1),
            macroWeighted=np.uint32(1),
            timeOffset=np.float64(0),
            unitDimension=np.zeros(7, dtype=np.float64),
            weightingPower=np.float64(0)
        )
        self.write_settings(tracked_group['id'], id_attr)

        for i, axis in enumerate(self.axis_labels):
            # X
            _path = f'position/{axis}'
            X = particle_group[_path]
            tracked_group.create_dataset(_path, data=X[list(particle_indices)],
                                         **_create_dataset_kwargs)
            tracked_group[_path].attrs['unitSI'] = \
                np.float64(self.cell_size[i])

            # U
            # velocity is saved as momentum (required by openPMD)
            _path = f'momentum/{axis}'
            U = particle_group[_path]
            tracked_group.create_dataset(_path, data=U[list(particle_indices)],
                                         **_create_dataset_kwargs)
            tracked_group[_path].attrs['unitSI'] = np.float64(m)

    @property
    def magnetic_E(self):
        raise NotImplementedError()

    @property
    def electric_E(self):
        raise NotImplementedError()

    def setup_stats(self, h5f, iteration=0):
        magnetic_E, induced_magnetic_E = self.magnetic_E
        electric_E, induced_electric_E = self.electric_E
        field_E = magnetic_E + electric_E

        n_particles = []
        kinetic_E = []
        for grid in self.particles.values():
            n_particles.append(grid['n_particles'])
            kinetic_E.append(grid.get('kinetic_E'))

        stats_path = self.base_path(h5f, iteration) + np.string_('stats')
        stats_group = h5f.require_group(stats_path)

        stats_attrs = dict(
            n_particles=np.array(n_particles),
            kinetic_E=np.array(kinetic_E),
            electric_E=electric_E,
            induced_electric_E=induced_electric_E,
            magnetic_E=magnetic_E,
            induced_magnetic_E=induced_magnetic_E,
            total_E=sum(kinetic_E) + field_E
        )

        self.write_settings(stats_group, stats_attrs)

        return SavedFlag.stats

    @property
    def grid_global_offset(self):
        raise NotImplementedError()

    def write_state(self, fields_group, particle_name, grid_n,
                    grid_U, grid_T):
        axis_labels = np.array([np.string_(v) for v in self.axis_labels])
        n_attrs = dict(
            geometry=np.string_(self.coordinate_system),
            gridSpacing=self.cell_size,
            gridGlobalOffset=self.grid_global_offset,
            gridUnitSI=np.float64(1),
            dataOrder=np.string_('C'),
            axisLabels=axis_labels,
            unitDimension=np.array(
                [-3, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            fieldSmoothing=np.string_('none'),
            timeOffset=0.,
            position=np.array([0, 0, 0], dtype=np.float64),
            unitSI=np.float64(1)
        )
        fields_group.create_dataset(f'{particle_name}_n', data=grid_n,
                                    **self.create_dataset_kwargs)
        self.write_settings(fields_group[f'{particle_name}_n'], n_attrs)

        T_attrs = dict(
            geometry=np.string_(self.coordinate_system),
            gridSpacing=self.cell_size,
            gridGlobalOffset=self.grid_global_offset,
            gridUnitSI=np.float64(1),
            dataOrder=np.string_('C'),
            axisLabels=axis_labels,
            unitDimension=np.array(
                [2, 1, -3, -1, 0, 0, 0], dtype=np.float64),
            fieldSmoothing=np.string_('none'),
            timeOffset=0.
        )
        T_group = fields_group.require_group(f'{particle_name}_T')
        self.write_settings(fields_group[f'{particle_name}_T'], T_attrs)
        for i, axis in enumerate(axis_labels):
            T_group.create_dataset(axis, data=grid_T[..., i],
                                   **self.create_dataset_kwargs)
            T_group[axis].attrs['position'] = \
                np.array([0, 0, 0], dtype=np.float64)
            T_group[axis].attrs['unitSI'] = np.float64(1)

        T_group.create_dataset('mean', data=grid_T.mean(axis=-1),
                               **self.create_dataset_kwargs)
        T_group['mean'].attrs['position'] = \
            np.array([0, 0, 0], dtype=np.float64)
        T_group['mean'].attrs['unitSI'] = np.float64(1)

        U_attrs = dict(
            geometry=np.string_(self.coordinate_system),
            gridSpacing=self.cell_size,
            gridGlobalOffset=self.grid_global_offset,
            gridUnitSI=np.float64(1),
            dataOrder=np.string_('C'),
            axisLabels=axis_labels,
            unitDimension=np.array(
                [1, 0, -1, 0, 0, 0, 0], dtype=np.float64),
            fieldSmoothing=np.string_('none'),
            timeOffset=0.,
        )
        U_group = fields_group.require_group(f'{particle_name}_U')
        self.write_settings(fields_group[f'{particle_name}_U'], U_attrs)
        for i, axis in enumerate(axis_labels):
            U_group.create_dataset(axis, data=grid_U[..., i],
                                   **self.create_dataset_kwargs)
            U_group[axis].attrs['position'] = np.array([0, 0, 0],
                                                       dtype=np.float64)
            U_group[axis].attrs['unitSI'] = np.float64(1)

    def setup_state(self, h5f, iteration=0, density_threshold=1e-10,
                    _e=1.602e-19, _m=9.1093837e-31, c=2.99792458e8):
        raise NotImplementedError()
