import numpy as np

from ..lib.common import node_to_center_3d
from .base import BaseInitializer


class CylindricalInitializer(BaseInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r0 = self.environment_config['r0']

        cell_size_phi = 2 * np.pi / self.grid_shape[2]
        self.cell_size = np.append(self.cell_size, cell_size_phi)

    @property
    def env_attrs(self):
        return dict(
            grid_shape=self.grid_shape,
            cell_size=self.cell_size,
            permittivity=self.permittivity,
            permeability=self.permeability,
            scale_length=self.environment_config['scale_length'],
            r0=self.r0
        )

    @property
    def grid_vertex_shape(self):
        nz, nr, nphi = self.grid_shape
        return nz + 1, nr + 1, nphi

    @property
    def B_shape(self):
        nz, nr, nphi = self.grid_shape
        return nz + 1, nr + 1, nphi, 3
    
    @property
    def E_shape(self):
        nz, nr, nphi = self.grid_shape
        return nz + 1, nr + 1, nphi, 3

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

    def position_offset_attrs(self, n_particles):
        return dict(
            macroWeighted=np.uint32(1),
            weightingPower=0.,
            timeOffset=0.,
            unitDimension=np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            **{axis: dict(
                value=np.float32(0)+self.r0 if axis =='r' else np.float32(0),
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

    @property
    def magnetic_E(self):
        grid_center_shape = np.array((*(self.grid_shape), 3))
        B_center = np.empty(grid_center_shape)

        node_to_center_3d(self.B_external + self.B_induced,
                          B_center, self.coordinate_system)
        magnetic_E = 0.5 * self.cell_size.prod() * \
            self.permeability * (B_center * B_center).sum()

        node_to_center_3d(self.B_induced, B_center,
                          self.coordinate_system)
        induced_magnetic_E = 0.5 * self.cell_size.prod() * \
            self.permeability * (B_center * B_center).sum()

        return magnetic_E, induced_magnetic_E
