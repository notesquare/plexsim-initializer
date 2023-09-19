import numpy as np

from ..lib.common import node_to_center_3d
from .base import BaseInitializer


class CartesianInitializer(BaseInitializer):
    @property
    def env_attrs(self):
        return dict(
            grid_shape=self.grid_shape,
            cell_size=self.cell_size,
            permittivity=self.permittivity,
            permeability=self.permeability
        )

    @property
    def grid_vertex_shape(self):
        return (*(self.grid_shape + 1),)

    @property
    def B_shape(self):
        # defined at cell center
        return (*(self.grid_shape), 3)

    @property
    def E_shape(self):
        # defined at grid point
        return (*(self.grid_shape + 1), 3)

    @property
    def cell_volume(self):
        return self.cell_size.prod()

    @property
    def axis_labels(self):
        dimension = len(self.grid_shape)
        if dimension == 3:
            return ['x', 'y', 'z']
        else:
            raise NotImplementedError()

    @property
    def grid_global_offset(self):
        dimension = len(self.grid_shape)
        return np.zeros(dimension, dtype=np.float64)

    @property
    def B_attrs(self):
        return dict(
            geometry=np.string_(self.coordinate_system),
            gridSpacing=self.cell_size,
            gridGlobalOffset=self.grid_global_offset + 0.5,
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
                value=np.float32(0),
                shape=np.array([n_particles], dtype=np.uint64),
                unitSI=np.float64(1)
            ) for axis in self.axis_labels}
        )

    def write_particle_patches_offset(self, patches_group, n_splits):
        offset = patches_group.require_group('offset')
        offset.attrs['unitDimension'] = np.array(
            [1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        for i, axis in enumerate(self.axis_labels):
            offset.create_dataset(axis, data=np.full(n_splits, 0.))
            offset[axis].attrs['unitSI'] = np.float64(self.cell_size[i])

    @property
    def magnetic_E(self):
        B_total = self.B_external + self.B_induced
        magnetic_E = 0.5 * self.cell_volume / \
            self.permeability * (B_total * B_total).sum()

        induced_magnetic_E = 0.5 * self.cell_volume / \
            self.permeability * (self.B_induced * self.B_induced).sum()

        return magnetic_E, induced_magnetic_E

    @property
    def electric_E(self):
        grid_center_shape = np.array((*(self.grid_shape), 3))
        E_center = np.empty(grid_center_shape)

        node_to_center_3d(self.E_external + self.E_induced,
                          E_center, self.coordinate_system)
        electric_E = 0.5 * self.cell_volume * \
            self.permittivity * (E_center * E_center).sum()

        node_to_center_3d(self.E_induced, E_center,
                          self.coordinate_system)
        induced_electric_E = 0.5 * self.cell_volume * \
            self.permittivity * (E_center * E_center).sum()

        return electric_E, induced_electric_E