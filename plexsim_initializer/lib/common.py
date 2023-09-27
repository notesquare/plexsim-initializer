from enum import Flag

import numpy as np

from .cartesian import node_to_center_cartesian
from .cylindrical import node_to_center_cylindrical

class SavedFlag(Flag):
    empty = 0x0
    particles = 0x1
    fields = 0x2
    tracked = 0x4
    stats = 0x8
    state = 0x10

    @property
    def value(self):
        return np.uint8(super().value)


def node_to_center_3d(V, V_center, coordinate_system):
    assert len(V.shape) == len(V_center.shape)
    if coordinate_system == 'cartesian':
        node_to_center_cartesian(V, V_center)
    elif coordinate_system == 'cylindrical':
        node_to_center_cylindrical(V, V_center)
    else:
        raise NotImplementedError()


def remove_cycle_pattern_from_filename(fp):
    # test_%T.h5 -> test.pmd
    # test.%T.h5 -> test.pmd
    # test%T.h5 -> test.pmd
    name = fp.name
    name = name.replace('_%T', '')
    name = name.replace('.%T', '')
    name = name.replace('%T', '')
    return fp.parent / name
