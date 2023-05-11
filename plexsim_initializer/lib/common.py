from enum import Flag

import numpy as np
from numba import njit, float32


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


@njit
def node_to_center_3d(V, V_center):
    assert len(V.shape) == len(V_center.shape)
    for i in range(3):
        assert V.shape[i] == V_center.shape[i] + 1

    for i in range(V_center.shape[0]):
        for j in range(V_center.shape[1]):
            for k in range(V_center.shape[2]):
                V_center[i, j, k] = V[i, j, k] + V[i+1, j, k] + V[i, j+1, k]\
                    + V[i, j, k+1] + V[i+1, j+1, k] + V[i+1, j, k+1]\
                    + V[i, j+1, k+1] + V[i+1, j+1, k+1]
    V_center *= .125


@njit
def weight_function(X):
    x, y, z = X
    xi = (1 - x, x)
    eta = (1 - y, y)
    zeta = (1 - z, z)

    w000 = float32(xi[0] * eta[0] * zeta[0])
    w001 = float32(xi[0] * eta[0] * zeta[1])
    w010 = float32(xi[0] * eta[1] * zeta[0])
    w011 = float32(xi[0] * eta[1] * zeta[1])
    w100 = float32(xi[1] * eta[0] * zeta[0])
    w101 = float32(xi[1] * eta[0] * zeta[1])
    w110 = float32(xi[1] * eta[1] * zeta[0])
    w111 = float32(xi[1] * eta[1] * zeta[1])

    return (((w000, w001), (w010, w011)), ((w100, w101), (w110, w111)))


@njit
def add_density_velocity_disjunct(cell_index, U, grid_n,
                                  grid_U, grid_U2, weight):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                _i = cell_index[0] * 2 + i
                _j = cell_index[1] * 2 + j
                _k = cell_index[2] * 2 + k

                grid_n[_i, _j, _k] += weight[i][j][k]
                grid_U[_i, _j, _k] += U * weight[i][j][k]
                grid_U2[_i, _j, _k] += (U * U).sum() * weight[i][j][k]


@njit
def compute_grid_velocity_disjunct(X, U, C_idx, grid_n, grid_U, grid_U2):
    for i in range(X.shape[0]):
        cell_index = C_idx[i]
        if np.any(cell_index == -1):
            continue

        weight = weight_function(X[i])

        add_density_velocity_disjunct(
            cell_index, U[i], grid_n, grid_U, grid_U2, weight)


@njit
def conv3d_2x2_s2(output_arr, input_arr):
    input_shape = input_arr.shape
    output_shape = output_arr.shape

    for i in range(0, output_shape[0]):
        i1 = i * 2 - 1
        i2 = i1 + 1
        i1, i2 = max(i1, 0), min(i2, input_shape[0]-1)
        for j in range(0, output_shape[1]):
            j1 = j * 2 - 1
            j2 = j1 + 1
            j1, j2 = max(j1, 0), min(j2, input_shape[1]-1)
            for k in range(0, output_shape[2]):
                k1 = k * 2 - 1
                k2 = k1 + 1
                k1, k2 = max(k1, 0), min(k2, input_shape[2]-1)
                output_arr[i, j, k] += (
                    input_arr[i1, j1, k1]
                    + input_arr[i1, j1, k2]
                    + input_arr[i1, j2, k1]
                    + input_arr[i1, j2, k2]
                    + input_arr[i2, j1, k1]
                    + input_arr[i2, j1, k2]
                    + input_arr[i2, j2, k1]
                    + input_arr[i2, j2, k2]
                )


def conv3d_array(output_arr, input_arr_list):
    output_shape = np.array(output_arr.shape)
    for input_arr in input_arr_list:
        input_shape = np.array(input_arr.shape)
        assert (2 * (output_shape[:3] - 1) == input_shape[:3]).all(), \
            f'from {output_shape} to {input_shape}'
        conv3d_2x2_s2(output_arr, input_arr)


def remove_cycle_pattern_from_filename(fp):
    # test_%T.h5 -> test.pmd
    # test.%T.h5 -> test.pmd
    # test%T.h5 -> test.pmd
    name = fp.name
    name = name.replace('_%T', '')
    name = name.replace('.%T', '')
    name = name.replace('%T', '')
    return fp.parent / name
