import numpy as np
from numba import njit, float64


@njit
def node_to_center_cartesian(V, V_center):
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

    w000 = float64(xi[0] * eta[0] * zeta[0])
    w001 = float64(xi[0] * eta[0] * zeta[1])
    w010 = float64(xi[0] * eta[1] * zeta[0])
    w011 = float64(xi[0] * eta[1] * zeta[1])
    w100 = float64(xi[1] * eta[0] * zeta[0])
    w101 = float64(xi[1] * eta[0] * zeta[1])
    w110 = float64(xi[1] * eta[1] * zeta[0])
    w111 = float64(xi[1] * eta[1] * zeta[1])

    return (((w000, w001), (w010, w011)), ((w100, w101), (w110, w111)))


@njit
def add_density_velocity(cell_index, U, grid_n, grid_U, grid_U2, weight):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                _i = cell_index[0] + i
                _j = cell_index[1] + j
                _k = cell_index[2] + k

                grid_n[_i, _j, _k] += weight[i][j][k]
                grid_U[_i, _j, _k] += U * weight[i][j][k]
                grid_U2[_i, _j, _k] += U * U * weight[i][j][k]


@njit
def compute_grid_velocity(X, U, C_idx, grid_n, grid_U, grid_U2):
    for i in range(X.shape[0]):
        cell_index = C_idx[i]
        if np.any(cell_index == -1):
            continue

        weight = weight_function(X[i])

        add_density_velocity(
            cell_index, U[i], grid_n, grid_U, grid_U2, weight)
