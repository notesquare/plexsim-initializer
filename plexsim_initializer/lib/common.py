import math

from numba import njit, float32
from numba import cuda


# TODO: remove GPU dependency
@cuda.jit
def is_in_grid_kernel(C_idx, mask, coords):
    i = cuda.grid(1)
    if i >= C_idx.shape[0]:
        return

    for j in range(coords.shape[0]):
        flag = True
        for k in range(coords.shape[1]):
            if C_idx[i, k] != coords[j, k]:
                flag = False
                break
        if flag:
            break
    mask[i] = flag


def is_in_grid(C_idx, mask, coords, stream=None):
    if C_idx.shape[0] == 0:
        return
    threadsperblock = (32,)
    blockspergrid_x = math.ceil(C_idx.shape[0] / threadsperblock[0])
    blockspergrid = (blockspergrid_x,)
    is_in_grid_kernel[blockspergrid, threadsperblock, stream](
        C_idx, mask, coords)


@njit
def node_to_center_3d(V, V_center):
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


@cuda.jit
def add_density_velocity(cell_index, U, grid_n, grid_U, grid_N, weight):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                _i = cell_index[0] + i
                _j = cell_index[1] + j
                _k = cell_index[2] + k

                cuda.atomic.add(grid_N, (_i, _j, _k), 1)
                cuda.atomic.add(grid_n, (_i, _j, _k), weight[i][j][k])
                for m in range(3):
                    cuda.atomic.add(grid_U, (_i, _j, _k, m),
                                    U[m] * weight[i][j][k])


@cuda.jit
def compute_grid_velocity_kernel(X, U, C_idx, grid_n, grid_U, grid_N):
    i = cuda.grid(1)
    if i >= X.shape[0]:
        return

    cell_index = C_idx[i]
    if cell_index[0] == -1:
        return

    weight = weight_function(X[i])

    add_density_velocity(cell_index, U[i], grid_n, grid_U, grid_N, weight)


def compute_grid_velocity(X, U, C_idx, grid_n, grid_U, grid_N):
    if X.shape[0] == 0:
        return
    threadsperblock = 32
    blockspergrid = math.ceil(X.shape[0] / threadsperblock)
    compute_grid_velocity_kernel[blockspergrid, threadsperblock](
        X, U, C_idx, grid_n, grid_U, grid_N
    )


@cuda.jit
def add_temperature(cell_index, U, grid_T, grid_U, grid_N, weight, q, m):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                _i = cell_index[0] + i
                _j = cell_index[1] + j
                _k = cell_index[2] + k

                n = grid_N[_i, _j, _k]
                assert n != 0

                c = 0
                for m in range(3):
                    _c = (U[m] - grid_U[_i, _j, _k, m]) * weight[i][j][k]
                    c += math.pow(_c, 2)
                t = c * m / q
                cuda.atomic.add(grid_T, (_i, _j, _k), t / n)


@cuda.jit
def compute_grid_temperature_kernel(X, U, C_idx, grid_T, grid_U, grid_N, q, m):
    i = cuda.grid(1)
    if i >= X.shape[0]:
        return

    cell_index = C_idx[i]
    if cell_index[0] == -1:
        return

    weight = weight_function(X[i])

    add_temperature(cell_index, U[i], grid_T, grid_U, grid_N, weight, q, m)


def compute_grid_temperature(X, U, C_idx, grid_T, grid_U, grid_N, q, m):
    if X.shape[0] == 0:
        return
    threadsperblock = 32
    blockspergrid = math.ceil(X.shape[0] / threadsperblock)
    compute_grid_temperature_kernel[blockspergrid, threadsperblock](
        X, U, C_idx, grid_T, grid_U, grid_N, q, m
    )
