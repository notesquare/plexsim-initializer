import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib


def gen_arg(arg, n_workers):
    for _ in range(n_workers):
        yield arg
    while True:
        yield None


def create_shm_array(shape, comm, data=None, main=False, dtype=None):
    if data is not None:
        if data.dtype == np.float32:
            datatype = MPI.FLOAT
        elif data.dtype == np.float64:
            datatype = MPI.DOUBLE
        else:
            raise NotImplementedError()
    elif dtype is not None:
        if dtype == 'fp32' or dtype == np.float32:
            datatype = MPI.FLOAT
        elif dtype == 'fp64' or dtype == np.float64:
            datatype = MPI.DOUBLE
        elif dtype == 'int':
            datatype = MPI.INT
        else:
            raise NotImplementedError()
    else:
        raise ValueError()

    np_dtype = dtlib.to_numpy_dtype(datatype)
    itemsize = datatype.Get_size()

    if main:
        win_size = np.prod(shape) * itemsize
    else:
        win_size = 0
    win = MPI.Win.Allocate_shared(win_size, itemsize, comm=comm)
    buf, itemsize = win.Shared_query(0)
    ary = np.ndarray(buffer=buf, dtype=np_dtype, shape=shape)
    if data is not None:
        ary[:] = data
    return ary
