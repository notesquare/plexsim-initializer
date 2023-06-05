import os

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from mpi4py.futures import _lib
# monkey patch get_python_flags function
_get_python_flags = _lib.get_python_flags
_lib.get_python_flags = lambda: []
from mpi4py.futures._lib import get_max_workers, get_python_flags, get_spawn_module  # noqa


def gen_arg(arg, n_workers):
    for _ in range(n_workers):
        yield arg
    while True:
        yield None


class MPIInitializer:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if MPI.COMM_WORLD.Get_size() > 1:
            # static mode
            max_workers = MPI.COMM_WORLD.Get_size() - 1
        else:
            # dynamic mode
            max_workers = get_max_workers()
            if max_workers < 1:
                max_workers = 1

        self.max_workers = max_workers
        kwargs = {
            'max_workers': max_workers
        }

        # check if running under singularity or apptainer
        if 'APPTAINER_CONTAINER' in os.environ:
            kwargs = {
                'python_exe': "apptainer",
                'python_args': [
                    '--silent', 'exec', os.environ['APPTAINER_CONTAINER'],
                    'python', *_get_python_flags()
                ]
            }
        elif 'SINGULARITY_CONTAINER' in os.environ:
            kwargs = {
                'python_exe': 'singularity',
                'python_args': [
                    '--silent', 'exec', os.environ['SINGULARITY_CONTAINER'],
                    'python', *_get_python_flags()
                ]
            }

        self.executor = MPIPoolExecutor(**kwargs)
