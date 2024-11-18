import os
import sys

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


FLAG_OPT_MAP = {
    'debug': 'd',
    'inspect': 'i',
    'interactive': 'i',
    'optimize': 'O',
    'dont_write_bytecode': 'B',
    'no_user_site': 's',
    'no_site': 'S',
    'ignore_environment': 'E',
    'verbose': 'v',
    'bytes_warning': 'b',
    'quiet': 'q',
    'hash_randomization': 'R',
    'isolated': 'I',
    # 'dev_mode': 'Xdev',
    # 'utf8_mode': 'Xutf8',
    # 'warn_default_encoding': 'Xwarn_default_encoding',
    'safe_path': 'P',
    # 'int_max_str_digits': 'Xint_max_str_digits=0'
}


def get_python_flags():
    args = []
    for flag, opt in FLAG_OPT_MAP.items():
        val = getattr(sys.flags, flag, 0)
        val = val if opt[0] != 'i' else 0
        val = val if opt[0] != 'Q' else min(val, 1)
        if val > 0:
            args.append('-' + opt * val)
    for opt in sys.warnoptions:  # pragma: no cover
        args.append('-W' + opt)
    sys_xoptions = getattr(sys, '_xoptions', {})
    for opt, val in sys_xoptions.items():  # pragma: no cover
        args.append('-X' + opt if val is True else
                    '-X' + opt + '=' + val)
    return args


def get_max_workers():
    max_workers = os.environ.get('MAX_WORKERS')
    if max_workers is not None:
        return int(max_workers)
    if MPI.UNIVERSE_SIZE != MPI.KEYVAL_INVALID:  # pragma: no branch
        universe_size = MPI.COMM_WORLD.Get_attr(MPI.UNIVERSE_SIZE)
        if universe_size is not None:  # pragma: no cover
            world_size = MPI.COMM_WORLD.Get_size()
            return max(universe_size - world_size, 1)
    return 1


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
                    'python', *get_python_flags()
                ]
            }
        elif 'SINGULARITY_CONTAINER' in os.environ:
            kwargs = {
                'python_exe': 'singularity',
                'python_args': [
                    '--silent', 'exec', os.environ['SINGULARITY_CONTAINER'],
                    'python', *get_python_flags()
                ]
            }

        self.executor = MPIPoolExecutor(**kwargs)
