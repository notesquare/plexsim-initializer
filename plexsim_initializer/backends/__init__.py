import yaml


def initialize(input_fp, out_fp, **kwargs):
    with open(input_fp, 'r') as f:
        config = yaml.safe_load(f)
        sim_config = config['simulation']

    sim_backend = sim_config.get('backend', 'local')
    if sim_backend == 'mpi':
        from .mpi.sim import initialize as MPI_initialize
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            # static mode, mpi4py.futures is required
            from mpi4py.futures._lib import SharedPoolCtx
            with SharedPoolCtx() as context:
                if context is not None:
                    MPI_initialize(input_fp, out_fp, **kwargs)
        else:
            # dynamic mode
            MPI_initialize(input_fp, out_fp, **kwargs)

    elif sim_backend == 'local':
        from .local.sim import initialize as local_initialize
        local_initialize(input_fp, out_fp, **kwargs)
    else:
        raise NotImplementedError(sim_backend)
