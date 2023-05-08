import yaml


def initialize(input_fp, out_fp, **kwargs):
    with open(input_fp, 'r') as f:
        config = yaml.safe_load(f)
        sim_config = config['simulation']

    sim_backend = sim_config.get('backend', 'local')
    if sim_backend == 'mpi':
        from .mpi.sim import initialize as MPI_initialize
        MPI_initialize(input_fp, out_fp, **kwargs)
    elif sim_backend == 'local':
        from .local.sim import initialize as local_initialize
        local_initialize(input_fp, out_fp, **kwargs)
    else:
        raise NotImplementedError(sim_backend)
