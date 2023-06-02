#!/bin/bash
#SBATCH --job-name=small_test         # Job name
#SBATCH --comment inhouse
#SBATCH --time 00:05:00         # Time limit hrs:min:sec
#SBATCH --partition debug1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=5

APPTAINER_CONTAINER='apptainer/plexsim_initializer.sif'
PYTHON_ARGS="config_random.yaml test.h5"

MPIRUN_ARGS="--bind-to core --map-by slot:PE=3"
PLEXSIM_N_SUBSETS_PER_NODE=2  # number of gpus per node

# system-specific settings (evergreen,firmworld)
SLURM_HOME=${SLURM_HOME:-/shared/lib/slurm-21.08.8}
MPI_HOME=${MPI_HOME:-/shared/lib/ompi-4.1.1}
# ompi_info --all --parsable | grep mca:mca:base:param:mca_param_files:value
OMPI_MCA_PARAM_FILES_PATH=/etc/openmpi

# other common settings
export UCX_POSIX_USE_PROC_LINK=n  # enables RMA in UCX
export APPTAINER_BIND="$MPI_HOME:/opt/ompi,$OMPI_MCA_PARAM_FILES_PATH,$SLURM_HOME/lib:/host_slurm_lib,/usr/lib/:/host_usr_lib"

# debug settings
# MPIRUN_ARGS="$MPIRUN_AGRS --report-bindings --display-map --display-allocation"
# export OMPI_MCA_pml_base_verbose=10
# export OMPI_MCA_btl_base_verbose=10
# export OMPI_MCA_pml_ucx_verbose=10
# export OMPI_MCA_pmix_base_verbose=10
# export OMPI_MCA_plm_base_verbose=10

mpirun -n 1 $MPIRUN_ARGS \
    -x MPI4PY_RC_THREAD_LEVEL=serialized \
    $APPTAINER_CONTAINER $PYTHON_ARGS
