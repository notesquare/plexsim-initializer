# PLEXsim

PLasma EXtensible Simulation (PLEXsim) Solver 사용 설명입니다.


## Usage

```bash
$ run.sh [HDF5_FILE] [CONFIG_FILE] [OPTIONS]
```

전체 실행 옵션 출력
```sh
$ run.sh --help
```


## Config
### 1. HDF5 File

e.g. `proj.0.h5`

- PLEXsim Initializer를 통해 만들어지는 파일입니다.
- 전체 입자 정보(`/data/{cycle}/particles/{particle_name}`) 및 field 정보 (`/data/{cycle}/fields/(B, E)`)를 반드시 포함하고 있어야 합니다.


### 2. Config File

e.g. `config_run.yaml`

```yaml
grids:
  -
    backend: gpu
    n_splits: 2
    max_n_particles_per_subgrid: 3.e+8

  -
    backend: gpu
    n_splits: 2
    max_n_particles_per_subgrid: 3.e+8
    
simulation:
  delta_time: 1.e-10
  n_cycles: 200
  theta: 0.5
  dynamic_balancing_frequency: 100
  backend: mpi
  use_shared_memory: No
  checkpoint_strategy:
    save_particles_frequency: 20
    save_field_frequency: 1
    save_tracked_frequency: 1
    save_state_frequency: -1
    chunk_size: 1.e+6
  gmres:
    rtol: 0
    atol: 1.e-6
```

#### grids
각 입자에 대한 설정을 목록(`-`)으로 나누어 지정합니다. PLEXsim Initializer에서 설정한 수와 순서가 일치해야 합니다.

- **backend**: _{'cpu', 'gpu'}_
- **n_splits**: _int_
  - PLEXsim Initializer에서 설정한 값과 일치해야 합니다. 추후 `n_splits`를 변경할 수 있도록 업데이트 예정입니다.
- **max_n_particles_per_subgrid**: _int_
  - 각 SubGrid(`n_splits` 에 의해 입자를 나누어 갖는 단위)의 최대 입자 수를 지정합니다.

#### simulation
시뮬레이션에 관련 설정입니다. `_frequency`가 붙은 항목의 경우 `0`이나 `-1`로 설정하면 옵션을 끌 수 있습니다.

- **delta_time**: _float_
  - 각 cycle의 시간 간격(s)을 지정합니다.
- **n_cycles**: _int_
  - 시뮬레이션 cycle 수를 지정합니다.
- **theta**: _float, optional_
  - 시뮬레이션의 theta-scheme을 지정합니다. 입력하지 않을 경우 0.5로 설정됩니다.
- **dynamic_balancing_frequency**: _int_
  - 몇 cycle마다 dynamic balancing을 할지 설정합니다. 각 SubGrid에 입자들이 불균등하게 분포하는 경우 속도와 메모리 최적화를 위해 설정할 수 있습니다.
- **backend**: _{'mpi, local'}_
  - _'mpi'_: PLEXsim Distributed를 실행합니다. 다중 노드 및 다중 GPU를 사용할 수 있습니다.
  - _'local'_: PLEXsim Core를 실행합니다. 단일 노드, 단일 GPU를 사용할 수 있습니다.
- **checkpoint_strategy**
  - 저장 옵션을 지정합니다.
  - **save_particles_frequency**: _int_
    - 전체 입자의 위치와 속도를 몇 cycle마다 저장할지 설정합니다.
    - 결과 파일의 `/data/{cycle}/particles/{particle_name}`에 저장됩니다.
  - **save_field_frequency**: _int_
    - 전기장과 자기장을 몇 cycle마다 저장할지 설정합니다.
    - 결과 파일의 `/data/{cycle}/fields/(B, E)`에 저장됩니다.
  - **save_tracked_frequency**: _int_
    - 추적하고 있는 입자의 정보를 몇 cycle마다 저장할지 설정합니다.
    - 결과 파일의 `/data/{cycle}/particles/{particle_name}_tracked`에 저장됩니다.
  - **save_state_frequency**: _int_
    - 각 grid에 대한 격자점의 temperature(`T`)[eV], density(`n`)[m^-3^], avg_velocity(`U`)[m/s]를 몇 cycle마다 저장할지 설정합니다.
    - 결과 파일의 `/data/{cycle}/fields/{particle_name}_(n, U, T)`에 저장됩니다.
- **gmres**
  - field solver의 [GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lgmres.html) 관련 parameter를 설정합니다.
  - **rtol, atol**: _float, optional_
    - gmres의 tolerance를 설정합니다. `norm(residual) <= max(rtol*norm(b), atol)` 을 만족하는 해를 찾습니다. 입력하지 않으면 `rtol=0`, `atol=1e-6`으로 설정됩니다.


## Serialize Structure

### 파일 구조

- Format: HDF-5 (openPMD)
- 각각의 입자 species 마다 하나의 HDF-5 파일(e.g. `proj.g0.h5`)을 만들고, 이를 총괄하는 HDF-5 파일(e.g. `proj.h5`)이 하나 있습니다. 총괄 HDF-5 파일만 열어도 모든 정보를 확인할 수 있지만, 일부는 species HDF-5 파일에 External Link로 연결되어 있기 때문에 파일들이 같은 디렉토리에 있어야 합니다.

### 저장 그룹 구조

e.g. `proj.h5`
```
/data
  /{cycle}
    /fields
      /B  # magnetic field
      /E  # electric field
      electron_T  # temperature[eV] of electron on node points
      electron_n  # density[m^-3] of electron on node points
      /electron_U  # avg velocity[m/s] of electron on node points
    /particles
      /electron  # all particles
        /position
        /momentum  # dataset x,y,z equal to velocity,
                   # and unitSI equals to mass
        ...
      /electron_tracked  # tracking particles
      /ion
      /ion_tracked
      ...
    /stats
      @n_particles  # same order as config
      @kinetic_E
      ...
/settings
```


## GPU 메모리 테스트

시뮬레이션 가능한 최대 입자 수를 계산하기 위한 메모리 테스트 기능입니다. MPI 환경에서 사용 가능하며 단일 노드, 단일/다중 GPU를 지원합니다. PLEXsim Initializer를 실행하기 전 대략적인 capacity를 계산할 수 있습니다.

### Usage
```bash
$ MPIRUN=[MPI_OPTIONS] maxmem_test.sh [CONFIG_FILE] [OPTIONS]
```
`maxmem_test.sh` 파일을 실행하면 현재 사용 가능한 GPU 메모리 크기를 기준으로 `--test-ratio`에 비례하는 임의의 입자 수를 로드하여 2사이클 동안 시뮬레이션을 진행합니다. 정상적으로 실행이 종료된 경우 `EXIT_CODE=0`을 리턴하며 `--test-ratio`를 늘려 더 많은 입자 수로 다시 테스트를 진행합니다. 만약 실행 도중 GPU `OutOfMemoryError` 에러 등이 발생하면 `EXIT_CODE=2`를 리턴해 더 적은 입자 수로 테스트를 진행합니다. 알려지지 않은 버그 등으로 실행이 종료되거나(`EXIT_CODE=1`) 정해진 횟수만큼 반복하면 테스트가 종료됩니다. (`EXIT_CODE=1`이 출력된 경우 알려주시면 반영하겠습니다.)

### Config
PLEXsim Initializer에 사용되는 `.yaml` 파일을 그대로 사용하실 수 있습니다. 

e.g. `config_init.yaml`
```yaml
environment:
  grid_shape: [70, 70, 32]
  cell_size: [0.01, 0.01, 0.01]
  field_dtype: fp64

grids:
  -
    species: electron
    dtype:
      X: fp32
      U: fp32
    n_splits: 2
  -
    species: ion
    dtype:
      X: fp32
      U: fp32
    n_splits: 2

```
Config 파일에서 위의 값만 설정하면 나머지 값들은 임의의 값으로 초기화됩니다. `n_splits`는 현재 장치의 GPU 수를 고려하여 설정해주시면 됩니다.

### Options
- `-n`, `--n-simulations`: 시뮬레이션 반복 횟수를 지정합니다.
- `-T`, `--test-ratio`: 값이 클수록 많은 입자수로 테스트를 합니다. GPU의 현재 사용 가능한 메모리 및 Config를 기준으로 하므로 입자 수와 완전히 대응되지는 않습니다.
- `-L`, `--lower-bound`: `--test-ratio`의 하한을 설정합니다. 테스트를 이어서 할 경우 사용할 수 있습니다.
- `-U`, `--upper-bound`: `--test-ratio`의 상한을 설정합니다.
