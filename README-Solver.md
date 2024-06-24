# PLEXsim

PLasma EXtensible Simulation (PLEXsim) Solver 사용 설명입니다.


## Usage

```bash
$ MPIHOME=[MPIHOME] PLEXSIM_BIN=[PLEXSIM_BIN] MPI_OPTIONS=[MPI_OPTIONS]\
  ./run.sh [HDF5_FILE] [CONFIG_FILE] [OPTIONS]
```
- **MPIHOME**: Open MPI 설치 경로
  - e.g. `/shared/lib/ompi4.1.1-gcc`
- **PLEXSIM_BIN**: PLEXsim Distributed 실행 파일 경로
  - e.g. `plexsim-1.2.1.sif`
- **MPI_OPTIONS**: `mpirun` 실행 옵션
  - [공식 문서 참조](https://www.open-mpi.org/doc/current/man1/mpirun.1.php)

- **OPTIONS**
  - 전체 실행 옵션 출력
    ```sh
    $ [PLEXSIM_BIN] run --help
    ```
  - [**positional arguments**](#positional-arguments)
    - HDF5 File
    - Config File
  - **optional arguments**
    - `--out`: 출력 파일 경로 설정
    - `--author`: 출력 파일에 작성자 기록
    - `--ui`, `--no-ui`: 시뮬레이션 진행 상황을 화면에 출력


## Positional Arguments
### 1. HDF5 File

e.g. `proj.0.h5`

- [PLEXsim Initializer](https://github.com/notesquare/plexsim-initializer)를 통해 만들어지는 파일입니다.
- 시뮬레이션을 이어서 하는 경우 PLEXsim Solver를 통해서 만들어지는 파일도 사용할 수 있습니다.
  - 이 경우 전체 입자 정보(`/data/{cycle}/particles/{particle_name}`) 및 field 정보 (`/data/{cycle}/fields/[B, E]`)를 반드시 포함하고 있어야 합니다.


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
  solver: explicit
  delta_time: 1.e-2
  n_cycles: 200
  theta: 0.5
  field_smoothing: 2
  field_damping_factor: 0.8
  dynamic_balancing_frequency: 100
  backend: mpi
  use_shared_memory: No
  loading_strategy:
    chunk_size: 1.e+6
  checkpoint_strategy:
    save_particles_frequency: 20
    save_field_frequency: 1
    save_tracked_frequency: 1
    save_state_frequency: -1
    chunk_size: 1.e+6
  check_stats_frequency: 0
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

- **solver**: _{'explicit', 'implicit'}_
  - `cartesian`: `implicit` method만을 지원
  - `cylindrical`: `implicit` 및 `explicit` method 지원
- **delta_time**: _float_
  - 각 cycle의 시간 간격을 지정합니다.
  - `cartesian`: [s]
  - `cylindrical`: `delta_time[s] = (scale_length / (2 * pi * c)) * delta_time`
- **n_cycles**: _int_
  - 시뮬레이션 cycle 수를 지정합니다.
- **theta**: _float, optional_
  - `cartesian` only
  - 시뮬레이션의 theta-scheme을 지정합니다. 입력하지 않을 경우 0.5로 설정됩니다.
- **field_smoothing**: _{0, 1, 2}, optional_
  - `cylindrical` only
  - _0_: smoothing off
  - _1_: r, z축 경계면을 포함하여 smoothing
  - _2_: r, z축 경계면 안쪽의 값만 smoothing
- **field_damping_factor**: _float, optional_
  - `cylindrical-explicit` only
  - LCFS 바깥쪽 field(B, E)를 lambda-damping. PLEXsim Initializer에서 `vacuum_cell_mask` 옵션을 준 경우에만 적용됨.
  - `E[vacuum_cell_mask] *= damping_factor`
  - `B[vacuum_cell_mask] = damping_factor * B[vacuum_cell_mask] + (1 - damping_factor) * B_fixed[vacuum_cell_mask]` (`B_fixed`는 Cycle 0에서의 total B)
- **dynamic_balancing_frequency**: _int_
  - 몇 cycle마다 dynamic balancing을 할지 설정합니다. 각 SubGrid에 입자들이 불균등하게 분포하는 경우 속도와 메모리 최적화를 위해 설정할 수 있습니다.
- **backend**: _{'mpi, local'}_
  - _'mpi'_: PLEXsim Distributed를 실행합니다. 다중 노드 및 다중 GPU를 사용할 수 있습니다.
  - _'local'_: PLEXsim Core를 실행합니다. 단일 노드, 단일 GPU를 사용할 수 있습니다.
- **loading_strategy**
  - **chunk_size**: _int, optional_
    - particle loading시 한 번에 읽어올 chunk의 크기 지정
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
- **check_stats_frequency**: _int, optional_
  - stats(n_particles, field 및 kinetic energy)를 몇 cycle마다 계산할지 설정합니다. 입력하지 않을 경우 1로 설정되며, 파일 저장을 하는 cycle에서는 항상 계산합니다.
- **gmres**
  - `implicit` method only (`cartesian`, `cylindrical`)
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
      /electron_T  # temperature[eV] of electron on node points
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

### Unit
- `cartesian`: SI unit
- `cylindrical`
  - SI unit: , `data/0/fields/[J_vacuum, {particle}_T, {particle}_U, {particle}_n`, `data/0/stats`
  - Normalized unit: `data/0/fields/[B, E, {particle}_J]`, `data/0/particles/{particle}/[momentum, position]`
    - position: `[0, grid_shape[i])` 사이의 값으로 normalized
    - momentum: `U[m/s] = U * c` (array의 값은 momentum이 아닌 velocity임에 주의)
    - fields: config에서 정의된 바와 같이 normalized

### 그 외 유의사항
- [mpi](#simulation) backend를 선택하여 Initializer를 실행할 경우 `/data/0/particles/{particle}/[momentum, position]`의 array는 가장 마지막 element를 NaN 값으로 채우고 있습니다.
  - e.g. 입자 수가 1000일 경우 1001의 크기를 갖는 배열이 만들어지며 index [0, 999]까지만 값이 채워집니다.
- PLEXsim Solver를 `cylindrical-explicit`으로 실행할 경우 입자 속도가 Lorentz transform되어 저장됩니다. 이는 `data/{cycle}/particles/{particle}/momentum`의 `_is_Lorentz_transformed` attribute 값이 1인 걸로 확인하실 수 있으며 없거나 0인 경우 로렌츠 변환되지 않은 것입니다.


## ~~GPU 메모리 테스트~~ (Not implemented in v1.3.0)

시뮬레이션 가능한 최대 입자 수를 계산하기 위한 메모리 테스트 기능입니다. MPI 환경에서 사용 가능하며 다중 노드, 다중 GPU를 지원합니다. PLEXsim Initializer를 실행하기 전 대략적인 capacity를 계산할 수 있습니다.

### Usage
```bash
$ MPIHOME=[MPIHOME] PLEXSIM_BIN=[PLEXSIM_BIN] MPI_OPTIONS=[MPI_OPTIONS]\
  ./maxmem_test.sh [CONFIG_FILE] [OPTIONS]
```

- **MPIHOME**: Open MPI 설치 경로
  - e.g. `/shared/lib/ompi4.1.1-gcc`
- **PLEXSIM_BIN**: PLEXsim Distributed 실행 파일 경로
  - e.g. `plexsim-1.2.1.sif`
- **MPI_OPTIONS**: `mpirun` 실행 옵션
  - [공식 문서 참조](https://www.open-mpi.org/doc/current/man1/mpirun.1.php)
- [**CONFIG_FILE**](#config)
- **OPTIONS**
  - `-n`, `--n-simulations`: 시뮬레이션 반복 횟수를 지정합니다.
  - `-T`, `--test-ratio`: 값이 클수록 많은 입자수로 테스트를 합니다. GPU의 현재 사용 가능한 메모리 및 Config를 기준으로 하므로 입자 수와 완전히 대응되지는 않습니다.
  - `-L`, `--lower-bound`: `--test-ratio`의 하한을 설정합니다. 테스트를 이어서 할 경우 사용할 수 있습니다.
  - `-U`, `--upper-bound`: `--test-ratio`의 상한을 설정합니다.

`maxmem_test.sh` 파일을 실행하면 현재 사용 가능한 GPU 메모리 크기를 기준으로 `--test-ratio`에 비례하는 임의의 입자 수를 로드하여 2사이클 동안 시뮬레이션을 진행합니다. 정상적으로 실행이 종료된 경우 `EXIT_CODE=0`을 리턴하며 `--test-ratio`를 늘려 더 많은 입자 수로 다시 테스트를 진행합니다. 만약 실행 도중 GPU `OutOfMemoryError` 에러 등이 발생하면 `EXIT_CODE=2`를 리턴해 더 적은 입자 수로 테스트를 진행합니다. 알려지지 않은 버그 등으로 실행이 종료되거나(`EXIT_CODE=1`) 정해진 횟수만큼 반복하면 테스트가 종료됩니다. `EXIT_CODE=1`이 출력된 경우 알려주시면 반영하겠습니다.

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
