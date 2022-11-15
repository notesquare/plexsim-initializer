# PLEXsim

Plasma Extensible Simualtion (PLEXsim) Solver 사용 설명입니다.


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
  dynamic_balancing_frequency: 100
  backend: mpi
  checkpoint_strategy:
    save_particles_frequency: 20
    save_field_frequency: 1
    save_tracked_frequency: 1
    save_state_frequency: -1
    chunk_size: 1.e+6
```

#### grids
각 입자에 대한 설정을 목록(`-`)으로 나누어 지정합니다. PLEXsim Initializer에서 설정한 수와 순서가 일치해야 합니다.

- **backend**: _{'cpu', 'gpu'}_
- **n_splits**: _int_
  - PLEXsim Initializer에서 설정한 값과 일치해야 합니다. 추후 `n_splits`를 변경할 수 있도록 업데이트 예정입니다.
- **max_n_particles_per_subgrid**: _int_
  - 각 SubsetGrid(`n_splits` 에 의해 입자를 나누어 갖는 단위)의 최대 입자 수를 지정합니다.

#### simulation
시뮬레이션에 관련 설정입니다. `_frequency`가 붙은 항목의 경우 `0`이나 `-1`로 설정하면 옵션을 끌 수 있습니다.

- **delta_time**: _float_
  - 각 cycle의 시간 간격(s)을 지정합니다.
- **n_cycles**: _int_
  - 시뮬레이션 cycle 수를 지정합니다.
- **dynamic_balancing_frequency**: _int_
  - 몇 cycle마다 dynamic balancing을 할지 설정합니다. 각 SubsetGrid에 입자들이 불균등하게 분포하는 경우 속도와 메모리 최적화를 위해 설정할 수 있습니다.
- **backend**: _{'mpi'}_
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
