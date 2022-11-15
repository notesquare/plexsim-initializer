# PLEXsim Initializer

Plasma Extensible Simualtion (PLEXsim) Initializer 사용 설명입니다.


## Install
1. git clone
```sh
$ git clone https://github.com/notesquare/plexsim-initializer.git
```

2. 의존 라이브러리 설치

```sh
$ pip install -r requirements.txt

# or

$ conda env update -n plexsim --file environment.yaml
```


## Usage

전체 실행 옵션 출력
```sh
$ python -m plexsim_initializer --help
```


`examples/random` 예시
```sh
$ python -m plexsim_initializer examples/random/config.yaml random_test.h5
```


`examples/maxwellian` 예시
```sh
$ python -m plexsim_initializer examples/maxwellian/config.yaml maxwellian_test.h5
```


## Config

### 예시
e.g. `config.yaml`
```yaml
initializer: maxwellian

environment:
  coordinate_system: cartesian
  grid_shape: [138, 138, 64]
  cell_size: [0.00925677, 0.00925677, 0.01144446]
  external_magnetic_field: ../common/B.npy
  external_electric_field: [0, 0, 0]
  field_dtype: fp64
  valid_cell_coords: ../common/valid_cell_coords.npy
  constant_field_coords: ../common/constant_field_coords.npy

grids:
  -
    species: electron
    dtype:
      X: fp32
      U: fp32
    n_splits: 2
    n_computational_to_physical: 1.e+10
    initial_condition:
      temperature: ./data/Te.npy  # [eV]
      density: ./data/ne.npy  # [m^-3]
      current_density: ./data/J.npy

      tracking:
        n_particles: 20
        sample: random

  -
    species: ion
    dtype:
      X: fp32
      U: fp32
    n_splits: 2
    n_computational_to_physical: 1.e+10
    initial_condition:
      temperature: ./data/Ti.npy  # [eV]
      density: ./data/ni.npy  # [m^-3]

      tracking:
        n_particles: 20
        sample: random

simulation:
  chunk_size: 1.e+6
  iteration_encoding: file

```

### 설명

#### Initializer
_{​​​​​​​'​​​​​​maxwellian', 'random'}​​​​​​_   
- _'maxwellian'_ : `grids.initial_condition`에 `temperature` 및 `density` 입력 필요
- _'random'_: `grids.initial_condition`에 `n_particles` 및 `avg_velocity` 입력 필요

#### Environment
```yaml
environment:
  coordinate_system: cartesian
  grid_shape: [138, 138, 64]
  cell_size: [0.00925677, 0.00925677, 0.01144446]
  external_magnetic_field: ../common/B.npy
  external_electric_field: [0, 0, 0]
  field_dtype: fp64
  valid_cell_coords: ../common/valid_cell_coords.npy
  constant_field_coords: ../common/constant_field_coords.npy
```
- **coordinate_system**: _'cartesian'_
  - 좌표계를 설정합니다. _'cylindrical'_ 업데이트 예정
- **grid_shape**: _list_
  - 전체 격자 `grid`의 크기를 지정합니다. 단위 격자 `cell`의 x, y, z축 방향 개수를 리스트로 받습니다.
- **cell_size**: _list_
  - 단위 격자의 x, y, z축 방향 길이[m]를 지정합니다.
- **external_magnetic_field**: _list or file_
  - 초기 외부 자기장을 설정합니다.
    - _list_: x, y, z 방향의 자기장 [T]을 지정합니다. 모든 격자점(node)에 같은 크기로 할당됩니다.
    - _file_: 각 격자점의 자기장 정보를 담고 있는 `.npy` 파일의 경로를 지정합니다.
      - e.g. `grid_shape == [138, 138, 64]` 인 경우 `external_magnetic_field의 shape == [139, 139, 65, 3]`이어야 합니다.
- **external_electric_field**: _list or file_
  - 초기 외부 전기장 [V/m]을 설정합니다. 입력 형식은 `external_magnetic_field`와 같습니다.
- **field_dtype**: _{'fp32', 'fp64'}_
  - 전기장과 자기장 배열의 부동 소수점 형식을 지정합니다. 32비트 단일 정밀도 형식과 64비트 이중 정밀도 형식 중 선택할 수 있습니다.
- **valid_cell_coords**: _file, optional_
  - 시뮬레이션 연산을 수행할 cell 좌표 정보를 담고 있는 `.npy`파일 경로. 설정한 좌표를 벗어나 입자가 이동하면 사라집니다.
  - e.g. `[[0, 0, 0], [0, 0, 1], ..., [137, 137, 63]]`
  - 입력하지 않을 경우 `grid_shape` 크기의 직육면체가 시뮬레이션 공간이 됩니다.
- **constant_field_coords**: _file, optional_
  - 초기 입력값으로 자기장, 전기장 값을 고정할 격자점 좌표를 담고 있는 `.npy`파일 경로. 장이 경계면에 반사되는 것을 막기 위해 사용.
  - e.g `[[0, 0, 0], [0, 0, 1], ..., [138, 138, 64]]`


#### Grids

```yaml
grids:
  -
    species: electron
    dtype:
      X: fp32
      U: fp32
    n_splits: 2
    n_computational_to_physical: 1.e+10
    initial_condition:
      temperature: ./data/Te.npy  # [eV]
      density: ./data/ne.npy  # [m^-3]
      current_density: ./data/J.npy
      #   n_particles: 1.e+3
      #   avg_velocity: 3.e+6  # [m/s]

      tracking:
        n_particles: 20
        sample: random
  -
    species: ion
    dtype:
      X: fp32
      U: fp32
    n_splits: 2
    n_computational_to_physical: 1.e+10
    initial_condition:
      temperature: ./data/Ti.npy  # [eV]
      density: ./data/ni.npy  # [m^-3]
      tracking:
        n_particles: 20
        sample: random
```
각 입자에 대한 설정을 목록(`-`)으로 나누어 지정합니다. 목록의 요소를 추가하거나 제거하여 몇 종류의 입자를 포함할지 선택할 수 있습니다.
- **species**: _{'electron', 'ion'}_
- **dtype**: _{'fp32', 'fp64'}_
  - 입자 위치(X) 및 속도(U) 배열의 부동 소수점 형식을 지정합니다.
- **n_splits**: _**int**, optional._
  - 각 `grid`를 몇 개의 부분공간으로 나눌지 지정합니다. 각 부분공간은 공간을 채우는 hilbert curve를 따라 cell을 기준으로 입자를 균등하게 나누며, `plexsim-solver` 에서 GPU를 사용하는 경우 GPU 수보다 각 입자의 `n_splits`의 합이 작거나 같으면 GPU를 하나씩 할당합니다.
  - e.g. GPU가 4개 있고 입자가 두 종류이면서 그 수가 비슷하다면, 각각의 `n_splits: 2`로 설정하면 GPU를 최대로 사용할 수 있습니다.
- **n_computational_to_physical**: _int, optional_
- **initial_condition**
  - **temperature**: _file, required for maxwellian initializer_
    - 각 격자점에 해당하는 온도[eV] 정보를 담고 있는 `.npy` 파일 경로. [`initializer`](#initializer) 가 `maxwellian`일 경우 필요합니다.
    - e.g. `grid_shape == [138, 138, 64]`인 경우 `temperature shape == [139, 139, 65]`이어야 합니다.
  - **density**: _file, required for maxwellian initializer_
    - 각 격자점에 해당하는 입자 밀도[m^-3^] 정보를 담고 있는 `.npy` 파일 경로. [`initializer`](#initializer) 가 `maxwellian`일 경우 필요합니다.
    - e.g. `grid_shape == [138, 138, 64]`인 경우 `density shape == [139, 139, 65]`이어야 합니다.
  - **current_density**: _file, optional for maxwellian initializer_
    - 각 격자점에 해당하는 전류 밀도[Am^-2^] 정보를 담고 있는 `.npy` 파일 경로. [`initializer`](#initializer) 가 `maxwellian`일 경우 입력할 수 있습니다. 입력하지 않으면, 0으로 초기화됩니다.
    - e.g. `grid_shape == [138, 138, 64]`인 경우 `current_density shape == [139, 139, 65, 3]`이어야 합니다.

  - **n_particles**: _int, required for random initializer_
    - 초기 입자의 수를 지정합니다. [`initializer`](#initializer) 가 `random`일 경우 필요합니다.
  - **avg_velocity**: _float, required for random initializer_
    - 입자의 초기 평균 속력을 지정합니다. [`initializer`](#initializer) 가 `random`일 경우 필요합니다.
  - **tracking**: _optional_
    - **n_particles**: _int_
      - 궤적을 추적하기 위한 입자의 수를 설정합니다.
    - **sample**: _'random'_


#### Simulation

```yaml
simulation:
  chunk_size: 1.e+6
  iteration_encoding: file
```
- **chunk_size**: _int_
  - `.h5` 파일 dataset의 `chunk size`를 지정합니다.
- **iteration_encoding**: _{'file', 'group'}_
  - `iteration_encoding` 타입을 지정합니다. (https://github.com/openPMD/openPMD-standard/blob/latest/STANDARD.md#iterations-and-time-series 참고)


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
