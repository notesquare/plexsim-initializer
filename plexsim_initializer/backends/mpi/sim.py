import sys

import yaml
from tqdm.auto import tqdm

from .maxwellian import MaxwellianInitializer
from .random import RandomInitializer
from ...initializers import (
    CartesianInitializer,
    CylindricalInitializer
)


def initialize(input_fp, out_fp, author='unspecified'):
    with open(input_fp, 'r') as f:
        config = yaml.safe_load(f)
    initializer = config.get('initializer', 'random')
    coordinate_system = config['environment'].get(
        'coordinate_system', 'cartesian')

    t = tqdm(total=7, bar_format='{desc}{bar}|{n_fmt}/{total_fmt}')

    if initializer == 'maxwellian':
        if coordinate_system == 'cartesian':
            class Initializer(MaxwellianInitializer, CartesianInitializer):
                pass
        elif coordinate_system == 'cylindrical':
            class Initializer(MaxwellianInitializer, CylindricalInitializer):
                pass
        else:
            print(f'Unkown coordinate system: {coordinate_system}',
                  file=sys.stderr)
            sys.exit(1)
    elif initializer == 'random':
        if coordinate_system == 'cartesian':
            class Initializer(RandomInitializer, CartesianInitializer):
                pass
        elif coordinate_system == 'cylindrical':
            class Initializer(RandomInitializer, CylindricalInitializer):
                pass
        else:
            print(f'Unkown coordinate system: {coordinate_system}',
                  file=sys.stderr)
            sys.exit(1)
    else:
        print(f'Unkown Initializer: {initializer}', file=sys.stderr)
        sys.exit(1)

    init = Initializer.load(input_fp)
    t.update()

    init.serialize(out_fp, author=author, t=t)
