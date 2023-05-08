import sys

import yaml
from tqdm.auto import tqdm

from .maxwellian import MaxwellianInitializer
from .random import RandomInitializer


def initialize(input_fp, out_fp, author='unspecified'):
    with open(input_fp, 'r') as f:
        config = yaml.safe_load(f)
        initializer = config.get('initializer', 'random')

    t = tqdm(total=7, bar_format='{desc}{bar}|{n_fmt}/{total_fmt}')

    if initializer == 'maxwellian':
        init = MaxwellianInitializer.load(input_fp)
    elif initializer == 'random':
        init = RandomInitializer.load(input_fp)
    else:
        print('Unkown Initializer: {initializer}', file=sys.stderr)
        sys.exit(1)

    t.update()

    init.serialize(out_fp, author=author, t=t)
