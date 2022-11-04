import argparse
from pathlib import Path

import yaml

from .initializers import (
    MaxwellianInitializer,
    RandomInitializer
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='plexsim_initializer')

    parser.add_argument('init_config_file', type=Path)
    parser.add_argument('out_file', type=Path)
    args = parser.parse_args()

    input_fp = args.init_config_file
    with open(input_fp, 'r') as f:
        config = yaml.safe_load(f)
        initializer = config.get('initializer', 'random')

    if initializer == 'maxwellian':
        init = MaxwellianInitializer.load(input_fp)
    elif initializer == 'random':
        init = RandomInitializer.load(input_fp)
    else:
        raise NotImplementedError(initializer)

    init.serialize(args.out_file)
