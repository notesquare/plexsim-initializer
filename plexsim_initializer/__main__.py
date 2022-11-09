import os
import sys
import argparse
from pathlib import Path
import warnings

import yaml
from numba.core.errors import NumbaPerformanceWarning
from tqdm.auto import tqdm

from .initializers import (
    MaxwellianInitializer,
    RandomInitializer
)


# disable numba warning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='plexsim_initializer')

    parser.add_argument('init_config_file', type=Path)
    parser.add_argument('out_file', type=Path)
    args = parser.parse_args()

    input_fp = args.init_config_file
    if not input_fp.is_file():
        print(f'File Not Found: {input_fp}', file=sys.stderr)
        sys.exit(1)

    with open(input_fp, 'r') as f:
        config = yaml.safe_load(f)
        initializer = config.get('initializer', 'random')

    # make sure out_file's parent directory is accessible
    out_fp = args.out_file
    if not os.access(out_fp.parent, os.W_OK):
        print(f'Permission Error: Cannot write a file to {out_fp.parent}',
              file=sys.stderr)
        sys.exit(1)

    # make sure out_file's extension is h5 (lowercase).
    if not out_fp.name.lower().endswith('.h5'):
        out_fp = out_fp.parent / (out_fp.name + '.h5')
    out_fp = out_fp.with_suffix('.h5')

    t = tqdm(total=7, bar_format='{desc}{bar}|{n_fmt}/{total_fmt}')

    if initializer == 'maxwellian':
        init = MaxwellianInitializer.load(input_fp)
    elif initializer == 'random':
        init = RandomInitializer.load(input_fp)
    else:
        print('Unkown Initializer: {initializer}', file=sys.stderr)
        sys.exit(1)

    t.update()
    init.serialize(out_fp, t=t)
