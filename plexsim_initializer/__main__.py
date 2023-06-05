import os
import sys
import argparse
from pathlib import Path
import warnings

from numba.core.errors import NumbaPerformanceWarning

from .backends import initialize


# disable numba warning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='plexsim_initializer')

    parser.add_argument('init_config_file', type=Path)
    parser.add_argument('out_file', type=Path)
    parser.add_argument('--author', type=str, default='unspecified')
    args = parser.parse_args()

    input_fp = args.init_config_file
    if not input_fp.is_file():
        print(f'File Not Found: {input_fp}', file=sys.stderr)
        sys.exit(1)

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

    initialize(input_fp, out_fp, author=args.author)
