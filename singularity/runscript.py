#!/usr/bin/env python3

import subprocess

"""This script is started automatically by the Singularity container.
"""


def main(argv):
    if len(argv) <= 1:
        arguments = ["ipython"]
    else:
        arguments = argv[1:]

    subprocess.run(arguments)


if __name__ == "__main__":
    import sys

    main(sys.argv)
