import os
import sys


def get_realpath(path):
    return os.path.abspath(os.path.expanduser(path))


def warning(msg):
    print(msg, file=sys.stderr)
