# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys, argparse, enum

from enum import Enum,unique
from .UtilGeneral import GenUtilities

@unique
class Step(Enum):
    READ = 0
    FILTERED = 1
    CORRECTED = 2
    MANUAL = 3
    SANITIZED = 4
    ALIGNED = 5
    REDUCED = 6
    POLISH = 7

def default_parser(default_dir):
    parser = argparse.ArgumentParser(description="Part of pipeline")
    parser.add_argument('--base', type=str, metavar='base',
                        help="Where data base directory lives",
                        default=default_dir)
    return parser

def _base_dir_from_cmd(default):
    parser = default_parser(default)
    args = parser.parse_args()
    return args.base

def _cache_dir(base, enum):
    return "{}cache_{}_{}/".format(base, enum.value, enum.name.lower())


def _plot_subdir(base, enum, extra_str=""):
    to_ret = _cache_dir(base, enum) + "_cache_plot_{}/".format(extra_str)
    GenUtilities.ensureDirExists(to_ret)
    return to_ret
