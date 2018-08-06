# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np


def assert_consistent_FEC(retract):
    """
    :param retract: see  _assert_consistent_zeroing
    :return: nothing, throws error if the FEC has been corrupted.
    """
    _assert_consistent_force(retract)

def assert_consistent_split_FEC(split):
    """
    :param split: with approach and retract attributes
    :return: nothing, but see assert_consistent_FEC
    """
    _assert_consistent_force(split.retract)
    _assert_consistent_force(split.approach)

def _assert_consistent_force(retract):
    """
    :param retract: FEC to check. should have SpringConstant, Separation,
    and Zsnsr
    :return: nothing, throws error if force != k * (q - z)
    """
    k = retract.SpringConstant
    q = retract.Separation
    z = retract.ZSnsr
    should_be_force = k*(q-z)
    should_be_near_zero = should_be_force - retract.Force
    # make sure the force is zero within a pN/1e6 or so
    np.testing.assert_allclose(should_be_near_zero,0,atol=1e-18,rtol=1e-6)