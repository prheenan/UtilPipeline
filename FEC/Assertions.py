# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np


def assert_consistent_FEC(retract,**kw):
    """
    :param retract: see  _assert_consistent_zeroing
    :return: nothing, throws error if the FEC has been corrupted.
    """
    _assert_consistent_force(retract,**kw)

def assert_consistent_split_FEC(split,**kw):
    """
    :param split: with approach and retract attributes
    :return: nothing, but see assert_consistent_FEC
    """
    _assert_consistent_force(split.retract,**kw)
    _assert_consistent_force(split.approach,**kw)

def _assert_consistent_force(retract,rtol=1e-3,atol=1e-12/200):
    """
    :param retract: FEC to check. should have SpringConstant, Separation,
    and Zsnsr
    :return: nothing, throws error if force != k * (q - z)
    """
    k = retract.SpringConstant
    q = retract.Separation
    z = retract.ZSnsr
    # note that we don't really care about the overall sign
    should_be_abs_force = np.abs(k*(q-z))
    abs_force = np.abs(retract.Force)
    should_be_near_zero = should_be_abs_force - abs_force
    # make sure the force is zero within a small fraction of a pN..
    np.testing.assert_allclose(should_be_near_zero,0,atol=atol,rtol=rtol)