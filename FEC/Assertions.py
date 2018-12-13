# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np

def assert_list_consistent(list_v,**kw):
    [assert_consistent_FEC(e,**kw) for e in list_v ]

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

def _assert_consistent_force(retract,rtol=1e-6,atol=1e-20,print_info=False):
    """
    :param retract: FEC to check. should have SpringConstant, Separation,
    and Zsnsr
    :param rtol: relative tolerance to use
    :param atol: absolute tolerance to use
    :param print_info: if true, prints debugging information
    :return: tuple of min and maximum error
    """
    k = retract.SpringConstant
    q = retract.Separation
    z = retract.ZSnsr
    # note that we don't really care about the overall sign
    should_be_abs_force = np.abs(k*(q-z))
    abs_force = np.abs(retract.Force)
    should_be_near_zero = np.abs(should_be_abs_force - abs_force)
    min_e = min(should_be_near_zero)
    max_e = max(should_be_near_zero)
    if print_info:
        print("Assertions.py::_assert_consistent_force. " + \
              "Min/Max force error (N) is: {:.3g}/{:.3g}".format(min_e,max_e))
    # make sure the force is zero within a small fraction of a pN..
    np.testing.assert_allclose(should_be_near_zero,0,atol=atol,rtol=rtol)
    return min_e, max_e