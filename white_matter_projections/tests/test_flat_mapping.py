import os
import numpy as np
from white_matter_projections import flat_mapping
from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_allclose
import utils


def test_make_flat_id_region_map():
    flat_map = utils.fake_flat_map()
    ret = flat_map.make_flat_id_region_map(['two', 'thirty'])
    expected = np.full(flat_map.shape, -1, dtype=int)
    expected[1, 0] = expected[1, 1] = 2
    expected[2, 2] = 30
    assert_allclose(expected, ret)

    ret = flat_map.make_flat_id_region_map(['two', ])
    expected = np.full(flat_map.shape, -1, dtype=int)
    expected[1, 0] = expected[1, 1] = 2
    expected[2, 2] = -1
    assert_allclose(expected, ret)
