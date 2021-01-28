import numpy as np
from white_matter_projections import flat_mapping
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


def test_mask_in_2d_flatmap():
    ret = flat_mapping.FlatMap.mask_in_2d_flatmap(np.array([[0, 0],
                                                            [1, 1],
                                                            [0, -1],
                                                            [-1, 0],
                                                            [-1, -1]]))
    assert_allclose(list(ret), [True, True, False, False, False])

    ret = flat_mapping.FlatMap.mask_in_2d_flatmap(np.array([[0.1, 0.9],
                                                            [0.9, 0.1],
                                                            [1.9, 1.1],
                                                            [1.1, 1.9],
                                                            [0.1, -1.9],
                                                            [0.9, -1.1],
                                                            [-1.1, 0.9],
                                                            [-1.9, 0.1],
                                                            [-1.1, -1.9],
                                                            [-1.9, -1.1],
                                                            [-0.5, -0.5]
                                                            ]))
    assert_allclose(list(ret), [True, True, True, True, False, False, False, False, False, False, False])
