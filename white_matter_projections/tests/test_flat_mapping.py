import shutil
import os
import numpy as np
from white_matter_projections import flat_mapping
from voxcell.hierarchy import Hierarchy
from voxcell.voxel_data import VoxelData
from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_allclose


class TestFlatMap(object):
    def __init__(self):
        raw = np.zeros((5, 5, 5), dtype=np.int)
        brain_regions = VoxelData(raw, np.zeros(3), offset=np.zeros(3))
        raw[1, :, 0:2] = 2
        raw[2, :4, 2:3] = 30

        hierarchy = Hierarchy(
            {'id': 1, 'acronym': 'one',
             'children': [
                 {'id': 2, 'acronym': 'two', 'children': []},
                 {'id': 20, 'acronym': 'twenty', 'children': [
                     {'id': 30, 'acronym': 'thirty', 'children': []}
                 ]}
             ]},
        )

        view_lookup = -1 * np.ones((5, 5), dtype=int)
        view_lookup[1, 0] = 0
        view_lookup[1, 1] = 1
        view_lookup[2, 2] = 2
        paths = np.array([[25, 26, 30, 31, 35, ],
                          [36, 40, 41, 45, 46, ],
                          [52, 57, 62, 67, 0, ],
                          ])
        self.flat_map = flat_mapping.FlatMap(brain_regions, hierarchy, view_lookup, paths)

    # keep load on external servers low
    #def test_load(self):
    #    self.config = test_utils.get_config()
    #    config = test_utils.get_config()
    #    path = os.path.join(config.config['cache_dir'],
    #                        flat_mapping.FlatMap.CORTICAL_MAP_PATH)
    #    if os.path.exists(path):
    #        shutil.rmtree(path)
    #    fm = self.config.flat_map
    #    eq_(fm.brain_regions.shape, (5, 5, 5))

    def test_make_flat_id_region_map(self):
        ret = self.flat_map.make_flat_id_region_map(['two', 'thirty'])
        expected = np.zeros((5, 5,), dtype=int)
        expected[1, 0] = expected[1, 1] = 2
        expected[2, 2] = 30
        assert_allclose(expected, ret)

        ret = self.flat_map.make_flat_id_region_map(['two', ])
        expected = np.zeros((5, 5,), dtype=int)
        expected[1, 0] = expected[1, 1] = 2
        expected[2, 2] = -1
        assert_allclose(expected, ret)

    def test_get_voxel_indices_from_flat(self):
        idx = self.flat_map.get_voxel_indices_from_flat((1, 1))
        np.testing.assert_allclose(self.flat_map.brain_regions.raw[idx],
                                   [2] * 5)

        idx = self.flat_map.get_voxel_indices_from_flat((2, 2))
        np.testing.assert_allclose(self.flat_map.brain_regions.raw[idx],
                                   [30] * 4)
