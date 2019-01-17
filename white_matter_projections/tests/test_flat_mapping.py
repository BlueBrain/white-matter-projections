import shutil
import os
import numpy as np
from white_matter_projections import flat_mapping
from voxcell.hierarchy import Hierarchy
from voxcell.voxel_data import VoxelData
from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_allclose
from mock import Mock, patch
import utils


class TestFlatMap(object):
    def __init__(self):
        self.flat_map = utils.fake_flat_map()

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
        assert_allclose(self.flat_map.brain_regions.raw[idx],
                        [2] * 5)

        idx = self.flat_map.get_voxel_indices_from_flat((2, 2))
        assert_allclose(self.flat_map.brain_regions.raw[idx],
                        [30] * 4)

def test__fit_path():
    path = np.array([[-1, 0.2, 0.9, 2.1],
                     [-1, 0.2, 0.9, 2.1],
                     [-1, 0.2, 0.9, 2.1], ]).T
    fit = flat_mapping._fit_path(path)
    assert_allclose(fit, [-0.95, -0.95, -0.95, 1., 1., 1.])


def test__fit_paths():
    flat_map = utils.fake_flat_map()
    fit_paths = flat_mapping._fit_paths(flat_map)

    eq_(sorted(fit_paths.columns),
        sorted(['ox', 'oy', 'oz', 'fx', 'fy', 'fz', 'x', 'y']))

    xy = [tuple(v) for v in fit_paths[['x', 'y']].values]
    for coord in ((1, 0), (1, 1), (2, 2)):
        ok_(coord in xy)


def test__paths_intersection_distances():
    flat_map = utils.fake_flat_map()
    fit_paths = flat_mapping._fit_paths(flat_map)

    bounds = [0, 0, 0]
    distances = flat_mapping._paths_intersection_distances(bounds, fit_paths)
    assert_allclose(distances, [0., 0., 0.])  # none of the paths hit the origin

    for i in range(5):
        bounds = [2, i, 2]
        distances = flat_mapping._paths_intersection_distances(bounds, fit_paths)
        assert_allclose(distances, [0., 0., 1.])  # directly passes through third path


def test__voxel2flat_helper():
    flat_map = utils.fake_flat_map()
    path_fits = flat_mapping._fit_paths(flat_map)
    with patch('white_matter_projections.flat_mapping.Config') as Config:
        Config.return_value = config = Mock()
        config.flat_map = flat_map
        config.regions = ['two', 'twenty', 'thirty', ]
        locs = np.array([[2, 1, 2], ])
        v2f = flat_mapping._voxel2flat_helper('config_path', path_fits, locs)
        assert_allclose(v2f, [[2, 2, ], ])

def test__voxel2flat():
    flat_map = utils.fake_flat_map()
    regions = ['two', 'twenty', 'thirty', ]
    path_fits = flat_mapping._fit_paths(flat_map)
    locs = np.array([[2, 1, 2], ])
    v2f = flat_mapping._voxel2flat(flat_map, regions, path_fits, locs)
    assert_allclose(v2f, [[2, 2, ], ])


def test_get_voxel_to_flat_mapping():
    flat_map = utils.fake_flat_map()

    with patch('white_matter_projections.flat_mapping.Config') as Config:
        Config.return_value = config = Mock()
        config.flat_map = flat_map
        config.regions = ['two', 'twenty', 'thirty', ]
        v2f = flat_mapping.get_voxel_to_flat_mapping(config, n_jobs=1)

        eq_(v2f.shape, (5, 5, 5))
        eq_(np.count_nonzero(v2f.raw), 14)
        mapping = np.array(np.unravel_index(v2f.raw[np.nonzero(v2f.raw)],
                                            flat_map.view_lookup.shape)
                           ).T
        assert_allclose(mapping[-4:, :], [[2, 2]] * 4)
