from white_matter_projections import cortical_mapping
from nose.tools import eq_, ok_
import numpy as np
from numpy.testing import assert_allclose
import utils


CORTICAL_MAP = utils.CorticalMap.fake_cortical_map()


def test__fit_path():
    path = np.array([[-1, 0.2, 0.9, 2.1],
                     [-1, 0.2, 0.9, 2.1],
                     [-1, 0.2, 0.9, 2.1], ]).T
    fit = cortical_mapping._fit_path(path)
    assert_allclose(fit, [-0.95, -0.95, -0.95, 1., 1., 1.])


def test__fit_paths():
    fit_paths = cortical_mapping._fit_paths(CORTICAL_MAP.load_brain_regions(),
                                            CORTICAL_MAP.load_cortical_view_lookup(),
                                            CORTICAL_MAP.load_cortical_paths())

    eq_(sorted(fit_paths.columns),
        sorted(['ox', 'oy', 'oz', 'fx', 'fy', 'fz', 'x', 'y']))

    xy = [tuple(v) for v in fit_paths[['x', 'y']].values]
    for coord in ((1, 0), (1, 1), (2, 2)):
        ok_(coord in xy)


def test__paths_intersection_distances():
    fit_paths = cortical_mapping._fit_paths(CORTICAL_MAP.load_brain_regions(),
                                            CORTICAL_MAP.load_cortical_view_lookup(),
                                            CORTICAL_MAP.load_cortical_paths())

    bounds = [0, 0, 0]
    distances = cortical_mapping._paths_intersection_distances(bounds, fit_paths)
    assert_allclose(distances, [0., 0., 0.])  # none of the paths hit the origin

    for i in range(5):
        bounds = [2, i, 2]
        distances = cortical_mapping._paths_intersection_distances(bounds, fit_paths)
        assert_allclose(distances, [0., 0., 1.])  # directly passes through third path


def test_create_cortical_to_flatmap():
    regions = ['two', 'twenty', 'thirty', ]
    v2f = cortical_mapping.create_cortical_to_flatmap(CORTICAL_MAP, regions, n_jobs=1, backfill=False)

    eq_(v2f.shape, (5, 5, 5))
    eq_(v2f.raw.shape, (5, 5, 5, 2))
    nz_idx = np.nonzero((v2f.raw[:, :, :, 0] > 0) | (v2f.raw[:, :, :, 1] > 0))
    eq_(len(nz_idx[0]), 14)
    assert_allclose(v2f.raw[nz_idx][-4:, :], [[2, 2]] * 4)

    # test backfill
    wanted_ids = {'two': [2]}
    v2f.raw[1, 0, 0, :] = 0  # this should be filled in with the original value
    cortical_mapping._backfill_voxel_to_flat_mapping(v2f,
                                                     CORTICAL_MAP.load_brain_regions(),
                                                     CORTICAL_MAP.center_line_2d,
                                                     CORTICAL_MAP.center_line_3d,
                                                     wanted_ids)
    eq_(v2f.raw[1, 0, 0, 0], 1)
    eq_(v2f.raw[1, 0, 0, 1], 1)

    #check we handle edge values
    cortical_mapping._backfill_voxel_to_flat_mapping(v2f,
                                                     CORTICAL_MAP.load_brain_regions(),
                                                     CORTICAL_MAP.center_line_2d,
                                                     CORTICAL_MAP.center_line_3d,
                                                     wanted_ids)
    eq_(v2f.raw[1, 0, 0, 0], 1)
    eq_(v2f.raw[1, 0, 0, 1], 1)


def test__voxel2flat_helper():
    path_fits = cortical_mapping._fit_paths(CORTICAL_MAP.load_brain_regions(),
                                            CORTICAL_MAP.load_cortical_view_lookup(),
                                            CORTICAL_MAP.load_cortical_paths())
    regions = ['two', 'twenty', 'thirty', ]
    locs = np.array([[2, 1, 2], ])
    v2f = cortical_mapping._voxel2flat_worker(CORTICAL_MAP, regions, path_fits, locs)
    assert_allclose(v2f, [[2, 2, ], ])
