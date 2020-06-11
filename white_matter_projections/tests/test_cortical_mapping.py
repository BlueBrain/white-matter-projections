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
    v2f.raw[1, 0, 0, :] = -1  # this should be filled in with the original value
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


def test__find_histogram_idx():
    counts = np.array([10, 0, 0, 0])
    idxs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])
    needed_count = 5
    eq_(cortical_mapping._find_histogram_idx(needed_count, counts, idxs), [0, 1, 2, 3, 4])

    counts = np.array([10, 10, 0, 0, 0])
    idxs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ])
    needed_count = 5
    eq_(cortical_mapping._find_histogram_idx(needed_count, counts, idxs), [10, 0, 11, 1, 12])

    counts = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, ])
    idxs = np.array([0,
                     1, 1,
                     2, 2, 2,
                     3, 3, 3, 3,
                     4, 4, 4, 4, 4,  # i = 10
                     5, 5, 5, 5,
                     6, 6, 6,
                     7, 7,
                     8, ])
    needed_count = 8
    eq_(cortical_mapping._find_histogram_idx(needed_count, counts, idxs),
        [10, 11, 15, 6, 12, 16, 7, 19])


def test__clamp_known_values():
    regions = ['two', 'twenty', 'thirty', ]
    flat_map = cortical_mapping.create_cortical_to_flatmap(CORTICAL_MAP, regions, n_jobs=1, backfill=False)

    # replace half the values point at (1,1) w/ (1, 0), and the other w/ (2, 2)
    idx = np.nonzero(((flat_map.raw[:, :, :, 0] == 1) & (flat_map.raw[:, :, :, 1] == 1)))
    flat_map.raw[idx] = 1, 0
    flat_map.raw[tuple(np.array(idx).T[::2].T)] = 2, 2

    view_lookup = CORTICAL_MAP.load_cortical_view_lookup()
    cortical_paths = CORTICAL_MAP.load_cortical_paths()

    cortical_mapping._clamp_known_values(view_lookup, cortical_paths, flat_map)
    # we only replace values along the original cortical path; in this case it only
    # intersects the linear fit paths in one place
    eq_(np.count_nonzero(((flat_map.raw[:, :, :, 0] == 1) & (flat_map.raw[:, :, :, 1] == 1))), 2)
