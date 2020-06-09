import math
import os
import numpy as np
import pandas as pd
from bluepy.v2.index import SegmentIndex

from nose.tools import ok_, eq_, assert_raises
from white_matter_projections import sampling, utils as white_matter_utils
from numpy.testing import assert_allclose, assert_equal
from pandas.testing import assert_frame_equal
import utils as test_utils
from mock import patch, Mock

def _full_sample_worker_mock(min_xyzs, index_path, voxel_dimensions):
    df = pd.DataFrame(
        np.array([[101, 201, 10., 0., 10., 0., 10., 0., 10., 31337], ]),
        columns=sampling.SEGMENT_COLUMNS)

    df = pd.concat([df for _ in min_xyzs], ignore_index=True, sort=False)
    return df


def test__ensure_only_flatmap_segments():
    config = Mock()
    config.flat_map = test_utils.fake_flat_map()
    columns = ['segment_x1', 'segment_y1', 'segment_z1',
               'segment_x2', 'segment_y2', 'segment_z2', ]
    segments = pd.DataFrame([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, ],  # maps to 0, 0, 0 -> -1, -1
                             [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, ],  # maps to 1, 1, 1 -> 1, 1
                             [2.0, 0.0, 1.0, 3.0, 1.0, 2.0, ],  # maps to 2, 0, 1 -> -1, -1
                             [2.0, 1.0, 2.0, 3.0, 2.0, 3.0, ],  # maps to 2, 1, 2 -> 2, 2

                             [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, ],  # maps to 0, 0, 0 -> -1, -1
                             [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, ],  # maps to 1, 1, 1 -> 1, 1
                             [2.0, 0.0, 1.0, 3.0, 1.0, 2.0, ],  # maps to 2, 0, 1 -> -1, -1
                             [2.0, 1.0, 2.0, 3.0, 2.0, 3.0, ],  # maps to 2, 1, 2 -> 2, 2
                             ],
                            columns=columns)
    ret = sampling._ensure_only_flatmap_segments(config, segments)
    eq_(len(ret), 4)


def test__ensure_only_segments_from_region():
    config = Mock()
    config.flat_map.center_line_3d = 10.
    config.get_cells.return_value = pd.DataFrame(
        {'region': ['region0', 'region1', 'region0', 'region1'],
         'z': [1, 1, 20, 20],
        },
        index=[10, 20, 30, 40])
    region, side = 'region0', 'right'
    df = pd.DataFrame({'tgid': [10, 20, 30, 40],})

    ret = sampling._ensure_only_segments_from_region(config, region, side, df)
    assert_allclose(ret.tgid.to_numpy(), [30])


@patch('white_matter_projections.sampling.synapses')
def test__full_sample_worker(patch_synapses):
    min_xyzs = np.array([(.5, .5, .5),  # single voxel
                        ])
    index_path = 'fake_index_path'
    dims = (10., 10., 10.)

    segs_df = np.array([(0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 1., 2., 1, 10, 100, 2),  # axon
                        (0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 1., 2., 1, 10, 100, 3),  # basal
                        ])
    segs_df = SegmentIndex._wrap_result(segs_df)
    patch_synapses._sample_with_flat_index.return_value = segs_df
    df = sampling._full_sample_worker(min_xyzs, index_path, dims)
    eq_(len(df), 1)  # axon is skipped
    ok_(isinstance(df, pd.core.frame.DataFrame))
    assert_allclose(df.segment_length.values, np.linalg.norm((0.75 - .5, 0.75 - .5, 0.75 - .5)))
    eq_(sorted(df.columns), sampling.SEGMENT_COLUMNS)

    #only axon
    segs_df = np.array([(0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 1., 2., 1, 10, 100, 2),  # axon
                        ])
    segs_df = SegmentIndex._wrap_result(segs_df)
    patch_synapses._sample_with_flat_index.return_value = segs_df
    df = sampling._full_sample_worker(min_xyzs, index_path, dims)
    eq_(len(df), 0)  # axon is skipped
    eq_(sorted(df.columns), sampling.SEGMENT_COLUMNS)

    #empty return from _sample_with_flat_index
    patch_synapses._sample_with_flat_index.return_value = pd.DataFrame(columns=segs_df.columns)
    df = sampling._full_sample_worker(min_xyzs, index_path, dims)
    eq_(len(df), 0)  # axon is skipped
    eq_(sorted(df.columns), sampling.SEGMENT_COLUMNS)


def test__dilate_region():
    brain_regions, _ = test_utils.fake_brain_regions()
    ret = sampling._dilate_region(brain_regions, [30, ], 1)
    ok_(30 in ret)
    # original indices exist
    eq_(set(zip(*np.nonzero(brain_regions.raw == 30))) - set(zip(*tuple(ret[30].T))), set())
    eq_(len(ret[30]),
        4 +  # original column
        4 +  4 + 4 + 4 + # columns north/south/east/west
        1) # top

    #dilate the whole region
    ret = sampling._dilate_region(brain_regions, [30, ], 10)
    ok_(30 in ret)
    eq_(len(ret[30]), 5 * 5 * 5)

    brain_regions, _ = test_utils.fake_brain_regions()
    brain_regions.raw[0, 0, 0] = 1
    brain_regions.raw[2, 2, 2] = 3
    brain_regions.raw[0, 1, 0] = 4
    ids = [1, 3, 4, 30, ]
    ret = sampling._dilate_region(brain_regions, ids, 1)
    for id_ in ids:
        ok_(id_ in ret)

    # mutual intersection should be zero
    eq_(len(set.intersection(*[set(zip(*tuple(s))) for s in ret.values()])), 0)


def test_sample_all():
    population = pd.DataFrame([(1, 'FRP', 'l1'),
                               (2, 'FRP', 'l2'),
                               (30, 'FRP', 'l3'),
                               (30, 'FRP', 'l3')],
        columns=['id', 'region', 'subregion'])
    brain_regions, _ = test_utils.fake_brain_regions()

    df = pd.DataFrame(
        np.array([[101, 201, 10., 0., 10., 0., 10., 0., 10., 31337], ]),
        columns=sampling.SEGMENT_COLUMNS)

    side = 'right'
    with test_utils.tempdir('test_sample_all') as tmp:
        index_base = os.path.join(tmp, 'fake_index_base')
        os.makedirs(os.path.join(index_base, 'FRP@left'))
        os.makedirs(os.path.join(index_base, 'FRP@right'))
        with open(os.path.join(index_base, 'FRP@left', 'SEGMENT_index.dat'), 'w') as fd:
            fd.write('FRP@left')
        with open(os.path.join(index_base, 'FRP@right', 'SEGMENT_index.dat'), 'w') as fd:
            fd.write('FRP@right')

        with patch('white_matter_projections.sampling._full_sample_parallel') as mock_fsp:
            mock_fsp.return_value = df
            sampling.sample_all(tmp, None, index_base, population, brain_regions, side, )
            for l in ('l2', 'l3'):  # note: 'l1' skipped b/c id doesn't exist
                ok_(os.path.exists(os.path.join(tmp, sampling.SAMPLE_PATH, 'FRP_%s_right.feather' % l)))
            eq_(mock_fsp.call_count, 3)

            mock_fsp.reset_mock()
            sampling.sample_all(tmp, None, index_base, population, brain_regions, side)
            eq_(mock_fsp.call_count, 0)


def test_load_all_samples():
    with test_utils.tempdir('test_load_all_samples') as tmp:
        white_matter_utils.ensure_path(os.path.join(tmp, sampling.SAMPLE_PATH))
        path = os.path.join(tmp, sampling.SAMPLE_PATH, 'Fake_l1_right.feather')
        df_right = pd.DataFrame(np.array([[0, 0., 10.],
                                          [1, -10, 10.]]),
                                columns=['tgid', 'segment_z1', 'segment_z2'])
        white_matter_utils.write_frame(path, df_right)

        df_left = pd.DataFrame(np.array([[0, 0., -10.],
                                         [1, -10, -10.]]),
                               columns=['tgid', 'segment_z1', 'segment_z2'])
        path = os.path.join(tmp, sampling.SAMPLE_PATH, 'Fake_l1_left.feather')
        white_matter_utils.write_frame(path, df_left)

        region_tgt = 'Fake'
        ret = sampling.load_all_samples(tmp, region_tgt)
        eq_(ret.keys(), ['l1'])
        assert_frame_equal(ret['l1']['right'], df_right)
        assert_frame_equal(ret['l1']['left'], df_left, check_names=False)


def test__add_random_position_and_offset():
    np.random.seed(42)
    columns = (sampling.SEGMENT_START_COLS +
               sampling.SEGMENT_END_COLS +
               ['segment_length', 'section_id', 'segment_id', 'tgid'])
    length = math.sqrt(100 + 1 + 1)
    syns = pd.DataFrame(np.array([[0., 0., 0.,
                                   10., 1., 1.,
                                   length, 10, 11, 12]]),
                        columns=columns)
    ret = sampling._add_random_position_and_offset(syns, n_jobs=1)
    expected = pd.DataFrame([[6.2545986, 0.6254599, 0.6254599, 3.7826698, 10.099505, 10, 11, 12]],
                            columns=['x', 'y', 'z',
                                     'segment_offset', 'segment_length',
                                     'section_id', 'segment_id', 'tgid'])
    assert_frame_equal(ret, expected, check_dtype=False)


def test__mask_xyzs_by_vertices_worker():
    from white_matter_projections.utils import in_2dtriangle
    with patch('white_matter_projections.sampling.utils') as utils:
        utils.in_2dtriangle = in_2dtriangle
        utils.Config.return_value = config = Mock()
        config.flat_map = test_utils.fake_flat_map()

        vertices = np.array(zip([0., 10., 0.], [0., 0., 10.]))
        xyzs = np.array([[1.1, 0.1, 0, ],
                         [2.3, 2.3, 2.3, ],
                         ])
        res = sampling._mask_xyzs_by_vertices_worker('config_path', vertices, xyzs, sl=slice(None))
        eq_([True, True], list(res))

        vertices = np.array(zip([0., 1., 0.], [0., 0., 1.]))
        xyzs = np.array([[1.1, 0.1, 0, ],
                         ])
        res = sampling._mask_xyzs_by_vertices_worker('config_path', vertices, xyzs, sl=slice(None))
        eq_([False, ], list(res))

        vertices = np.array(zip([0., 2., 0.], [0., 0., 2.]))
        xyzs = np.array([[1.1, 0.1, 0, ],
                         ])
        res = sampling._mask_xyzs_by_vertices_worker('config_path', vertices, xyzs, sl=slice(None))
        eq_([True, ], list(res))


def test_mask_xyzs_by_vertices():
    vertices = np.array(zip([0., 10., 0.], [0., 0., 10.]))
    xyzs = np.array([[1.1, 0.1, 0, ],
                     [2.3, 2.3, 2.3, ],
                     ])
    with patch('white_matter_projections.sampling.utils') as utils:
        utils.in_2dtriangle = white_matter_utils.in_2dtriangle
        utils.Config.return_value = config = Mock()
        config.flat_map = test_utils.fake_flat_map()
        res = sampling._mask_xyzs_by_vertices(
            'fake_config_path', vertices, xyzs, n_jobs=1, chunk_size=1000000)
        eq_([True, True], list(res))


def test_calculate_constrained_volume():
    with patch('white_matter_projections.sampling.utils') as utils:
        utils.in_2dtriangle = white_matter_utils.in_2dtriangle
        utils.Config.return_value = config = Mock()
        config.flat_map = test_utils.fake_flat_map()
        brain_regions = config.flat_map.brain_regions
        config_path = 'fake'

        vertices = np.array(zip([0., 10., 0.], [0., 0., 10.]))

        region_id = 314159  # fake
        ret = sampling.calculate_constrained_volume(config_path, brain_regions, region_id, vertices)
        eq_(ret, 0)

        region_id = 2
        ret = sampling.calculate_constrained_volume(config_path, brain_regions, region_id, vertices)
        eq_(ret, 10.)  # 10 voxels of 1 unit each

        vertices = np.array(zip([0., 1., 0.], [0., 0., 1.]))
        ret = sampling.calculate_constrained_volume(config_path, brain_regions, region_id, vertices)
        eq_(ret, 5.)  # removes 5, compared to above


def test__pick_syns():
    np.random.seed(37)

    syns = pd.DataFrame([10.], columns=['segment_length'])
    ret = sampling._pick_syns(syns, count=1)
    eq_(list(ret), [0])

    ret = sampling._pick_syns(syns, count=2)
    eq_(list(ret), [0, 0])

    syns = pd.DataFrame([0.000000000001, 10], columns=['segment_length'])
    ret = sampling._pick_syns(syns, count=1)
    eq_(list(ret), [1])


def test__subsample_per_source():
    config = Mock()
    config.config_path = 'fake'
    config.flat_map = test_utils.fake_flat_map()
    config.atlas.load_data.return_value, _ = test_utils.fake_brain_regions()
    config.recipe.get_projection.return_value = projection = Mock()
    projection.source_population = 'source_population0'
    config.recipe.projections_mapping = test_utils.fake_projection_mapping()

    columns = ['segment_x1', 'segment_y1', 'segment_z1',
               'segment_x2', 'segment_y2', 'segment_z2',
               'segment_offset', 'segment_length', 'section_id', 'segment_id', 'tgid', ]
    data = [[0.0, 0.0, 0.0,
             1.0, 1.0, 1.0,
             0.1, 10, 1, 1, 1],
            ]
    samples = {'two': {'left': pd.DataFrame(data, columns=columns),
                       'right': pd.DataFrame(data, columns=columns),
                       },
               'thirty': {'left': pd.DataFrame(data, columns=columns),
                          'right': pd.DataFrame(data, columns=columns),
                          },
               }
    projection_name = 'projection0'
    side = 'left'
    region_tgt = 'region_tgt'
    hemisphere = 'contra'

    def _pick_candidate_synapse_locations_mock(*args, **kwargs):
        count = kwargs['syns_count']
        ret = pd.DataFrame([[0, 2, ]] * count,
                           columns=['section_id', 'segment_id', ])

        return ret

    with test_utils.tempdir('test__subsample_per_source') as output:
        output_path = os.path.join(output, sampling.SAMPLE_PATH, side, '%s_%s.feather' % (projection_name, region_tgt))

        with patch('white_matter_projections.sampling._pick_candidate_synapse_locations') as syn_locations, \
            patch('white_matter_projections.sampling.utils.Config') as mock_config:

            mock_config.return_value = config

            syn_locations.side_effect = _pick_candidate_synapse_locations_mock
            np.random.seed(37)

            vertices = np.array(zip([0., 1., 0.], [0., 0., 1.]))

            densities = pd.DataFrame([['two', 2, 0.14],  # densities aren't the same, but when added (0.14 + 0.15) * 5. (volume)
                                      ['two', 2, 0.15]   # they are larger than 1
                                      ], columns=['subregion_tgt', 'id_tgt', 'density'])

            syn_count = sampling._subsample_per_source(config, vertices,
                                                       projection_name, region_tgt, densities, hemisphere, side,
                                                       samples, output)
            eq_(syn_count, 1)  # int((0.14 + 0.15) * 5.) == 1

            ok_(os.path.exists(output_path))
            res = white_matter_utils.read_frame(output_path)
            eq_(len(res), 1)  # int((0.14 + 0.15) * 5.) == 1

            #already sampled, file exists
            syn_count = sampling._subsample_per_source(config, vertices,
                                                       projection_name, region_tgt, densities, hemisphere, side,
                                                       samples, output)
            eq_(syn_count, 0)

            # duplicated densities
            densities = pd.DataFrame([['two', 2, 1.],  # duplicates
                                      ['two', 2, 1.]   # duplicates
                                      ], columns=['subregion_tgt', 'id_tgt', 'density'])

            os.unlink(output_path)
            syn_count = sampling._subsample_per_source(config, vertices,
                                                       projection_name, region_tgt, densities, hemisphere, side,
                                                       samples, output)
            eq_(syn_count, 5)  # int(1. * 5.), where 5 is the volume

            ok_(os.path.exists(output_path))
            res = white_matter_utils.read_frame(output_path)
            eq_(len(res), 5)  # int(1. * 5.), where 5 is the volume


def test__pick_candidate_synapse_locations_by_function():
    np.random.seed(37)

    columns = ['segment_x1', 'segment_y1', 'segment_z1',
               'segment_x2', 'segment_y2', 'segment_z2',
               'segment_offset', 'segment_length', 'section_id', 'segment_id', 'tgid', ]
    data = [[0.0, 0.0, 0.0,
             1.0, 1.0, 1.0,
             0.1, 10, 1, 1, 1],
            ]
    segment_samples = pd.DataFrame(data, columns=columns)

    mask_function = lambda(samples): [True, ] * len(samples)
    ret = sampling._pick_candidate_synapse_locations_by_function(mask_function,
                                                                 segment_samples,
                                                                 syns_count=1,
                                                                 min_to_pick=1,
                                                                 times=1)
    ok_(isinstance(ret, pd.DataFrame))
    eq_(len(ret), 1)

    mask_function = lambda(samples): [False, ] * len(samples)
    ret = sampling._pick_candidate_synapse_locations_by_function(mask_function,
                                                                 segment_samples,
                                                                 syns_count=1,
                                                                 min_to_pick=1,
                                                                 times=1)
    eq_(len(ret), 0)


def test__mask_xyzs_by_compensation():
    config_path = 'Mocked'
    vertices = np.array(zip([0., 10., 0.], [0., 0., 10.]))
    xyzs = np.array([[1.1, 0.1, 0, ],
                     [2.3, 2.3, 2.3, ],
                     ])
    sl = slice(None)
    with patch('white_matter_projections.sampling.utils') as utils:
        utils.Config.return_value = config = Mock()
        config.flat_map = test_utils.fake_flat_map()

        with test_utils.tempdir('test__mask_xyzs_by_compensation') as tmp:
            src_uvs_path = os.path.join(tmp, 'src_uvs_path.csv')
            pd.DataFrame({'u': [1., 2, 3,],
                          'v': [1., 2, 3,]}).to_csv(src_uvs_path, index=False)
            ret = sampling._mask_xyzs_by_compensation_worker(config_path, src_uvs_path,
                                                             xyzs, sl, sigma=10)
            assert_equal(ret, [True, True])

            ret = sampling._mask_xyzs_by_compensation_worker(config_path, src_uvs_path,
                                                             xyzs, sl, sigma=1)
            assert_equal(ret, [False, True])


def test__calculate_compensation():
    config = Mock()
    config.flat_map = test_utils.fake_flat_map()
    config.atlas.load_data.return_value, _ = test_utils.fake_brain_regions()
    config.recipe.projections_mapping = test_utils.fake_projection_mapping()

    src_ids = [2, 30]
    tgt_locations = np.array([[1.1, 0.1, 0, ],
                              [2.3, 2.3, 2.3, ],
                              ])
    side = 'left'
    hemisphere = 'ipsi'
    source_population = 'source_population0'
    projection_name = 'projection0'
    src_uvs, src_uvs_mapped, tgt_uvs, within_cutoff = sampling._calculate_compensation(
        config, src_ids, tgt_locations, side, hemisphere, source_population, projection_name)
    assert_allclose(src_uvs, src_uvs_mapped)  # the test mapping is the identity one
    assert_equal(tgt_uvs, np.array([[1.1, 0. ], [2.3, 2.3]]))

    assert_equal(within_cutoff, [True, True, ])


def test_compensation_paths():
    ret = sampling.get_compensation_path('output', 'left')
    eq_(ret, 'output/density_compensation_left.csv')
    ret = sampling.get_compensation_src_uvs_path('output', 'right', 'projection_name')
    eq_(ret, 'output/COMPENSATION/projection_name_right.csv')

#def test_calculate_compensation():
#    ret = sampling.calculate_compensation(config, projection_name, side, sample_size=10000)
#def subsample_per_target
