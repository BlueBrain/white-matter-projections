import math
import os
import h5py
import numpy as np
import pandas as pd
from bluepy.v2.index import SegmentIndex
from joblib import parallel_backend

from nose.tools import ok_, eq_
from white_matter_projections import sampling, utils
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from utils import (tempdir, gte_,
                   fake_brain_regions, fake_flat_map,
                   )
from mock import patch, Mock


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


def test__full_sample_parallel():
    brain_regions, _ = fake_brain_regions()
    brain_regions.raw[0, 0, 0] = 1
    index_path = 'fake_index_path'

    df = pd.DataFrame(
        np.array([[101, 201, 10., 0., 10., 0., 10., 0., 10., 31337], ]),
        columns=sampling.SEGMENT_COLUMNS)

    with patch('white_matter_projections.sampling._full_sample_worker') as mock_fsw:
        mock_fsw.return_value = df
        region_id = 1
        ret = sampling._full_sample_parallel(brain_regions, region_id, index_path, n_jobs=1)
        eq_(len(ret), np.count_nonzero(brain_regions.raw == region_id))

        region_id = 2
        nz_count = np.count_nonzero(brain_regions.raw == region_id)
        ret = sampling._full_sample_parallel(brain_regions, region_id, index_path, n_jobs=1, chunks=nz_count)
        eq_(len(ret), nz_count)

        region_id = 12345  # does not exist
        ret = sampling._full_sample_parallel(brain_regions, region_id, index_path, n_jobs=1)
        ok_(ret is None)

def test_sample_all():
    population = pd.DataFrame(
        np.array([(1, 'FRP', 'l1'),
                  (2, 'FRP', 'l2'),
                  (30, 'FRP', 'l3')]),
        columns=['id', 'region', 'layer'])
    brain_regions, _ = fake_brain_regions()

    df = pd.DataFrame(
        np.array([[101, 201, 10., 0., 10., 0., 10., 0., 10., 31337], ]),
        columns=sampling.SEGMENT_COLUMNS)

    side = 'right'
    with tempdir('test_sample_all') as tmp:
        index_base = os.path.join(tmp, 'fake_index_base')
        os.makedirs(os.path.join(index_base, 'FRP@left'))
        os.makedirs(os.path.join(index_base, 'FRP@right'))

        with patch('white_matter_projections.sampling._full_sample_parallel') as mock_fsp:
            mock_fsp.return_value = df
            sampling.sample_all(tmp, index_base, population, brain_regions, side)
            for l in ('l1', 'l2', 'l3'):
                ok_(os.path.exists(os.path.join(tmp, sampling.SAMPLE_PATH, 'FRP_%s_right.feather' % l)))
            eq_(mock_fsp.call_count, 3)

            mock_fsp.reset_mock()
            sampling.sample_all(tmp, index_base, population, brain_regions, side)
            eq_(mock_fsp.call_count, 0)


def test_load_all_samples():
    with tempdir('test_load_all_samples') as tmp:
        utils.ensure_path(os.path.join(tmp, sampling.SAMPLE_PATH))
        path = os.path.join(tmp, sampling.SAMPLE_PATH, 'Fake_l1_right.feather')
        df_right = pd.DataFrame(np.array([[0, 0., 10.],
                                          [1, -10, 10.]]),
                          columns=['tgid', 'segment_z1', 'segment_z2'])
        utils.write_frame(path, df_right)

        df_left = pd.DataFrame(np.array([[0, 0., -10.],
                                         [1, -10, -10.]]),
                          columns=['tgid', 'segment_z1', 'segment_z2'])
        path = os.path.join(tmp, sampling.SAMPLE_PATH, 'Fake_l1_left.feather')
        utils.write_frame(path, df_left)

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


def test__mask_xyzs_by_vertices():
    config = Mock()
    config.flat_map = fake_flat_map()

    vertices = np.array(zip([0., 10., 0.], [0., 0., 10.]))
    xyzs = np.array([[1.1, 0.1, 0, ],
                     [2.3, 2.3, 2.3, ],
                     ])
    res = sampling._mask_xyzs_by_vertices(config, vertices, xyzs, sl=slice(None))
    eq_([True, True], list(res))

    vertices = np.array(zip([0., 1., 0.], [0., 0., 1.]))
    xyzs = np.array([[1.1, 0.1, 0, ],
                     ])
    res = sampling._mask_xyzs_by_vertices(config, vertices, xyzs, sl=slice(None))
    eq_([False, ], list(res))

    vertices = np.array(zip([0., 2., 0.], [0., 0., 2.]))
    xyzs = np.array([[1.1, 0.1, 0, ],
                     ])
    res = sampling._mask_xyzs_by_vertices(config, vertices, xyzs, sl=slice(None))
    eq_([True, ], list(res))


def test__mask_xyzs_by_vertices():
    vertices = np.array(zip([0., 10., 0.], [0., 0., 10.]))
    xyzs = np.array([[1.1, 0.1, 0, ],
                     [2.3, 2.3, 2.3, ],
                     ])
    from white_matter_projections.utils import in_2dtriangle
    with patch('white_matter_projections.sampling.utils') as utils:
        utils.in_2dtriangle = in_2dtriangle
        utils.Config.return_value = config = Mock()
        config.flat_map = fake_flat_map()
        res = sampling._mask_xyzs_by_vertices_helper('fake_config', vertices, xyzs, slice(None))
        eq_([True, True], list(res))


def test_mask_xyzs_by_vertices():
    vertices = np.array(zip([0., 10., 0.], [0., 0., 10.]))
    xyzs = np.array([[1.1, 0.1, 0, ],
                     [2.3, 2.3, 2.3, ],
                     ])
    from white_matter_projections.utils import in_2dtriangle
    with patch('white_matter_projections.sampling.utils') as utils:
        utils.in_2dtriangle = in_2dtriangle
        utils.Config.return_value = config = Mock()
        config.flat_map = fake_flat_map()
        res = sampling.mask_xyzs_by_vertices(
            'fake_config_path', vertices, xyzs, n_jobs=1, chunk_size=1000000)
        eq_([True, True], list(res))


def test_calculate_constrained_volume():
    config = Mock()
    config.flat_map = fake_flat_map()
    brain_regions = config.flat_map.brain_regions
    region_id = 314159  # fake
    vertices = np.array(zip([0., 10., 0.], [0., 0., 10.]))
    ret = sampling.calculate_constrained_volume(config, brain_regions, region_id, vertices)
    eq_(ret, 0)

    region_id = 2
    ret = sampling.calculate_constrained_volume(config, brain_regions, region_id, vertices)
    eq_(ret, 10.)  # 10 voxels of 1 unit each

    vertices = np.array(zip([0., 1., 0.], [0., 0., 1.]))
    ret = sampling.calculate_constrained_volume(config, brain_regions, region_id, vertices)
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
    vertices = np.array(zip([0., 1., 0.], [0., 0., 1.]))
    densities = pd.DataFrame([['l1', 2, 0.145739],
                              ['l1', 2, 0.145739]
                              ], columns=['layer_tgt', 'id_tgt', 'density'])
    config = Mock()
    config.flat_map = fake_flat_map()
    config.atlas.load_data.return_value, _ = fake_brain_regions()

    hemisphere = 'contra'

    columns = ['segment_x1', 'segment_y1', 'segment_z1',
               'segment_x2', 'segment_y2', 'segment_z2',
               'segment_offset', 'segment_length', 'section_id', 'segment_id', 'tgid', ]
    data = [[0.0, 0.0, 0.0,
             1.0, 1.0, 1.0,
             0.1, 10, 1, 1, 1],
            ]
    samples = {'l1': {'left': pd.DataFrame(data, columns=columns),
                      'right': pd.DataFrame(data, columns=columns),
                      },
               }
    projection_name = 'projection_name'
    side = 'left'
    with tempdir('test__subsample_per_source') as output:
        with patch('white_matter_projections.sampling.mask_xyzs_by_vertices') as mask_xyzs:
            mask_xyzs.return_value = np.array([True], dtype=bool)
            np.random.seed(37)
            sampling._subsample_per_source(config, vertices,
                                           projection_name, densities, hemisphere, side,
                                           samples, output)

        output_path = os.path.join(output, sampling.SAMPLE_PATH, side, projection_name + '.feather')
        ok_(os.path.exists(output_path))
        res = utils.read_frame(output_path)
        eq_(len(res), 1)

#def subsample_per_target
