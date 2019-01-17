import os
import h5py
import numpy as np
import pandas as pd
from bluepy.v2.index import SegmentIndex
from joblib import parallel_backend

from voxcell.voxel_data import VoxelData
from nose.tools import ok_, eq_
from white_matter_projections import sampling
from numpy.testing import assert_allclose
from utils import tempdir, gte_
from mock import patch


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


def fake_brain_regions():
    raw = np.zeros((5, 5, 5), dtype=np.int)
    raw[0, 0, 0] = 1
    raw[1, :, 0:2] = 2
    raw[2, :4, 2:3] = 30
    brain_regions = VoxelData(raw, np.ones(3), offset=np.zeros(3))
    return brain_regions


def test__full_sample_parallel():
    brain_regions = fake_brain_regions()
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
    index_base = 'fake_index_base'
    population = pd.DataFrame(
        np.array([(1, 'FRP', 'l1'),
                  (2, 'FRP', 'l2'),
                  (30, 'FRP', 'l3')]),
        columns=['id', 'region', 'layer'])
    brain_regions = fake_brain_regions()

    df = pd.DataFrame(
        np.array([[101, 201, 10., 0., 10., 0., 10., 0., 10., 31337], ]),
        columns=sampling.SEGMENT_COLUMNS)

    with tempdir('test_sample_all') as tmp:
        with patch('white_matter_projections.sampling._full_sample_parallel') as mock_fsp:
            mock_fsp.return_value = df
            sampling.sample_all(tmp, index_base, population, brain_regions)
            for l in ('l1', 'l2', 'l3'):
                ok_(os.path.exists(os.path.join(tmp, sampling.SAMPLE_PATH, 'FRP_%s.feather' % l)))
            eq_(mock_fsp.call_count, 3)

            mock_fsp.reset_mock()
            sampling.sample_all(tmp, index_base, population, brain_regions)
            eq_(mock_fsp.call_count, 0)
