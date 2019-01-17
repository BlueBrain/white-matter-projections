'''sampling of the circuit morphologies to create potential synapses based on segments'''
import os
import logging
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from neurom import NeuriteType
from projectionizer import synapses
from white_matter_projections import utils


L = logging.getLogger(__name__)
SAMPLE_PATH = 'SAMPLED'
SEGMENT_COLUMNS = ['section_id', 'segment_id', 'segment_length',
                   'segment_x1', 'segment_x2',
                   'segment_y1', 'segment_y2',
                   'segment_z1', 'segment_z2',
                   'tgid']


def _full_sample_worker(min_xyzs, index_path, dims):
    '''
    '''
    start_cols = ['Segment.X1', 'Segment.Y1', 'Segment.Z1']
    end_cols = ['Segment.X2', 'Segment.Y2', 'Segment.Z2', ]

    chunks = []
    for min_xyz in min_xyzs:
        max_xyz = min_xyz + dims
        df = synapses._sample_with_flat_index(  # pylint: disable=protected-access
            index_path, min_xyz, max_xyz)

        if df is None or len(df) == 0:
            continue

        df.columns = map(str, df.columns)

        df = df[df['Section.NEURITE_TYPE'] != NeuriteType.axon].copy()

        if df is None or len(df) == 0:
            continue

        starts, ends = df[start_cols].values, df[end_cols].values
        df['segment_length'] = np.linalg.norm(ends - starts, axis=1).astype(np.float32)

        #  { need to get rid of memory usage as quickly as possible
        #    MOs5 (the largest region by voxel count) *barely* fits into 300GB
        def fix_name(name):
            '''convert pandas column names to snake case'''
            return name.lower().replace('.', '_')

        # float64 -> float32
        for name in start_cols + end_cols:
            df[fix_name(name)] = df[name].values.astype(np.float32)
            del df[name]

        # uint -> smallest uint needed
        for name in ('Section.ID', 'Segment.ID', ):
            df[fix_name(name)] = pd.to_numeric(df[name], downcast='unsigned')
            del df[name]

        df['tgid'] = pd.to_numeric(df['gid'], downcast='unsigned')

        del df['Section.NEURITE_TYPE'], df['Segment.R1'], df['Segment.R2'], df['gid']
        #  }

        chunks.append(df)

    if len(chunks):
        df = pd.concat(chunks).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=SEGMENT_COLUMNS)
    return df


def _full_sample_parallel(brain_regions, region_id, index_path, n_jobs=-2, chunks=None):
    '''Sample *all* segments of type region_id

    Args:
        brain_regions(VoxelData): brain regions
        region_id(int): single region id to sample
        index_path(str): directory where FLATIndex can find SEGMENT_*
    '''
    nz = np.array(np.nonzero(brain_regions.raw == region_id)).T
    if len(nz) == 0:
        return None

    positions = brain_regions.indices_to_positions(nz)
    positions = np.unique(positions, axis=0)
    if chunks is None:
        chunks = (len(positions) // 500) + 1

    worker = delayed(_full_sample_worker)
    # TODO: check if using multiprocessing backend is faster here
    p = Parallel(n_jobs=n_jobs)
    df = p(worker(xyzs, index_path, brain_regions.voxel_dimensions)
           for xyzs in np.array_split(positions, chunks, axis=0))
    df = pd.concat(df).reset_index(drop=True)
    return df


def sample_all(output, index_base, population, brain_regions):
    '''sample all segments per region & layer for a population: ie: VISam_l5

    Args:
        output(str):
        index_base(str): path to segment indices base
        population(population dataframe): with only the target population
        brain_regions(voxcell.VoxelData): tagged regions

    Output:
        Feather files written to output/$SAMPLE_PATH/$population_$layer.feather
        containing all the sample segments
    '''
    output = os.path.join(output, SAMPLE_PATH)
    utils.ensure_path(output)

    for id_, region, layer in population[['id', 'region', 'layer']].values:
        path = os.path.join(output, '%s_%s.feather' % (region, layer))
        if os.path.exists(path):
            L.debug('Already sampled %s[%s] (%s), skipping', region, layer, path)
            continue

        L.debug('Sampling %s[%s] -> %s', region, layer, path)

        index_path = os.path.join(index_base, region)
        df = _full_sample_parallel(brain_regions, id_, index_path)
        if df is not None:
            utils.write_frame(path, df)
