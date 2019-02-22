'''sampling of the circuit morphologies to create potential synapses based on segments'''
import collections
import itertools as it
import os
import logging
from glob import glob

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from neurom import NeuriteType
from projectionizer import synapses
from projectionizer.utils import (ErrorCloseToZero, normalize_probability,
                                  )
from white_matter_projections import utils, mapping


L = logging.getLogger(__name__)
SAMPLE_PATH = 'SAMPLED'

SEGMENT_START_COLS = ['segment_x1', 'segment_y1', 'segment_z1', ]
SEGMENT_END_COLS = ['segment_x2', 'segment_y2', 'segment_z2', ]
SEGMENT_COLUMNS = sorted(['section_id', 'segment_id', 'segment_length', ] +
                         SEGMENT_START_COLS + SEGMENT_END_COLS +
                         ['tgid']
                         )


def _full_sample_worker(min_xyzs, index_path, voxel_dimensions):
    '''for every voxel defined by the lower coordinate in min_xzys, gather segments

    Args:
        min_xyzs(np.array of (Nx3):
        index_path(str): path to FlatIndex indices
        voxel_dimensions(1x3 array): voxel dimensions
    '''
    start_cols = ['Segment.X1', 'Segment.Y1', 'Segment.Z1']
    end_cols = ['Segment.X2', 'Segment.Y2', 'Segment.Z2', ]

    chunks = []
    for min_xyz in min_xyzs:
        max_xyz = min_xyz + voxel_dimensions
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
        df = pd.concat(chunks, ignore_index=True, sort=False)
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
    df = pd.concat(df, ignore_index=True, sort=False)
    return df


def sample_all(output, index_base, population, brain_regions):
    '''sample all segments per region & layer for a population: ie: VISam_l5

    Args:
        output(str):
        index_base(str): path to segment indices base
        population(population dataframe): with only the target population
        brain_regions(voxcell.VoxelData): tagged regions

    Output:
        Feather files written to output/$SAMPLE_PATH/$population_$layer_$side.feather
        containing all the sample segments
    '''
    output = os.path.join(output, SAMPLE_PATH)
    utils.ensure_path(output)

    populations = population[['id', 'region', 'layer']].values
    for (id_, region, layer), side in it.product(populations, utils.SIDES):
        # if format changes, need to change in: load_all_samples
        path = os.path.join(output, '%s_%s_%s.feather' % (region, layer, side))
        if os.path.exists(path):
            L.debug('Already sampled %s@%s[%s] (%s), skipping', region, side, layer, path)
            continue

        L.debug('Sampling %s[%s@%s] -> %s', region, layer, side, path)

        index_path = os.path.join(index_base, region + '@' + side)
        if not os.path.exists(index_path):
            L.warning('Index %s is missing, skipping: %s[%s@%s]',
                      index_path, region, layer, side)
            continue

        df = _full_sample_parallel(brain_regions, id_, index_path)
        if df is not None:
            utils.write_frame(path, df)


def load_all_samples(path, region_tgt):
    '''load samples for `region_tgt` from `path`, in parallel, separating into left/right'''
    files = glob(os.path.join(path, SAMPLE_PATH, '%s_*.feather' % region_tgt))
    work = []
    # format is: '%s_%s_%s.feather' % (region, layer, side))
    for file_ in files:
        assert len(os.path.basename(file_).split('_')) == 3
        work.append(delayed(utils.read_frame)(file_))

    # reduce data serialization by only using threading ~6m -> 1.3m for VISp
    kwargs = {'n_jobs': len(work),
              'backend': 'threading',
              }
    work = Parallel(**kwargs)(work)

    ret = collections.defaultdict(dict)
    for file_, segments in zip(files, work):
        file_ = os.path.basename(file_).split('_')
        layer, side = file_[1], file_[2][:-8]

        ret[layer][side] = segments
    return dict(ret)


def _add_random_position_and_offset_worker(segments, output, sl):
    '''Take start and end positions of `segments`, and create a random synapse position

    Args:
        segments(np.array Nx7): 0:3 starts, 3:6 ends, 6: segment_length
        output(np.array to write to): Nx4: :3 columns are XYZ position, 3 is the offset
        sl(slice): DataFrame input slice and output slice

    Note: output is written modified
    '''
    starts, ends, segment_length = segments[sl, 0:3], segments[sl, 3:6], segments[sl, 6]
    alpha = np.random.random_sample((starts.shape[0], 1))
    output[sl, :3] = alpha * starts + (1. - alpha) * ends
    output[sl, 3] = alpha.ravel() * segment_length  # segment_offset


def _add_random_position_and_offset(segments, chunk_size=1000000, n_jobs=-2):
    '''parallelize creating a synapse position on a segments

    Read _add_random_position_and_offset_worker for more info
    '''
    cols = SEGMENT_START_COLS + SEGMENT_END_COLS + ['segment_length']
    data = segments[cols].values
    slices = [slice(start, start + chunk_size)
              for start in range(0, len(segments), chunk_size)]
    output = np.empty((len(segments), 4), dtype=np.float32)
    worker = delayed(_add_random_position_and_offset_worker)
    p = Parallel(n_jobs=n_jobs,
                 backend='threading',
                 # 'verbose': 60,
                 )
    p(worker(data, output, sl) for sl in slices)

    syns = pd.DataFrame(output, columns=['x', 'y', 'z', 'segment_offset'])
    syns['segment_length'] = segments['segment_length'].values
    for c in ('section_id', 'segment_id', 'tgid', ):
        syns[c] = pd.to_numeric(segments[c].values, downcast='unsigned')

    return syns


def _mask_xyzs_by_vertices_helper(config_path, vertices, xyzs, sl):
    '''wrap _mask_xyzs_by_vertices so full config doesn't need to be serialized'''
    config = utils.Config(config_path)
    return _mask_xyzs_by_vertices(config, vertices, xyzs, sl)


def _mask_xyzs_by_vertices(config, vertices, xyzs, sl):
    '''create mask of `xyzs` that fall whithin `vertices`

    Args:
        config(utils.Config): configuration
        vertices(array): vertices in flat_space such where the rows of `xyzs` are masked to
        xyzs(array of positions): one position per row
        sl(slice): to operate on

    Returns:
        array of bools masking xyzs by rows
    '''
    position_to_voxel = mapping.PositionToVoxel(config.flat_map.brain_regions)
    voxel_to_flat = mapping.VoxelToFlat(config.voxel_to_flat(), config.flat_map.view_lookup.shape)

    voxel_ijks, offsets = position_to_voxel(xyzs[sl])
    pos, offset = voxel_to_flat(voxel_ijks, offsets)

    mask = utils.in_2dtriangle(vertices, pos + offset)

    return mask


def mask_xyzs_by_vertices(config_path, vertices, xyzs, n_jobs=36, chunk_size=1000000):
    '''parallize find `xyzs` that are in `vertices`

    Args:
        config_path(str): path to config file
        vertices(array): vertices in flat_space such where the rows of `xyzs` are masked to
        xyzs(array of positions): one position per row
        n_jobs(int): number of jobs
        chunk_size(int): size of the chunks to be passed off
    '''
    p = Parallel(n_jobs=n_jobs,
                 # the multiprocessing backend is 50x faster than 'loky';
                 # I *think* it has to do w/ loky startup being slow GPFS
                 # coupled w/ its use of semaphores, but I haven't been able
                 # to prove that.  For debugging, 'loky' is *much* nicer, use
                 # that when you can
                 backend='multiprocessing',
                 # verbose=51
                 )
    worker = delayed(_mask_xyzs_by_vertices_helper)
    slices = [slice(start, start + chunk_size)
              for start in range(0, len(xyzs), chunk_size)]
    masks = p(worker(config_path, vertices, xyzs, sl) for sl in slices)
    mask = np.hstack(masks)
    return mask


def calculate_constrained_volume(config, brain_regions, region_id, vertices):
    '''calculate the total volume in layer 'constrained' by vertices'''
    idx = np.array(np.nonzero(brain_regions.raw == region_id)).T
    if len(idx) == 0:
        return 0
    xyzs = brain_regions.indices_to_positions(idx)
    count = np.sum(_mask_xyzs_by_vertices(config, vertices, xyzs, slice(None)))
    return count * brain_regions.voxel_volume


def _subsample_per_source(config, target_vertices,
                          projection_name, densities, hemisphere, side,
                          segment_samples, output):
    '''Given all region's `segment_samples`, pick segments that satisfy the
    desired density in constrained by `vertices`

    Args:
        config(utils.Config): configuration
        target_vertices(array): vertices in flat_space to constrain the samples in the target
        flat_space
        projection_name(str): name of projection
        densities(DataFrame): with columns 'layer_tgt', 'id_tgt', 'density'
        hemisphere(str): either 'ipsi' or 'contra'
        side(str): either 'left' or 'right'
        segment_samples(dict of 'left'/'right'): full sample of regions's segments
        output(path): where to output files
    '''
    # pylint: disable=too-many-locals

    center_line = config.flat_map.center_line_2d
    brain_regions = config.atlas.load_data('brain_regions')

    base_path = os.path.join(output, SAMPLE_PATH, side)
    utils.ensure_path(base_path)
    path = os.path.join(base_path, projection_name + '.feather')
    if os.path.exists(path):
        L.info('Already subsampled: %s', path)
        return

    L.debug('Subsampling for %s[%s]', projection_name, side)
    mirrored_vertices = target_vertices.copy()
    if utils.is_mirror(side, hemisphere):
        mirrored_vertices = utils.mirror_vertices_y(mirrored_vertices, center_line)

    groupby = densities.groupby(['layer_tgt', 'id_tgt']).density.sum().iteritems()
    all_syns = []
    for (layer, id_tgt), density in groupby:
        volume = calculate_constrained_volume(config, brain_regions, id_tgt, mirrored_vertices)
        if volume <= 0.1:
            L.info('No synapses found for: %s %s', projection_name, side)
            return

        syns = _add_random_position_and_offset(segment_samples[layer][side])

        mask = mask_xyzs_by_vertices(config.config_path,
                                     mirrored_vertices,
                                     syns[utils.XYZ].values)
        syns = syns[mask]

        picked = _pick_syns(syns, count=int(volume * density))
        all_syns.append(syns.iloc[picked])

        del syns, picked  # drop memory usage as quickly as possible

    if not len(all_syns):
        L.info('No synapses found for: %s %s', projection_name, side)
        return

    all_syns = pd.concat(all_syns, ignore_index=True, sort=False)

    all_syns['section_id'] = pd.to_numeric(all_syns['section_id'], downcast='unsigned')
    all_syns['segment_id'] = pd.to_numeric(all_syns['segment_id'], downcast='unsigned')

    utils.write_frame(path, all_syns)


def _pick_syns(syns, count):
    '''pick (with replacement) `count` syns, using syns.segment_length as weighting'''
    prob_density = syns.segment_length.values.astype(np.float64)
    try:
        prob_density = normalize_probability(prob_density)
    except ErrorCloseToZero:
        return None

    picked = np.random.choice(len(syns), size=count, replace=True, p=prob_density)
    return picked


def subsample_per_target(output, config, target_population, side):
    '''Create feathers files in `output` for projections targeting `target_population`'''
    norm_layer_profiles = utils.normalize_layer_profiles(config.region_layer_heights,
                                                         config.recipe.layer_profiles)
    target_population = target_population  # trick pylint since used in pandas query
    densities = (config.recipe.
                 calculate_densities(norm_layer_profiles)
                 .query('target_population == @target_population')
                 )

    projections_mapping = config.recipe.projections_mapping
    region_tgt = str(densities.region_tgt.unique()[0])
    segment_samples = load_all_samples(output, region_tgt)
    gb = densities.groupby(['source_population', 'region_tgt', 'projection_name', 'hemisphere', ])
    for keys, densities in gb:
        source_population, region_tgt, projection_name, hemisphere = keys

        tgt_vertices = projections_mapping[source_population][projection_name]['vertices']

        _subsample_per_source(config, tgt_vertices,
                              projection_name, densities, hemisphere, side,
                              segment_samples, output)
