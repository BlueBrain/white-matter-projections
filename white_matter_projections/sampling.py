'''sampling of the circuit morphologies to create potential synapses based on segments'''
import collections
import functools
import os
import logging
from glob import glob

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import KDTree

from joblib import Parallel, delayed
from neurom import NeuriteType
from white_matter_projections import utils, mapping, flat_mapping


L = logging.getLogger(__name__)
SAMPLE_PATH = 'SAMPLED'
COMPENSATION_PATH = 'COMPENSATION'

SEGMENT_START_COLS = ['segment_x1', 'segment_y1', 'segment_z1', ]
SEGMENT_END_COLS = ['segment_x2', 'segment_y2', 'segment_z2', ]
SEGMENT_COLUMNS = (['afferent_section_type', 'section_id', 'segment_id', 'segment_length'] +
                   SEGMENT_START_COLS + SEGMENT_END_COLS +
                   ['tgid']
                   )


def _ensure_only_flatmap_segments(config, segments):
    '''Make sure 3D locations map to sensible 2D flatmap locations'''
    xyzs = (segments[SEGMENT_START_COLS].to_numpy() + segments[SEGMENT_END_COLS].to_numpy()) / 2.
    mapper = mapping.CommonMapper.load_default(config)
    uvs = mapper.map_points_to_flat(xyzs)
    inside_mask = flat_mapping.FlatMap.mask_in_2d_flatmap(uvs)

    segments = segments[inside_mask]
    L.debug('Removed %d of %d (%.2f%%) locations due to not having a flatmap location',
            len(inside_mask) - len(segments), len(inside_mask),
            100. * (len(inside_mask) - len(segments)) / float(len(inside_mask)))

    return segments


def _ensure_only_segments_from_region(config, region, side, df):
    '''Check that the segments in `df` are only from region@side'''
    region = region  # trick pylint
    cells = config.get_cells().query('region == @region')
    cells = utils.partition_left_right(cells, side, config.flat_map.center_line_3d)

    orig_len = len(df)
    df = df[np.isin(df.tgid.to_numpy(), cells.index.to_numpy())]
    L.debug('Removed %d of %d (%.2f %%) locations due to not being of the correct region',
            orig_len - len(df), orig_len, (orig_len - len(df)) / float(orig_len))

    return df


def _sample_with_flat_index(index_path, min_xyz, max_xyz):
    '''use flat index to get segments within min_xyz, max_xyz'''
    #  this is loaded late so that multiprocessing loads it outside of the main
    #  python binary - at one point, this was necessary, as there was shared state
    import libFLATIndex as FI  # pylint:disable=import-outside-toplevel
    from bluepy.index import SegmentIndex
    try:
        index = FI.loadIndex(str(os.path.join(index_path, 'SEGMENT')))  # pylint: disable=no-member
        min_xyz_ = tuple(map(float, min_xyz))
        max_xyz_ = tuple(map(float, max_xyz))
        segs_df = FI.numpy_windowQuery(index, *(min_xyz_ + max_xyz_))  # pylint: disable=no-member
        segs_df = SegmentIndex._wrap_result(segs_df)  # pylint: disable=protected-access
        FI.unLoadIndex(index)  # pylint: disable=no-member
        del index
    except Exception:  # pylint: disable=broad-except
        return None

    return segs_df.sort_values(segs_df.columns.tolist())


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
        df = _sample_with_flat_index(index_path, min_xyz, max_xyz)

        if df is None or len(df) == 0:
            continue

        df.columns = map(str, df.columns)

        df = df[df['Section.NEURITE_TYPE'] != NeuriteType.axon].copy()

        if df is None or len(df) == 0:
            continue

        del df['Segment.R1'], df['Segment.R2']

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

        df['afferent_section_type'] = pd.to_numeric(
            df['Section.NEURITE_TYPE'].apply(int),
            downcast='unsigned')
        del df['Section.NEURITE_TYPE']

        # uint -> smallest uint needed
        for name in ('Section.ID', 'Segment.ID', ):
            df[fix_name(name)] = pd.to_numeric(df[name], downcast='unsigned')
            del df[name]

        df['tgid'] = pd.to_numeric(df['gid'], downcast='unsigned')
        del df['gid']
        # }

        chunks.append(df)

    if len(chunks):
        df = pd.concat(chunks, ignore_index=True, sort=False)
    else:
        df = pd.DataFrame(columns=SEGMENT_COLUMNS)
    return df


def _full_sample_parallel(positions,
                          voxel_dimensions,
                          index_path,
                          region,
                          side,
                          config,
                          n_jobs=-2,
                          chunks=None):
    '''Parallel sample of all `positions`

    Args:
        positions(iterable of float positions): the lower corner of the voxel to be sampled
        voxel_dimensions: 3D values for the voxel size
        index_path(str): directory where FLATIndex can find SEGMENT_*
        region(str): region of intererest
        side(str): 'left' or 'right'
        config(utils.Config): configuration
        n_jobs(int): number of jobs
        chunks(int): number of chunks
    '''
    if chunks is None:
        chunks = max(32, (len(positions) // 500) + 1)

    L.debug('_full_sample_parallel: Using %s chunks', chunks)

    worker = delayed(_full_sample_worker)
    # TODO: check if using multiprocessing backend is faster here
    p = Parallel(n_jobs=n_jobs,
                 # verbose=150,
                 )
    df = p(worker(xyzs, index_path, voxel_dimensions)
           for xyzs in np.array_split(positions, chunks, axis=0))
    df = pd.concat(df, ignore_index=True, sort=False)

    if(not _is_split_index(index_path, region, side) and
       config.config.get('only_segments_from_region', False)):
        df = _ensure_only_segments_from_region(config, region, side, df)

    df = _ensure_only_flatmap_segments(config, df)

    return df


def _generate_dilatation_structure(dimension, dilatation_size):
    """ Create a voxel sphere like binary structure of radius `nb_pixels` """
    # pylint: disable=assignment-from-no-return
    output = np.fabs(np.indices([dilatation_size * 2 + 1] * dimension) - dilatation_size)
    output = np.add.reduce(output, 0)  # pylint: disable=assignment-from-no-return
    return output <= dilatation_size


def _dilate_region(brain_regions, region_ids, dilation_size):
    '''dilate voxels in `region_ids` by size `dilation_size`'''
    raw = np.zeros_like(brain_regions.raw, dtype=np.bool)
    raw[np.isin(brain_regions.raw, list(region_ids))] = True

    L.debug('_dilate_region: start dilation: dilation_size: %d', dilation_size)
    dilated = ndimage.binary_dilation(
        raw, _generate_dilatation_structure(brain_regions.ndim, dilation_size))
    dilated = dilated ^ raw  # only consider dilated voxels for assignment

    # Create a 'shell' around the region; this reduces the target points the
    # KDTree includes, which is faster for large regions
    raw = ndimage.distance_transform_cdt(raw) == 1

    # assign each of the dilated voxels to region of the closest point on the shell
    raw_idx = np.array(np.nonzero(raw)).T
    dilated_idx = np.array(np.nonzero(dilated)).T

    L.debug('_dilate_region: start assign: shell: %d, dilated: %d',
            len(raw_idx), len(dilated_idx))

    _, row_ids = KDTree(raw_idx).query(dilated_idx, 1)

    L.debug('_dilate_region: end assign')

    region_idx = brain_regions.raw[tuple(raw_idx[row_ids].T)]

    ret = {region_id: np.vstack((dilated_idx[region_idx == region_id, :],
                                 np.array(np.nonzero(brain_regions.raw == region_id)).T))
           for region_id in region_ids}

    return ret


def _is_split_index(index_base, region, side):
    '''Check if flatindex indices are split'''
    return os.path.exists(os.path.join(index_base, region + '@' + side, 'SEGMENT_index.dat'))


def _get_flatindices_path(index_base, region, side):
    '''Handle the 'split' or 'monolithic' index case'''
    index_path = None
    if _is_split_index(index_base, region, side):
        index_path = os.path.join(index_base, region + '@' + side)
    elif os.path.exists(os.path.join(index_base, 'SEGMENT_index.dat')):
        index_path = index_base

    return index_path


def sample_all(output, config, index_base, population, brain_regions, side, dilation_size=0):
    '''sample all segments per region and side for a population

    Args:
        output(str): path to output
        config(utils.Config): config
        index_base(str): path to segment indices base
        population(population dataframe): with only the target population
        brain_regions(voxcell.VoxelData): tagged regions
        side(str): either 'left' or 'right'
        dilation_size(int): number of pixels used to dilate each brain sub-regions

    Output:
        Feather files written to output/$SAMPLE_PATH/$population_$layer_$side.feather
        containing all the sample segments

    Notes:
        for dilation: https://en.wikipedia.org/wiki/Dilation_%28morphology%29
    '''
    output = os.path.join(output, SAMPLE_PATH)
    utils.ensure_path(output)

    to_sample = []
    for row in population.itertuples():
        # if format changes, need to change in: load_all_samples
        path = os.path.join(output, '%s_%s_%s.feather' % (row.region, row.subregion, side))
        if os.path.exists(path):
            L.debug('Already sampled %s@%s[%s] (%s), skipping',
                    row.region, side, row.subregion, path)
            continue

        to_sample.append((path, row))

    if dilation_size:
        indices = _dilate_region(brain_regions, list(population.id), dilation_size)
    else:
        indices = {row.id: np.array(np.nonzero(brain_regions.raw == row.id)).T
                   for _, row in to_sample}

    for path, row in to_sample:
        L.debug('Sampling %s[%s@%s] -> %s', row.region, row.subregion, side, path)

        index_path = _get_flatindices_path(index_base, row.region, side)
        if index_path is None:
            L.warning('Index %s is missing, skipping: %s[%s@%s]',
                      index_path, row.region, row.subregion, side)
            continue

        if not len(indices[row.id]):
            L.warning('Region ID %s is missing, skipping: %s[%s@%s]',
                      index_path, row.region, row.subregion, side)
            continue

        positions = brain_regions.indices_to_positions(indices[row.id])
        positions = np.unique(positions, axis=0)

        df = _full_sample_parallel(
            positions, brain_regions.voxel_dimensions, index_path, row.region, side, config)

        if df is None:
            continue

        L.debug('Sampled %s[%s@%s] -> %d', row.region, row.subregion, side, len(df))

        df.sort_values(['tgid', 'section_id', 'segment_id', 'segment_x1'],
                       kind='stable',
                       inplace=True)

        utils.write_frame(path, df)


def load_all_samples(path, region_tgt):
    '''load samples for `region_tgt` from `path`, in parallel, separating into left/right'''
    files = sorted(glob(os.path.join(path, SAMPLE_PATH, '%s_*.feather' % region_tgt)))
    L.debug('load_all_samples: %s', files)

    work = []
    # format is: '%s_%s_%s.feather' % (region, subregion, side))
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
        subregion, side = file_[1], file_[2][:-8]

        ret[subregion][side] = segments

    return dict(ret)


def _add_random_position_and_offset_worker(segments, output, sl, rng):
    '''Take start and end positions of `segments`, and create a random synapse position

    Args:
        segments(np.array Nx7): 0:3 starts, 3:6 ends, 6: segment_length
        output(np.array to write to): Nx4: :3 columns are XYZ position, 3 is the offset
        sl(slice): DataFrame input slice and output slice
        rng(np.random.Generator): used for random number generation

    Note: `output` is written modified
    '''
    starts, ends, segment_length = segments[sl, 0:3], segments[sl, 3:6], segments[sl, 6]
    alpha = rng.random(size=(len(starts), 1))
    output[sl, :3] = (1 - alpha) * starts + alpha * ends
    output[sl, 3] = alpha.ravel() * segment_length  # segment_offset


def _add_random_position_and_offset(segments, seed_sequence, chunk_size=1000000, n_jobs=-2):
    '''parallelize creating a synapse position on a segments

    Read _add_random_position_and_offset_worker for more info
    '''
    cols = SEGMENT_START_COLS + SEGMENT_END_COLS + ['segment_length']
    data = segments[cols].to_numpy()
    slices = [slice(start, start + chunk_size)
              for start in range(0, len(segments), chunk_size)]
    output = np.empty((len(segments), 4), dtype=np.float32)
    worker = delayed(_add_random_position_and_offset_worker)
    p = Parallel(n_jobs=n_jobs,
                 backend='threading',
                 # 'verbose': 60,
                 )
    p(worker(data, output, sl, np.random.default_rng(seed))
      for sl, seed in zip(slices, seed_sequence.spawn(len(slices)))
      )

    syns = pd.DataFrame(output, columns=['x', 'y', 'z', 'segment_offset'])
    syns['segment_length'] = segments['segment_length'].values
    syns['afferent_section_type'] = segments['afferent_section_type'].values

    for c in ('section_id', 'segment_id', 'tgid', ):
        syns[c] = segments[c].to_numpy()

    return syns


def _mask_xyzs_by_compensation_worker(config_path, src_uvs_path, xyzs, sl, sigma):
    '''using the flatmap UVs from `src_uvs_path`, make sure `xyzs` are within the sigma distance'''
    config = utils.Config(config_path)

    mapper = mapping.CommonMapper.load_default(config)
    pos = mapper.map_points_to_flat(xyzs[sl])

    # ignore positions outside of flatmap
    inside_mask = flat_mapping.FlatMap.mask_in_2d_flatmap(pos)
    pos = pos[inside_mask, :]

    src_uvs_mapped = pd.read_csv(src_uvs_path)
    kd_fibers = KDTree(src_uvs_mapped[['u', 'v']].to_numpy())
    distances, _ = kd_fibers.query(pos, 1)

    mask = np.zeros(len(xyzs[sl]), dtype=bool)
    mask[inside_mask] = distances <= sigma
    return mask


def _mask_xyzs_by_compensation(xyzs, config_path, src_uvs_path, sigma,
                               n_jobs=-2,
                               chunk_size=10000):
    '''parallize find `xyzs` that are in `vertices`

    Args:
        xyzs(array of positions): one position per row
        config_path(str): path to config file
        src_uvs_path(str): path to csv file w/ the flat-mapped source fibers used in compensation
        n_jobs(int): number of jobs
        chunk_size(int): size of the chunks to be passed off
    '''
    L.debug('mask_xyzs_by_compensation: with %s candidates', len(xyzs))
    p = Parallel(n_jobs=n_jobs,
                 # the multiprocessing backend is 50x faster than 'loky';
                 # I *think* it has to do w/ loky startup being slow on GPFS
                 # coupled w/ its use of semaphores, but I haven't been able
                 # to prove that.  For debugging, 'loky' is *much* nicer, use
                 # that when you can
                 backend='multiprocessing',
                 # verbose=51
                 )
    worker = delayed(_mask_xyzs_by_compensation_worker)
    slices = [slice(start, start + chunk_size)
              for start in range(0, len(xyzs), chunk_size)]
    masks = p(worker(config_path, src_uvs_path, xyzs, sl, sigma) for sl in slices)
    mask = np.hstack(masks)
    L.debug('mask_xyzs_by_compensation done: with %s candidates', len(xyzs))
    return mask


def _mask_xyzs_by_vertices_worker(config_path, vertices, xyzs, sl):
    '''create mask of `xyzs` that fall whithin `vertices`

    Args:
        config_path(str): path to yaml config
        vertices(array): vertices in flat_space such where the rows of `xyzs` are masked to
        xyzs(array of positions): one position per row
        sl(slice): slice to operate on

    Returns:
        array of bools masking xyzs by rows
    '''
    config = utils.Config(config_path)

    position_to_voxel = mapping.PositionToVoxel(config.flat_map.brain_regions)
    voxel_to_flat = mapping.VoxelToFlat(config.flat_map.flat_map, config.flat_map.shape)

    voxel_ijks, offsets = position_to_voxel(xyzs[sl])
    pos, offset = voxel_to_flat(voxel_ijks, offsets)

    mask = utils.in_2dtriangle(vertices, pos + offset)

    return mask


def _mask_xyzs_by_vertices(xyzs, config_path, vertices, n_jobs=36, chunk_size=1000000):
    '''parallize find `xyzs` that are in `vertices`

    Args:
        xyzs(array of positions): one position per row
        config_path(str): path to config file
        vertices(array): vertices in flat_space such where the rows of `xyzs` are masked to
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
    worker = delayed(_mask_xyzs_by_vertices_worker)
    slices = [slice(start, start + chunk_size)
              for start in range(0, len(xyzs), chunk_size)]
    masks = p(worker(config_path, vertices, xyzs, sl) for sl in slices)
    mask = np.hstack(masks)
    return mask


def calculate_constrained_volume(config_path, brain_regions, region_id, vertices):
    '''calculate the total volume in subregion 'constrained' by vertices'''
    idx = np.array(np.nonzero(brain_regions.raw == region_id)).T
    if len(idx) == 0:
        return 0
    xyzs = brain_regions.indices_to_positions(idx)
    count = np.sum(_mask_xyzs_by_vertices_worker(config_path, vertices, xyzs, slice(None)))
    return count * brain_regions.voxel_volume


def _pick_candidate_synapse_locations(  # pylint: disable=too-many-arguments
    output,
    config,
    segment_samples,
    projection_name,
    side,
    mirrored_vertices,
    syns_count,
    use_compensation,
    seed
):
    '''Pick potential synapse locations

    From `segment_samples`, get `syns_count` unique locations where synapses can be placed
    '''
    if use_compensation:
        L.info('Using compensation masking')
        source_population = config.recipe.get_projection(projection_name).source_population

        mask_function = functools.partial(
            _mask_xyzs_by_compensation,
            config_path=config.config_path,
            src_uvs_path=get_compensation_src_uvs_path(output, side, projection_name),
            sigma=_get_projection_sigma(config, source_population, projection_name))
    else:
        L.info('Using triangle masking')
        mask_function = functools.partial(_mask_xyzs_by_vertices,
                                          config_path=config.config_path,
                                          vertices=mirrored_vertices,
                                          )

    return _pick_candidate_synapse_locations_by_function(
        mask_function, segment_samples, syns_count, seed=seed)


def _pick_candidate_synapse_locations_by_function(mask_function,
                                                  segment_samples,
                                                  syns_count,
                                                  min_to_pick=25000,
                                                  times=20,
                                                  seed=0):
    '''Repeatedly call `mask_function` to pick synpases, until syns_count is met'''
    ret = []
    to_pick = syns_count
    seed_sequences = np.random.SeedSequence(seed)
    for i, seed_sequence in zip(range(times), seed_sequences.spawn(times)):
        L.debug('_pick_candidate_synapse_locations_by_function try %d, to go %d', i, to_pick)
        L.debug('    start add random: %d', len(segment_samples))
        syns = _add_random_position_and_offset(segment_samples, seed_sequence)
        L.debug('    end random')

        # when we are getting close to the count we want picked, oversample to make it faster
        # and thus we don't need to traverse the loop many times just to get the last few
        L.debug('    start picking %d', max(min_to_pick, to_pick))
        rng = np.random.default_rng(seed_sequence.spawn(1)[0])
        picked = _pick_syns(syns, count=max(min_to_pick, to_pick), rng=rng)

        if picked is None:
            return None

        syns = syns.iloc[picked]
        L.debug('    end picking')

        mask = mask_function(syns[utils.XYZ].to_numpy())
        L.debug('Done masking')

        syns = syns[mask]

        if len(syns) > to_pick:
            syns = syns.sample(to_pick, random_state=rng.bit_generator)

        ret.append(syns)
        to_pick -= len(syns)
        if to_pick <= 0:
            break

    ret = pd.concat(ret, ignore_index=True, sort=False)
    return ret


def _subsample_per_source(  # pylint: disable=too-many-arguments
    config,
    target_vertices,
    projection_name,
    region_tgt,
    densities,
    hemisphere,
    side,
    segment_samples,
    output,
    seed
):
    '''Given all region's `segment_samples`, pick segments that satisfy the
    desired density in constrained by `vertices`

    Args:
        config(utils.Config): configuration
        target_vertices(array): vertices in flat_space to constrain the samples in the target
        flat_space
        projection_name(str): name of projection
        densities(DataFrame): with columns 'subregion_tgt', 'id_tgt', 'density'
        hemisphere(str): either 'ipsi' or 'contra'
        side(str): either 'left' or 'right'
        segment_samples(dict of 'left'/'right'): full sample of regions's segments
        output(path): where to output files
        seed(int): seed do use

    Returns:
        number of synapses subsampled
    '''
    # pylint: disable=too-many-locals

    brain_regions = config.atlas.load_data('brain_regions')

    base_path = os.path.join(output, SAMPLE_PATH, side)
    utils.ensure_path(base_path)
    path = os.path.join(base_path, '%s_%s.feather' % (projection_name, region_tgt))
    if os.path.exists(path):
        L.info('Already subsampled: %s', path)
        return 0

    mirrored_vertices = target_vertices.copy()
    if utils.is_mirror(side, hemisphere):
        center_line = config.flat_map.center_line_2d
        mirrored_vertices = utils.mirror_vertices_y(mirrored_vertices, center_line)

    densities = densities[['subregion_tgt', 'id_tgt', 'density']].drop_duplicates()
    groupby = densities.groupby(['subregion_tgt', 'id_tgt']).density.sum().iteritems()
    all_syns, zero_volume = [], []
    for i, ((layer, id_tgt), density) in enumerate(groupby):
        volume = calculate_constrained_volume(
            config.config_path, brain_regions, id_tgt, mirrored_vertices)

        L.debug('  %s[%s %s]: %s', projection_name, layer, side, volume)

        if volume <= 0.1:
            zero_volume.append((projection_name, side, layer, ))
            continue

        syns = _pick_candidate_synapse_locations(
            output,
            config,
            segment_samples[layer][side],
            projection_name,
            side,
            mirrored_vertices,
            syns_count=int(volume * density),
            use_compensation=config.config.get('compensation', False),
            seed=seed + i)

        if syns is not None:
            L.debug('Got %d syns', len(syns))
            all_syns.append(syns)

    if zero_volume:
        L.info('No volume found for: %s', zero_volume)
    elif len(all_syns) == 0:
        L.info('No synapses found')
        return 0

    all_syns = pd.concat(all_syns, ignore_index=True, sort=False)

    if not len(all_syns):
        L.info('No synapses found for: %s %s', projection_name, side)
    else:
        all_syns['section_id'] = pd.to_numeric(all_syns['section_id'], downcast='unsigned')
        all_syns['segment_id'] = pd.to_numeric(all_syns['segment_id'], downcast='unsigned')

        utils.write_frame(path, all_syns)

    return len(all_syns)


def _pick_syns(syns, count, rng):
    '''pick (with replacement) `count` syns, using syns.segment_length as weighting'''
    prob_density = syns.segment_length.values.astype(np.float64)
    try:
        prob_density = utils.normalize_probability(prob_density)
    except utils.ErrorCloseToZero:
        L.warning('ErrorCloseToZero %s', prob_density.shape)
        return None

    picked = rng.choice(len(syns), size=count, replace=True, p=prob_density)
    return picked


def subsample_per_target(output,
                         config,
                         target_population,
                         side,
                         use_compensation,
                         rank,
                         max_rank=0):
    '''Create feathers files in `output` for projections targeting `target_population`

    Args:
        output(str): path to output directory
        config(utils.Config): config
        target_population(str):
        side(str): 'left' or 'right'
        use_compensation(bool): whether to use compensation
        rank(int): which worker number this is
        max_rank(int): total number of workers
    '''
    # pylint: disable=too-many-locals
    L.debug('Sub-sampling for target: %s', target_population)
    target_population = target_population  # trick pylint since used in pandas query
    densities = (config.recipe.
                 calculate_densities(utils.normalize_layer_profiles(config.region_layer_heights,
                                                                    config.recipe.layer_profiles))
                 .query('target_population == @target_population')
                 )

    if not len(densities):
        L.warning('Densities are empty, did you select a target?')
        return

    if use_compensation:
        compensation = get_compensation_path(output, side)
        L.info('Using compensation from: %s', compensation)
        compensation = pd.read_csv(compensation, index_col='projection_name')

    regions = list(densities.region_tgt.unique())

    for region_tgt in regions:
        segment_samples = load_all_samples(output, region_tgt)

        gb = list(densities
                  .query('region_tgt == @region_tgt')
                  .groupby(['source_population', 'projection_name', 'hemisphere', ]))

        for i, (keys, density) in enumerate(gb):
            if max_rank and i % max_rank != rank:
                continue

            source_population, projection_name, hemisphere = keys
            tgt_vertices = config.recipe.projections_mapping[
                source_population][projection_name]['vertices']

            updated_density = density.copy()

            if use_compensation:
                comp = compensation.loc[projection_name]
                comp = comp.total / float(comp.within_cutoff)
                L.debug('Compensation: %s', comp)
                updated_density.density *= comp

            seed = utils.generate_seed(source_population, projection_name, hemisphere)

            L.debug('Subsampling for %s[%s][%s] (%s of %s), seed: %s',
                    projection_name, side, region_tgt, i + 1, len(gb), seed)
            _subsample_per_source(config, tgt_vertices,
                                  projection_name, region_tgt,
                                  updated_density, hemisphere, side,
                                  segment_samples, output, seed)


def calculate_compensation(config, projection_name, side, sample_size=10000):
    '''calculate the compensation of synapses needed for incomplete targeting

    Args:
        config(utils.Config): config instance
        projection_name(str): name of the projection
        side: 'left' or 'right'
        sample_size(int=10000):


    As discussed in: https://bbpteam.epfl.ch/project/issues/browse/BBPP82-62
    and in doc/source/concepts.rst
    '''
    from white_matter_projections import micro

    projection = config.recipe.get_projection(projection_name)
    source_population, target_population, hemisphere = projection[
        ['source_population', 'target_population', 'hemisphere']]

    tgt_gids = micro.get_gids_by_population(
        config.recipe.populations, config.get_cells, target_population)
    left_cells, right_cells = micro.partition_cells_left_right(
        config.get_cells().loc[tgt_gids], config.flat_map.center_line_3d)

    if side == 'left':
        tgt_locations = left_cells
    else:
        tgt_locations = right_cells

    rng = np.random.default_rng(
        seed=config.seed + utils.generate_seed(source_population, projection_name, hemisphere))
    tgt_locations = tgt_locations.sample(sample_size, random_state=rng.bit_generator)
    tgt_locations = tgt_locations[utils.XYZ].to_numpy()

    src_ids = config.recipe.populations.set_index('population').loc[source_population]['id']

    if isinstance(src_ids, pd.Series):
        src_ids = np.array(src_ids)
    else:
        src_ids = np.array([src_ids, ])

    return _calculate_compensation(config,
                                   src_ids,
                                   tgt_locations,
                                   side,
                                   hemisphere,
                                   source_population,
                                   projection_name
                                   )


def _get_projection_sigma(config, source_population, projection_name):
    '''get the sigma for a `projection_name`'''
    variance = config.recipe.projections_mapping[source_population][projection_name]['variance']
    return variance**0.5


def _calculate_compensation(config,
                            src_ids,
                            tgt_locations,
                            side,
                            hemisphere,
                            source_population,
                            projection_name
                            ):
    '''calculate the compensation of synapses needed for incomplete targeting

    As discussed in: https://bbpteam.epfl.ch/project/issues/browse/BBPP82-62
    and in doc/source/concepts.rst
    '''
    # pylint: disable=too-many-locals

    # sample the voxels in the source region, transform them to the flatmap
    brain_regions = config.atlas.load_data('brain_regions')
    src_locations = brain_regions.indices_to_positions(
        np.array(np.nonzero(np.isin(brain_regions.raw, src_ids))).T)

    mirrored = utils.is_mirror(side, hemisphere)

    if mirrored:
        src_locations = src_locations[src_locations[:, utils.Z] < config.flat_map.center_line_3d]
    else:
        src_locations = src_locations[src_locations[:, utils.Z] > config.flat_map.center_line_3d]

    mapper = mapping.CommonMapper.load_default(config)

    src_uvs = np.unique(mapper.map_points_to_flat(src_locations), axis=0)
    src_uvs = src_uvs[flat_mapping.FlatMap.mask_in_2d_flatmap(src_uvs), :]

    src_uvs_mapped = mapper.map_flat_to_flat(source_population, projection_name, src_uvs, mirrored)

    tgt_uvs = mapper.map_points_to_flat(tgt_locations)

    distances, _ = KDTree(src_uvs_mapped).query(tgt_uvs, 1)

    within_cutoff = distances <= _get_projection_sigma(config, source_population, projection_name)

    return src_uvs, src_uvs_mapped, tgt_uvs, within_cutoff


def get_compensation_path(output, side):
    '''ibid'''
    return os.path.join(output, 'density_compensation_%s.csv' % side)


def get_compensation_src_uvs_path(output, side, projection_name):
    '''ibid'''
    return os.path.join(output, COMPENSATION_PATH, '%s_%s.csv' % (projection_name, side))
