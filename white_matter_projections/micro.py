'''operations related to 'micro' connectome, as defined in the white-matter paper

'''
from glob import glob
import logging
import math
import os

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from white_matter_projections import sampling, utils, streamlines, mapping


ASSIGNMENT_PATH = 'ASSIGNED'
L = logging.getLogger(__name__)


def _append_side_to_regions_hack_ncx176(regions):
    '''due to hemisphere not being encoded in atlas, MVD had hemisphere appended

    # XXX: NCX-176 hack
    '''
    ret = []
    for region in regions:
        ret.append(region + '@left')
        ret.append(region + '@right')
    return ret


def get_gids_by_population(populations, get_cells, population):
    '''for a `population`, get all the gids from `cells` that are in that population'''
    population = populations.set_index('population').loc[population]

    if isinstance(population, pd.DataFrame):
        region_names = set(population.region)
        subregion = set(population.subregion)
        population_filter = population.population_filter[0]
    else:
        region_names = {population.region}
        subregion = {population.subregion}
        population_filter = population.population_filter

    #  Due to the cells dataframe contaning numeric layers, need to convert to int
    subregion = [s for s in subregion if s]
    try:
        subregion = [int(s[0]) for s in subregion]
    except Exception as e:  # noqa
        raise ValueError(f'Non numeric layer names: {subregion}') from e

    cells = get_cells(population_filter)

    if len(cells) and (cells.iloc[0].region.endswith('left') or
                       cells.iloc[0].region.endswith('right')):
        region_names = _append_side_to_regions_hack_ncx176(region_names)

    gids = cells.query('region in @region_names and layer in @subregion').index.values
    return gids


def partition_cells_left_right(cells, center_line_3d):
    '''return left and right cells'''
    left_cells = cells[cells.z <= center_line_3d]
    right_cells = cells[cells.z > center_line_3d]
    return left_cells, right_cells


def separate_source_and_targets(left_cells, right_cells, sgids, hemisphere, side):
    '''based on the hemisphere and side, return the source cell positions, and synapses

    Args:
        left_cells(df): cells in the left hemisphere
        right_cells(df): cells in the right hemisphere
        sgids(np.array of int): potential source GIDs
        hemisphere(str): 'ipsi' or 'contra'; which type of projection
        side(str): 'left' or 'right', which hemisphere we're working in

    Returns:
       source cells
       synapses dataframe
    '''
    assert hemisphere in utils.HEMISPHERES
    assert side in utils.SIDES

    if side == 'right':
        if hemisphere == 'ipsi':
            cells = right_cells
        else:
            cells = left_cells
    else:
        if hemisphere == 'ipsi':
            cells = left_cells
        else:
            cells = right_cells

    sgids = sgids[np.isin(sgids, cells.index)]
    source_cells = cells.loc[sgids]

    return source_cells


def _assign_groups_worker(src_flat, tgt_flat, sigma, closest_count, rng):
    '''helper for multiprocessing

    Args:
        src_flat(array(Nx2)): source 'fiber' locations in flat coordinates
        tgt_flat(array(Nx2)) target synapses locations in flat coordinates
        sigma(float): sigma for normal distribution for weights
        closest_count(int): number of fibers in the kd-tree
        rng(np.random.Generator): used for random number generation

    Returns:
        index into src_flat for each of the tgt_flat

    '''
    from scipy.spatial import KDTree
    from scipy.stats import norm

    kd_fibers = KDTree(src_flat)
    distances, sgids = kd_fibers.query(tgt_flat, closest_count)

    # From: 'White matter workflow and recipe creation':
    #  calculate the probabilities that each neuron is mapped to innervates
    #  each connection as a gaussian of the pairwise distances with a
    #  specified variance $\sigma_M$.

    prob = norm.pdf(distances, loc=0, scale=sigma)
    prob = np.nan_to_num(prob)
    idx = utils.choice(prob, rng)

    return sgids[np.arange(len(sgids)), idx]


def assign_groups(src_flat, tgt_flat, sigma, closest_count, seed, n_jobs=-2, chunks=None):
    '''

    Args:
        src_flat(DataFrame with sgid as index): index is used to return sgids
        tgt_flat(array(Nx2)) target synapses locations in flat coordinates
        sigma(float): sigma for normal distribution for weights
        closest_count(int): number of fibers in the kd-tree
        seed(int): used to seed the `np.random.SeedSequence`
        n_jobs(int): number of jobs
        chunks(int): number of chunks to use

    Returns:
        index into src_xy for each of the tgt_xy
    '''
    seed_sequences = np.random.SeedSequence(seed)

    if chunks is None:
        chunks = len(tgt_flat) // 10000 + 1

    p = Parallel(n_jobs=n_jobs,
                 # the multiprocessing backend is 50x faster than 'loky';
                 # I *think* it has to do w/ loky startup being slow GPFS
                 # coupled w/ its use of semaphores, but I haven't been able
                 # to prove that.  For debugging, 'loky' is *much* nicer, use
                 # that when you can
                 backend='multiprocessing',
                 # verbose=51
                 )
    worker = delayed(_assign_groups_worker)
    ids = p(worker(src_flat.values, uv, sigma, closest_count, np.random.default_rng(seed_sequence))
            for uv, seed_sequence in zip(np.array_split(tgt_flat, chunks),
                                         seed_sequences.spawn(chunks)))
    ids = np.concatenate(ids)

    return src_flat.index[ids]


def _calculate_delay_dive(src_cells, syns, conduction_velocity, atlas):
    '''make the 'dive' connection delay

    diving menas that the signal (per SSCXDIS-193):
        'goes straight down into the white matter, then straight towards the
        target region, then up towards the target.'
    '''
    ph_y = atlas.load_data('[PH]y')

    def depth(xyz):
        '''find the depth of position `xyz`'''
        idx = ph_y.positions_to_indices(xyz)
        return ph_y.raw[tuple(idx.T)]

    src_distance = depth(src_cells[utils.XYZ].loc[syns.sgid].to_numpy())
    tgt_distance = depth(syns[utils.XYZ].to_numpy())

    delay = ((src_distance + tgt_distance) / conduction_velocity['intra_region'] +
             _calculate_delay_direct(src_cells, syns, conduction_velocity))

    return delay.astype(np.float32)


def _calculate_delay_direct(src_cells, syns, conduction_velocity):
    '''make the 'direct' (ie: straight) connection delay

    Note: assumes the cells are *directly* connected with inter_region conduction velocity
    '''
    src_cells = src_cells[utils.XYZ].loc[syns.sgid].to_numpy()
    delay = np.linalg.norm(src_cells - syns[utils.XYZ].to_numpy(), axis=1)
    delay /= conduction_velocity['inter_region']
    return delay.astype(np.float32)


def _calculate_delay_streamline(src_cells, syns, streamline_metadata, conduction_velocity, rng):
    '''for all the synapse locations, assign a streamline

    Args:
        src_cells(DataFrame): positions with index of GID
        syns(DataFrame): target synapses positions and sgid
        streamline_metadata(DataFrame): describes the inter-region distance
        between points within the regions; a row within this dataset
        is chosen, and saved so that the streamlines can be visualized
        conduction_velocity(dict): values for inter and intra region velocities
            in um/ms
        rng(np.random.Generator): used for random number generation
    '''
    # pylint: disable=too-many-locals
    START_COLS = ['start_x', 'start_y', 'start_z']
    END_COLS = ['end_x', 'end_y', 'end_z']
    NEEDED_COLS = START_COLS + END_COLS + ['length']

    metadata = streamline_metadata.set_index('path_row')
    src_cells = src_cells[utils.XYZ]

    path_rows = metadata.index.values

    path_rows = rng.choice(path_rows, size=len(syns))
    gid2row = np.vstack((syns.sgid.values, path_rows.astype(np.int64))).T
    gid2row = pd.DataFrame(gid2row, columns=['sgid', 'row']).drop_duplicates()

    metadata = metadata.loc[path_rows, NEEDED_COLS]
    src_start = metadata[START_COLS].values
    src_end = src_cells.loc[syns.sgid].values
    src_distance = np.linalg.norm(src_start - src_end, axis=1)

    tgt_start = metadata[END_COLS].values
    tgt_distance = np.linalg.norm(tgt_start - syns[utils.XYZ].values, axis=1)

    inter_distance = metadata['length'].values

    delay = ((src_distance + tgt_distance) / conduction_velocity['intra_region'] +
             inter_distance / conduction_velocity['inter_region'])

    return delay.astype(np.float32), gid2row


def _load_subsamples(samples_path, side, projection_name):
    '''load all samples for `projection_name`

    Note: filename convenction has to match sampling.py:_subsample_per_source
    '''
    path = os.path.join(samples_path, side, projection_name + '_*.feather')
    files = sorted(glob(path))
    L.debug('load_subsamples: %s -> %s', path, files)
    all_syns = [utils.read_frame(p) for p in files]
    return pd.concat(all_syns, ignore_index=True, sort=False)


def assignment(output,
               config,
               allocations,
               side,
               reverse):
    '''perform assignment

    Args:
        output(str): path to output to
        config(utils.Config): config
        allocations(pd.DataFrame): allocations
        side(str): 'left' or 'right'
        reverse(bool): whether to reverse the order of assignment
    '''
    # pylint: disable=too-many-locals
    assert side in utils.SIDES, f'unknown side: {side}'

    if config.delay_method == 'streamlines':
        streamline_metadata = streamlines.load(output, only_metadata=True)

    count = len(allocations)
    allocations = allocations[['projection_name', 'source_population', 'target_population',
                               'sgids', 'hemisphere', ]]

    src_base_system = {config.recipe.projections_mapping[s]['base_system']
                       for s in allocations['source_population']}
    assert len(src_base_system) == 1, f'want only one src base system: {src_base_system}'
    src_base_system = next(iter(src_base_system))

    src_mapper = mapping.CommonMapper.load(config, src_base_system)
    left_cells, right_cells = partition_cells_left_right(config.get_cells(),
                                                         src_mapper.flat_map.center_line_3d)

    tgt_base_system = {config.recipe.projections_mapping[s][pn]['base_system']
                       for _, pn, s in allocations[['projection_name',
                                                    'source_population']].itertuples()}
    assert len(tgt_base_system) == 1, f'want only one src base system: {tgt_base_system}'
    tgt_base_system = next(iter(tgt_base_system))
    tgt_mapper = mapping.CommonMapper.load(config, tgt_base_system)

    allocations = allocations.to_numpy()
    if reverse:
        allocations = reversed(allocations)

    for i, keys in enumerate(allocations):
        projection_name, source_population, target_population, sgids, hemisphere = keys

        output_path = os.path.join(output, ASSIGNMENT_PATH, side)
        utils.ensure_path(output_path)
        output_path = os.path.join(output_path, projection_name + '.feather')
        if os.path.exists(output_path):
            L.debug('Skipping %s, already have %s', projection_name, output_path)
            continue

        seed = utils.generate_seed(source_population, projection_name, hemisphere)
        L.debug('Assigning %s -> %s (%s of %s): seed: %s',
                projection_name, output_path, i, count, seed)

        # src coordinates in flat space
        src_cells = separate_source_and_targets(
            left_cells, right_cells, sgids, hemisphere, side)
        flat_src_uvs = src_mapper.map_points_to_flat(src_cells[utils.XYZ].to_numpy())
        src_coordinates = src_mapper.map_flat_to_flat(
            source_population, projection_name, flat_src_uvs, utils.is_mirror(side, hemisphere))
        src_coordinates = pd.DataFrame(src_coordinates, index=src_cells.index)

        # tgt synapses in flat space
        syns = _load_subsamples(os.path.join(output, sampling.SAMPLE_PATH), side, projection_name)

        syns = utils.partition_left_right(syns, side, tgt_mapper.flat_map.center_line_3d)
        flat_tgt_uvs = tgt_mapper.map_points_to_flat(syns[utils.XYZ].to_numpy())

        syns['sgid'] = assign_groups(
            src_coordinates,
            flat_tgt_uvs,
            math.sqrt(
                config.recipe.projections_mapping[source_population][projection_name]['variance']),
            config.config['assignment']['closest_count'],
            seed=seed)

        if config.delay_method == 'streamlines':
            source_region = utils.population2region(config.recipe.populations, source_population)
            target_region = utils.population2region(config.recipe.populations, target_population)
            source_region, target_region = source_region, target_region  # trick pylint
            metadata = streamline_metadata.query('source == @source_region and '
                                                 'target == @target_region and '
                                                 'target_side == @side and '
                                                 'hemisphere == @hemisphere'
                                                 )

            syns['delay'], gid2row = _calculate_delay_streamline(
                src_cells,
                syns,
                metadata,
                config.config['conduction_velocity'],
                config.rng
            )

            utils.write_frame(output_path.replace('.feather', '_gid2row.feather'), gid2row)
        elif config.delay_method == 'dive':
            syns['delay'] = _calculate_delay_dive(
                src_cells, syns, config.config['conduction_velocity'], config.atlas)
        else:
            syns['delay'] = _calculate_delay_direct(
                src_cells, syns, config.config['conduction_velocity'])

        syns['delay'] += config.neural_transmitter_release_delay
        utils.write_frame(output_path, syns)
