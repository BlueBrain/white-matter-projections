'''operations related to 'micro' connectome, as defined in the white-matter paper

This includes
    Allocations:
        The 'p-types' section of the white-matter recipe defines the
        'allocations' as fractions based on the source population and the target
        projection.  The idea is to then take the gids in the source population,
        and divide them into groups based on their target projection fraction.
        Only first order interactions are accounted for in the recipe, with
        potentially a multiplier to increase from the 'baseline' version.

        For example, a projection target group A with fraction 0.5, and B with
        fraction 0.5 will have a baseline overlap/interaction of 0.5 * 0.5.  If
        there is a multipler, that would increase the baseline by that multiple.

'''
from functools import partial
import itertools as it
import logging
import math
import os

import h5py
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from white_matter_projections import sampling, utils, streamlines, mapping


ASSIGNMENT_PATH = 'ASSIGNED'
L = logging.getLogger(__name__)


def save_allocations(allocations_path, allocations):
    '''save allocations to an h5 file'''
    with h5py.File(allocations_path, 'w') as h5:
        _serialize_allocations(h5, allocations)


def load_allocations(allocations_path, projections_mapping=None):
    '''load allocations from an h5 file'''
    with h5py.File(allocations_path, 'r') as h5:
        allocations = _deserialize_allocations(h5)

    if projections_mapping:
        allocations = transpose_allocations(allocations, projections_mapping)

    return allocations


def _deserialize_allocations(h5):
    '''return a dictionary with allocations

    Format of dictionary is:
        source_population -> projection_name -> array of source_gids
    '''
    ret = {}
    for source_population in h5:
        ret[source_population] = projection_names = {}
        for projection_name in h5[source_population]:
            projection_names[projection_name] = h5[source_population][projection_name][:]
    return ret


def _serialize_allocations(h5, allocations):
    ''' serialize allocations from dictionary to h5

        H5 layout is:
            /{source_population, ...}/{projection_name, ....}/gids
    '''
    for source_population, allocation in allocations.items():
        grp = h5.create_group(source_population)
        for projection_name, gids in allocation.items():
            grp.create_dataset(projection_name, data=gids, dtype='i')


def transpose_allocations(allocations, projections_mapping):
    '''return allocations, in a dataframe, keyed by source rather than target_populiations

    target_population -> {('source_population', 'projection_name'): sgids,

    }

    recipe.projections_mapping: source_population -> {'vertices',
                                                      'projection_name' -> sgids
                                                            ...
                                                      }
    '''
    data = []
    for source_population, allocation in allocations.items():
        source_mapping = projections_mapping[source_population]
        for projection_name, sgids in allocation.items():
            if projection_name not in source_mapping:
                L.warning('projection %s missing from source %s',
                          projection_name, source_population)
                continue
            target_population = source_mapping[projection_name]['target_population']
            data.append((projection_name, source_population, target_population, np.array(sgids)))

    columns = ['projection_name', 'source_population', 'target_population', 'sgids']
    return pd.DataFrame(data, columns=columns)


def _make_numeric_groups(total_counts, overlap_counts):
    '''convert to numeric names to make sets and other operations simpler/more efficient

    Args:
        total_counts(dict): group_name -> count (int)
        overlap_counts(dict): tuple(group_name, group_name) -> how many overlapping sgids there are
    Returns:
        names: list of names
        name_map: map from original name to number
        total_counts: original total_counts with names converted to numbers
        overlap_counts: original overlap_counts with names converted to numbers
    '''
    names = sorted(total_counts)
    name_map = {k: v for k, v in zip(names, range(len(names)))}
    total_counts = [total_counts[k] for k in names]
    overlap_counts = {tuple(sorted(name_map[k] for k in key)): v
                      for key, v in overlap_counts.items()}

    return names, name_map, total_counts, overlap_counts


def _allocate_groups(total_counts, overlap_counts, gids):
    '''Greedy allocation of gids to groups

    Args:
        total_counts(dict): group_name -> count (int)
        overlap_counts(dict): tuple(group_name, group_name) -> how many overlapping sgids there are
        gids(np.array of gids): gids to allocate to different groups

    Returns:
        dict: group_name -> gids

    Note: This is not optimal by any means; I haven't tried to prove, but my gut
    feeling is that this partitioning problem is NP-complete, and because some
    of the groupings in the recipe are large (ie: 70+ [MOs_5it]), the number of
    first order interactions is thus (70 choose 2 = 2415), and being able to
    satisfy all the allocations simultaneously becomes hard.  I have tried using
    Google's `ortools` to SAT solve it, but even that was blowing up just
    entering the problem into solver.

    For the 1p10 recipe, the breakdown looks like this (for the top ones):
        source_pop  total_counts  overlap_counts  total
            MOs_23            46            1035   1081
          ORBl_5it            46            1035   1081
           TEa_5it            49            1176   1225
         ORBvl_5it            50            1225   1275
           MOp_5it            53            1378   1431
          ACAd_5it            57            1596   1653
       SSp-bfd_5it            62            1891   1953
           SSs_5it            65            2080   2145
           MOs_5it            75            2775   2850

    Refer to test with a case that fails.
    '''
    # pylint: disable=too-many-locals
    choice = partial(np.random.choice, replace=False)

    names, _, total_counts, overlap_counts = _make_numeric_groups(
        total_counts, overlap_counts)

    ret = {}
    for key, count in enumerate(total_counts):
        ret[names[key]] = choice(gids, size=count)

    for g0, g1 in sorted(overlap_counts):
        ret_g0 = ret[names[g0]]
        ret_g1 = ret[names[g1]]

        set_g0, set_g1 = set(ret_g0), set(ret_g1)
        current_overlap = len(set_g0 & set_g1)
        half_needed = (overlap_counts[g0, g1] - current_overlap) // 2

        if half_needed > 0:
            ret_g0[:half_needed] = choice(list(set_g1 - set_g0), size=half_needed)
            ret_g1[:half_needed] = choice(list(set_g0 - set_g1), size=half_needed)

            # reduce chance these overlapping gids are removed by subsequent interactions
            np.random.shuffle(ret_g0)
            np.random.shuffle(ret_g1)

        assert len(ret_g0) == total_counts[g0]
        assert len(ret_g1) == total_counts[g1]
        # XXX else: can be negative? we should drop the overlap potentially...

    return ret


def _ptype_to_counts(cell_count, ptype, interactions):
    '''extract from the ptypes, the cell counts required for each group

    Args:
        cell_count(int):
        ptype():
        interactions():
    Returns:
        tuple(total_counts, overlap_counts):
            total_counts:
            overlap_counts:
    '''
    ptype = ptype.set_index('projection_name')
    total_counts = {projection_name: int(cell_count * fraction)
                    for projection_name, fraction in ptype['fraction'].items()}

    overlap_counts = {}
    for g0, g1 in it.combinations(total_counts, 2):
        if g0 == g1:
            continue

        key = tuple(sorted((g0, g1)))

        weight = 1.
        if interactions is not None and g0 in interactions.index and g1 in interactions.index:
            weight = interactions.loc[g0, g1]

        if isinstance(ptype.loc[g0], pd.Series):
            base_fraction = ptype.loc[g0].fraction * ptype.loc[g1].fraction
        else:
            fraction0 = ptype.loc[g0].fraction
            fraction1 = ptype.loc[g1].fraction
            msg = 'Should only have one fraction%d %s'
            assert len(fraction0.unique()) == 1, msg % (0, ptype.loc[g0].fraction.unique())
            assert len(fraction1.unique()) == 1, msg % (1, ptype.loc[g1].fraction.unique())
            base_fraction = fraction0[0] * fraction1[0]

        overlap_counts[key] = int(cell_count * weight * base_fraction)

    return total_counts, overlap_counts


def _append_side_to_regions(regions):
    '''due to hemisphere not being encoded in atlas, MVD had hemisphere appended

    # XXX: NCX-176 hack
    '''
    ret = []
    for region in regions:
        ret.append(region + '@left')
        ret.append(region + '@right')
    return ret


def get_gids_by_population(populations, get_cells, source_population):
    '''for a `population`, get all the gids from `cells` that are in that population'''
    population = populations.set_index('population').loc[source_population]

    if isinstance(population, pd.DataFrame):
        region_names = set(population.region)
        layers = set(population.layer)
        population_filter = population.population_filter[0]
    else:
        region_names = {population.region}
        layers = {population.layer}
        population_filter = population.population_filter

    layers = [int(l[1]) for l in layers]

    cells = get_cells(population_filter)
    region_names = _append_side_to_regions(region_names)
    gids = cells.query('region in @region_names and layer in @layers').index.values
    return gids


def allocate_projections(recipe, get_cells):
    '''Allocate *all* projections in recipe based on ptypes in recipe

    Args:
        recipe(MacroConnections): recipe
        cells(DataFrame): as returned by bluepy.v2.Circuit.cells.get()

    Returns:
        dict of source_population -> dict projection_name -> np array of source gids
    '''
    ret = {}

    ptypes = recipe.ptypes.merge(recipe.populations,
                                 left_on='source_population', right_on='population')

    skipped_populations = []
    for source_population, ptype in ptypes.groupby('source_population'):
        if not len(ptype):
            skipped_populations.append(source_population)
            continue

        L.info('Allocating for source population: %s', source_population)

        gids = get_gids_by_population(recipe.populations, get_cells, source_population)
        interactions = recipe.ptypes_interaction_matrix.get(source_population, None)

        total_counts, overlap_counts = _ptype_to_counts(len(gids), ptype, interactions)
        ret[source_population] = _allocate_groups(total_counts, overlap_counts, gids)

    if skipped_populations:
        L.warning('Skipping populations because empty p-types: %s', sorted(skipped_populations))

    return ret


def allocation_stats(recipe, get_cells, allocations, source_population):
    '''calculate the expected and allocated gids

    useful to verify how well `allocate_projections` works

    Args:
        recipe(MacroConnections): the recipe
        cells(DataFrame): potential source cells, from circuit
        allocations(DataFrame): allocated cells
        source_population(str): name of source population

    Returns:
        tuple of dfs:
            counts: for each group, 'expected' and 'allocated' counts
            interactions: first order interactions w/ 'expected' and 'allocated'
    '''
    # pylint: disable=too-many-locals
    ptypes = (recipe.ptypes
              .merge(recipe.populations, left_on='source_population', right_on='population')
              .query('source_population == @source_population')
              )
    interactions = recipe.ptypes_interaction_matrix.get(source_population, None)
    gids = get_gids_by_population(recipe.populations, get_cells, source_population)

    total_counts, overlap_counts = _ptype_to_counts(len(gids), ptypes, interactions)

    allocations = (allocations
                   .query('source_population == @source_population')
                   .set_index('projection_name')
                   )

    counts = pd.DataFrame.from_dict(total_counts, orient='index', columns=['expected'])
    counts['allocated'] = allocations.sgids.apply(len)
    counts['absolute_difference'] = np.abs((counts.allocated - counts.expected) / counts.expected)

    rows = []
    for (g0, g1), overlap_count in overlap_counts.items():
        actual_overlap = len(set(allocations.loc[g0].sgids) & set(allocations.loc[g1].sgids))
        rows.append((g0, g1, overlap_count, actual_overlap))
    df = pd.DataFrame(rows, columns=['Group0', 'Group1', 'expected', 'actual'])
    df['absolute_difference'] = np.abs((df.actual - df.expected) / df.expected)

    return counts, df


def partition_cells_left_right(cells, center_line_3d):
    '''return left and right cells'''
    left_cells = cells[cells.z <= center_line_3d]
    right_cells = cells[cells.z > center_line_3d]
    return left_cells, right_cells


def partition_syns(syns, side, center_line_3d):
    '''return synapses from `side` based on `center_line_3d`'''
    assert side in ('left', 'right', )
    if side == 'right':
        syns_mask = center_line_3d < syns.z
    else:
        syns_mask = syns.z <= center_line_3d
    return syns[syns_mask]


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
    assert hemisphere in ('ipsi', 'contra', )
    assert side in ('left', 'right', )

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


def _assign_groups(src_flat, tgt_flat, sigma, closest_count):
    '''helper for multiprocessing

    Args:
        src_flat(array(Nx2)): source 'fiber' locations in flat coordinates
        tgt_flat(array(Nx2)) target synapses locations in flat coordinates
        sigma(float): sigma for normal distribution for weights
        closest_count(int): number of fibers in the kd-tree

    Returns:
        index into src_flat for each of the tgt_flat

    '''
    from scipy.spatial import KDTree
    from scipy.stats import norm
    from projectionizer.utils import choice

    kd_fibers = KDTree(src_flat)
    distances, sgids = kd_fibers.query(tgt_flat, closest_count)

    # From: 'White matter workflow and recipe creation':
    #  calculate the probabilities that each neuron is mapped to innervates
    #  each connection as a gaussian of the pairwise distances with a
    #  specified variance $\sigma_M$.

    prob = norm.pdf(distances, 0, sigma)
    prob = np.nan_to_num(prob)
    idx = choice(prob)

    return sgids[np.arange(len(sgids)), idx]


def assign_groups(src_flat, tgt_flat, sigma, closest_count, n_jobs=-2, chunks=None):
    '''

    Args:
        src_flat(DataFrame with sgid as index): index is used to return sgids
        tgt_flat(array(Nx2)) target synapses locations in flat coordinates
        sigma(float): sigma for normal distribution for weights
        closest_count(int): number of fibers in the kd-tree
        n_jobs(int): number of jobs

    Returns:
        index into src_xy for each of the tgt_xy
    '''

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
    worker = delayed(_assign_groups)
    ids = p(worker(src_flat.values, uv, sigma, closest_count)
            for uv in np.array_split(tgt_flat, chunks))
    ids = np.concatenate(ids)

    return src_flat.index[ids]


def _calculate_delay_direct(src_cells, syns, conduction_velocity):
    '''make the 'direct' (ie: straight) connection delay

    Note: assumes the cells are *directly* connected with inter_region conduction velocity
    '''
    src_cells = src_cells[utils.XYZ].loc[syns.sgid].to_numpy()
    delay = np.linalg.norm(src_cells - syns[utils.XYZ].to_numpy(), axis=1)
    delay /= conduction_velocity['inter_region']
    return delay.astype(np.float32)


def _calculate_delay_streamline(src_cells, syns, streamline_metadata, conduction_velocity):
    '''for all the synapse locations, assign a streamline

    Args:
        src_cells(DataFrame): positions with index of GID
        syns(DataFrame): target synapses positions and sgid
        streamline_metadata(DataFrame): describes the inter-region distance
        between points within the regions; a row within this dataset
        is chosen, and saved so that the streamlines can be visualized
        conduction_velocity(dict): values for inter and intra region velocities
        in um/ms
    '''
    # pylint: disable=too-many-locals
    START_COLS = ['start_x', 'start_y', 'start_z']
    END_COLS = ['end_x', 'end_y', 'end_z']
    NEEDED_COLS = START_COLS + END_COLS + ['length']

    metadata = streamline_metadata.set_index('path_row')
    src_cells = src_cells[utils.XYZ]

    path_rows = metadata.index.values

    path_rows = np.random.choice(path_rows, size=len(syns))
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


def assignment(output, config, allocations, projections_mapping, side,
               closest_count, reverse, use_streamlines):
    '''perform ssignment'''
    # pylint: disable=too-many-locals
    samples_path = os.path.join(output, sampling.SAMPLE_PATH)

    left_cells, right_cells = partition_cells_left_right(config.get_cells(),
                                                         config.flat_map.center_line_3d)

    if use_streamlines:
        streamline_metadata = streamlines.load(output, only_metadata=True)

    mapper = mapping.CommonMapper.load_default(config)

    conduction_velocity = config.config['conduction_velocity']

    count = len(allocations)
    alloc = allocations[['projection_name', 'source_population', 'target_population',
                         'sgids', 'hemisphere', ]].values
    if reverse:
        alloc = reversed(alloc)

    for i, keys in enumerate(alloc):
        projection_name, source_population, target_population, sgids, hemisphere = keys

        output_path = os.path.join(output, ASSIGNMENT_PATH, side)
        utils.ensure_path(output_path)
        output_path = os.path.join(output_path, projection_name + '.feather')
        if os.path.exists(output_path):
            L.debug('Skipping %s, already have %s', projection_name, output_path)
            continue

        L.debug('Assigning %s -> %s (%s of %s)', projection_name, output_path, i, count)

        # src coordinates in flat space
        src_cells = separate_source_and_targets(
            left_cells, right_cells, sgids, hemisphere, side)

        flat_src_uvs = mapper.map_points_to_flat(src_cells[utils.XYZ].values)
        src_coordinates = mapper.map_flat_to_flat(
            source_population, projection_name, flat_src_uvs, utils.is_mirror(side, hemisphere))
        src_coordinates = pd.DataFrame(src_coordinates, index=src_cells.index)

        # tgt synapses in flat space
        syns = utils.read_frame(os.path.join(samples_path, side, projection_name + '.feather'))
        syns = partition_syns(syns, side, config.flat_map.center_line_3d)
        flat_tgt_uvs = mapper.map_points_to_flat(syns[utils.XYZ].values)

        sigma = math.sqrt(projections_mapping[source_population][projection_name]['variance'])
        syns['sgid'] = assign_groups(src_coordinates, flat_tgt_uvs, sigma, closest_count)

        if use_streamlines:
            source_region = utils.population2region(config.recipe.populations, source_population)
            target_region = utils.population2region(config.recipe.populations, target_population)
            source_region, target_region = source_region, target_region  # trick pylint
            metadata = streamline_metadata.query('source == @source_region and '
                                                 'target == @target_region and '
                                                 'target_side == @side and '
                                                 'hemisphere == @hemisphere'
                                                 )
            syns['delay'], gid2row = _calculate_delay_streamline(src_cells,
                                                                 syns,
                                                                 metadata,
                                                                 conduction_velocity)
            utils.write_frame(output_path.replace('.feather', '_gid2row.feather'), gid2row)
        else:
            syns['delay'] = _calculate_delay_direct(src_cells, syns, conduction_velocity)
        utils.write_frame(output_path, syns)
