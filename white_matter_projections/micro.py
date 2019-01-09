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

import h5py
import numpy as np
import pandas as pd


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
            assert len(fraction0.unique()) == 1, \
                'Should only have one fraction0 %s' % ptype.loc[g0].fraction.unique()
            assert len(fraction1.unique()) == 1, \
                'Should only have one fraction1 %s' % ptype.loc[g1].fraction.unique()
            base_fraction = fraction0[0] * fraction1[0]

        overlap_counts[key] = int(cell_count * weight * base_fraction)

    return total_counts, overlap_counts


def get_gids_py_population(populations, cells, source_population):
    '''for a `population`, get all the gids from `cells` that are in that population'''
    source_population = source_population  # trick pylint so variable can be used in query
    region_names = set(populations.query('population == @source_population').subregion)

    layers = populations.query('population == @source_population').layer
    layers = [int(l[1]) for l in layers]

    region_names = region_names  # trick pylint so variable can be used in query
    gids = cells.query('region in @region_names and layer in @layers').index.values
    return gids


def allocate_projections(recipe, cells):
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

        gids = get_gids_py_population(recipe.populations, cells, source_population)
        interactions = recipe.ptypes_interaction_matrix.get(source_population, None)

        total_counts, overlap_counts = _ptype_to_counts(len(gids), ptype, interactions)
        ret[source_population] = _allocate_groups(total_counts, overlap_counts, gids)

    if skipped_populations:
        L.warning('Skipping populations because empty p-types: %s', sorted(skipped_populations))

    return ret


def allocation_stats(ptypes, populations, ptypes_interaction_matrix, cells,
                     allocations, population):
    '''calculate the expected and allocated gids

    useful to verify how well `allocate_projections` works

    Args:
        ptypes
        populations
        ptypes_interaction_matrix
        cells
        allocations
        population

    Returns:
        tuple of dfs:
            counts: for each group, 'expected' and 'allocated' counts
            interactions: first order interactions w/ 'expected' and 'allocated'
    '''
    # pylint: disable=too-many-locals
    ptypes = (ptypes
              .merge(populations, left_on='source_population', right_on='population')
              .query('source_population == @population')
              )
    interactions = ptypes_interaction_matrix.get(population, None)
    gids = get_gids_py_population(populations, cells, population)

    total_counts, overlap_counts = _ptype_to_counts(len(gids), ptypes, interactions)

    allocations = (allocations
                   .query('source_population == @population')
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
