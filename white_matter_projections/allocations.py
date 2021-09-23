''' two implementations of allocations - 'greedy' and 'stochastic_tree_model'

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
from collections import defaultdict
from enum import Enum
import itertools as it
import logging
import warnings

import h5py
import numpy as np
import pandas as pd

from white_matter_projections import micro, ptypes_generator

L = logging.getLogger(__name__)


class Algorithm(Enum):
    '''
    Class holding the algorithms that one
    can use to allocate gids to target groups.
    '''
    STOCHASTIC_TREE_MODEL = 0
    GREEDY = 1


def save_allocations(allocations_path, allocations):
    '''save allocations to an h5 file'''
    with h5py.File(allocations_path, 'w') as h5:
        _serialize_allocations(h5, allocations)


def load_allocations(allocations_path, projections_mapping=None):
    '''load allocations from an h5 file'''
    with h5py.File(allocations_path, 'r') as h5:
        allocations = _deserialize_allocations(h5)

    if projections_mapping:
        allocations = _transpose_allocations(allocations, projections_mapping)

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
            grp.create_dataset(projection_name, data=np.sort(gids), dtype='i')


def _transpose_allocations(allocations, projections_mapping):
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
    name_map = dict(zip(names, range(len(names))))
    total_counts = [total_counts[k] for k in names]
    overlap_counts = {tuple(sorted(name_map[k] for k in key)): v
                      for key, v in overlap_counts.items()}

    return names, name_map, total_counts, overlap_counts


def _create_completed_interaction_matrix(recipe_interaction_matrix, fractions):
    '''Convert the recipe's interaction matrix into a 2D float array. Ones replace missing entries.

    The interaction matrix is made compliant, in a weak sense,
    with the ptype-generating tree model before it is used to build such a tree.
    This only means that the inverse of each interaction strength can be interpreted
    as the innervation probability of an internal node of the model.
    The output matrix dimensions match the size of the fractions list, i.e.,
    the number of target regions.

    Missing entries are filled with ones. Diagonal entries are zeroed.

    Args:
        recipe_interaction_matrix(pandas.DataFrame): DataFrame of float values where rows
            and columns are labelled by target region names. The entries of the matrix are the
            stastistical interaction strengths
                I_S(A, B) := P(S --> A) P(S --> B) / P(S --> A intersection B)
            with S the source region and (A, B) any pair of target regions.
            Some target pairs may be missing with respect to the full list of target regions
            given in the fractions dict. For each missing target region A, the values I_S(A, .)
            are set to 1.0. This means that A can be innervated by S independently of any other
            target region.
            The DataFrame recipe_interaction_matrix can be None, in which case the innervations of
            the different target regions are assumed to all be independent.
        fractions(list): list of innervation probabilities for the target regions, interpreted
            as the expected fractions of the source neurons that will innervate the corresponding
            target regions. The mapping between indices and names is provided by regions.

    Returns:
        completed_interaction_matrix(pandas.DataFrame): a 2D float array holding the values
            of I_S(A, B) for all pairs of target region indices (i_A, i_B) in accordance with
            the input name map.
    '''
    regions = fractions.index

    if recipe_interaction_matrix is None:
        df = pd.DataFrame()
    else:
        df = recipe_interaction_matrix

    # Independence is assumed for missing entries.
    # Note however that the interpretation in terms of probabilities
    # imposes much more constraints on the matrices than those above.
    # These constraints are much more involved but could be checked
    # a posteriori once the tree is built.
    # These lines fills all missing rows and columns with ones
    df = df.reindex(index=regions, columns=regions, fill_value=1)
    np.fill_diagonal(df.values, 0)

    # Make sure the interaction matrix is compatible with its interpretation
    # as the inverse of the innervation probability of the lowest common ancestor in
    # the tree model.
    # See formula (4) of https://www.nature.com/articles/s41467-019-11630-x.
    for i, j in it.permutations(df.columns, 2):
        df.loc[i, j] = min(1.0 / max(fractions[i], fractions[j]), df.loc[i, j])

    return df


def _ptypes_to_target_groups(ptypes_array, regions, gids):
    '''Convert an array of ptypes into groups of gids labelled by target regions.

    Args:
        ptypes_array(list): one-dimensional array of ptypes.
            A ptype is a set of target region indices.
            The map between target region names and indices is given by regions.
        region_names(list): list of target region names
        gids(numpy.ndarray): one-dimensional array of integer identifiers for
            the neurons of the source region of interest. These neurons identifiers
            will be allocated to the different target groups based on the input array of ptypes.
        regions(dict): dict whose keys are integer indices and
            whose values are target region names.

    Returns:
        target_groups(dict): dict whose keys are the target regions names and whose values are
            one-dimensional arrays of gids allocated from the input gids array.
    '''
    target_groups = defaultdict()
    for region_name in regions.values():
        target_groups[region_name] = []
    for gid, ptype in zip(gids, ptypes_array):
        for region_index in ptype:
            region_name = regions[region_index]
            target_groups[region_name].append(gid)

    return dict(target_groups)


def _allocate_gids_randomly_to_targets(targets, recipe_interaction_matrix, gids, rng):
    '''Random allocation of gids to target groups in accordance to the tree
    model of 'A null model of the mouse whole-neocortex micro-connectome',
    see Section 'A model to generate projection types' of
    https://www.nature.com/articles/s41467-019-11630-x.

    Args:
        targets(pandas.DataFrame): DataFrame holding the projection_name section of the
            recipe for a given source of interest.
        recipe_interaction_matrix(pandas.DataFrame): DataFrame holding the
            interaction matrix section of the recipe for a given source of interest.
        gids(numpy.ndarray): 1D array of gids to be allocated
            to different target populations.
        rng(np.random.Generator): used for random number generation

    Returns:
        target_groups(dict): dict whose keys are target region names and whose values are
            list of allocated gids taken from the input gids array.
    '''
    target_fractions = targets.set_index('projection_name')['fraction']
    # We need to handle the missing matrix entries as well.
    full_interaction_matrix = _create_completed_interaction_matrix(
        recipe_interaction_matrix, target_fractions)
    generator = ptypes_generator.PtypesGenerator(
        list(target_fractions), full_interaction_matrix.values, rng)
    # We create a generator instance based on the tree model.
    ptypes = generator.generate_random_ptypes(len(gids))
    regions = dict(enumerate(target_fractions.keys()))
    target_groups = _ptypes_to_target_groups(ptypes, regions, gids)

    return target_groups


def allocate_gids_to_targets(
    targets, recipe_interaction_matrix, gids, rng, algorithm=Algorithm.GREEDY
):
    '''Allocation of gids to target groups

    Args:
        targets(pandas.DataFrame): DataFrame holding the projection_name section of the
            recipe for a given source of interest.
        recipe_interaction_matrix(pandas.DataFrame): DataFrame holding the
            interaction matrix section of the recipe for a given source of interest.
        gids(numpy.ndarray): 1D array of gids to be allocated to different target populations.
            algorithm(string): (optional) algorithm to be used so as to populate target groups.
            Defaults to STOCHASTIC_TREE_MODEL, the allocation schema based on the tree model of
            'A null model of the mouse whole-neocortex micro-connectome' by M. Reimann et al.
        rng(np.random.Generator): used for random number generation

    Returns:
        target_groups(dict): dict whose keys are target region names and whose values are
            list of allocated gids taken from the input gids array.
    '''
    if not isinstance(algorithm, Algorithm):
        raise ValueError(
            ' The algorithm {} is not supported. For gids allocation to target groups,'
            ' you can use one of the following options: {}'.format(algorithm, list(Algorithm)))
    algorithm_function = {
        Algorithm.STOCHASTIC_TREE_MODEL: _allocate_gids_randomly_to_targets,
        Algorithm.GREEDY: _allocate_gids_greedily_to_targets
    }
    return algorithm_function[algorithm](targets, recipe_interaction_matrix, gids, rng)


def _allocate_gids_greedily_to_targets(targets, recipe_interaction_matrix, gids, rng):
    '''Greedy allocation of gids to target groups

    Args:
        targets(pandas.DataFrame): DataFrame holding the projection_name section of the
            recipe for a given source of interest.
            recipe_interaction_matrix(pandas.DataFrame): DataFrame holding
            the interaction matrix section of the recipe for a given source of interest.
        gids(np.array of gids): gids to allocate to different target populations
        rng(np.random.Generator): used for random number generation

    Returns:
        target_groups(dict): dict whose keys are target region names and whose values are
            list of allocated gids taken from the input gids array.
    '''
    warnings.warn(
        ' The function _allocate_gids_greedily_to_targets will be deprecated in version 0.0.2.'
        ' It is currently used for the benchmarking of _allocate_gids_randomly_to_targets.'
        ' Use _allocate_gids_randomly_to_targets instead.',
        PendingDeprecationWarning
    )
    total_counts, overlap_counts = _ptype_to_counts(len(gids), targets, recipe_interaction_matrix)
    target_groups = _greedy_gids_allocation_from_counts(total_counts, overlap_counts, gids, rng)

    return target_groups


def _greedy_gids_allocation_from_counts(total_counts, overlap_counts, gids, rng):
    '''Greedy allocation of gids to target groups

    Args:
        total_counts(dict): dictionary whose keys are target region names
            and whose values are the expected sizes of the corresponding target groups.
            overlap_counts(dict): dictionary whose keys are pairs target region names
            and whose values are the expected overlap counts of the corresponding
            target groups.
        gids(np.array of gids): gids to allocate to different target populations
        rng(np.random.Generator): used for random number generation

    Returns:
        target_groups(dict): dict whose keys are target region names and whose values are
            list of allocated gids taken from the input gids array.

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

    choice = partial(rng.choice, replace=False)

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
            rng.shuffle(ret_g0)
            rng.shuffle(ret_g1)

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


def allocate_projections(recipe, get_cells, rng):
    '''Allocate *all* projections in recipe based on ptypes in recipe

    Args:
        recipe(MacroConnections): recipe
        cells(pandas.DataFrame): as returned by bluepy.Circuit.cells.get()
        rng(np.random.Generator): used for random number generation

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

        L.info('Allocating for source population: %s...', source_population)

        gids = micro.get_gids_by_population(recipe.populations, get_cells, source_population)
        interaction_matrix = recipe.ptypes_interaction_matrix.get(source_population, None)

        ret[source_population] = allocate_gids_to_targets(ptype, interaction_matrix, gids, rng)

        used = set.union(*map(set, ret[source_population].values()))
        L.info('... has %d gids, %d used, %0.3f',
               len(gids), len(used), len(used) / float(len(gids)))

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
    gids = micro.get_gids_by_population(recipe.populations, get_cells, source_population)

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
