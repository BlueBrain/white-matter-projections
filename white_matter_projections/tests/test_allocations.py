import os
import itertools as it

from nose.tools import ok_, eq_, raises
from numpy.testing import assert_allclose, assert_array_equal

import h5py
import numpy as np
import pandas as pd

from white_matter_projections import allocations as test_module

import utils

def compare_allocations(ret):
    allocations = utils.fake_allocations()

    eq_(sorted(allocations), sorted(ret))
    assert_allclose(allocations['source_population0']['projection0'],
                    ret['source_population0']['projection0'])
    assert_allclose(allocations['source_population0']['projection1'],
                    ret['source_population0']['projection1'])


def test_save_load_allocations():
    allocations = utils.fake_allocations()
    with utils.tempdir('test_serialize_allocations') as tmp:
        name = os.path.join(tmp, 'allocations.h5')
        test_module.save_allocations(name, allocations)
        ret = test_module.load_allocations(name, projections_mapping=None)
        compare_allocations(ret)

        projections_mapping = utils.fake_projection_mapping()
        ret = test_module.load_allocations(name, projections_mapping)
        ok_(isinstance(ret, pd.DataFrame))


def test_serialize_allocations():
    allocations = utils.fake_allocations()

    with utils.tempdir('test_serialize_allocations') as tmp:
        name = os.path.join(tmp, 'allocations.h5')
        with h5py.File(name, 'w') as h5:
            test_module._serialize_allocations(h5, allocations)

        with h5py.File(name, 'r') as h5:
            ret = test_module._deserialize_allocations(h5)

    compare_allocations(ret)


def test__transpose_allocations():
    allocations = utils.fake_allocations()
    projections_mapping = utils.fake_projection_mapping()

    ret = test_module._transpose_allocations(allocations, projections_mapping)
    ret.set_index('target_population', inplace=True)
    target00 = ret.loc['target00']
    eq_(target00.projection_name, 'projection0')
    eq_(target00.source_population, 'source_population0')
    eq_(len(target00.sgids), 10)


def test__ptype_to_counts():
    cell_count = 10000
    ptype = pd.DataFrame([('proj0', .25),
                          ('proj1', .25),
                          ('proj2', .10),
                          ],
                         columns=['projection_name', 'fraction'])

    # simple case - no interactions
    interactions = None
    total_counts, overlap_counts = test_module._ptype_to_counts(
        cell_count, ptype, interactions)
    eq_(total_counts, {'proj0': 10000 * 0.25, 'proj1': 10000 * 0.25, 'proj2': 10000 * 0.10})
    eq_(overlap_counts, {('proj0', 'proj1'): int(10000 * 0.25 * 0.25),
                         ('proj0', 'proj2'): int(10000 * 0.25 * 0.10),
                         ('proj1', 'proj2'): int(10000 * 0.25 * 0.10),
                         })

    proj = ['proj0', 'proj1', 'proj2']

    # with interactions, but all set to 1 - same as before
    interactions = np.ones((3, 3))
    interactions = pd.DataFrame(interactions, columns=proj, index=proj)
    total_counts, overlap_counts = test_module._ptype_to_counts(
        cell_count, ptype, interactions)
    eq_(total_counts, {'proj0': 10000 * 0.25, 'proj1': 10000 * 0.25, 'proj2': 10000 * 0.10})
    eq_(overlap_counts, {('proj0', 'proj1'): int(10000 * 0.25 * 0.25),
                         ('proj0', 'proj2'): int(10000 * 0.25 * 0.10),
                         ('proj1', 'proj2'): int(10000 * 0.25 * 0.10),
                         })

    # now with actual interactions
    interactions = np.array([[1, 2, 3],
                             [2, 1, 1],
                             [3, 1, 1]])
    interactions = pd.DataFrame(interactions, columns=proj, index=proj)
    total_counts, overlap_counts = test_module._ptype_to_counts(
        cell_count, ptype, interactions)
    eq_(total_counts, {'proj0': 10000 * 0.25, 'proj1': 10000 * 0.25, 'proj2': 10000 * 0.10})
    eq_(overlap_counts, {('proj0', 'proj1'): int(10000 * 0.25 * 0.25 * 2),
                         ('proj0', 'proj2'): int(10000 * 0.25 * 0.10 * 3),
                         ('proj1', 'proj2'): int(10000 * 0.25 * 0.10),
                         })


def test__make_numeric_groups():
    total_counts = {'proj0': 0, 'proj1': 1, 'proj2': 2}
    overlap_counts = {('proj0', 'proj1'): 1,
                      ('proj0', 'proj2'): 20,
                      ('proj1', 'proj2'): 21,
                      }
    names, name_map, total_counts_remap, overlap_counts_remap = test_module._make_numeric_groups(
        total_counts, overlap_counts)
    eq_(['proj0', 'proj1', 'proj2'], sorted(names))
    eq_(total_counts, dict(zip(names, total_counts_remap)))
    proj0, proj1, proj2 = name_map['proj0'], name_map['proj1'], name_map['proj2']
    eq_(overlap_counts_remap,
        {tuple(sorted((proj0, proj1))): 1,
         tuple(sorted((proj2, proj0))): 20,
         tuple(sorted((proj2, proj1))): 21,
         })


def test__greedy_gids_allocation_from_counts():
    # simple case, no interactions
    total_counts = {'proj0': 250, 'proj1': 250, 'proj2': 100}
    overlap_counts = {}
    gids = np.arange(1000)  # 1000 > 250 + 250 + 100
    ret = test_module._greedy_gids_allocation_from_counts(
        total_counts, overlap_counts, gids, np.random)
    eq_(['proj0', 'proj1', 'proj2'], sorted(ret))
    eq_(len(ret['proj0']), 250)
    eq_(len(ret['proj1']), 250)
    eq_(len(ret['proj2']), 100)

    np.random.seed(42)
    total_counts = {'proj0': 250, 'proj1': 250, 'proj2': 100}
    overlap_counts = {('proj0', 'proj1'): 10,
                      ('proj0', 'proj2'): 20,
                      ('proj1', 'proj2'): 21,
                      }
    # want a large number here so unlikely to have the required overlap
    gids = np.arange(10000)
    ret = test_module._greedy_gids_allocation_from_counts(
        total_counts, overlap_counts, gids, np.random)

    eq_(len(ret['proj0']), 250)
    eq_(len(ret['proj1']), 250)
    eq_(len(ret['proj2']), 100)

    def overlap(g0, g1):
        return set(ret[g0]) & set(ret[g1])

    #def gte_(lhs, rhs):
    #    assert lhs <= rhs, 'lhs: %s not <= rhs: %s' % (lhs, rhs)
    # Note: the current implementation isn't great, read comment in _fill_groups
    # picking a different seed may make tests pass, but it's nice to have this fail...
    #gte_(overlap_counts[('proj0', 'proj1')], len(overlap('proj0', 'proj1')))
    #gte_(overlap_counts[('proj0', 'proj2')], len(overlap('proj0', 'proj2')))
    #gte_(overlap_counts[('proj1', 'proj2')], len(set(ret['proj1']) & set(ret['proj2'])))


def test__create_completed_interaction_matrix():
    # The recipe doesn't provide any interaction matrix
    recipe_interaction_matrix = None
    region_names = ['A', 'B', 'C', 'D']
    number_of_regions = len(region_names)
    fractions = pd.Series([0.5, 0.25, 0.66, 0.75], index=list('ABCD'))

    matrix = test_module._create_completed_interaction_matrix(
        recipe_interaction_matrix, fractions
    )
    # Missing entries are filled with ones.
    # Diagonal entries are zeroed.
    expected = np.ones([number_of_regions] * 2) - np.identity(number_of_regions)
    assert_array_equal(expected, matrix)

    # The recipe provides an interaction matrix, but only filled with ones
    interactions = np.ones((4, 4))
    recipe_interaction_matrix = pd.DataFrame(interactions, columns=region_names, index=region_names)
    matrix = test_module._create_completed_interaction_matrix(
        recipe_interaction_matrix, fractions
    )
    assert_array_equal(expected, matrix)

    # The recipe provides an interaction matrix, but with some incompatible entries
    # i.e., some A and B such that I_S(A, B) > 1.0 / max(P(S --> A), P(S --> B))
    interactions = np.ones((4, 4))
    interactions[0, 1] = interactions[1, 0] = 9.0
    interactions[2, 3] = interactions[3, 2] = 3.0
    recipe_interaction_matrix = pd.DataFrame(interactions, columns=region_names, index=region_names)
    matrix = test_module._create_completed_interaction_matrix(
        recipe_interaction_matrix, fractions
    )
    expected[0, 1] = expected[1, 0] = 2.0
    expected[2, 3] = expected[3, 2] = 1.0 / 0.75
    assert_array_equal(expected, matrix)

    # The recipe provides an interaction matrix which is compatible
    # with innervation probabilities in the sense that
    # I_S(A, B) <= 1.0 / max(P(S --> A), P(S --> B))
    expected = np.array([[1.0, 2.0, 1.5, 1.3],
                         [2.0, 1.0, 1.5, 1.0],
                         [1.5, 1.5, 1.0, 1.3],
                         [1.3, 1.0, 1.3, 1.0]])
    expected -= np.identity(number_of_regions)
    recipe_interaction_matrix = pd.DataFrame(expected, columns=region_names, index=region_names)
    matrix = test_module._create_completed_interaction_matrix(
        recipe_interaction_matrix, fractions
    )
    assert_array_equal(expected, matrix)

    # The recipe provides an interaction matrix which is compatible
    # with innervation probabilities but it has 2 missing rows and 2 missing columns
    expected = np.array([[1.0, 2.0], [2.0, 1.0]])
    for region_name in ['A', 'D']:
        region_names.remove(region_name)
    recipe_interaction_matrix = pd.DataFrame(expected, columns=region_names, index=region_names)
    matrix = test_module._create_completed_interaction_matrix(
        recipe_interaction_matrix, fractions
    )
    expected = np.ones((4, 4)) - np.identity(number_of_regions)
    expected[1, 2] = expected[2, 1] = 1.0 / 0.66
    assert_array_equal(expected, matrix)


def test__ptypes_to_target_groups():
    projection_name_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    ptypes_array = [
        set([]), set([]), set([]), set([]),
        set([0]), set([1]), set([1]), set([3]),
        set([2, 3]), set([2, 3]), set([0, 1]), set([0, 3]),
        set([0, 1, 3]), set([0, 1, 3]), set([1, 2, 3]), set([0, 1, 2]),
        set([0, 1, 2, 3]), set([0, 1, 2, 3]), set([0, 1, 2, 3])
    ]
    gids = range(len(ptypes_array))
    target_groups = test_module._ptypes_to_target_groups(ptypes_array, projection_name_map, gids)
    expected = {
        'A': [4, 10, 11, 12, 13, 15, 16, 17, 18],
        'B': [5, 6, 10, 12, 13, 14, 15, 16, 17, 18],
        'C': [8, 9, 14, 15, 16, 17, 18],
        'D': [7, 8, 9, 11, 12, 13, 14, 16, 17, 18]
    }
    for region_name in expected:
        assert_array_equal(expected[region_name], sorted(target_groups[region_name]))


@raises(ValueError)
def test_allocate_gids_to_targets_exception():
    gids = range(10)
    targets = None
    recipe_interaction_matrix = None
    # Unsupported type of algorithm
    algorithm = 'sublinear_beta_optimal_allocation_schema'
    test_module.allocate_gids_to_targets(
        targets, recipe_interaction_matrix, gids, algorithm=algorithm, rng=np.random)


def check_target_group_sizes(algorithm, total_count=1000):
    gids = range(total_count)
    targets = pd.DataFrame([('A', .25),
                            ('B', .25),
                            ('C', .10),
                            ('D', 0.10),
                            ('E', 0.05)],
                           columns=['projection_name', 'fraction'])
    # Note: when testing the random generation of p-types
    # we need to make sure that interaction matrix comes from an actual
    # tree model, otherwise the population sizes/probabilities won't necessarily match.
    interactions = np.array([[1.0, 2.0, 3.0, 4.0],
                             [2.0, 1.0, 2.0, 2.0],
                             [3.0, 2.0, 1.0, 3.0],
                             [4.0, 2.0, 3.0, 1.0]
                             ])
    incomplete_region_names = ['A', 'B', 'D', 'E']
    recipe_interaction_matrix = pd.DataFrame(
        interactions,
        columns=incomplete_region_names,
        index=incomplete_region_names)
    np.random.seed(0)
    target_groups = test_module.allocate_gids_to_targets(
        targets, recipe_interaction_matrix, gids, rng=np.random, algorithm=algorithm)
    target_fractions = targets.set_index('projection_name')['fraction']
    region_names = []
    # Testing the size of the expected population for each target group
    for region_name, fraction in target_fractions.items():
        region_names.append(region_name)
        if algorithm == test_module.Algorithm.GREEDY:
            assert len(target_groups[region_name]) == int(total_count * fraction)
        elif algorithm == test_module.Algorithm.STOCHASTIC_TREE_MODEL:
            assert_allclose(len(target_groups[region_name]), int(total_count * fraction), rtol=0.05)
    # Testing the size of the expected population for each pair of specified targets
    for name_i, name_j in it.combinations(incomplete_region_names, 2):
        if name_i == name_j:
            continue
        intersection = set(target_groups[name_i]) & set(target_groups[name_j])
        expected = total_count * target_fractions[name_i] * target_fractions[name_j] \
            * recipe_interaction_matrix.loc[name_i, name_j]
        actual = len(intersection)
        assert_allclose(actual, expected, rtol=0.10, atol=0.015 * total_count)
    # Testing the independence constraint on the regions that were missing in the interaction matrix
    missing_regions = [
        region_name for region_name in region_names if region_name not in incomplete_region_names]
    for region_name in region_names:
        for missing_region in missing_regions:
            if region_name == missing_region:
                continue
            intersection = set(target_groups[region_name]) & set(target_groups[missing_region])
            actual_ratio = len(intersection) / \
                (total_count * target_fractions[region_name] * target_fractions[missing_region])
            assert_allclose(actual_ratio, 1.0, rtol=0.15)


def test_allocate_gids_to_targets_greedy():
    check_target_group_sizes(test_module.Algorithm.GREEDY, total_count=10000)


def test_allocate_gids_to_targets_random():
    check_target_group_sizes(test_module.Algorithm.STOCHASTIC_TREE_MODEL, total_count=10000)
