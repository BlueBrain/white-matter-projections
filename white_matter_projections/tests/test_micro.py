import os
import h5py
import itertools as it
import numpy as np
import pandas as pd

import yaml
from nose.tools import ok_, eq_, raises
from white_matter_projections import macro, micro, utils as wmp_utils
from numpy.testing import assert_allclose, assert_array_equal
import utils


def fake_allocations():
    return {'source_population0': {'projection0': np.arange(10),
                                   'projection1': np.arange(10, 20),
                                   },
            'source_population1': {'projection0': np.arange(20, 30),
                                   'projection1': np.arange(30, 40),
                                   },
            }


def fake_projection_mapping():
    ret = {'source_population0': {'projection0': {'target_population': 'target00'},
                                  'projection1': {'target_population': 'target01'},
                                  },
           'source_population1': {'projection0': {'target_population': 'target10'},
                                  'projection1': {'target_population': 'target11'},
                                  },
           }
    return ret


def compare_allocations(ret):
    allocations = fake_allocations()

    eq_(sorted(allocations), sorted(ret))
    assert_allclose(allocations['source_population0']['projection0'],
                    ret['source_population0']['projection0'])
    assert_allclose(allocations['source_population0']['projection1'],
                    ret['source_population0']['projection1'])


def test_save_load_allocations():
    allocations = fake_allocations()
    with utils.tempdir('test_serialize_allocations') as tmp:
        name = os.path.join(tmp, 'allocations.h5')
        micro.save_allocations(name, allocations)
        ret = micro.load_allocations(name, projections_mapping=None)
        compare_allocations(ret)

        projections_mapping = fake_projection_mapping()
        ret = micro.load_allocations(name, projections_mapping)
        ok_(isinstance(ret, pd.DataFrame))


def test_serialize_allocations():
    allocations = fake_allocations()

    with utils.tempdir('test_serialize_allocations') as tmp:
        name = os.path.join(tmp, 'allocations.h5')
        with h5py.File(name, 'w') as h5:
            micro._serialize_allocations(h5, allocations)

        with h5py.File(name, 'r') as h5:
            ret = micro._deserialize_allocations(h5)

    compare_allocations(ret)


def test_transpose_allocations():
    allocations = fake_allocations()
    projections_mapping = fake_projection_mapping()

    ret = micro.transpose_allocations(allocations, projections_mapping)
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
    total_counts, overlap_counts = micro._ptype_to_counts(
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
    total_counts, overlap_counts = micro._ptype_to_counts(
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
    total_counts, overlap_counts = micro._ptype_to_counts(
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
    names, name_map, total_counts_remap, overlap_counts_remap = micro._make_numeric_groups(
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
    ret = micro._greedy_gids_allocation_from_counts(total_counts, overlap_counts, gids)
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
    ret = micro._greedy_gids_allocation_from_counts(total_counts, overlap_counts, gids)

    eq_(len(ret['proj0']), 250)
    eq_(len(ret['proj1']), 250)
    eq_(len(ret['proj2']), 100)

    def overlap(g0, g1):
        return set(ret[g0]) & set(ret[g1])

    # Note: the current implementation isn't great, read comment in _fill_groups
    # picking a different seed may make tests pass, but it's nice to have this fail...
    #gte_(overlap_counts[('proj0', 'proj1')], len(overlap('proj0', 'proj1')))
    #gte_(overlap_counts[('proj0', 'proj2')], len(overlap('proj0', 'proj2')))
    #gte_(overlap_counts[('proj1', 'proj2')], len(set(ret['proj1']) & set(ret['proj2'])))


def test_get_gids_by_population():
    populations = yaml.load('''\
- name: POP1_ALL_LAYERS
  atlas_region:
      name: ECT
      subregions: [l1, l2, l3, l4, l5, l6]
  filters: []
''', Loader=yaml.FullLoader)
    _, populations = macro._parse_populations(populations,
                                              utils.REGION_MAP,
                                              utils.SUBREGION_TRANSLATION,
                                              utils.REGION_SUBREGION_FORMAT)

    def cells(_):
        return pd.DataFrame({'layer': [1, 1, 2, 3, ],
                             'region': ['ECT', 'ECT', 'FRP', 'ECT', ],
                             })

    source_population = 'POP1_ALL_LAYERS'
    ret = micro.get_gids_by_population(populations, cells, source_population)
    assert_array_equal(ret, [0, 1, 3])

    def cells(_):
        return pd.DataFrame({'layer': [1, 1, 2, 3, ],
                             'region': ['ECT@left', 'ECT@right', 'FRP@left', 'ECT@right', ],
                             })
    source_population = 'POP1_ALL_LAYERS'
    ret = micro.get_gids_by_population(populations, cells, source_population)
    assert_array_equal(ret, [0, 1, 3])

# def test_allocate_projections():
#    micro.allocate_projections(recipe, cells)

# def test_allocation_stats():
#    ret = micro.allocation_stats(ptype, interactions, cell_count, allocations)

def test_create_completed_interaction_matrix():
    # The recipe doesn't provide any interaction matrix
    recipe_interaction_matrix = None
    region_names = ['A', 'B', 'C', 'D']
    number_of_regions = len(region_names)
    fractions = pd.Series([0.5, 0.25, 0.66, 0.75], index=list('ABCD'))

    matrix = micro.create_completed_interaction_matrix(
        recipe_interaction_matrix, fractions
    )
    # Missing entries are filled with ones.
    # Diagonal entries are zeroed.
    expected = np.ones([number_of_regions] * 2) - np.identity(number_of_regions)
    assert_array_equal(expected, matrix)

    # The recipe provides an interaction matrix, but only filled with ones
    interactions = np.ones((4, 4))
    recipe_interaction_matrix = pd.DataFrame(interactions, columns=region_names, index=region_names)
    matrix = micro.create_completed_interaction_matrix(
        recipe_interaction_matrix, fractions
    )
    assert_array_equal(expected, matrix)

    # The recipe provides an interaction matrix, but with some incompatible entries
    # i.e., some A and B such that I_S(A, B) > 1.0 / max(P(S --> A), P(S --> B))
    interactions = np.ones((4, 4))
    interactions[0, 1] = interactions[1, 0] = 9.0
    interactions[2, 3] = interactions[3, 2] = 3.0
    recipe_interaction_matrix = pd.DataFrame(interactions, columns=region_names, index=region_names)
    matrix = micro.create_completed_interaction_matrix(
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
    matrix = micro.create_completed_interaction_matrix(
        recipe_interaction_matrix, fractions
    )
    assert_array_equal(expected, matrix)

    # The recipe provides an interaction matrix which is compatible
    # with innervation probabilities but it has 2 missing rows and 2 missing columns
    expected = np.array([[1.0, 2.0], [2.0, 1.0]])
    for region_name in ['A', 'D']:
        region_names.remove(region_name)
    recipe_interaction_matrix = pd.DataFrame(expected, columns=region_names, index=region_names)
    matrix = micro.create_completed_interaction_matrix(
        recipe_interaction_matrix, fractions
    )
    expected = np.ones((4, 4)) - np.identity(number_of_regions)
    expected[1, 2] = expected[2, 1] = 1.0 / 0.66
    assert_array_equal(expected, matrix)


def test_ptypes_to_target_groups():
    projection_name_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    ptypes_array = [
        set([]), set([]), set([]), set([]),
        set([0]), set([1]), set([1]), set([3]),
        set([2, 3]), set([2, 3]), set([0, 1]), set([0, 3]),
        set([0, 1, 3]), set([0, 1, 3]), set([1, 2, 3]), set([0, 1, 2]),
        set([0, 1, 2, 3]), set([0, 1, 2, 3]), set([0, 1, 2, 3])
    ]
    gids = range(len(ptypes_array))
    target_groups = micro.ptypes_to_target_groups(ptypes_array, projection_name_map, gids)
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
    micro.allocate_gids_to_targets(targets, recipe_interaction_matrix, gids, algorithm=algorithm)


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
    target_groups = micro.allocate_gids_to_targets(
        targets, recipe_interaction_matrix, gids, algorithm=algorithm)
    target_fractions = targets.set_index('projection_name')['fraction']
    region_names = []
    # Testing the size of the expected population for each target group
    for region_name, fraction in target_fractions.items():
        region_names.append(region_name)
        if algorithm == micro.Algorithm.GREEDY:
            assert len(target_groups[region_name]) == int(total_count * fraction)
        elif algorithm == micro.Algorithm.STOCHASTIC_TREE_MODEL:
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
    check_target_group_sizes(micro.Algorithm.GREEDY, total_count=10000)


def test_allocate_gids_to_targets_random():
    check_target_group_sizes(micro.Algorithm.STOCHASTIC_TREE_MODEL, total_count=10000)


def test_partition_cells_left_right():
    cells = pd.DataFrame(np.arange(10) + .1, columns=['z', ])
    left, right = micro.partition_cells_left_right(cells, center_line_3d=5.1)
    eq_(len(left), 6)
    eq_(len(right), 4)


def test_partition_syns():
    syns = pd.DataFrame(np.arange(10) + .1, columns=['z', ])
    left = micro.partition_syns(syns, side='left', center_line_3d=5.1)
    eq_(len(left), 6)

    right = micro.partition_syns(syns, side='right', center_line_3d=5.1)
    eq_(len(right), 4)


def test_separate_source_and_targets():
    cells = pd.DataFrame(np.arange(10) + .1, columns=['z', ])
    cells['x'] = 1
    cells['y'] = 2
    sgids = cells.index.values
    left_cells, right_cells = micro.partition_cells_left_right(cells, center_line_3d=5.1)

    ret = micro.separate_source_and_targets(left_cells, right_cells, sgids,
                                            hemisphere='ipsi', side='left')
    eq_(len(ret), 6)

    ret = micro.separate_source_and_targets(left_cells, right_cells, sgids,
                                            hemisphere='contra', side='left')
    eq_(len(ret), 4)

    ret = micro.separate_source_and_targets(left_cells, right_cells, sgids,
                                            hemisphere='ipsi', side='right')
    eq_(len(ret), 4)

    ret = micro.separate_source_and_targets(left_cells, right_cells, sgids,
                                            hemisphere='contra', side='right')
    eq_(len(ret), 6)


def test__assign_groups():
    tgt_flat = np.array([[-10., -10.],
                         [10., 10.],
                         ])

    src_flat = np.array([[0., 0.], ])
    res = micro._assign_groups(src_flat, tgt_flat, sigma=10, closest_count=10)
    eq_(list(res), [0, 0])

    src_flat = np.array([[-10., -10.],
                         [10., 10.],
                         ])
    res = micro._assign_groups(src_flat, tgt_flat, sigma=10, closest_count=10)
    eq_(list(res), [0, 1])


def test_assign_groups():
    tgt_flat = np.array([[-10., -10.],
                         [10., 10.],
                         ])
    src_flat = pd.DataFrame([[-10., -10.],
                             [10., 10.],
                             ], index=(10, 20))
    res = micro.assign_groups(src_flat, tgt_flat, sigma=10, closest_count=10, n_jobs=1)
    eq_(list(res), [10, 20])


def test__calculate_delay_streamline():
    np.random.seed(42)
    src_cells = pd.DataFrame([(55., 0., 0.,),
                              (65., 0., 0.,),
                              (75., 0., 0.,),
                              (85., 0., 0.,), ],
                             columns=wmp_utils.XYZ,
                             index=[1, 2, 3, 4])
    syns = pd.DataFrame([(1, 1., 0., 0.),
                         (2, 2., 0., 0.),
                         (2, 3., 0., 0.),
                         (4, 4., 0., 0.), ],
                        columns=['sgid', ] + wmp_utils.XYZ)
    columns = ['path_row', 'length', 'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z', ]
    streamline_metadata = pd.DataFrame([(3, 3000., 0., 0., 0., 3., 0., 0.),
                                        (4, 4000., 1., 0., 0., 4., 0., 0.),
                                        (5, 5000., 2., 0., 0., 5., 0., 0.),
                                        ],
                                       columns=columns)
    conduction_velocity = {'inter_region': 10.,
                           'intra_region': 1.
                           }

    delay, gid2row = micro._calculate_delay_streamline(src_cells, syns, streamline_metadata,
                                                       conduction_velocity=conduction_velocity)
    eq_(list(gid2row.row), [5, 3, 5, 5, ])  # based on random picking
    assert_allclose(gid2row.sgid.values, syns.sgid.values)
    eq_(list(delay), [557., 366., 565., 584.])


def test__calculate_delay_direct():
    conduction_velocity = {'inter_region': 10.,
                           'intra_region': 1.
                           }

    src_cells = pd.DataFrame([(55., 0., 0.,),
                              (65., 0., 0.,),
                              (75., 0., 0.,),
                              (85., 0., 0.,), ],
                             columns=wmp_utils.XYZ,
                             index=[1, 2, 3, 4])
    syns = pd.DataFrame([(1, 1., 0., 0.),
                         (2, 2., 0., 0.),
                         (2, 3., 0., 0.),
                         (4, 4., 0., 0.), ],
                        columns=['sgid', ] + wmp_utils.XYZ)

    delay = micro._calculate_delay_direct(src_cells, syns, conduction_velocity)
    assert_allclose(delay, np.array([54, 63, 62, 81, ]) / 10.)

# def test_assignment():
#    config
#    allocations
#    side
#    with utils.tempdir('test_assignment') as output:
#        micro.assignment(output, config, allocations, side)
