
from nose.tools import raises
import networkx as nx
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from utils import \
    create_statistical_interaction_strength_matrix,\
    create_innervation_probability_row
from white_matter_projections import ptypes_generator_utils as utils

def test_is_leave():
    tree = nx.DiGraph()
    number_of_nodes = 8
    tree.add_nodes_from(range(number_of_nodes))
    tree.add_edges_from([(5, 0), (5, 1), (6, 5), (6, 2), (6, 7), (7, 3), (7, 4)])
    # Here is the tree we have just built:
    #  4  3
    #   \/
    #   7 2
    #   |/
    #   6
    #   |
    #   5
    #  /\
    # 0  1
    for node in range(5):
        assert utils.is_leave(tree, node)
    for node in range(5, number_of_nodes):
        assert not utils.is_leave(tree, node)


def test_generate_random_ptype():
    tree = nx.DiGraph()
    number_of_nodes = 8
    source_id = 7
    tree.add_nodes_from(range(number_of_nodes))
    tree.add_edges_from([(source_id, 6), (6, 4), (6, 5), (4, 0), (4, 1), (5, 2), (5, 3)])
    # Here is the tree we have just built:
    #    7 <-- source
    #    |
    #    6
    #    /\
    #   4  5
    #  /\  /\
    # 0 1  2 3

    rng = np.random
    # All edges with probability 1.0
    for edge in tree.edges:
        tree.edges[edge]['crossing_probability'] = 1.0
    assert utils.generate_random_ptype(tree, source_id, rng) == set([0, 1, 2, 3])

    # The edge issued from the source has probability 0.0
    tree.edges[(source_id, 6)]['crossing_probability'] = 0.0
    assert utils.generate_random_ptype(tree, source_id, rng) == set([])

    # All depth 1 edges have probability 0.0
    tree.edges[(7, 6)]['crossing_probability'] = 1.0
    tree.edges[(6, 4)]['crossing_probability'] = 0.0
    tree.edges[(6, 5)]['crossing_probability'] = 0.0
    assert utils.generate_random_ptype(tree, source_id, rng) == set([])

    # All depth 2 edges have probability 0.0
    tree.edges[(6, 4)]['crossing_probability'] = 1.0
    tree.edges[(6, 5)]['crossing_probability'] = 1.0
    tree.edges[(4, 0)]['crossing_probability'] = 0.0
    tree.edges[(4, 1)]['crossing_probability'] = 0.0
    tree.edges[(5, 2)]['crossing_probability'] = 0.0
    tree.edges[(5, 3)]['crossing_probability'] = 0.0
    assert utils.generate_random_ptype(tree, source_id, rng) == set([])

    # Different patterns depending on each terminal edge assignment
    tree.edges[(4, 0)]['crossing_probability'] = 1.0
    tree.edges[(4, 1)]['crossing_probability'] = 0.0
    tree.edges[(5, 2)]['crossing_probability'] = 1.0
    tree.edges[(5, 3)]['crossing_probability'] = 0.0
    assert utils.generate_random_ptype(tree, source_id, rng) == set([0, 2])

    tree.edges[(4, 0)]['crossing_probability'] = 0.0
    tree.edges[(4, 1)]['crossing_probability'] = 1.0
    tree.edges[(5, 2)]['crossing_probability'] = 0.0
    tree.edges[(5, 3)]['crossing_probability'] = 1.0
    assert utils.generate_random_ptype(tree, source_id, rng) == set([1, 3])

def test_generate_random_ptypes():
    # The seed below has been choosen so as to minimize the number of generated p-types
    # required to get the desired expectation accuracy
    np.random.seed(20000)
    tree = nx.DiGraph()
    number_of_nodes = 10
    source_id = 9
    tree.add_nodes_from(range(number_of_nodes))
    tree.add_edges_from([(source_id, 8), (8, 5), (8, 6), (8, 7),\
         (6, 0), (6, 1), (6, 2), (7, 3), (7, 4)])
    # Here is the tree we have just built:
    #     9 <-- source
    #     |
    #     8
    #    /|\
    #   6 5 7
    #  /|\  /\
    # 0 1 2 3 4
    # Setting crossing probability for each edge
    tree.edges[(9, 8)]['crossing_probability'] = 0.8
    tree.edges[(8, 5)]['crossing_probability'] = 0.66
    tree.edges[(8, 6)]['crossing_probability'] = 0.75
    tree.edges[(8, 7)]['crossing_probability'] = 0.55
    tree.edges[(6, 0)]['crossing_probability'] = 0.25
    tree.edges[(6, 1)]['crossing_probability'] = 0.50
    tree.edges[(6, 2)]['crossing_probability'] = 0.30
    tree.edges[(7, 3)]['crossing_probability'] = 0.70
    tree.edges[(7, 4)]['crossing_probability'] = 0.20
    # In the array below, the entry (i, i) represents the innervation probability of
    # the target region of index i, while the entry (i, j), for i distinct from j, represents
    # the probability that one axon from the source innervates the target regions with index i and
    # j simultaneously.
    innervation_probabilities = np.array([
        [0.15, 0.075, 0.045, 0.05775, 0.0165, 0.099],
        [0.075, 0.3, 0.09, 0.1155, 0.033, 0.198],
        [0.045, 0.09, 0.18, 0.0693, 0.0198, 0.1188],
        [0.05775, 0.1155, 0.0693, 0.308, 0.0616, 0.20328],
        [0.0165, 0.033, 0.0198, 0.0616, 0.088, 0.05808],
        [0.099, 0.198, 0.1188, 0.20328, 0.05808, 0.528]
    ], dtype=np.float)
    number_of_leaves = 6
    number_of_ptypes = 41500
    ptypes = utils.generate_random_ptypes(tree, source_id, number_of_ptypes, np.random)
    actual_counts = np.zeros([number_of_leaves] * 2)
    for ptype in ptypes:
        for i in ptype:
            for j in ptype:
                actual_counts[i, j] += 1
    # We compare actual counts with expected counts.
    # They should be closed to each other by Borel's law of large numbers,
    # see https://en.wikipedia.org/wiki/Law_of_large_numbers.
    # The rate of convergence of the mean estimator is controlled by the inequalities in
    # "Inequalities for the r-th absolute moment of a sum of random variables, 1 <= r <= 2"
    # by von Bahr and Essen, 1965.
    # To get a precision epsilon with probability 1.0 - eta, it is sufficient to have
    # a sample of size >= 2.0 * var(1_E) / (eta * epsilon^2) where var(1_E) = P(E)(1 - P(E))
    # is the variance of the random variable 1_E defined by
    # 1_E(omega) = 1 if omega in E, 0 otherwise.
    # The mesurable set E can be any event, e.g., "an axon from the source innervates
    # simultaneously the target regions A, B and C".
    for i in range(number_of_leaves):
        for j in range(i, number_of_leaves):
            actual = actual_counts[i, j] / float(number_of_ptypes)
            expected = innervation_probabilities[i][j]
            assert_allclose(actual, expected, rtol=0.05)

    triple_intersections_probabilities = {
        '0,1,2': 0.0225,
        '0,1,3': 0.028875,
        '1,2,5': 0.0594
    }
    actual_counts = {
        '0,1,2': 0.0,
        '0,1,3': 0.0,
        '1,2,5': 0.0
    }
    for ptype in ptypes:
        if set([0, 1, 2]).issubset(ptype):
            actual_counts['0,1,2'] += 1
        if set([0, 1, 3]).issubset(ptype):
            actual_counts['0,1,3'] += 1
        if set([1, 2, 5]).issubset(ptype):
            actual_counts['1,2,5'] += 1

    for triple_intersection, expected in triple_intersections_probabilities.items():
        actual = actual_counts[triple_intersection] / float(number_of_ptypes)
        assert_allclose(actual, expected, rtol=0.05)


def test_get_max_indices():
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0],
        [9.0, 8.0, 7.0]
    ])
    # A unique index pair is optimal
    actual = utils.get_max_indices(matrix)
    expected = [2, 0]
    assert_array_equal(expected, actual)
    # Two index pairs achieve the maximum
    matrix[0, 2] = 9.0
    actual = utils.get_max_indices(matrix)
    expected = [0, 2]
    assert_array_equal(expected, actual)

def test_get_leaves():
    tree = nx.DiGraph()
    tree.add_nodes_from(range(10))
    tree.add_edges_from([(10, 9), (9, 8), (9, 5), (9, 7),\
         (8, 0), (8, 1), (8, 2), (7, 3), (7, 6), (6, 4)])
    # Here is the tree we have just built:
    #     10 <-- source
    #     |
    #     9
    #    /|\
    #   8 5 7
    #  /|\  /\
    # 0 1 2 3 6
    #         |
    #         4
    assert_array_equal([0, 1, 2, 3, 4, 5], utils.get_leaves(tree))

def test_get_root():
    tree = nx.DiGraph()
    tree.add_nodes_from(range(10))
    tree.add_edges_from([(10, 9), (9, 8), (9, 5), (9, 7),\
         (8, 0), (8, 1), (8, 2), (7, 3), (7, 6), (6, 4)])
    # Here is the tree we have just built:
    #     10 <-- source
    #     |
    #     9
    #    /|\
    #   8 5 7
    #  /|\  /\
    # 0 1 2 3 6
    #         |
    #         4
    assert_array_equal(10, utils.get_root(tree))

@raises(ValueError)
def test_get_root_raise():
    tree = nx.DiGraph()
    tree.add_nodes_from(range(10))
    tree.add_edges_from([(9, 10), (10, 9), (9, 8), (9, 5), (9, 7),\
         (8, 0), (8, 1), (8, 2), (7, 3), (7, 6), (6, 4)])
    # Here is the tree we have just built:
    #     10 <-- source with an incoming edge from 9
    #     |
    #     9
    #    /|\
    #   8 5 7
    #  /|\  /\
    # 0 1 2 3 6
    #         |
    #         4
    utils.get_root(tree)

@raises(ValueError)
def test_get_root_raise():
    tree = nx.DiGraph()
    tree.add_nodes_from(range(10))
    tree.add_edges_from([(11, 9), (10, 9), (9, 8), (9, 5), (9, 7),\
         (8, 0), (8, 1), (8, 2), (7, 3), (7, 6), (6, 4)])
    # Here is the tree we have just built:
    #     10 <-- first source
    #     |
    #     9 -- 11 <-- second source
    #    /|\
    #   8 5 7
    #  /|\  /\
    # 0 1 2 3 6
    #         |
    #         4
    utils.get_root(tree)

def test_contract_ineluctable_edges():
    tree = nx.DiGraph()
    tree.add_nodes_from(range(10))
    tree.add_edges_from([(9, 8), (8, 5), (8, 6), (8, 7),\
         (6, 0), (6, 1), (6, 2), (7, 3), (7, 4)])
    # Here is the tree we have just built:
    #     9 <-- source
    #     |
    #     8
    #    /|\
    #   6 5 7
    #  /|\  /\
    # 0 1 2 3 4
    # Setting crossing probability for each edge
    tree.edges[(9, 8)]['crossing_probability'] = 1.0 # ineluctable edge
    tree.edges[(8, 5)]['crossing_probability'] = 1.0
    tree.edges[(8, 6)]['crossing_probability'] = 1.0 # ineluctable edge
    tree.edges[(8, 7)]['crossing_probability'] = 0.55
    tree.edges[(6, 0)]['crossing_probability'] = 0.25
    tree.edges[(6, 1)]['crossing_probability'] = 1.0
    tree.edges[(6, 2)]['crossing_probability'] = 0.30
    tree.edges[(7, 3)]['crossing_probability'] = 0.70
    tree.edges[(7, 4)]['crossing_probability'] = 0.20
    # Expected tree
    expected_tree = nx.DiGraph()
    expected_tree.add_nodes_from(range(8))
    expected_tree.add_edges_from([(7, 0), (7, 1), (7, 2), (7, 5), (7, 6), (6, 3), (6, 4)])
    # Here is the tree we have just built:
    #     7
    #   ////|\
    #  0 1 2 5 6
    #          /\
    #         3  4
    actual_tree = utils.contract_ineluctable_edges(tree)
    assert nx.is_isomorphic(expected_tree, actual_tree)

def test_create_statistical_interaction_strength_matrix():
    tree = nx.DiGraph()
    number_of_nodes = 10
    tree.add_nodes_from(range(number_of_nodes))
    tree.add_edges_from([(9, 8), (8, 5), (8, 6), (8, 7),\
         (6, 0), (6, 1), (6, 2), (7, 3), (7, 4)])
    # Here is the tree we have just built:
    #     9 <-- source
    #     |
    #     8
    #    /|\
    #   6 5 7
    #  /|\  /\
    # 0 1 2 3 4
    # Setting crossing probability for each edge
    tree.edges[(9, 8)]['crossing_probability'] = 0.8
    tree.edges[(8, 5)]['crossing_probability'] = 0.7
    tree.edges[(8, 6)]['crossing_probability'] = 0.6
    tree.edges[(8, 7)]['crossing_probability'] = 0.4
    tree.edges[(6, 0)]['crossing_probability'] = 0.1
    tree.edges[(6, 1)]['crossing_probability'] = 0.5
    tree.edges[(6, 2)]['crossing_probability'] = 0.3
    tree.edges[(7, 3)]['crossing_probability'] = 0.2
    tree.edges[(7, 4)]['crossing_probability'] = 0.2
    expected_innervation_probability_row = np.array(
        [0.048, 0.24, 0.144, 0.064, 0.064, 0.56]
    )
    a = 1.0 / (0.8 * 0.6)
    b = 1.0 / 0.8
    c = 1.0 / (0.8 * 0.4)
    expected_interaction_matrix = np.array([
            [0.0, a, a, b, b, b],
            [a, 0.0, a, b, b, b],
            [a, a, 0.0, b, b, b],
            [b, b, b, 0.0, c, b],
            [b, b, b, c, 0.0, b],
            [b, b, b, b, b, 0.0]])
    actual_innervation_probability_row = create_innervation_probability_row(tree)
    actual_interaction_matrix = create_statistical_interaction_strength_matrix(tree)
    assert_allclose(
        expected_innervation_probability_row, actual_innervation_probability_row, atol=1e-16)
    assert_allclose(
        expected_interaction_matrix, actual_interaction_matrix, atol=1e-15)
