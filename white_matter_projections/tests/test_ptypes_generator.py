import itertools
import numpy as np
from numpy.testing import assert_allclose
import networkx as nx
from utils import (
    create_statistical_interaction_strength_matrix,
    create_innervation_probability_row
    )
from white_matter_projections import ptypes_generator
from white_matter_projections import ptypes_generator_utils as utils


def _get_defining_matrices():
    tree = nx.DiGraph()
    tree.add_nodes_from(range(10))
    tree.add_edges_from([(9, 8), (8, 6), (8, 7),\
         (6, 0), (6, 1), (6, 2), (7, 3), (7, 4), (7, 5)])
    # Here is the tree we have just built:
    #      9 <-- source
    #      |
    #      8
    #    /   \
    #   6     7
    #  /|\   /|\
    # 0 1 2 3 4 5
    # Setting crossing probability for each edge
    tree.edges[(9, 8)]['crossing_probability'] = 1.00 # ineluctable edge
    tree.edges[(8, 6)]['crossing_probability'] = 0.75
    tree.edges[(8, 7)]['crossing_probability'] = 1.0 # ineluctable edge
    tree.edges[(6, 0)]['crossing_probability'] = 0.45
    tree.edges[(6, 1)]['crossing_probability'] = 0.50
    tree.edges[(6, 2)]['crossing_probability'] = 0.15
    tree.edges[(7, 3)]['crossing_probability'] = 0.30
    tree.edges[(7, 4)]['crossing_probability'] = 0.60
    tree.edges[(7, 5)]['crossing_probability'] = 1.0

    # Compute the interaction matrix and the innervation probabilities
    # corresponding to this tree model
    interaction_matrix = create_statistical_interaction_strength_matrix(tree)
    innervation_probability_row = create_innervation_probability_row(tree)

    return interaction_matrix, innervation_probability_row


def _get_expected_tree():
    expected_tree = nx.DiGraph()
    expected_tree.add_nodes_from(range(8))
    expected_tree.add_edges_from([(7, 3), (7, 4), (7, 5),\
        (7, 6), (6, 0), (6, 1), (6, 2)])
    # Here is the tree we have just built:
    #      7 <-- source
    #    / |\\
    #   6  3 4 5
    #  /|\
    # 0 1 2
    # Setting crossing probability for each edge
    expected_tree.edges[(7, 3)]['crossing_probability'] = 0.3
    expected_tree.edges[(7, 4)]['crossing_probability'] = 0.6
    expected_tree.edges[(7, 5)]['crossing_probability'] = 1.0
    expected_tree.edges[(7, 6)]['crossing_probability'] = 0.75
    expected_tree.edges[(6, 0)]['crossing_probability'] = 0.45
    expected_tree.edges[(6, 1)]['crossing_probability'] = 0.5
    expected_tree.edges[(6, 2)]['crossing_probability'] = 0.15

    return expected_tree

def test_matrices_consistency():
    interaction_matrix, innervation_probability_row = _get_defining_matrices()
    # Generate the tree
    generator = ptypes_generator.PtypesGenerator(
        innervation_probability_row, interaction_matrix, np.random)

    actual_tree = generator.tree
    # Consistency check: the matrices built from the generated tree must coincide with the matrices
    # used to build it.
    number_of_leaves = 6
    assert utils.get_root(actual_tree) == number_of_leaves
    actual_innervation_probability_row = create_innervation_probability_row(actual_tree)
    actual_interaction_matrix = create_statistical_interaction_strength_matrix(actual_tree)
    assert_allclose(
        actual_innervation_probability_row, innervation_probability_row, atol=1e-15)
    assert_allclose(
        actual_interaction_matrix, interaction_matrix, atol=1e-15)

def test_tree_isomorphism():
    interaction_matrix, innervation_probability_row = _get_defining_matrices()
    # Generate the tree
    generator = ptypes_generator.PtypesGenerator(
        innervation_probability_row, interaction_matrix, np.random)

    # Compare with the expected tree
    expected_tree = _get_expected_tree()
    actual_tree = generator.tree
    assert nx.is_isomorphic(expected_tree, actual_tree)


def test_Ptypes_generator():
    interaction_matrix, innervation_probability_row = _get_defining_matrices()
    # Generate the tree
    generator = ptypes_generator.PtypesGenerator(
        innervation_probability_row, interaction_matrix, np.random)

    # In the array below, the entry (i, i) represents the innervation probability of
    # the target region of index i, while the entry (i, j), for i distinct from j, represents
    # the probability that one axon from the source innervates the target regions with index i and
    # j simultaneously.
    expected_innervation_probabilities = np.array([
        [0.3375, 0.16875, 0.050625, 0.10125, 0.2025, 0.3375],
        [0.16875, 0.375, 0.05625, 0.1125, 0.225, 0.375],
        [0.050625, 0.05625, 0.1125, 0.03375, 0.0675, 0.1125],
        [0.10125, 0.1125, 0.03375, 0.3, 0.18, 0.3],
        [0.2025, 0.225, 0.0675, 0.18, 0.6, 0.6],
        [0.3375, 0.375, 0.1125, 0.3, 0.6, 1.0]
    ], dtype=np.float)

    # Testing the p-types generator
    number_of_ptypes = 13000
    np.random.seed(0)
    ptypes = generator.generate_random_ptypes(number_of_ptypes)
    number_of_leaves = 6
    actual_counts = np.zeros([number_of_leaves] * 2)
    for ptype in ptypes:
        for i, j in itertools.product(ptype, repeat=2):
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
            expected = expected_innervation_probabilities[i][j]
            assert_allclose(actual, expected, rtol=0.05)

    triple_intersections_probabilities = {
        '0,1,2': 0.0253125,
        '0,3,4': 0.06075,
        '2,3,5': 0.03375
    }
    actual_counts = {
        '0,1,2': 0.0,
        '0,3,4': 0.0,
        '2,3,5': 0.0
    }
    for ptype in ptypes:
        if set([0, 1, 2]).issubset(ptype):
            actual_counts['0,1,2'] += 1
        if set([0, 3, 4]).issubset(ptype):
            actual_counts['0,3,4'] += 1
        if set([2, 3, 5]).issubset(ptype):
            actual_counts['2,3,5'] += 1

    for triple_intersection, expected in triple_intersections_probabilities.items():
        actual = actual_counts[triple_intersection] / float(number_of_ptypes)
        assert_allclose(actual, expected, rtol=0.05)
