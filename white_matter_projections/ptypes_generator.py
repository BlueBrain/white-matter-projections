'''Class and functions related to the p-types generating tree model defined in
"A null model of the mouse whole-neocortex micro-connectome",
see the section "A model to generate projection types"
in https://www.nature.com/articles/s41467-019-11630-x .
A p-type of a source region is one of its projection types, that is, a subset of the target regions
innervated by some of the source neurons.

This includes
   - The creation of the p-types generating tree out of the innervation probablities
   (a.k.a fractions in the recipe) and the interaction strengths matrix (a.k.a
   matrix of increases in conditional innervation probability in the paper)

   - The generation of random samples of p-types based on the tree model.
'''

import numpy
import networkx as nx
from white_matter_projections import ptypes_generator_utils as utils


def _append_parent_node(matrix, children_indices):
    '''Append a new column and a new row corresponding to a parent node.

    A parent node is an internal node in the binary tree construction.
    It replaces its two children in the ongoing build process. Hence the interaction matrix needs
    to be updated after its creation by adding a new row and a new column.

    Args:
        matrix(numpy.ndarray): 2D float matrix

    Returns:
        matrix(numpy.ndarray): matrix with one more row and one more column.
            The new entries are set by taking the maximum component-wise of
            the rows and columns specified by the input index pair, i.e., children_indices.
        children_indices(list): pair of indices indicating which child columns and
            child rows should be used to compute the new matrix entries.
    '''
    matrix = numpy.pad(matrix, (0, 1), mode='constant')
    (left, right) = children_indices
    new_index = matrix.shape[0] - 1
    matrix[new_index, :] = numpy.amax([matrix[left, :], matrix[right, :]], axis=0)
    matrix[:, new_index] = numpy.amax([matrix[:, left], matrix[:, right]], axis=0)
    matrix[new_index, new_index] = 0.0
    return matrix


def _remove_child_nodes(matrix, indices):
    '''Remove rows and columns with specified indices.

    When a parent node is created in the binary tree construction, it replaces its two children
    in the ongoing build process. Hence the interaction matrix needs to be updated
    by removing the two rows and the columns corresponding to the children nodes.

    Args:
        matrix(numpy.ndarray): 2D float matrix
        indices(list): list of indices, just a pair when called in the
            context of the binary tree construction.

    Returns:
        matrix(numpy.ndarray): matrix with as many rows and columns removed as
            specified by the indices array.
    '''
    for axis in range(2):
        matrix = numpy.delete(matrix, indices, axis)
    return matrix


def _create_full_binary_tree(innervation_probabilities, interaction_matrix):
    '''Create a full binary tree based on innervation probabilites and interaction strengths.

    Create a full binary tree with as many leaves as target regions and extend it with
    an extra source node. This source node becomes the unique node of outer degree one
    and acts as a new root. Each edge is assigned a probability.

    This rooted and weighted tree is used to define a stochastic process
    that generates random p-types, i.e., random subsets of innervated regions.
    See Section "A model to generate projection types"
    of https://www.nature.com/articles/s41467-019-11630-x.

    The gist of this stochastic process can be summarized as follows.
    An axon orginating from the source node spreads along the edges of the tree.
    The probability that the axon crosses an edge is
    determined by the probability assigned to this edge. The generated p-type is the set of leaves
    which have been eventually reached by the axon.

    The algorithm below assumes that the innervation probabilities P(S --> T) and the interaction
    strengths I_S(A, B) for the source S and for all target regions T, A and B are consistent, i.e.,
    1.0/I_S(A, B) >= max (P(S --> A), P(S --> B)) holds true for every A, B.

    The build process is a loop which breaks down into 5 steps:
    - (1) select two target regions which maximize I_S(. , .).
    - (2) insert these regions as two leaves A and B of the tree under construction.
    - (3) create a parent node P for these two leaves and assign probabilities
    to the newly created edges in agreement with P(S --> .) and I_S(., .), i.e.,
    the probability of the edge joining P to A is P(S --> A) * I_S(A, B);
    likewise the probability assigned to the edge joining P to B is P(S --> B) * I_S(A, B).
    Insert P in I_S(. ,.) by defining I_S(P, T) = max (I_S(A, T), I_S(B, T))
    for all regions distinct from A and B.
    - (4) remove the two columns and the two rows corresponding to A and B.
    - (5) Consider the parent node as a new leave and re-iterate from (1)
    until all leaves have been consumed, i.e., until I_S(. ,.) has been reduced to
    an empty matrix of shape (0, 0).

    Note: After further contractions of ineluctable edges (see definition below),
    we obtain the unique rooted and weighted tree which agrees with the values of
    P(S --> .) and I_S(.,.) and whose number of nodes is minimal.

    Args:
        innervation_probabilities(list): 1D array of float
        interaction_matrix(numpy.ndarray): 2D float matrix

    Returns:
        tuple(networkx.DiGraph, int): the p-types generating tree together
            with the identifier of its source node.
    '''
    innervation_probabilities_dict = dict(enumerate(innervation_probabilities))
    source_id = len(innervation_probabilities_dict)  # coincides with the number of leaves
    assert source_id == interaction_matrix.shape[0] == interaction_matrix.shape[1]
    parent_id = source_id + 1
    tree = nx.DiGraph()

    while True:
        nodes = list(innervation_probabilities_dict.keys())
        max_indices = utils.get_max_indices(interaction_matrix)
        max_node_ids = [nodes[max_index] for max_index in max_indices]
        maximum = interaction_matrix[max_indices]
        # Insert nodes
        for child_id in max_node_ids:
            if child_id not in tree.nodes:
                tree.add_node(child_id)
        parent_id += 1
        tree.add_node(parent_id)
        # Set the parent node innervation probability as the inverse of the interaction strength
        # of its children and update the interaction matrix
        innervation_probabilities_dict[parent_id] = parent_innervation_probability =\
            1.0 / maximum
        # Insert edges
        for child_id in max_node_ids:
            child_innervation_probability = innervation_probabilities_dict[child_id]
            crossing_probability = child_innervation_probability / parent_innervation_probability
            # Numerical tests indicate that the above ratio can be slightly
            # above 1.0 (1e-6) because of round-off errors.
            tree.add_edge(parent_id, child_id, crossing_probability=min(crossing_probability, 1.0))
        # Add a last node of degree 1 on top of the current root and exit
        if len(nodes) == 2:
            tree.add_node(source_id)
            tree.add_edge(source_id, parent_id, crossing_probability=parent_innervation_probability)
            break
        interaction_matrix = _append_parent_node(interaction_matrix, max_indices)
        # Remove children from the innervation probabilities dict and from the interaction matrix
        for child_id in max_node_ids:
            del innervation_probabilities_dict[child_id]
        interaction_matrix = _remove_child_nodes(interaction_matrix, max_indices)

    return tree, source_id


class PtypesGenerator(object):
    '''Class handling the random generation of p-types.

    The PtypesGenerator class is responsible for building the
    p-type generating tree model as described in the section
    "A model to generate projection types" of https://www.nature.com/articles/s41467-019-11630-x.

    Instances are built out of
    - the row of innervation probabilities
    P(S --> T), where P(S --> T) is the probability that the source of concern,
    innervates the target region T
    - the interaction strengths matrix I_S(. , .).

    Args:
        tree(networkx.DiGraph): directed rooted tree with weighted edges
        source_id(int): unique integer id of the source node
            from which random axons are cast.

    Returns:
        matrix(numpy.ndarray): matrix with as many rows and columns removed as
            specified by the indices array.
    '''

    def __init__(self, innervation_probabilities, interaction_matrix):
        '''Build the p-type generating tree.

        Note: innervation_probabilities is 1D array whose size matches
        the size of the square interaction_matrix.
        The build process assumes that the innervation probabilities P(S --> T)
        and the interaction strengths I_S(A, B) for the source S
        and all target regions T, A and B are compatible,
        i.e., 1.0 / I_S(A, B) >= max (P(S --> A), P(S --> B)) holds true.
        These conditions are not sufficient however to guarantee that the
        returned tree has an interaction matrix which matches the original one.
        This happens only if the two provided arrays originate from an actual
        tree model. This latter condition should be ensured by the recipe.

        Args:
            innervation_probabilities(list): 1D array of float
            interaction_matrix(numpy.ndarray): 2D float matrix
        '''
        self.tree, self.source_id = _create_full_binary_tree(
            innervation_probabilities, interaction_matrix
        )
        # We shall remove superfluous edges. This will put the generating tree in
        # normal form and will make the random generator more efficient.
        self.tree = utils.contract_ineluctable_edges(self.tree)

    def generate_random_ptypes(self, number_of_ptypes):
        '''Generate the specidied number of p-types based on the tree model.

        Args:
            number_of_ptypes(int): number of p-types to be generated
                based on the tree model.

        Returns:
            ptypes(list): list of ptypes. A p-type is a set of leave identifiers
                which corresponds to a set of target region indices.
        '''
        return utils.generate_random_ptypes(self.tree, self.source_id, number_of_ptypes)
