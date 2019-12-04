'''Helper functions related to the PtypesGenerator class defined in ptypes_generator.py.

This includes
   - functions acting on arbitrary rooted and directed trees
   - the generation of random samples of p-types based on a rooted tree provided
   as a networkx.DiGraph
'''

import numpy as np
import networkx as nx


def is_leave(tree, node_id):
    '''Determine whether a node of directed tree is a leave or not.

    Args:
        tree(networkx.DiGraph): directed rooted tree
        node_id(int): unique integer id of the queried node

    Returns:
        True if the node_id identifies a leave, False otherwise.
    '''
    return tree.out_degree[node_id] == 0


def generate_random_ptype(tree, source_id):
    '''Generate a unique random p-type based on the tree model.

    Args:
        tree(networkx.DiGraph): directed rooted tree
        source_id(int): unique integer id of the source node
            from which a random axon is cast.

    Returns:
        ptype(set): set of ids of the leaves reached by the axon,
            i.e., list of indices of the target regions
            innervated be the source.
    '''

    ptype = set()
    # Breadth-first search
    node_stack = [source_id]
    while node_stack:
        current_node = node_stack.pop()
        outward_edges = tree.out_edges(current_node)
        for edge in outward_edges:
            crossing_probability = tree.edges[edge]['crossing_probability']
            if np.random.uniform() <= crossing_probability:
                successor = edge[1]
                if is_leave(tree, successor):
                    ptype.add(successor)
                else:
                    node_stack.append(successor)

    return ptype


def generate_random_ptypes(tree, source_id, number_of_ptypes):
    '''Generate the specidied number of p-types based on the tree model.

    Args:
        tree(networkx.DiGraph): directed rooted tree
        source_id(int): unique integer id of the source node
            from which random axons are cast.
        number_of_ptypes(int): number of p-types to be generated.

    Returns:
        ptypes(list): list of ptypes. A ptype is a set of leave identifiers,
            .i.e, a set of target region indices.
    '''
    return [generate_random_ptype(tree, source_id) for _ in range(number_of_ptypes)]


def get_max_indices(matrix):
    '''Get the first pair of indices maximizing the matrix entries.

    Args:
        matrix(np.ndarray): 2D float matrix

    Returns:
        max_indices(list): a pair of indices corresponding to
            the maximum of all matrix entries.
    '''
    return np.unravel_index(np.nanargmax(matrix), matrix.shape)


def get_root(tree):
    '''Get the root of a directed tree whose edges flow downward.
    Args:
        tree(networkx.DiGraph): directed rooted tree.

    Returns:
        root(int): id of the root node.
    '''
    roots = [node for node in tree.nodes if tree.in_degree(node) == 0]
    error_msg = 'The input graph is not a tree or has an invalid orientation.'
    if len(roots) > 1:
        raise ValueError('Multiple root candidates: {}. {}'.format(roots, error_msg))
    if not roots:
        raise ValueError('Root not found. {}'.format(error_msg))
    return roots[0]


def get_leaves(tree):
    '''Get the leaves of a rooted tree.

    Note: A rooted tree is naturally a directed graph: edges flow
    downward from the root to the leaves. Hence the leaves are defined
    as nodes of outer degree zero.
    Args:
        tree(networkx.DiGraph): directed rooted tree

    Returns:
        leaves(list): sorted list of the
            node ids corresponding to the leaves
    '''
    leaves = [node for node in tree.nodes if is_leave(tree, node)]
    return sorted(leaves)


def contract_ineluctable_edges(tree):
    '''Contract the edges with probability 1.0 which are superfluous.

    Contract the edges with crossing probability equal to 1.0
    which join either two internal nodes (i.e., non-leaf nodes)
    or which join the root to an internal node.

    Args:
        tree(networkx.DiGraph): directed rooted tree with weighted edges

    Returns:
        matrix(np.ndarray): matrix with as many rows and columns removed as
            specified by the indices array.
    '''
    while True:
        ineluctable_edges = []
        for edge in tree.edges:
            is_internal_edge = not is_leave(tree, edge[1])
            if tree.edges[edge]['crossing_probability'] >= 1.0 and is_internal_edge:
                ineluctable_edges.append(edge)
        for edge in ineluctable_edges:
            if edge in tree.edges:
                tree = nx.contracted_edge(tree, edge, self_loops=False)
        if not ineluctable_edges:
            break

    return tree
