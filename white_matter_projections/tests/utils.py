import os
import shutil
import tempfile
import yaml

import numpy as np
from contextlib import contextmanager
from pandas.api.types import CategoricalDtype
from voxcell.hierarchy import Hierarchy
from voxcell.voxel_data import VoxelData


BASEDIR = os.path.dirname(__file__)
DATADIR = os.path.join(BASEDIR, 'data')
with open(os.path.join(DATADIR, 'recipe.yaml')) as fd:
    RECIPE_TXT = fd.read()
RECIPE = yaml.load(RECIPE_TXT, Loader=yaml.FullLoader)

POP_CAT = CategoricalDtype(categories=['POP1_ALL_LAYERS',
                                       'POP2_ALL_LAYERS',
                                       'POP3_ALL_LAYERS',
                                       'POP4_ALL_LAYERS',
                                       'SUB_POP4_L23',
                                       ])

HIER_DICT = {"id": 65535,
             "acronym": "root",
             "name": "root",
             "children": [
                 {"id": 895,
                  "acronym": "ECT",
                  "name": "Ectorhinal area",
                  "children": [
                      {"id": 836, "acronym": "ECT1", "name": "Ectorhinal area/Layer 1"},
                      {"id": 426, "acronym": "ECT2", "name": "Ectorhinal area/Layer 2"},
                      {"id": 427, "acronym": "ECT3", "name": "Ectorhinal area/Layer 3"},
                      {"id": 428, "acronym": "ECT4", "name": "Ectorhinal area/Layer 4"},
                      {"id": 988, "acronym": "ECT5", "name": "Ectorhinal area/Layer 5"},
                      {"id": 977, "acronym": "ECT6", "name": "Ectorhinal area/Layer 6"}
                  ]},
                 {"id": 184,
                  "acronym": "FRP",
                  "name": "Frontal pole, cerebral cortex",
                  "children": [
                      {"id": 68, "acronym": "FRP1", "name": "Frontal pole, layer 1"},
                      {"id": 666, "acronym": "FRP2", "name": "Frontal pole, layer 2"},
                      {"id": 667, "acronym": "FRP3", "name": "Frontal pole, layer 3"},
                      {"id": 526322264, "acronym": "FRP4", "name": "Frontal pole, layer 4"},
                      {"id": 526157192, "acronym": "FRP5", "name": "Frontal pole, layer 5"},
                      {"id": 526157196, "acronym": "FRP6", "name": "Frontal pole, layer 6"}
                  ]},
                 {"name": "Anterior cingulate area, dorsal part",
                  "id": 39,
                  "acronym": "ACAd",
                  "children": [
                      {"name": "Anterior cingulate area, dorsal part, layer 1", "id": 935, "acronym": "ACAd1"},
                      {"name": "Anterior cingulate area, dorsal part, layer 2", "id": 20211, "acronym": "ACAd2"},
                      {"name": "Anterior cingulate area, dorsal part, layer 3", "id": 30211, "acronym": "ACAd3"},
                      {"name": "Anterior cingulate area, dorsal part, layer 4", "id": 30212, "acronym": "ACAd4"},
                      {"name": "Anterior cingulate area, dorsal part, layer 5", "id": 1015, "acronym": "ACAd5"},
                      {"name": "Anterior cingulate area, dorsal part, layer 6", "id": 20919, "acronym": "ACAd6"}
                  ]},
                 {"name": "Secondary motor area",
                  "id": 993,
                  "acronym": "MOs",
                  "children": [
                      {"name": "Secondary motor area, layer 1", "id": 656, "acronym": "MOs1"},
                      {"name": "Secondary motor area, layer 2", "id": 20962, "acronym": "MOs2"},
                      {"name": "Secondary motor area, layer 3", "id": 30962, "acronym": "MOs3"},
                      {"name": "Secondary motor area, layer 4", "id": 30969, "acronym": "MOs4"},
                      {"name": "Secondary motor area, layer 5", "id": 767, "acronym": "MOs5"},
                      {"name": "Secondary motor area, layer 6", "id": 21021, "acronym": "MOs6"}
                  ]}
]}
HIER = Hierarchy(HIER_DICT)


def get_config():
    from white_matter_projections import utils
    config_path = os.path.join(DATADIR, 'config.yaml')
    return utils.Config(config_path)


@contextmanager
def tempdir(prefix):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def gte_(lhs, rhs):
    assert lhs <= rhs, 'lhs: %s not <= rhs: %s' % (lhs, rhs)


def fake_brain_regions():
    raw = np.zeros((5, 5, 5), dtype=np.int)
    raw[1, :, 0:2] = 2
    raw[2, :4, 2:3] = 30
    brain_regions = VoxelData(raw, np.ones(3), offset=np.zeros(3))

    hierarchy = Hierarchy(
        {'id': 1, 'acronym': 'one',
         'children': [
             {'id': 2, 'acronym': 'two', 'children': []},
             {'id': 20, 'acronym': 'twenty', 'children': [
                 {'id': 30, 'acronym': 'thirty', 'children': []}
             ]}
         ]},
    )

    return brain_regions, hierarchy


class CorticalMap(object):
    def __init__(self, paths, view_lookup,
                 hierarchy, brain_regions,
                 center_line_2d, center_line_3d):
        self.paths = paths
        self.view_lookup = view_lookup
        self.hierarchy = hierarchy
        self.brain_regions = brain_regions
        self.center_line_2d = center_line_2d
        self.center_line_3d = center_line_3d

    def load_cortical_view_lookup(self):
        return self.view_lookup

    def load_cortical_paths(self):
        return self.paths

    def load_hierarchy(self):
        return self.hierarchy

    def load_brain_regions(self):
        return self.brain_regions

    @classmethod
    def fake_cortical_map(cls):
        brain_regions, hierarchy = fake_brain_regions()

        view_lookup = -1 * np.ones((5, 5), dtype=int)
        view_lookup[1, 0] = 0
        view_lookup[1, 1] = 1
        view_lookup[2, 2] = 2
        paths = np.array([[25, 26, 30, 31, 35, ],
                          [36, 40, 41, 45, 46, ],
                          [52, 57, 62, 67, 0, ],
                          ])

        center_line_2d = view_lookup.shape[1] / 2.
        center_line_3d = (brain_regions.voxel_dimensions * brain_regions.shape + brain_regions.offset) / 2.
        center_line_3d = center_line_3d[2]
        return cls(paths, view_lookup,
                   hierarchy, brain_regions,
                   center_line_2d, center_line_3d)


def fake_flat_map():
    from white_matter_projections import flat_mapping
    flat_map = VoxelData.load_nrrd(os.path.join(DATADIR, '5x5x5_flat_map.nrrd'))

    brain_regions, hierarchy = fake_brain_regions()
    center_line_2d = 2.5
    center_line_3d = (brain_regions.voxel_dimensions * brain_regions.shape +
                      brain_regions.offset) / 2.

    return flat_mapping.FlatMap(flat_map,
                                brain_regions,
                                hierarchy,
                                center_line_2d,
                                center_line_3d[2])


'''Below are several utils functions used for consistency checks of the methods from
ptypes_generator.py. These functions are only run by unit tests.

This includes
   - The creation of the creation of the interaction strength matrix out
   of a specified tree model
   - The creation of the innervation probability row out
   of a specified tree model.
'''

from networkx.algorithms import shortest_path
from white_matter_projections.ptypes_generator_utils import get_root, get_leaves, is_leave

def get_statistical_interaction_strength(tree, source_id, id1, id2):
    '''Get the statistical interaction strength between to target regions.

    The interaction strength between two regions represented by two leaves
    of the p-types generating tree is the inverse of the innervation probability
    of the lowest common ancestor of the two leaves. The innervation probability
    of a node is the product of the crossing probabilities for each edge of the path
    joining the source to this node.

    Args:
        tree(networkx.DiGraph): directed rooted tree with weighted edges
        source_id(int): node identifier of the source from which axons are cast.
        id1(int): identifier of the first leaf
        id2(int): identifier of the second leaf

    Returns:
        interaction_strength(float): the statistical interaction strength
        I_S(T_1, T_2) between the two target regions T_1 and T_2
        corresponding to the identifiers id1 and id2.
    '''
    path_to_node_1 = shortest_path(tree, source_id, id1)
    path_to_node_2 = shortest_path(tree, source_id, id2)
    lowest_common_ancestor_index = np.nonzero(
        [node in path_to_node_2 for node in path_to_node_1])[0][-1]
    lowest_common_ancestor = path_to_node_1[lowest_common_ancestor_index]
    path_to_lowest_common_ancestor = shortest_path(tree, source_id, lowest_common_ancestor)
    path_length = len(path_to_lowest_common_ancestor)
    innervation_probability = 1.0
    for i in range(path_length - 1):
        node = path_to_lowest_common_ancestor[i]
        next_node = path_to_lowest_common_ancestor[i + 1]
        crossing_probability = tree.edges[(node, next_node)]['crossing_probability']
        innervation_probability *= crossing_probability
    return 1.0 / innervation_probability


def create_statistical_interaction_strength_matrix(tree):
    '''Get the statistical interaction matrix from the p-type generating tree.

    This method retrieves the matrix I_S(A, B) from the p-types generating tree.
    The interaction strength I_S(A, B) between any two target regions A and B
    represented by two leaves of the p-types generating tree
    is the inverse of the innervation probability
    of the lowest common ancestor of the two leaves.
    The innervation probability of a node is the product of
    the crossing probabilities for each edge of the path joining the source to this node.

    Note: diagonal entries are zeroed as self-interactions are excluded.

    Args:
        tree(networkx.DiGraph): directed rooted tree with weighted edges

    Returns:
        interaction_matrix(numpy.ndarray): the statistical interaction strength
        matrix I_S(., .) of shape (number of leaves, number of leaves).

    '''
    source_id = get_root(tree)
    leaves = get_leaves(tree)
    number_of_leaves = len(leaves)
    M = np.zeros([number_of_leaves] * 2)
    for i, id1 in enumerate(leaves):
        for j, id2 in enumerate(leaves):
            if i == j:
                M[i, j] = 0.0  # self-interactions are excluded
            else:
                M[i, j] = get_statistical_interaction_strength(tree, source_id, id1, id2)
    return M


def create_innervation_probability_row(tree):
    '''Get the innervation probabilities from the p-type generating tree.

        This method retrieves the row of innervation probabilities P(S --> T)
        from the p-types generating tree. The innervation probability of a
        leaf is given by the product of the crossing probabilities for each edge
        of the path joining the source to that leaf. It represents the probability that
        an axon issued from the source S innervates the target region T corresponding
        represented as a leaf of the generating tree.

        Note: the location of the source is inferred from the tree structure. The source
        is the only node with in-degree 0.

        Args:
            tree(networkx.DiGraph): directed rooted tree with weighted edges.
            The weight of an edge is the probability that an axon which have
            reached the edge origin crosses it, reaching also the other end.

        Returns:
            row(numpy.ndarray): 1D float array holding the innervation probabilities
            P(S --> T) for every target region T.
    '''
    row_size = len(get_leaves(tree))
    row = np.full(row_size, np.nan)
    # Breadth-first search
    source_id = get_root(tree)
    node_stack = [{'id': source_id, 'innervation_probability': 1.0}]
    while len(node_stack) > 0:
        current_node = node_stack.pop()
        outward_edges = tree.out_edges(current_node['id'])
        for edge in outward_edges:
            crossing_probability = tree.edges[edge]['crossing_probability']
            path_probability = current_node['innervation_probability'] * crossing_probability
            successor = edge[1]
            if is_leave(tree, successor):
                row[successor] = path_probability
            else:
                node_stack.append({
                    'id': successor,
                    'innervation_probability': path_probability
                })
    return row
