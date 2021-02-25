import os
import shutil
import tempfile
import yaml

from mock import Mock

import numpy as np
import voxcell
from contextlib import contextmanager
from pandas.api.types import CategoricalDtype


BASEDIR = os.path.dirname(__file__)
DATADIR = os.path.join(BASEDIR, 'data')
with open(os.path.join(DATADIR, 'recipe.yaml')) as fd:
    RECIPE_TXT = fd.read()
RECIPE = yaml.load(RECIPE_TXT, Loader=yaml.FullLoader)

FLAT_MAP_NAMES = ['Allen Dorsal Flatmap', ]

SUBREGION_TRANSLATION = {'l1': '1',
                         'l2': '2',
                         'l3': '3',
                         'l23': '23',
                         'l4': '4',
                         'l5': '5',
                         'l6': '6',
                         'l6a': '6a',
                         'l6b': '6b',
                         }
REGION_SUBREGION_FORMAT = '@{region}(?:_l|;|){subregion}'

REGION_SUBREGION_SEPARATION_FORMAT = '(?P<region>[^\d]+)(?:_l|;)(?P<subregion>\d+)'

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
                      {"id": 836, "acronym": "ECT;1", "name": "Ectorhinal area/Layer 1"},
                      {"id": 426, "acronym": "ECT;2", "name": "Ectorhinal area/Layer 2"},
                      {"id": 427, "acronym": "ECT;3", "name": "Ectorhinal area/Layer 3"},
                      {"id": 428, "acronym": "ECT;4", "name": "Ectorhinal area/Layer 4"},
                      {"id": 988, "acronym": "ECT;5", "name": "Ectorhinal area/Layer 5"},
                      {"id": 977, "acronym": "ECT;6", "name": "Ectorhinal area/Layer 6"}
                  ]},
                 {"id": 184,
                  "acronym": "FRP",
                  "name": "Frontal pole, cerebral cortex",
                  "children": [
                      {"id": 68, "acronym": "FRP_l1", "name": "Frontal pole, layer 1"},
                      {"id": 666, "acronym": "FRP_l2", "name": "Frontal pole, layer 2"},
                      {"id": 667, "acronym": "FRP_l3", "name": "Frontal pole, layer 3"},
                      {"id": 526322264, "acronym": "FRP_l4", "name": "Frontal pole, layer 4"},
                      {"id": 526157192, "acronym": "FRP_l5", "name": "Frontal pole, layer 5"},
                      {"id": 526157196, "acronym": "FRP_l6", "name": "Frontal pole, layer 6"}
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
                  ]},

                  {"name": "Rhomboid nucleus",  # childless region
                   "id": 189,
                   "acronym": "RH",
                   "children": [],
                   },

                  {"name": "Primary somatosensory area, barrel field",  # has layers and has an inner region (VISrll)
                   "id": 329,
                   "acronym": "SSp-bfd",
                   "children": [
                    {
                     "name": "Primary somatosensory area, barrel field, layer 1",
                     "children": [],
                     "id": 981,
                     "acronym": "SSp-bfd1",
                    },
                    {
                     "name": "Primary somatosensory area, barrel field, layer 2",
                     "children": [],
                     "id": 20201,
                     "acronym": "SSp-bfd2",
                    },
                    {
                     "name": "Primary somatosensory area, barrel field, layer 3",
                     "acronym": "SSp-bfd3",
                     "children": [],
                     "id": 30201,
                    },
                    {
                     "name": "Primary somatosensory area, barrel field, layer 4",
                     "children": [],
                     "id": 1047,
                     "acronym": "SSp-bfd4",
                    },
                    {
                     "name": "Primary somatosensory area, barrel field, layer 5",
                     "children": [],
                     "id": 1070,
                     "acronym": "SSp-bfd5",
                    },
                    {
                     "name": "Primary somatosensory area, barrel field, layer 6a",
                     "children": [],
                     "id": 1038,
                     "acronym": "SSp-bfd6a",
                    },
                    {
                     "name": "Primary somatosensory area, barrel field, layer 6b",
                     "children": [],
                     "id": 1062,
                     "acronym": "SSp-bfd6b",
                    },
                    {
                     "name": "Rostrolateral lateral visual area",
                     "id": 480149202,
                     "acronym": "VISrll",
                     "children": [
                      {
                       "name": "Rostrolateral lateral visual area, layer 1",
                       "children": [],
                       "id": 480149206,
                       "acronym": "VISrll1",
                      },
                      {
                       "name": "Rostrolateral lateral visual area, layer 2",
                       "children": [],
                       "id": 480169210,
                       "acronym": "VISrll2",
                      },
                      {
                       "name": "Rostrolateral lateral visual area, layer 3",
                       "acronym": "VISrll3",
                       "children": [],
                       "id": 480179210,
                      },
                      {
                       "name": "Rostrolateral lateral visual area, layer 4",
                       "children": [],
                       "id": 480149214,
                       "acronym": "VISrll4",
                      },
                      {
                       "name": "Rostrolateral lateral visual area,layer 5",
                       "children": [],
                       "id": 480149218,
                       "acronym": "VISrll5",
                      },
                      {
                       "name": "Rostrolateral lateral visual area, layer 6a",
                       "children": [],
                       "id": 480149222,
                       "acronym": "VISrll6a",
                      },
                      {
                       "name": "Rostrolateral lateral visual area, layer 6b",
                       "children": [],
                       "id": 480149226,
                       "acronym": "VISrll6b",
                      }
                     ],
                    }
                   ],
                  },

]}
REGION_MAP = voxcell.RegionMap.from_dict(HIER_DICT)


def get_config():
    from white_matter_projections import utils
    config_path = os.path.join(DATADIR, 'config.yaml')
    return utils.Config(config_path)


def get_region_subregion_translation():
    from white_matter_projections import utils
    return utils.RegionSubregionTranslation(
       region_subregion_format=REGION_SUBREGION_FORMAT,
       region_subregion_separation_format=REGION_SUBREGION_SEPARATION_FORMAT,
       subregion_translation=SUBREGION_TRANSLATION)


def recipe_brain_regions():
    raw = np.zeros((5, 5, 5), dtype=np.int)
    raw[0, 0, 0] = 836  # "ECT;1"
    raw[0, 0, 1] = 426  # "ECT;2"
    raw[0, 0, 2] = 427  # "ECT;3"
    raw[0, 0, 3] = 428  # "ECT;4"
    raw[0, 0, 4] = 988  # "ECT;5"
    raw[0, 1, 0] = 977  # "ECT;6"

    raw[1, 0, 0] = 68  # "FRP_l1"
    raw[1, 0, 1] = 666  # "FRP_l2"
    raw[1, 0, 2] = 667  # "FRP_l3"
    raw[1, 0, 3] = 526322264  # "FRP_l4"
    raw[1, 0, 4] = 526157192  # "FRP_l5"
    raw[1, 1, 0] = 526157196  # "FRP_l6"

    raw[2, 0, 0] = 935  # "ACAd1"
    raw[2, 0, 1] = 20211  # "ACAd2"
    raw[2, 0, 2] = 30211  # "ACAd3"
    raw[2, 0, 3] = 30212  # "ACAd4"
    raw[2, 0, 4] = 1015  # "ACAd5"
    raw[2, 1, 0] = 20919  # "ACAd6"

    raw[3, 0, 0] = 656  # "MOs1"
    raw[3, 0, 1] = 20962  # "MOs2"
    raw[3, 0, 2] = 30962  # "MOs3"
    raw[3, 0, 3] = 30969  # "MOs4"
    raw[3, 0, 4] = 767  # "MOs5"
    raw[3, 1, 0:2] = 21021  # "MOs6"

    raw[4, 0, 0] = 981  # "SSp-bfd1"
    raw[4, 0, 1] = 20201  # "SSp-bfd2"
    raw[4, 0, 2] = 30201  # "SSp-bfd3"
    raw[4, 0, 3] = 1047  # "SSp-bfd4"
    raw[4, 0, 4] = 1070  # "SSp-bfd5"
    raw[4, 1, 0] = 1038  # "SSp-bfd6a"
    raw[4, 1, 1] = 1062  # "SSp-bfd6b"
    raw[4, 1, 2] = 480149202  # "VISrll"
    raw[4, 1, 3] = 480149206  # "VISrll1"
    raw[4, 1, 4] = 480169210  # "VISrll2"
    raw[4, 2, 0] = 480179210  # "VISrll3"
    raw[4, 2, 1] = 480149214  # "VISrll4"
    raw[4, 2, 2] = 480149218  # "VISrll5"
    raw[4, 2, 3] = 480149222  # "VISrll6a"
    raw[4, 2, 4] = 480149226  # "VISrll6b"

    raw[4, 3, 0] = 189  # "RH"

    brain_regions = voxcell.VoxelData(raw, np.ones(3), offset=np.zeros(3))
    return brain_regions


def make_mock_atlas(brain_regions, region_map, have_ph=False, **kwargs):
    atlas = Mock()

    def load_atlas(name):
        if name == 'brain_regions':
            return brain_regions
        elif have_ph and name.startswith('[PH]'):
            raw = np.zeros((5, 5, 5, 2))
            raw[:, :, :, 1] = 1.
            ph = voxcell.VoxelData(raw, np.ones(3), offset=np.zeros(3))
            return ph
        elif name in kwargs:
            return kwargs[name]

        raise Exception('Unknown atlas: %s' % name)

    atlas.load_data = load_atlas
    atlas.region_map = region_map

    return atlas



@contextmanager
def tempdir(prefix):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def fake_brain_regions():
    raw = np.zeros((5, 5, 5), dtype=np.int)
    raw[1, :, 0:2] = 2
    raw[2, :4, 2:3] = 30
    brain_regions = voxcell.VoxelData(raw, np.ones(3), offset=np.zeros(3))

    region_map = voxcell.RegionMap.from_dict(
        {'id': 1, 'acronym': 'one',
         'children': [
             {'id': 2, 'acronym': 'two', 'children': []},
             {'id': 20, 'acronym': 'twenty', 'children': [
                 {'id': 30, 'acronym': 'thirty', 'children': []}
             ]}
         ]},
    )

    return brain_regions, region_map


class CorticalMap(object):
    def __init__(self, paths, view_lookup,
                 region_map, brain_regions,
                 center_line_2d, center_line_3d):
        self.paths = paths
        self.view_lookup = view_lookup
        self.region_map = region_map
        self.brain_regions = brain_regions
        self.center_line_2d = center_line_2d
        self.center_line_3d = center_line_3d

    def load_cortical_view_lookup(self):
        return self.view_lookup

    def load_cortical_paths(self):
        return self.paths

    def load_region_map(self):
        return self.region_map

    def load_brain_regions(self):
        return self.brain_regions

    @classmethod
    def fake_cortical_map(cls):
        brain_regions, region_map = fake_brain_regions()

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
                   region_map, brain_regions,
                   center_line_2d, center_line_3d)


def fake_flat_map():
    from white_matter_projections import flat_mapping
    flat_map = voxcell.VoxelData.load_nrrd(os.path.join(DATADIR, '5x5x5_flat_map.nrrd'))

    brain_regions, region_map = fake_brain_regions()
    center_line_2d = 2.5
    center_line_3d = (brain_regions.voxel_dimensions * brain_regions.shape +
                      brain_regions.offset) / 2.

    return flat_mapping.FlatMapBase(flat_map,
                                    brain_regions,
                                    region_map,
                                    center_line_2d,
                                    center_line_3d[2])

def fake_allocations():
    return {'source_population0': {'projection0': np.arange(10),
                                   'projection1': np.arange(10, 20),
                                   },
            'source_population1': {'projection0': np.arange(20, 30),
                                   'projection1': np.arange(30, 40),
                                   },
            }


def fake_projection_mapping():
    vertices = np.array(list(zip([0., 10., 0.], [0., 0., 10.])))

    ret = {'source_population0': {'projection0': {'target_population': 'target00',
                                                  'variance': 1.,
                                                  'vertices': vertices,
                                                  'base_system': 'base_system0',
                                                  },
                                  'projection1': {'target_population': 'target01',
                                                  'variance': 1.,
                                                  'vertices': vertices,
                                                  'base_system': 'base_system1',
                                                  },
                                  'vertices': vertices,
                                  'base_system': 'base_system0',
                                  },
           'source_population1': {'projection0': {'target_population': 'target10',
                                                  'variance': 1.,
                                                  'vertices': vertices,
                                                  'base_system': 'base_system0',
                                                  },
                                  'projection1': {'target_population': 'target11',
                                                  'variance': 1.,
                                                  'vertices': vertices,
                                                  'base_system': 'base_system1',
                                                  },
                                  'vertices': vertices,
                                  'base_system': 'base_system1',
                                  },
           }
    return ret



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
