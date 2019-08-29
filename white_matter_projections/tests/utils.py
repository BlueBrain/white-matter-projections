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
RECIPE = yaml.load(RECIPE_TXT)

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
    flat_map_shape = (5, 5, )

    brain_regions, hierarchy = fake_brain_regions()
    center_line_2d = flat_map.shape[1] / 2.
    center_line_3d = (brain_regions.voxel_dimensions * brain_regions.shape +
                      brain_regions.offset) / 2.

    return flat_mapping.FlatMap(flat_map,
                                flat_map_shape,
                                brain_regions,
                                hierarchy,
                                center_line_2d,
                                center_line_3d[2])
