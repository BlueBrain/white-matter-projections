import os
from pandas.api.types import CategoricalDtype
from voxcell.hierarchy import Hierarchy
import yaml
import shutil
import tempfile
from contextlib import contextmanager


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
