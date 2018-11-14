'''utils'''
import os

import itertools as it
import logging

from lazy import lazy
from voxcell.nexus import voxelbrain
import voxcell
import numpy as np
import pandas as pd
import yaml

L = logging.getLogger(__name__)

X, Y, Z = 0, 1, 2
cX, cY, cZ = np.s_[:, X], np.s_[:, Y], np.s_[:, Z]


class Config(object):
    '''encapsulates config.yaml

    makes it easier to load files pointed by the config, and wraps them in their proper types
    '''

    def __init__(self, config_path):
        assert os.path.exists(config_path), 'config path %s does not exist' % config_path
        self.config_path = config_path

    @lazy
    def config(self):
        '''dictionary containing all keys in config'''
        with open(self.config_path) as fd:
            config = yaml.load(fd)
        return config

    @lazy
    def circuit(self):
        '''circuit referenced by config'''
        from bluepy.v2.circuit import Circuit
        return Circuit(self.config['circuit_config'])

    @lazy
    def regions(self):
        '''regions referenced in config'''
        regions = list(it.chain.from_iterable(c[1] for c in self.config['module_grouping']))
        return regions

    @lazy
    def recipe(self):
        '''MacroConnections recipe referenced in config'''
        from white_matter_projections import macro

        recipe_path = self.config['projections_recipe']
        recipe_path = self._relative_to_config(self.config_path, recipe_path)

        with open(recipe_path) as fd:
            recipe = yaml.load(fd)

        recipe = macro.MacroConnections.load_recipe(recipe, self.hierarchy)
        return recipe

    @lazy
    def atlas(self):
        '''atlas referenced in config'''
        atlas = voxelbrain.Atlas.open(self.config['atlas_url'],
                                      cache_dir=self.config['cache_dir'])
        return atlas

    @lazy
    def hierarchy(self):
        '''heirarchy referenced via atlas in config'''
        if 'hierarchy' in self.config:
            path = self._relative_to_config(self.config_path, self.config['hierarchy'])
            hier = voxcell.hierarchy.Hierarchy.load_json(path)
        else:
            hier = self.atlas.load_hierarchy()
        return hier

    @lazy
    def region_layer_heights(self):
        '''converted dictionary in config to DataFrame'''
        return region_layer_heights(self.config['region_layer_heights'])

    @staticmethod
    def _relative_to_config(config_path, path):
        '''helper so full paths don't need to be embedded in config'''
        if not os.path.exists(path):
            relative = os.path.join(os.path.dirname(config_path), path)
            if os.path.exists(relative):
                path = relative
            else:
                raise Exception('Cannot find path: %s' % path)
        return path


def region_layer_heights(layer_heights, columns=('l1', 'l2', 'l3', 'l4', 'l5', 'l6')):
    '''convert layer heights dictionary to pandas DataFrame'''
    return pd.DataFrame.from_dict(layer_heights, orient='index', columns=columns)


def perform_module_grouping(df, module_grouping):
    '''group regions in df, a DataFrame into a multiindex based on `module_grouping`

    Args:
        df(DataFrame): dataframe to reindex
        module_grouping(list of [Module, [list of regions]]): provides ordering and grouping

    Note: it seems natural to have `module_grouping` be a dictionary, but then
    the ordering is lost
    '''
    tuples = tuple(it.chain.from_iterable(
        tuple(it.product([k], v)) for k, v in module_grouping))
    midx = pd.MultiIndex.from_tuples(tuples, names=['Module', 'Region'])

    ret = df.copy().reindex(index=midx, level=1)
    ret = ret.reindex(columns=midx, level=1)

    ret.index.name = 'Source Population'
    ret.columns.name = 'Target Population Density'
    return ret


def normalize_layer_profiles(layer_heights, profiles):
    '''calculate 'x' as described in white-matter-projections whitepaper

    Args:
        layer_heights: mean height of all layers in all regions
        profiles: profiles for each region, as defined in recipe

    As per Michael Reiman in NCX-121:
        Show overall density in each layer of the target region. Method: let w be the vector of
        layer widths of the target region, p the layer profile of a projection:
           x  = sum(w) / sum(w * p)
    '''
    ret = pd.DataFrame(index=layer_heights.index, columns=profiles.name.unique())
    ret.index.name = 'region'
    for profile_name, profile in profiles.groupby('name'):
        for region in layer_heights.index:
            w = layer_heights.loc[region].values
            p = profile['relative_density'].values
            ret.loc[region][profile_name] = np.sum(w) / np.dot(p, w)

    return ret


def get_region_layer_to_id(hier, region, layers):
    '''map `region` name and `layers` to to ids based on `hier`

    an ID of zero means it was not found
    '''
    ret = {}
    for i in layers:
        ids_ = hier.collect('acronym', '%s%d' % (region, i), 'id')
        if len(ids_) == 0:
            ret[i] = 0
        elif len(ids_) == 1:
            ret[i] = next(iter(ids_))
        else:
            L.warning('Got more than one id for region: %s, layer: %d', region, i)
            ret[i] = 0
    return ret
