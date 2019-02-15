'''utils'''
import os

import itertools as it
import logging

from lazy import lazy
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import pyarrow
from pyarrow import feather

import voxcell
from voxcell.nexus import voxelbrain
import yaml

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache


L = logging.getLogger(__name__)

X, Y, Z = 0, 1, 2
cX, cY, cZ, cXYZ = np.s_[:, X], np.s_[:, Y], np.s_[:, Z], np.s_[:, :3]
XYZ = list('xyz')
HEMISPHERE = CategoricalDtype(categories=['ipsi', 'contra'], ordered=True)
SIDE = CategoricalDtype(categories=['left', 'right'], ordered=True)


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
            recipe = fd.read()

        recipe = macro.MacroConnections.load_recipe(recipe, self.hierarchy,
                                                    cache_dir=self.cache_dir)
        return recipe

    @lazy
    def atlas(self):
        '''atlas referenced in config'''
        atlas = voxelbrain.Atlas.open(self.config['atlas_url'],
                                      cache_dir=self.cache_dir)
        return atlas

    @lazy
    def cache_dir(self):
        '''get the cache path'''
        path = self.config['cache_dir']
        ensure_path(path)
        return path

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

    @lazy
    def flat_map(self):
        '''conversion from voxel space to 2d flat space'''
        from white_matter_projections import flat_mapping
        config = self.config['flat_mapping']
        flat_map = flat_mapping.FlatMap.load(config['cortical_map'],
                                             config['brain_regions'],
                                             config['hierarchy'],
                                             config['center_line_2d'],
                                             config['center_line_3d'],
                                             self.cache_dir)
        return flat_map

    @lru_cache()
    def get_cells(self, population_filter=None):
        '''Get cells in circuit with the mtype in `projecting_mtypes` unless `include_all`'''
        cells = self.circuit.cells.get()

        if population_filter is not None:
            categories = self.config['populations_filters'][population_filter]
            categories = categories
            cells = cells.query('mtype in @categories')

        return cells

    def voxel_to_flat(self):
        '''get the voxel to flat mapping

        Returns: VoxelData with ravelled values corresponding to the
        x/y coordinates in the flat_map.view_lookup
        '''
        path = os.path.join(self.cache_dir, 'voxel_to_flat.nrrd')
        if os.path.exists(path):
            ret = voxcell.VoxelData.load_nrrd(path)
        else:
            from white_matter_projections import flat_mapping
            ret = flat_mapping.get_voxel_to_flat_mapping(self)
            ret.save_nrrd(path)

        return ret

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


def ensure_path(path):
    '''make sure path exists'''
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def write_frame(path, df):
    """Write a DataFrame to disk using feather serialization format

    Note: This performs destructive changes to the dataframe, caller must
    save it if they need an unchanged version
    """
    assert path.endswith('.feather'), 'Can only write feathers at the moment'

    df.columns = map(str, df.columns)
    df.reset_index(drop=True, inplace=True)
    feather.write_feather(df, path)


def read_frame(path, columns=None):
    '''read a dataframe, optionally specifying columns'''
    # this is a work around for so that pyarrow doesn't use mmap,
    # which causes *big* slowdown (more than 10x) on GPFS
    source = pyarrow.OSFile(path)

    if path.endswith('.feather'):
        return feather.FeatherReader(source).read_pandas(columns)
    assert False, 'Need to end with .feather: %s' % path
    return None


def mirror_vertices_y(vertices, center_line):
    '''vertices are only defined in recipe for right side of brain, transpose to left'''
    ret = vertices.copy()
    ret[cY] = 2 * center_line - ret[cY]
    return ret


def in_2dtriangle(vertices, points):
    '''check if points are in in triangled `vertices`'''
    def det(a, b, c):
        '''det'''
        return ((b[cX] - a[cX]) * (c[cY] - a[cY]) -
                (b[cY] - a[cY]) * (c[cX] - a[cX]))

    def _in_triangle(v0, v1, v2, p):
        '''check if p is in triangle defined by vertices v0, v1, v2'''
        # XXX: this probably doesn't short circuit, so might be excess computation?
        return ((det(v1[None, :], v2[None, :], p) >= 0.) &
                (det(v2[None, :], v0[None, :], p) >= 0.) &
                (det(v0[None, :], v1[None, :], p) >= 0.))

    v0, v1, v2 = vertices[0, :], vertices[1, :], vertices[2, :]

    # check triangle winding, centroid should be in triangle
    centroid = np.mean(vertices, axis=0)
    if _in_triangle(v0, v1, v2, centroid[None, :])[0]:
        ret = _in_triangle(v0, v1, v2, points)
    else:
        ret = _in_triangle(v0, v2, v1, points)  # swap last two vertices
    return ret


def raster_triangle(vertices):
    '''given verticies, return all indices that would rasterize the triangle'''
    min_x, min_y = int(np.min(vertices[cX], axis=0)), int(np.min(vertices[cY], axis=0))
    max_x, max_y = int(np.max(vertices[cX], axis=0)), int(np.max(vertices[cY], axis=0))

    x, y = np.mgrid[min_x:max_x + 1, min_y:max_y + 1]
    points = np.vstack((x.ravel(), y.ravel())).T
    return points[in_2dtriangle(vertices, points)]  # swapping the last two vertices


def is_mirror(side, hemisphere):
    '''decide whether vertices needs to be mirrored

    Recipe is organized by the 'source', so when the side (ie: where the
    synapses are) is 'right', we need to flip for contra such that the source is
    in the opposite hemisphere, and the synapses are in the right one

    Same logic for 'left'
    '''
    return ((side == 'right' and hemisphere == 'contra') or
            (side == 'left' and hemisphere == 'ipsi'))


def population2region(populations, population_name):
    '''returns the region acronym for the hierarchy based on the population'''
    populations = populations.query('population == @population_name')
    assert len(populations), 'Population %s not found in populations' % population_name
    return populations.iloc[0].region
