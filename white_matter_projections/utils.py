'''useful utils'''
import collections
import hashlib
import itertools as it
import json
import logging
import os
import re

from lazy import lazy
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import pyarrow
from pyarrow import feather

import voxcell
from voxcell.nexus import voxelbrain
import yaml


L = logging.getLogger(__name__)

X, Y, Z = 0, 1, 2
cX, cY, cZ, cXYZ = np.s_[:, X], np.s_[:, Y], np.s_[:, Z], np.s_[:, :3]
XYZ = list('xyz')

SIDES = ('left', 'right')
SIDE = CategoricalDtype(categories=SIDES, ordered=True)
HEMISPHERES = ('ipsi', 'contra', )
HEMISPHERE = CategoricalDtype(categories=HEMISPHERES, ordered=True)


class Config(object):
    '''encapsulates config.yaml

    makes it easier to load files pointed by the config, and wraps them in their proper types
    '''

    def __init__(self, config_path):
        assert os.path.exists(config_path), 'config path %s does not exist' % config_path
        self.config_path = os.path.abspath(config_path)

    @lazy
    def config(self):
        '''dictionary containing all keys in config'''
        with open(self.config_path) as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)

        return config

    @lazy
    def volume_dilatation(self):
        ''' Number of pixels used for the volume dilatation '''
        return self.config.get("volume_dilatation", 0)

    @lazy
    def circuit(self):
        '''circuit referenced by config'''
        from bluepy.circuit import Circuit
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

        recipe = macro.MacroConnections.load_recipe(
            recipe,
            self.region_map,
            cache_dir=self.cache_dir,
            region_subregion_translation=self.region_subregion_translation,
        )
        return recipe

    @lazy
    def region_subregion_translation(self):
        '''get atlas compatibility object'''
        translation = self.config.get('region_subregion_translation', {})
        region_subregion_format = translation.get('region_subregion_format',
                                                  '{region}_{subregion}')
        region_subregion_separation_format = translation.get('region_subregion_separation_format',
                                                             '')
        subregion_translation = translation.get('subregion_translation', {})

        return RegionSubregionTranslation(
            region_subregion_format,
            region_subregion_separation_format,
            subregion_translation)

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
    def delay_method(self):
        '''get the style of method, one of 'streamlines', 'direct', 'dive'

        Note: 'direct' is the default
        '''
        delay_method = self.config.get('delay_method', 'direct')
        methods = ('streamlines', 'dive', 'direct')
        assert delay_method in methods, '%s not in %s' % (delay_method, methods)
        return delay_method

    @lazy
    def region_map(self):
        '''heirarchy referenced via atlas in config'''
        if 'hierarchy' in self.config:
            path = self._relative_to_config(self.config_path, self.config['hierarchy'])
            ret = voxcell.region_map.RegionMap.load_json(path)
        else:
            ret = self.atlas.load_region_map()
        return ret

    @lazy
    def region_layer_heights(self):
        '''calculate and cache layer heights from atlas to DataFrame'''

        m = hashlib.sha256()
        m.update(self.config['atlas_url'].encode('utf-8'))
        hexdigest = m.hexdigest()

        path = os.path.join(self.cache_dir, 'region_layer_heights_%s.json' % hexdigest)
        if os.path.exists(path):
            with open(path) as fd:
                layer_heights = json.load(fd)
        else:
            layers = list(self.recipe.layer_profiles.subregion.unique())
            layer_heights = calculate_region_layer_heights(
                self.atlas, self.region_map, self.regions, layers, self.config['layer_splits'])
            with open(path, 'w') as fd:
                json.dump(layer_heights, fd)

        return region_layer_heights(layer_heights)

    @lazy
    def flat_map(self):
        '''conversion from voxel space to 2d flat space'''
        from white_matter_projections import flat_mapping
        config = self.config['flat_mapping']
        flat_map = flat_mapping.FlatMap.load(config['flat_map'],
                                             config['brain_regions'],
                                             config['hierarchy'],
                                             config['center_line_2d'],
                                             config['center_line_3d'])
        return flat_map

    def get_cells(self, population_filter=None):
        '''Get cells in circuit with the mtype in `projecting_mtypes` unless `include_all`'''
        m = hashlib.sha256()
        m.update(self.config['circuit_config'].encode('utf-8'))
        hexdigest = m.hexdigest()

        path = os.path.join(self.cache_dir, 'cells_%s.feather' % hexdigest)
        if os.path.exists(path):
            cells = read_frame(path)
        else:
            cells = self.circuit.cells.get()
            # orientation is removed since it historically hasn't been used
            # and it's a large *object* (not an array)
            del cells['orientation']
            write_frame(path, cells, reset_index=False)

        if population_filter is not None and population_filter != 'Empty':
            categories = self.config['populations_filters'][population_filter]
            categories = categories
            cells = cells.query('mtype in @categories')

        return cells

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


class RegionSubregionTranslation(object):
    '''adapt recipes to various hierarchies '''
    def __init__(self,
                 region_subregion_format='{region}_{subregion}',
                 region_subregion_separation_format=None,
                 subregion_translation=None):
        '''
        Args:
            region_subregion_format(str): format string describing the region/subregion ->
            acronym conversion
            region_subregion_separation_format(str):
            subregion_translation(dict): subregion alias -> subregion in atlas
        '''
        if region_subregion_format:
            if 'region' not in region_subregion_format:
                raise ValueError(
                    f'subregion_translation must contain "region": {region_subregion_format}')

        if subregion_translation is None:
            subregion_translation = {}

        self.region_subregion_format = region_subregion_format
        self.region_subregion_separation_format = region_subregion_separation_format
        self.subregion_translation = subregion_translation

    def extract_region_subregion_from_acronym(self, acronym):
        '''use the config to extract the region/subregion from an acronym

        Note: config key name is `region_subregion_separation_format`
        '''
        match = re.match(self.region_subregion_separation_format, acronym)

        assert match, f'Could not find {self.region_subregion_separation_format} in {acronym}'

        region = str(match['region'])
        subregion = str(match['subregion'])

        return region, subregion

    def translate_subregion(self, subregion):
        '''use the config `subregion_translation` to translate subregion name'''
        return self.subregion_translation.get(subregion, subregion)

    def get_region_layer_to_id(self, region_map, region, layers):
        '''map `region` name and `layers` to ids based on `region_map`

        an ID of -1 means it was not found
        '''
        return {layer: self.region_subregion_to_id(region_map, region, layer)
                for layer in layers}

    def region_subregion_to_id(self, region_map, region, subregion):
        '''Populate ids for acronyms

        Args:
            region_map(voxcell.RegionMap): hierarchy to verify population region against
            region(str): region to lookup
            subregion(str/num): subregion

        Returns:
            tuple of array of ids corresponding to rows in df and a dataframe w/ the removed rows
        '''
        acronym = self.region_subregion_format.format(region=region, subregion=subregion)
        id_ = region_map.find(acronym, 'acronym', with_descendants=True)

        if len(id_) == 0:
            L.warning('Missing region %s, subregion: %s: (%s)',
                      region, subregion, acronym)
            return -1
        elif len(id_) > 1:
            L.warning('Too many ids for region %s, subregion: %s: (%s)',
                      region, subregion, acronym)
            return -1

        return next(iter(id_))


def calculate_region_layer_heights(atlas, region_map, regions, layers, layer_splits):
    '''find region layer heights for layers

    Args:
        atlas(voxcell.nexus.voxelbrain): atlas to be used for region lookup
        region_map(voxcell.RegionMap): hierarchy for region lookup
        regions(list): regions to have their height calculated
        layers(list): the subregions of interest; this is 'post-split'
        layer_splits(dict): subregion name -> [(new_name, factor), ...]

    Note: the word 'layer' is used to follow the recipe/paper convention
    '''
    # pylint: disable=too-many-locals
    # set the layers to the pre-split ones
    layers = set(layers)

    for pre_split_layer, post_split_layers in layer_splits.items():
        for post_split_layer, _ in post_split_layers:
            if post_split_layer in layers:
                layers.discard(post_split_layer)
                layers.add(pre_split_layer)

    brain_regions = atlas.load_data('brain_regions')

    thicknesses = collections.defaultdict(dict)
    for region in regions:
        ids = region_map.find(region, 'acronym', 'id', with_descendants=True)
        mask = np.isin(brain_regions.raw, list(ids))
        for layer in layers:
            ph = atlas.load_data('[PH]%s' % layer).raw[mask]
            thickness = np.mean(ph[:, 1] - ph[:, 0])
            if layer in layer_splits:
                for name, fraction in layer_splits[layer]:
                    thicknesses[region][name] = thickness * fraction
            else:
                thicknesses[region][layer] = thickness

    return dict(thicknesses)


def region_layer_heights(layer_heights):
    '''convert layer heights dictionary to pandas DataFrame'''
    return pd.DataFrame.from_dict(layer_heights, orient='index')


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
    ret = pd.DataFrame(index=layer_heights.index, columns=profiles.name.unique(), dtype=np.float64)
    ret.index.name = 'region'

    for profile_name, profile in profiles.groupby('name'):
        for region in layer_heights.index:
            w = layer_heights.loc[region].to_numpy()
            p = (profile
                 .set_index('subregion')
                 .loc[layer_heights.columns]['relative_density']
                 .to_numpy()
                 )
            ret.loc[region][profile_name] = np.sum(w) / np.dot(p, w)

    return ret


def ensure_path(path):
    '''make sure path exists'''
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass
    return path


def write_frame(path, df, reset_index=True):
    """Write a DataFrame to disk using feather serialization format

    Note: This performs destructive changes to the dataframe, caller must
    save it if they need an unchanged version
    """
    assert path.endswith('.feather'), 'Can only write feathers at the moment'

    df.columns = map(str, df.columns)
    df.reset_index(drop=reset_index, inplace=True)
    feather.write_feather(df, path)


def read_frame(path, columns=None):
    '''read a dataframe, optionally specifying columns'''
    # this is a work around for so that pyarrow doesn't use mmap,
    # which causes *big* slowdown (more than 10x) on GPFS
    source = pyarrow.OSFile(path)

    if path.endswith('.feather'):
        df = feather.read_feather(source, columns=columns)
        try:
            df['index'] = feather.read_feather(source, columns=['index'])
        except pyarrow.lib.ArrowInvalid as e:
            assert str(e).startswith('Field named')

        if 'index' in df:
            df = df.set_index('index')
        return df

    assert False, 'Need to end with .feather: %s' % path
    return None


def partition_left_right(df, side, center_line_3d):
    '''return df from `side` based on `center_line_3d`'''
    assert side in ('left', 'right', )
    if side == 'right':
        mask = center_line_3d < df.z
    else:
        mask = df.z <= center_line_3d
    return df[mask]


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


def hierarchy_2_df(content):
    '''convert a AIBS hierarchy.json file into a DataFrame
    Args:
        content(str): content of heirarchy.json file

    Returns:
        pd.DataFrame with columns ['acronym', 'id', 'name', 'parent_id']
    '''
    if 'msg' in content:
        if len(content['msg']) > 1:
            raise Exception("Unexpected JSON layout (more than one 'msg' child)")
        content = content['msg'][0]

    def recurse(node, parent_id, res):
        '''helper to recursively add all children to `res`'''
        res.append((node['acronym'], node['id'], node['name'], parent_id))
        if 'children' in node:
            for child_node in node['children']:
                recurse(child_node, node['id'], res)

    res = [('root', 0, 'root', -1)]
    recurse(content, 0, res)

    ret = pd.DataFrame(res, columns=['acronym', 'id', 'name', 'parent_id'])
    assert len(ret) == len(ret.id.unique()), 'Duplicate ids in hierarchy'
    return ret.set_index('id')


def get_acronym_volumes(acronyms, brain_regions, region_map):
    '''
    Returns:
        pd.DataFrame with index `acronyms` with values for `volume`
    '''
    ret = []
    for acronym in acronyms:
        ids = region_map.find(acronym, 'acronym', with_descendants=True)
        count = np.count_nonzero(np.isin(brain_regions.raw, list(ids)))
        ret.append((acronym, count * brain_regions.voxel_volume))
    return pd.DataFrame(ret, columns=['acronym', 'volume']).set_index('acronym')
