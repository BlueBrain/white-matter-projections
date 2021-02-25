'''useful utils'''
import collections
import hashlib
import itertools as it
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
        with open(self.config_path, encoding='utf-8') as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)

        return config

    @property
    def seed(self):
        '''get seed for this config, otherwise use 42'''
        return self.config.get('seed', 42)

    @property
    def rng(self):
        '''get np.random.default_rng with seed for this config'''
        return np.random.default_rng(seed=self.seed)

    @property
    def volume_dilatation(self):
        ''' Number of pixels used for the volume dilatation '''
        return self.config.get("volume_dilatation", 0)

    @property
    def circuit(self):
        '''circuit referenced by config'''
        from bluepy import Circuit
        return Circuit(self.config['circuit_config'])

    @property
    def regions(self):
        '''regions referenced in config'''
        regions = list(it.chain.from_iterable(c[1] for c in self.config['module_grouping']))
        return regions

    @property
    def recipe_path(self):
        '''get the specified recipe path'''
        recipe_path = self.config['projections_recipe']
        recipe_path = self._relative_to_config(self.config_path, recipe_path)
        return recipe_path

    @lazy
    def recipe(self):
        '''MacroConnections recipe referenced in config'''
        from white_matter_projections import macro

        with open(self.recipe_path, encoding='utf-8') as fd:
            recipe = fd.read()

        recipe = macro.MacroConnections.load_recipe(
            recipe,
            self.region_map,
            cache_dir=self.cache_dir,
            region_subregion_translation=self.region_subregion_translation,
            flat_map_names=self.config['flat_mapping'].keys(),
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

    @property
    def atlas(self):
        '''atlas referenced in config'''
        atlas = voxelbrain.Atlas.open(self.config['atlas_url'],
                                      cache_dir=self.cache_dir)
        return atlas

    @property
    def cache_dir(self):
        '''get the cache path'''
        path = self.config['cache_dir']
        ensure_path(path)
        return path

    @property
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

    def flat_map(self, base_system):
        '''conversion from voxel space to 2d flat space'''
        from white_matter_projections import flat_mapping
        config = self.config['flat_mapping'][base_system]

        flat_map = flat_mapping.FlatMap(config['flat_map'],
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


def ensure_path(path):
    '''make sure path exists'''
    path = str(path)
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
        # TODO: this probably doesn't short circuit, so might be excess computation?
        # to reduce computation, we could do it iteratively, and only do a subset each time
        # of the values that pass the `>= 0.` test, but this would require more bookkeeping
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

    assert side in SIDES, f'unknown: {side}'
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
        content(str): content of hierarchy.json file

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


def get_acronym_volumes(acronyms, brain_regions, region_map, center_line_3d, side):
    ''' get the volume that is occupied by `acronyms` in the brain_regions

    Returns:
        pd.DataFrame with index `acronyms` with values for `volume`
    '''
    assert side in SIDES, f'unknown: {side}'

    raw = brain_regions.raw.copy()
    midline_idx = brain_regions.positions_to_indices([[0., 0., float(center_line_3d)]])
    midline_idx = midline_idx[0][Z]

    if side == 'right' and brain_regions.voxel_dimensions[Z] > 0.:
        raw = raw[:, :, midline_idx:]
    else:
        raw = raw[:, :, :midline_idx]

    ret = []
    for acronym in acronyms:
        ids = region_map.find(acronym, 'acronym', with_descendants=True)
        count = np.count_nonzero(np.isin(raw, list(ids)))
        ret.append((acronym, count * brain_regions.voxel_volume))
    ret = pd.DataFrame(ret, columns=['acronym', 'volume']).set_index('acronym')
    return ret


def generate_seed(*args):
    '''stringify args, and make a 32bit seed out of it'''
    m = hashlib.sha256()
    for arg in args:
        m.update(str(arg).encode('utf-8'))
    seed = int(m.hexdigest()[:8], base=16)
    return seed


def choice(probabilities, rng):
    '''Given an array of shape (N, M) of probabilities (not necessarily normalized)
    returns an array of shape (N), with one element choosen from every row according
    to the probabilities normalized on this row
    '''

    cum_distances = np.cumsum(probabilities, axis=1)
    cum_distances = cum_distances / np.sum(probabilities, axis=1, keepdims=True)

    rand_cutoff = rng.random((len(cum_distances), 1))

    idx = np.argmax(rand_cutoff < cum_distances, axis=1)
    return idx


class ErrorCloseToZero(Exception):
    '''Raised if normalizing if sum of probabilities is close to zero'''


def normalize_probability(p):
    ''' Normalize vector of probabilities `p` so that sum(p) == 1. '''
    norm = np.sum(p)
    if norm < 1e-7:
        raise ErrorCloseToZero("Could not normalize almost-zero vector")
    return p / norm


def ensure_iter(v):
    '''ibid'''
    if isinstance(v, collections.abc.Iterable):
        return v
    else:
        return (v, )
