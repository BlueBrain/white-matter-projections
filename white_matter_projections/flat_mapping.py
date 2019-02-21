'''voxel space to 2d flat space'''
import itertools as it
import json
import logging
import operator
import os

from functools import reduce  # pylint: disable=redefined-builtin

import h5py
import numpy as np
import pandas as pd
import requests
import voxcell

from joblib import Parallel, delayed
from white_matter_projections.utils import X, Y, Z, cX, cY, cZ, Config


L = logging.getLogger(__name__)
NEIGHBORS = np.array(list(set(it.product([-1, 0, 1], repeat=3)) - set([(0, 0, 0)])))

ORIGINS = ['ox', 'oy', 'oz']
BETAS = ['fx', 'fy', 'fz']


class FlatMap(object):
    '''Holds flat map, and related hierarchy and brain_regions'''

    CORTICAL_MAP_PATH = 'cortical_map'

    def __init__(self,
                 brain_regions, hierarchy,
                 view_lookup, paths,
                 center_line_2d, center_line_3d):
        '''init

        Args:
            brain_regions(VoxelData): brain regions dataset at the same
            resolution as view_lookup/paths
            hierarchy(voxcell.Hierarchy): associated with brain_regions
            view_lookup(2D np.array): 2d 'surface', where each location's value,
            when >=0, is associated with a row in `paths`
            paths(2D np.array): a unique path through the voxel dataset (ie:
            brain_regions, but can be others), where each non-zero element is
            the flat array index (ie: np.ravel_multi_index()) of the voxel dataset
            center_line_2d(float): defines the line separating the hemispheres in the flat map
            center_line_3d(float): defines the line separating the hemispheres in the brain_regions,
            in world coordiates

            view_lookup and paths both come from the Allen Institute, and
            are part of their `dorsal_flatmap_paths_xxx` datasets.

            100um is available here:
                https://github.com/AllenInstitute/mouse_connectivity_models/
                    tree/master/mcmodels/core/cortical_coordinates

            10um is available here:
                http://download.alleninstitute.org/informatics-archive/
                    current-release/mouse_ccf/cortical_coordinates/ccf_2017/
        '''
        self.brain_regions = brain_regions
        self.hierarchy = hierarchy
        self.view_lookup = view_lookup
        self.paths = paths
        self.center_line_2d = center_line_2d
        self.center_line_3d = center_line_3d

    def get_voxel_indices_from_flat(self, idx):
        '''idx of indices in the flat view'''
        paths = self.paths[self.view_lookup[idx]]
        paths = paths[paths.nonzero()]
        voxel_indices = np.unravel_index(paths, self.brain_regions.shape)
        return voxel_indices

    @classmethod
    def load(cls,
             cortical_map_url, brain_regions_url, hierarchy_url,
             center_line_2d, center_line_3d,
             cache_dir):  # pragma: no cover
        '''load the flat_mapping from path, caching in cache_dir

        Note: this should rely on neuroinformatics, getting directly from Allen for now
        '''
        # pylint: disable=too-many-locals

        base = os.path.join(cache_dir, cls.CORTICAL_MAP_PATH)
        if not os.path.exists(base):
            os.makedirs(base)

        brain_regions = os.path.join(base, 'annotation_100.nrrd')
        hierarchy = os.path.join(base, 'hierarchy.json')
        cortical_map = os.path.join(base, 'dorsal_flatmap_paths_100.h5')

        def get_file(url, path):
            '''download `url` and save it to `path`'''
            resp = requests.get(url)
            resp.raise_for_status()
            with open(path, 'wb') as fd:
                fd.write(resp.content)

        if not os.path.exists(brain_regions):
            L.info('Getting flat_map annotations')
            get_file(brain_regions_url, brain_regions)

        if not os.path.exists(hierarchy):
            L.info('Getting flat_map hierarchy')
            resp = requests.get(hierarchy_url)
            resp.raise_for_status()

            # The Allen Institute adds an extra wrapper around the contents need to strip that
            resp = resp.json()['msg'][0]
            with open(hierarchy, 'wb') as fd:
                json.dump(resp, fd, indent=2)

        if not os.path.exists(cortical_map):
            L.info('Getting flat_map cortical_map')
            get_file(cortical_map_url, cortical_map)

        brain_regions = voxcell.VoxelData.load_nrrd(brain_regions)
        hier = voxcell.hierarchy.Hierarchy.load_json(hierarchy)

        with h5py.File(cortical_map, 'r') as h5:
            view_lookup, paths = h5['view lookup'][:], h5['paths'][:]

        return cls(brain_regions, hier,
                   view_lookup, paths,
                   center_line_2d, center_line_3d)

    def make_flat_id_region_map(self, regions):
        '''find most popular *parent* region IDs for each flat_map value, based on path in voxels

        Args;
            regions(list of str): regions that are considered

        Return:
            ndarray of shape flat_map.view_lookup with the most popular id of region
        '''
        counts = np.count_nonzero(self.paths, axis=1)
        df = pd.DataFrame({'path_row': np.repeat(np.arange(len(self.paths)), counts),
                           'flat_index': self.paths[np.nonzero(self.paths)].ravel()})
        df['subregion_id'] = self.brain_regions.raw.ravel()[df['flat_index']]
        df = df[df.subregion_id > 0]

        nz = np.nonzero(self.view_lookup >= 0)
        row2flat_index = pd.Series(np.ravel_multi_index(nz, self.view_lookup.shape),
                                   index=self.view_lookup[nz], name='flat_idx')

        region2id = pd.DataFrame([(region, self.hierarchy.find('acronym', region)[0].data['id'])
                                  for region in regions],
                                 columns=['subregion', 'parent_region_id', ]).set_index('subregion')
        parent_ids = (pd.DataFrame([(id_, region)
                                    for region in regions
                                    for id_ in self.hierarchy.collect('acronym', region, 'id')],
                                   columns=['id', 'parent_region']).set_index('id')
                      .join(region2id, on='parent_region')
                      )
        df = df.join(parent_ids, on='subregion_id', how='inner')

        most_popular = df.groupby('path_row').parent_region_id.agg(
            lambda x: pd.Series.mode(x)[0])

        nz = np.nonzero(self.view_lookup >= 0)
        row2flat_index = pd.DataFrame(np.ravel_multi_index(nz, self.view_lookup.shape),
                                      index=self.view_lookup[nz], columns=['flat_idx', ])
        most_popular = row2flat_index.join(most_popular, how='inner')

        flat_id = np.zeros_like(self.view_lookup, dtype=int)
        flat_id[nz] = -1
        flat_id.ravel()[most_popular.flat_idx.values] = most_popular.parent_region_id.values
        return flat_id


def _fit_path(path):
    '''given path composed of voxel indices, fit a line through them

    Args:
        path(Nx3): for N voxel indices, the X, Y, Z locations on the path

    Returns:
        np.array(1x6): 0:3 are origins, 3:6 are betas (slope)
    '''
    independent = np.arange(len(path))
    bx, ox = np.polyfit(independent, path[:, X], 1)
    by, oy = np.polyfit(independent, path[:, Y], 1)
    bz, oz = np.polyfit(independent, path[:, Z], 1)

    return ox, oy, oz, bx, by, bz


def _fit_paths(flat_map):
    '''given paths, fit all of them, ignoring single voxel path

    Args:
        flat_map(FlatMap): flat map to find fitted paths
    '''
    ret, ids = [], []
    for i, path in enumerate(flat_map.paths):
        path = path[path.nonzero()]
        if len(path) < 2:
            continue

        # this is to follow what MR did, but I'm unsure of the format of
        # the dorsal path: does '0' mean 'no value' in the paths array?
        # That might be an oversight, or, since ijk = (0, 0, 0) isn't used in
        # the atlas, it's a safe sentinel for no value
        # path -= 1

        path = np.array(np.unravel_index(path, flat_map.brain_regions.shape)).T
        ret.append(_fit_path(path))
        ids.append(i)

    ret = pd.DataFrame(ret, columns=(ORIGINS + BETAS))
    ret.index = ids

    xy = pd.DataFrame.from_dict({flat_map.view_lookup[idx]: idx
                                 for idx in zip(*np.nonzero(flat_map.view_lookup >= 0))},
                                orient='index',
                                columns=['x', 'y'])

    ret = ret.join(xy)
    return ret


def _paths_intersection_distances(bounds, path_fits):
    '''calculate all fitted paths that intersect bounds

    Args:
        bounds(3 floats): location of center of voxel, the 'minimum' corner is
        calculated by subtracting 0.5 from this, the upper by adding 0.5
        path_fits(np.array of Nx6): N lines, 0:3 are the origins of each fit,
        3:6 are the betas (slopes)

    Returns:
        distance: distance travelled in voxel

    https://www.scratchapixel.com/lessons/3d-basic-rendering/
        minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    '''
    bounds = np.array(bounds) - 0.5

    origin = path_fits[ORIGINS].values
    invdir = 1. / path_fits[BETAS].values
    sign = (invdir < 0).astype(int)

    tmin = (bounds + sign - origin) * invdir
    tmax = (bounds + (1 - sign) - origin) * invdir

    hit = np.invert((tmin[cX] > tmax[cY]) | (tmin[cX] > tmax[cZ]) |
                    (tmin[cY] > tmax[cX]) | (tmin[cY] > tmax[cZ]) |
                    (tmin[cZ] > tmax[cX]) | (tmin[cZ] > tmax[cY])
                    )

    t = np.zeros(len(path_fits))
    dist = np.empty(len(path_fits))

    t[hit] = (np.abs(np.min(tmax[hit], axis=1) - np.max(tmin[hit], axis=1)))
    dist = np.linalg.norm(path_fits[BETAS].values * t[:, None], axis=1)

    return dist


def _voxel2flat_helper(config_path, path_fits, locs):
    '''helper, so the config path can be passed, and doesn't need to be serialized'''
    config = Config(config_path)
    return _voxel2flat(config.flat_map, config.regions, path_fits, locs)


def _voxel2flat(flat_map, regions, path_fits, locs):
    '''for each voxel, find the corresponding flat 2d position'''
    center_line_3d = flat_map.center_line_3d / flat_map.brain_regions.voxel_dimensions[Z]

    id_to_top_region = {id_: region
                        for region in regions
                        for id_ in flat_map.hierarchy.collect('acronym', region, 'id')}
    id_to_top_region = pd.DataFrame.from_dict(id_to_top_region,
                                              orient='index',
                                              columns=['region', ])

    def mask_paths_by_region_membership(loc, path_locs):
        '''make mask of whether region of loc is in paths

        since the brain is curved, it's possible that voxels are intersected
        by fitted lines that aren't even close to being in the same region.  Thus,
        we check if the parent region of loc (ie: FRP_l6 -> FRP) is in the
        set of parents of the paths
        '''
        loc_id = flat_map.brain_regions.raw[tuple(loc)]
        loc_region = id_to_top_region.loc[loc_id].values[0]

        paths = flat_map.paths[path_locs, :]
        voxel_indices = np.unravel_index(paths.ravel(), flat_map.brain_regions.shape)
        path_ids = flat_map.brain_regions.raw[voxel_indices]
        path_ids = id_to_top_region.reindex(path_ids)['region'].values
        path_ids = np.reshape(path_ids, (-1, paths.shape[-1]))
        mask = np.any(path_ids == loc_region, axis=1)
        return mask

    path_fits_left = path_fits[path_fits.y < flat_map.center_line_2d]
    path_fits_right = path_fits[path_fits.y > flat_map.center_line_2d]

    def lookup(loc):
        '''helper to lookup `loc`'''
        if loc[Z] >= center_line_3d:
            path_fit_side = path_fits_right
        else:
            path_fit_side = path_fits_left

        distances = _paths_intersection_distances(loc, path_fit_side)
        idx = np.nonzero(distances)
        distances = distances[idx]
        path_locs = path_fit_side.index[idx].values

        coordinates = path_fits.loc[path_locs][['x', 'y']].values

        mask = mask_paths_by_region_membership(loc, path_locs)
        coordinates = coordinates[mask]
        distances = distances[mask]

        weights = distances / np.sum(distances)
        return np.sum(coordinates * weights[:, None], axis=0)

    ret = []
    for loc in locs:
        ret.append(lookup(loc))

    ret = np.array(ret)

    return ret


def _create_voxcell_from_xy(locations, flat_map, flat_xy):
    '''
    Args:
        locations(array(Nx3)): locations
        flat_map(FlatMap): regions
        flat_xy(np.array(Nx3): containing coordinates in flat_map space
        of locations
    '''
    nz = np.count_nonzero(flat_xy) // 2
    if len(locations) > nz:
        L.debug('did not find values for %d of %d voxels (%0.2f percent)',
                len(locations) - nz, len(locations),
                (len(locations) - nz) / len(locations))

    ret = np.zeros_like(flat_map.brain_regions.raw)
    ret[tuple(locations.T)] = np.ravel_multi_index(tuple(flat_xy.T.astype(int)),
                                                   flat_map.view_lookup.shape)
    ret = flat_map.brain_regions.with_data(ret)
    return ret


def _backfill_voxel_to_flat_mapping(voxel_mapping, flat_map, wanted_ids):
    '''inplace filling of missing values

    these voxels weren't intersected by any of the fit lines; just average the position
    in the flat view of the neighbors to the voxel
    '''
    shape = flat_map.view_lookup.shape
    center_line_idx = flat_map.center_line_3d // flat_map.brain_regions.voxel_dimensions[Z]

    for ids in wanted_ids.values():
        ids = list(ids)
        mask = np.isin(flat_map.brain_regions.raw, ids)
        idx = np.nonzero(mask)
        missing = np.nonzero(voxel_mapping.raw[idx] <= 0)
        idx = np.array(idx).T
        for row in missing[0]:
            miss = tuple(idx[row, :])
            neighbors = NEIGHBORS + miss

            neighbor_values = voxel_mapping.raw[tuple(neighbors.T)]
            neighbor_values = neighbor_values[np.nonzero(neighbor_values)]
            neighbor_values = np.array(np.unravel_index(neighbor_values, shape)).T

            if miss[Z] <= center_line_idx:  # 'left'
                mask = neighbor_values[cY] <= flat_map.center_line_2d
            else:
                mask = neighbor_values[cY] > flat_map.center_line_2d

            neighbor_values = neighbor_values[mask]

            if not len(neighbor_values):
                continue

            flat_xy = np.mean(neighbor_values, axis=0).astype(int)

            voxel_mapping.raw[miss] = np.ravel_multi_index(tuple(flat_xy.T), shape)


def get_voxel_to_flat_mapping(config, backfill=True, n_jobs=-2, chunks=None):
    '''get the full voxel to flat mapping of regions in config.regions'''
    flat_map = config.flat_map

    wanted_ids = {region: flat_map.hierarchy.collect('acronym', region, 'id')
                  for region in config.regions}
    ids = list(reduce(operator.or_, wanted_ids.values()))

    path_fits = _fit_paths(flat_map)

    locations = np.array(np.nonzero(np.isin(flat_map.brain_regions.raw, ids))).T

    L.debug('There are %d locations', len(locations))

    p = Parallel(n_jobs=n_jobs,
                 # faster first run time than 'loky'; use loky for debugging
                 backend='multiprocessing',
                 # verbose=100,
                 )

    if chunks is None:
        chunks = p._effective_n_jobs()  # pylint: disable=protected-access

    worker = delayed(_voxel2flat_helper)
    flat_xy = p(worker(config.config_path, path_fits, locs)
                for locs in np.array_split(locations, chunks, axis=0))

    flat_xy = np.vstack(flat_xy)
    voxel_mapping = _create_voxcell_from_xy(locations, flat_map, flat_xy)

    if backfill:
        _backfill_voxel_to_flat_mapping(voxel_mapping, flat_map, wanted_ids)

    return voxel_mapping
