'''voxel space to 2d flat space'''
import collections
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
        '''find most popular region IDs for each flat_map value, based on path through voxels

        Args;
            regions(list of str): regions that are considered

        Return:
            ndarray of shape flat_map.view_lookup with the most popular id of region
        '''
        region2ids = {region: self.hierarchy.collect('acronym', region, 'id')
                      for region in regions}
        id2region = {id_: region
                     for region, ids in region2ids.items()
                     for id_ in ids}
        region2id = {region: self.hierarchy.find('acronym', region)[0].data['id']
                     for region in regions}
        region2id[Ellipsis] = -1

        def _get_most_popular_region(idx):
            path = self.paths[self.view_lookup[idx]]
            path = path[path.nonzero()]
            ids = self.brain_regions.raw.ravel()[path]
            count = collections.Counter((id2region.get(id_, Ellipsis) for id_ in ids))
            if Ellipsis in count:
                count.pop(Ellipsis)
            most_common = count.most_common(1)
            if not most_common:
                L.warning('%s does not have a mapping for regions', idx)
                return Ellipsis
            return most_common[0][0]

        flat_id = np.zeros_like(self.view_lookup, dtype=int)
        for idx in zip(*np.nonzero(self.view_lookup >= 0)):
            most_popular = _get_most_popular_region(idx)
            if most_popular not in region2id:
                L.warning('Most popular %s missing from region2id, skipping', most_popular)
                continue
            flat_id[idx] = region2id[most_popular]

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


def _voxel2flat(config_path, path_fits, locs):
    '''for each voxel, find the corresponding flat 2d position'''
    # avoid excessive serialization
    config = Config(config_path)
    flat_map = config.flat_map

    center_line_3d = int(flat_map.center_line_3d /
                         flat_map.brain_regions.voxel_dimensions[Z])

    id_to_top_region = {id_: region
                        for region in config.regions
                        for id_ in config.flat_map.hierarchy.collect('acronym', region, 'id')}
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
        loc_id = config.flat_map.brain_regions.raw[tuple(loc)]
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


def get_voxel_to_flat_mapping(config, n_jobs=-2, chunks=None):
    '''get the full voxel to flat mapping of regions in config.regions'''
    flat_map = config.flat_map

    ids = {region: flat_map.hierarchy.collect('acronym', region, 'id')
           for region in config.regions}
    ids = list(reduce(operator.or_, ids.values()))

    path_fits = _fit_paths(flat_map)

    locations = flat_map.brain_regions.raw
    locations = np.array(np.nonzero(np.isin(locations, ids))).T

    L.debug('There are %d locations', len(locations))

    p = Parallel(n_jobs=n_jobs,
                 # faster first run time than 'loky'; use loky for debugging
                 backend='multiprocessing',
                 # verbose=100,
                 )

    if chunks is None:
        chunks = p._effective_n_jobs()  # pylint: disable=protected-access

    worker = delayed(_voxel2flat)
    flat_xy = p(worker(config.config_path, path_fits, locs)
                for locs in np.array_split(locations, chunks, axis=0))

    flat_xy = np.vstack(flat_xy)

    L.debug('did not find values for %d of %d voxels',
            len(locations) - np.count_nonzero(flat_xy) // 2, len(locations))

    ret = np.zeros_like(flat_map.brain_regions.raw)
    ret[tuple(locations.T)] = np.ravel_multi_index(tuple(flat_xy.T.astype(int)),
                                                   flat_map.view_lookup.shape)
    ret = flat_map.brain_regions.with_data(ret)

    return ret
