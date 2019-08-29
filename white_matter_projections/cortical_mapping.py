'''helpers to deal with AIBS cortical map

The idea is to create a flat map (ie: mapping from each voxel in 3D space to a 2D plane)

The cortical map that Allen provides is composed of two pieces:
    view_lookup(2D np.array): 2d 'surface', where each location's value,
        when >=0, is associated with a row in `paths`
    paths(2D np.array): a unique path through the voxel dataset (ie:
        brain_regions, but can be others), where each non-zero element is
        the flat array index (ie: np.ravel_multi_index()) of the voxel dataset

The problem is twofold:
    * that the 'paths' that exist aren't dense: not every voxel is associated with a path
    * some of the voxels lie in multiple paths

Finally, one can 'backfill' to try and fully fill the region, so every voxel has
a mapping from 3D to 2D space
'''
import itertools as it
import functools
import logging
import operator

import h5py
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import voxcell
from white_matter_projections.utils import X, Y, Z, cX, cY, cZ

L = logging.getLogger(__name__)

NEIGHBORS = np.array(list(set(it.product([-1, 0, 1], repeat=3)) - {(0, 0, 0), }))

ORIGINS = ['ox', 'oy', 'oz', ]
BETAS = ['fx', 'fy', 'fz', ]


class CorticalMapParameters(object):
    '''hold paths for cortical mapping artifacts

    Note: artifacts are only loaded when load_* methods are called, so that
    serialization is simple
    '''
    def __init__(self,
                 cortical_map_path,
                 brain_regions_path,
                 hierarchy_path,
                 center_line_2d,
                 center_line_3d):
        self.cortical_map_path = cortical_map_path
        self.brain_regions_path = brain_regions_path
        self.hierarchy_path = hierarchy_path
        self.center_line_2d = center_line_2d
        self.center_line_3d = center_line_3d

    def load_cortical_view_lookup(self):
        '''load the view_lookup'''
        with h5py.File(self.cortical_map_path, 'r') as h5:
            return h5['view lookup'][:]

    def load_cortical_paths(self):
        '''load the paths'''
        with h5py.File(self.cortical_map_path, 'r') as h5:
            return h5['paths'][:]

    def load_hierarchy(self):
        '''load the hierarchy'''
        return voxcell.hierarchy.Hierarchy.load_json(self.hierarchy_path)

    def load_brain_regions(self):
        '''load the brain_regions'''
        return voxcell.VoxelData.load_nrrd(self.brain_regions_path)


def _fit_path(path):
    '''given path composed of voxel indices, fit a line through them

    Args:
        path(Nx3): for N voxel indices, the X, Y, Z locations on the path

    Returns:
        tuple(1x6): 0:3 are origins, 3:6 are betas (slope)
    '''
    independent = np.arange(len(path))
    bx, ox = np.polyfit(independent, path[:, X], 1)
    by, oy = np.polyfit(independent, path[:, Y], 1)
    bz, oz = np.polyfit(independent, path[:, Z], 1)

    return ox, oy, oz, bx, by, bz


def _fit_paths(brain_regions, cortical_view_lookup, cortical_paths):
    '''given paths, fit all of them, ignoring single voxel path

    Args:
        cortical_map_paths(CorticalMapParameters): paths of the cortical map
    '''
    ret, ids = [], []
    for i, path in enumerate(cortical_paths):
        path = path[path.nonzero()]
        if len(path) < 2:
            continue

        # this is to follow what MR did, but I'm unsure of the format of
        # the dorsal path: since '0' means 'no value' in the paths array, do I
        # have to subtract 1 from all the raveled path indices?
        # That might be an oversight, or, since ijk = (0, 0, 0) isn't used in
        # the atlas, it's a safe sentinel for no value

        path = np.array(np.unravel_index(path, brain_regions.shape)).T
        ret.append(_fit_path(path))
        ids.append(i)

    ret = pd.DataFrame(ret, columns=(ORIGINS + BETAS), index=ids)

    xy = pd.DataFrame.from_dict({cortical_view_lookup[idx]: idx
                                 for idx in zip(*np.nonzero(cortical_view_lookup >= 0))},
                                orient='index',
                                columns=['x', 'y'])

    ret = ret.join(xy)

    return ret


def _paths_intersection_distances(bounds, path_fits):
    '''calculate distance travelled in voxel of all `path_fits` that intersect bounds

    Args:
        bounds(3 floats): location of center of voxel, the 'minimum' corner is
        calculated by subtracting 0.5 from this, the upper by adding 0.5
        path_fits(Dataframe): N lines, containing ORIGINS of each fit,
        and BETAS (slopes)

    Returns:
        distance: distance travelled in voxel

    https://www.scratchapixel.com/lessons/3d-basic-rendering/
        minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    '''
    bounds = np.array(bounds) - 0.5

    origin = path_fits[ORIGINS].to_numpy()
    invdir = 1. / path_fits[BETAS].to_numpy()
    sign = (invdir < 0).astype(int)

    tmin = (bounds + sign - origin) * invdir
    tmax = (bounds + (1 - sign) - origin) * invdir

    hit = np.invert((tmin[cX] > tmax[cY]) | (tmin[cX] > tmax[cZ]) |
                    (tmin[cY] > tmax[cX]) | (tmin[cY] > tmax[cZ]) |
                    (tmin[cZ] > tmax[cX]) | (tmin[cZ] > tmax[cY])
                    )

    t = np.zeros(len(path_fits))

    t[hit] = (np.abs(np.min(tmax[hit], axis=1) - np.max(tmin[hit], axis=1)))
    dist = np.linalg.norm(path_fits[BETAS].to_numpy() * t[:, None], axis=1)

    return dist


def _voxel2flat_worker(cortical_map_paths, regions, path_fits, locs):
    '''for each voxel, find the corresponding flat 2d position'''

    brain_regions = cortical_map_paths.load_brain_regions()
    center_line_3d = cortical_map_paths.center_line_3d / brain_regions.voxel_dimensions[Z]

    id_to_top_region = {id_: region
                        for region in regions
                        for id_ in cortical_map_paths.load_hierarchy().collect('acronym',
                                                                               region,
                                                                               'id')}
    id_to_top_region = pd.DataFrame.from_dict(id_to_top_region,
                                              orient='index',
                                              columns=['region', ])

    cortical_paths = cortical_map_paths.load_cortical_paths()

    path_fits_left = path_fits[path_fits.y < cortical_map_paths.center_line_2d]
    path_fits_right = path_fits[path_fits.y > cortical_map_paths.center_line_2d]

    def lookup(loc):
        '''helper to lookup `loc`'''

        def mask_paths_by_region_membership(path_locs):
            '''make mask of whether region of loc is in paths

            since the brain is curved, it's possible that voxels are
            intersected by fitted lines that aren't even close to being in the
            same region.  Thus, we check if the parent region of loc
            (ie: FRP_l6 -> FRP) is in the set of parents of the paths
            '''
            loc_id = brain_regions.raw[tuple(loc)]
            loc_region = id_to_top_region.loc[loc_id].to_numpy()[0]

            paths = cortical_paths[path_locs, :]
            voxel_indices = np.unravel_index(paths.ravel(), brain_regions.shape)
            path_ids = brain_regions.raw[voxel_indices]
            path_ids = id_to_top_region.reindex(path_ids)['region'].to_numpy()
            path_ids = np.reshape(path_ids, (-1, paths.shape[-1]))
            mask = np.any(path_ids == loc_region, axis=1)
            return mask

        if loc[Z] >= center_line_3d:
            path_fit_side = path_fits_right
        else:
            path_fit_side = path_fits_left

        distances = _paths_intersection_distances(loc, path_fit_side)
        idx = np.nonzero(distances)
        distances = distances[idx]
        path_locs = path_fit_side.index[idx].to_numpy()

        coordinates = path_fits.loc[path_locs][['x', 'y']].to_numpy()

        mask = mask_paths_by_region_membership(path_locs)
        coordinates = coordinates[mask]
        distances = distances[mask]

        weights = distances / np.sum(distances)
        return np.sum(coordinates * weights[:, None], axis=0)

    ret = np.array([lookup(loc) for loc in locs])

    return ret


def _create_voxcell_from_xy(locations, brain_regions, view_lookup, flat_xy):
    '''
    Args:
        locations(array(Nx3)): locations
        brain_regions(3D np.array): labeled brain regions
        view_lookup(2D np.array): as defined in module docstring
        flat_xy(np.array(Nx3): containing coordinates in flat_map space
        of locations
    '''
    nz = np.count_nonzero(flat_xy) // 2
    if len(locations) > nz:
        L.debug('did not find values for %d of %d voxels (%0.2f percent)',
                len(locations) - nz, len(locations),
                (len(locations) - nz) / len(locations))

    ret = np.zeros_like(brain_regions.raw)
    ret[tuple(locations.T)] = np.ravel_multi_index(tuple(flat_xy.T.astype(int)),
                                                   view_lookup.shape)
    ret = brain_regions.with_data(ret)
    return ret


def _backfill_voxel_to_flat_mapping(voxel_mapping,
                                    brain_regions,
                                    view_lookup_shape,
                                    center_line_2d,
                                    center_line_3d,
                                    wanted_ids):
    '''inplace filling of missing values

    these voxels weren't intersected by any of the fit lines; just average the position
    in the flat view of the neighbors to the voxel
    '''
    # pylint: disable=too-many-locals
    center_line_idx = center_line_3d // brain_regions.voxel_dimensions[Z]

    count = 0
    for ids in wanted_ids.values():
        idx = np.nonzero(np.isin(brain_regions.raw, list(ids)))
        missing_rows = np.nonzero(voxel_mapping.raw[idx] <= 0)[0]
        idx = np.array(idx).T
        for missing_row in missing_rows:
            miss = tuple(idx[missing_row, :])

            neighbors = NEIGHBORS + miss
            in_bounds = np.all((neighbors >= 0) & (neighbors < voxel_mapping.raw.shape), axis=1)
            neighbors = neighbors[in_bounds]
            neighbor_values = voxel_mapping.raw[tuple(neighbors.T)]

            neighbor_values = neighbor_values[np.nonzero(neighbor_values)]
            neighbor_values = np.array(np.unravel_index(neighbor_values, view_lookup_shape)).T

            if miss[Z] <= center_line_idx:  # 'left'
                neighbor_values = neighbor_values[neighbor_values[cY] <= center_line_2d]
            else:
                neighbor_values = neighbor_values[neighbor_values[cY] > center_line_2d]

            if not len(neighbor_values):
                continue

            count += 1
            flat_xy = np.mean(neighbor_values, axis=0).astype(int)

            voxel_mapping.raw[miss] = np.ravel_multi_index(tuple(flat_xy.T), view_lookup_shape)

    L.info('Backfill updated %d values', count)


def create_cortical_to_flatmap(cortical_map_paths,
                               regions,
                               backfill=True,
                               n_jobs=-2,
                               chunks=None):
    '''get the full voxel to flat mapping of `regions`'''

    wanted_ids = {region: cortical_map_paths.load_hierarchy().collect('acronym', region, 'id')
                  for region in regions}

    brain_regions = cortical_map_paths.load_brain_regions()
    cortical_view_lookup = cortical_map_paths.load_cortical_view_lookup()

    path_fits = _fit_paths(brain_regions.raw,
                           cortical_view_lookup,
                           cortical_map_paths.load_cortical_paths())

    locations = np.array(np.nonzero(np.isin(
        brain_regions.raw,
        list(functools.reduce(operator.or_, wanted_ids.values()))))).T

    L.debug('There are %d locations', len(locations))

    p = Parallel(n_jobs=n_jobs,
                 # faster first run time than 'loky'; use loky for debugging
                 backend='multiprocessing',
                 # verbose=100,
                 )

    if chunks is None:
        chunks = p._effective_n_jobs()  # pylint: disable=protected-access

    worker = delayed(_voxel2flat_worker)
    flat_xy = p(worker(cortical_map_paths, regions, path_fits, locs)
                for locs in np.array_split(locations, chunks, axis=0))

    flat_xy = np.vstack(flat_xy)
    voxel_mapping = _create_voxcell_from_xy(locations,
                                            brain_regions,
                                            cortical_map_paths.load_cortical_view_lookup(),
                                            flat_xy)

    if backfill:
        _backfill_voxel_to_flat_mapping(voxel_mapping,
                                        brain_regions,
                                        cortical_view_lookup.shape,
                                        cortical_map_paths.center_line_2d,
                                        cortical_map_paths.center_line_3d,
                                        wanted_ids)
    return voxel_mapping, cortical_view_lookup.shape
