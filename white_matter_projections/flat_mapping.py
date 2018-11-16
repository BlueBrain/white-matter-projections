'''voxel space to 2d flat space'''
import collections
import itertools as it
import json
import logging
import os

import numpy as np
import voxcell
import h5py


L = logging.getLogger(__name__)
NEIGHBORS = np.array(list(set(it.product([-1, 0, 1], repeat=3)) - set([(0, 0, 0)])))


class FlatMap(object):
    '''Holds flat map, and related hierarchy and brain_regions'''

    CORTICAL_MAP_PATH = 'cortical_map'

    def __init__(self, brain_regions, hierarchy, view_lookup, paths):
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

    def get_voxel_indices_from_flat(self, idx):
        '''idx of indices in the flat view'''
        paths = self.paths[self.view_lookup[idx]]
        paths = paths[paths.nonzero()]
        voxel_indices = np.unravel_index(paths, self.brain_regions.shape)
        return voxel_indices

    @classmethod
    def load(cls,
             cortical_map_url, brain_regions_url, hierarchy_url, cache_dir):  # pragma: no cover
        '''load the flat_mapping from path, caching in cache_dir

        Note: this should rely on neuroinformatics, getting directly from Allen for now
        '''
        # pylint: disable=too-many-locals
        import requests

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

            #  The Allen Institute adds an extra wrapper around the contents
            #  need to strip that
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

        return cls(brain_regions, hier, view_lookup, paths)

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
