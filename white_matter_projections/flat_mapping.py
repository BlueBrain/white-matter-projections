'''voxel space to 2d flat space'''
import logging

import numpy as np
import pandas as pd
import voxcell
from lazy import lazy


L = logging.getLogger(__name__)


class FlatMapBase:
    '''Flat map that holds flat_map/brain_regions directly, instead of lazy loading them

    Note: used for testing'''
    def __init__(self,
                 flat_map,
                 brain_regions,
                 region_map,
                 center_line_2d, center_line_3d):
        '''init

        Args:
            flat_map(VoxelData): mapping of each voxel in 3D to a 2D pixel location
            brain_regions(VoxelData): brain regions dataset at the same resolution as the flatmap
            region_map(voxcell.region_map): associated with brain_regions
            center_line_2d(float): defines the line separating the hemispheres in the flat map
            center_line_3d(float): defines the line separating the hemispheres in the brain_regions,
            in world coordiates
        '''
        self._flat_map = flat_map
        self._brain_regions = brain_regions
        self._region_map = region_map
        self.center_line_2d = center_line_2d
        self.center_line_3d = center_line_3d

    @lazy
    def flat_map(self):
        '''mapping of each voxel in 3D to a 2D pixel location'''
        return self._flat_map

    @lazy
    def brain_regions(self):
        '''brain regions dataset at the same resolution as the flatmap'''
        return self._brain_regions

    @lazy
    def region_map(self):
        '''hierarchy_path associated with brain_regions'''
        return self._region_map

    @lazy
    def flat_idx(self):
        '''indices of 3d flatmap having a value'''
        return np.reshape(self.mask_in_2d_flatmap(np.reshape(self.flat_map.raw, (-1, 2))),
                          self.flat_map.shape)

    @lazy
    def shape(self):
        '''shape of 2d flatmap'''
        return np.max(self.flat_map.raw[self.flat_idx], axis=0) + 1

    @staticmethod
    def mask_in_2d_flatmap(uvs):
        '''return a mask for all values of `uv` that are inside the 2d-flatmap

        Args:
            uvs(np.array [n x 2])): of float/int to test

        Returns:
            np.array(size=n): boolean if outside or not

        Note: per doc/source/flatmap.rst:
            The 2D canvas is only in the positive quadrant.  Thus, negative
            values of either the x or y mean that the voxel does not have a
            valid value.
        '''
        mask = (uvs[:, 0] >= 0) & (uvs[:, 1] >= 0)
        return mask

    def make_flat_id_region_map(self, regions):
        '''find most popular *parent* region IDs for each flat_map value, based on path in voxels

        Args;
            regions(list of str): regions that are considered

        Return:
            ndarray of shape flat_map.view_lookup with the most popular id of region
        '''
        df = pd.DataFrame({'subregion_id': self.brain_regions.raw[self.flat_idx],
                           'flat_x': self.flat_map.raw[self.flat_idx, 0],
                           'flat_y': self.flat_map.raw[self.flat_idx, 1],
                           })

        region2id = pd.DataFrame(
            [(region, next(iter(self.region_map.find(region, 'acronym'))))
             for region in regions],
            columns=['subregion', 'parent_region_id', ]).set_index('subregion')
        parent_ids = (pd.DataFrame([(id_, region)
                                    for region in regions
                                    for id_ in self.region_map.find(region, 'acronym',
                                                                    with_descendants=True)],
                                   columns=['id', 'parent_region']).set_index('id')
                      .join(region2id, on='parent_region')
                      )
        df = df.join(parent_ids, on='subregion_id', how='inner')

        most_popular = df.groupby(['flat_x', 'flat_y']).parent_region_id.agg(
            lambda x: pd.Series.mode(x)[0])

        flat_id = np.full(self.shape, fill_value=-1, dtype=np.int64)
        idx = tuple(most_popular.reset_index()[['flat_x', 'flat_y']].to_numpy().T)
        flat_id[idx] = most_popular.to_numpy()
        return flat_id


class FlatMap(FlatMapBase):
    '''Holds flat map, and related region_map and brain_regions'''
    def __init__(self,
                 flat_map_path,
                 brain_regions_path,
                 hierarchy_path,
                 center_line_2d,
                 center_line_3d):
        '''init

        Args:
            flat_map_path(str): mapping of each voxel in 3D to a 2D pixel location
            brain_regions(str): brain regions dataset at the same resolution as the flatmap
            hierarchy_path(str): hierarchy_path associated with brain_regions
            center_line_2d(float): defines the line separating the hemispheres in the flat map
            center_line_3d(float): defines the line separating the hemispheres in the brain_regions,
            in world coordiates
        '''
        # pylint: disable=super-init-not-called
        self.flat_map_path = flat_map_path
        self.brain_regions_path = brain_regions_path
        self.hierarchy_path = hierarchy_path
        self.center_line_2d = center_line_2d
        self.center_line_3d = center_line_3d

    @lazy
    def flat_map(self):
        '''mapping of each voxel in 3D to a 2D pixel location'''
        return voxcell.VoxelData.load_nrrd(self.flat_map_path)

    @lazy
    def brain_regions(self):
        '''brain regions dataset at the same resolution as the flatmap'''
        return voxcell.VoxelData.load_nrrd(self.brain_regions_path)

    @lazy
    def region_map(self):
        '''hierarchy_path associated with brain_regions'''
        return voxcell.region_map.RegionMap.load_json(self.hierarchy_path)
