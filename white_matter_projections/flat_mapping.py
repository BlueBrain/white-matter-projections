'''voxel space to 2d flat space'''
import logging

import numpy as np
import pandas as pd
import voxcell


L = logging.getLogger(__name__)


class FlatMap(object):
    '''Holds flat map, and related region_map and brain_regions'''
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
        self.flat_map = flat_map
        self.brain_regions = brain_regions
        self.region_map = region_map
        self.center_line_2d = center_line_2d
        self.center_line_3d = center_line_3d

        self.flat_idx = (flat_map.raw[:, :, :, 0] > 0) | (flat_map.raw[:, :, :, 1] > 0)
        self.shape = np.max(flat_map.raw[self.flat_idx], axis=0) + 1

    @classmethod
    def load(cls,
             flat_map_path,
             brain_regions_path,
             hierarchy_path,
             center_line_2d,
             center_line_3d):  # pragma: no cover
        '''load the flat_mapping from paths'''
        flat_map = voxcell.VoxelData.load_nrrd(flat_map_path)
        brain_regions = voxcell.VoxelData.load_nrrd(brain_regions_path)
        region_map = voxcell.region_map.RegionMap.load_json(hierarchy_path)

        return cls(flat_map, brain_regions, region_map, center_line_2d, center_line_3d)

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
