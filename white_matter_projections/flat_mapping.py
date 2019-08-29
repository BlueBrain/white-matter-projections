'''voxel space to 2d flat space'''
import logging

import numpy as np
import pandas as pd
import voxcell


L = logging.getLogger(__name__)


class FlatMap(object):
    '''Holds flat map, and related hierarchy and brain_regions'''
    def __init__(self,
                 flat_map,
                 flat_map_shape,
                 brain_regions,
                 hierarchy,
                 center_line_2d, center_line_3d):
        '''init

        Args:
            flat_map(VoxelData): mapping of each voxel to a raveled coordinate in `shape`
            flat_map_shape(tuple): 2D tuple giving the dimensions of 2D flatmap
            brain_regions(VoxelData): brain regions dataset at the same resolution as the flatmap
            hierarchy(voxcell.Hierarchy): associated with brain_regions
            center_line_2d(float): defines the line separating the hemispheres in the flat map
            center_line_3d(float): defines the line separating the hemispheres in the brain_regions,
            in world coordiates
        '''
        self.flat_map = flat_map
        self.shape = flat_map_shape
        self.brain_regions = brain_regions
        self.hierarchy = hierarchy
        self.center_line_2d = center_line_2d
        self.center_line_3d = center_line_3d

    @classmethod
    def load(cls,
             flat_map_path,
             flat_map_shape,
             brain_regions_path,
             hierarchy_path,
             center_line_2d,
             center_line_3d):  # pragma: no cover
        '''load the flat_mapping from paths'''
        flat_map = voxcell.VoxelData.load_nrrd(flat_map_path)
        brain_regions = voxcell.VoxelData.load_nrrd(brain_regions_path)
        hier = voxcell.hierarchy.Hierarchy.load_json(hierarchy_path)

        return cls(flat_map, flat_map_shape, brain_regions, hier,
                   center_line_2d, center_line_3d)

    def make_flat_id_region_map(self, regions):
        '''find most popular *parent* region IDs for each flat_map value, based on path in voxels

        Args;
            regions(list of str): regions that are considered

        Return:
            ndarray of shape flat_map.view_lookup with the most popular id of region
        '''
        flat_id = np.full(self.shape, fill_value=-1, dtype=int)

        flat_idx = np.nonzero(self.flat_map.raw)
        df = pd.DataFrame({'subregion_id': self.brain_regions.raw[flat_idx],
                           'flat_idx': self.flat_map.raw[flat_idx]
                           })

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

        most_popular = df.groupby('flat_idx').parent_region_id.agg(
            lambda x: pd.Series.mode(x)[0])

        flat_id.ravel()[most_popular.index.values] = most_popular.values
        return flat_id
