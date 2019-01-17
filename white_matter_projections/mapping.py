'''Methods for handling mapping one cortical region to another

Terminology:
    *flat*: the 2D coordinate system used for mapping
    *voxel*: 3D voxelized coordinate system, generally use ijk as indices
    *position*: xyz position

General strategy is to move everything *to* the flat 2D coordinate system of the
target region, and perform all mapping/assignment operations there.

Thus:
    1) For the source axons:
        src axon location points -> src voxels -> src flat points -> tgt flat points

    2) For the synpase locations:
        tgt synapse location points -> tgt voxels -> tgt flat points
'''

import itertools as it
import logging

import numpy as np
from white_matter_projections.utils import (X, Y,
                                            mirror_vertices_y,
                                            )


L = logging.getLogger(__name__)
NEIGHBORS = np.array(list(set(it.product([-1, 0, 1], repeat=3)) - set([(0, 0, 0)])))
cXZ = np.s_[:, X:3:2]


class PositionToVoxel(object):
    '''helper to go from 3D position to 3D voxel

    Args:
        brain_regions(VoxelData): brain regions corresponding to the flat_map
    '''
    def __init__(self, brain_regions):
        self.brain_regions = brain_regions

    def __call__(self, positions):
        indices = self.brain_regions.positions_to_indices(positions, keep_fraction=True)
        ret = np.floor(indices).astype(np.int)
        return ret, indices - ret


class VoxelToFlat(object):
    '''Map from voxels to flat space

    Args:
        flat_map(flat_mapping.FlatMap): cortical mapping
    '''
    def __init__(self, voxel_to_flat_mapping, shape):
        self.voxel_to_flat_mapping = voxel_to_flat_mapping
        self.shape = shape

    def __call__(self, voxel_ijks, offsets):
        flat_ids = self.voxel_to_flat_mapping.raw[tuple(voxel_ijks.T)]
        idx = np.array(np.unravel_index(flat_ids, self.shape)).T

        offsets = offsets[cXZ]
        return idx.astype(float), offsets


class FlatToFlat(object):
    '''helper for mapping from flat to flat space using BarycentricCoordinates'''
    def __init__(self, projections_mapping, center_line_2d):
        self.projections_mapping = projections_mapping
        self.center_line_2d = center_line_2d

    def __call__(self, src_region, projection_name, src_flat_uvs, offsets, mirror):
        '''if mirror: the src/tgt vertices are mirrored'''
        src_verts = self.projections_mapping[src_region]['vertices']
        tgt_verts = self.projections_mapping[src_region][projection_name]['vertices']

        if mirror:
            src_verts = mirror_vertices_y(src_verts, self.center_line_2d)
            tgt_verts = mirror_vertices_y(tgt_verts, self.center_line_2d)

        bc_src = BarycentricCoordinates(src_verts)
        bc_tgt = BarycentricCoordinates(tgt_verts)

        pos = src_flat_uvs.astype(float) + offsets
        tgt_flat_uvs = bc_tgt.bary2cart(bc_src.cart2bary(pos))
        return tgt_flat_uvs


class BarycentricCoordinates(object):
    '''Handle conversion of barycentric coordinates for interpolation'''
    def __init__(self, vertices):
        '''init

        Args:
            vertices(np.array): 3x2 where x coordinates of the triangle are column 0, and y col 1
        '''
        self.vertices = vertices
        self.inv = np.linalg.inv(np.vstack((vertices[0:2, X] - vertices[2, X],
                                            vertices[0:2, Y] - vertices[2, Y],)))

    def cart2bary(self, points):
        '''convert points from cartesian to barycentric coords

        Args:
            points(np.array): Nx2 with X coordinates in column 0, and y in column 1

        Returns:
            np.array(Nx3): 'weights' (ie lambda_{1,2,3}) for the points
        '''
        # XXX: np.linalg.solve might be safer, but it's slower - check
        diff = points - self.vertices[2, :]
        res = np.einsum('ij,kj->ik', diff, self.inv)
        return np.hstack((res, (1.0 - res.sum(axis=1))[:, None]))

    def bary2cart(self, coords):
        '''convert points from barycentric to cartesian

        Args:
            coords(np.array): Nx3 with the 'weights' (ie lambda_{1,2,3}) of the barycentric coords

        Returns:
            np.array(Nx2): with X coordinates in column 0, and y in column 1
        '''
        return np.einsum('ij,jk->ik', coords, self.vertices)