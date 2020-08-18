import numpy as np
from white_matter_projections import mapping, flat_mapping
from numpy.testing import assert_allclose, assert_array_equal
import utils


def test_PositionToVoxel():
    brain_regions, _ = utils.fake_brain_regions()
    p2v = mapping._PositionToVoxel(brain_regions)
    voxel_ijks, offset = p2v(np.array([[0.5, 0.5, 0.5],
                                       [0, 0, 0],
                                       [1, 1, 1]]))
    assert_allclose(voxel_ijks, np.array([[0, 0, 0],
                                          [0, 0, 0],
                                          [1, 1, 1]]))
    assert_allclose(offset, np.array([[0.5, 0.5, 0.5],
                                      [0., 0., 0.],
                                      [0., 0., 0.]]))


def test_VoxelToFlat():
    flat_map = utils.fake_flat_map()

    v2fc = mapping._VoxelToFlat(flat_map.flat_map, flat_map.shape)

    # working path, [4, 4, 4] is far from everything, so should be missing
    ijks, offsets = np.array([[1, 1, 1], [2, 2, 2], [4, 4, 4]]), 0.5 * np.ones((3, 3))
    idx, idx_offsets = v2fc(ijks, offsets)
    assert_allclose(idx, np.array([[1, 1], [2, 2], [-1, -1], ]))
    assert_allclose(idx_offsets, np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], ]))


def test_FlatToFlat():
    verts = np.array([[0., 1., 2],
                      [0., 2., 0]]).T
    projections_mapping = {'Region1':
                           {'vertices': verts,
                            'Projection1':
                            {'vertices': 10 * verts,
                             }
                            },
                           }
    f2f = mapping._FlatToFlat(projections_mapping, center_line_2d=0.)

    src_flat_uvs = np.array([np.mean(verts, axis=0), ])
    tgt_flat_uvs = f2f(src_region='Region1', projection_name='Projection1',
                       flat_uvs=src_flat_uvs, mirror=False)
    assert_allclose(10 * src_flat_uvs, tgt_flat_uvs)


    # when mirroring, since all the vertices are on the same side, will get the same result
    tgt_flat_uvs = f2f(src_region='Region1', projection_name='Projection1',
                       flat_uvs=src_flat_uvs, mirror=True)
    assert_allclose(10 * src_flat_uvs, tgt_flat_uvs)


def test_BarycentricCoordinates():
    vertices = np.array([(0., 0), (10., 0), (0., 10.)])
    bc = mapping.BarycentricCoordinates(vertices)
    points = np.mean(vertices, axis=0)[None, :]

    # by definition, the midpoint should be 1/3 from each edge
    coords = bc.cart2bary(points)
    assert_allclose(coords, np.array([[0.33333333, 0.33333333, 0.33333333]]))
    ret = bc.bary2cart(coords)
    assert_allclose(ret, points)

    #a vertex point should be 1. from the opposing edge
    points = np.array([[0., 0.]])
    coords = bc.cart2bary(points)
    assert_allclose(coords, np.array([[1., 0., 0.]]))
    ret = bc.bary2cart(coords)
    assert_allclose(ret, points)
