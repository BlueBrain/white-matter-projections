import os
import h5py
import itertools as it
import numpy as np
import pandas as pd

import voxcell
import yaml
from nose.tools import ok_, eq_, raises
from white_matter_projections import macro, micro, utils as wmp_utils
from numpy.testing import assert_allclose, assert_array_equal
import utils


def test_get_gids_by_population():
    pop_cat, populations = macro._parse_populations(
        utils.RECIPE['populations'],
        utils.REGION_MAP,
        region_subregion_translation=utils.get_region_subregion_translation()
    )
    def cells(_):
        return pd.DataFrame({'layer': [1, 1, 2, 3, ],
                             'region': ['ECT', 'ECT', 'FRP', 'ECT', ],
                             })

    source_population = 'POP1_ALL_LAYERS'
    ret = micro.get_gids_by_population(populations, cells, source_population)
    assert_array_equal(ret, [0, 1, 3])

    def cells(_):
        return pd.DataFrame({'layer': [1, 1, 2, 3, ],
                             'region': ['ECT@left', 'ECT@right', 'FRP@left', 'ECT@right', ],
                             })
    source_population = 'POP1_ALL_LAYERS'
    ret = micro.get_gids_by_population(populations, cells, source_population)
    assert_array_equal(ret, [0, 1, 3])

# def test_allocate_projections():
#    micro.allocate_projections(recipe, cells)

# def test_allocation_stats():
#    ret = micro.allocation_stats(ptype, interactions, cell_count, allocations)

def test_partition_cells_left_right():
    cells = pd.DataFrame(np.arange(10) + .1, columns=['z', ])
    left, right = micro.partition_cells_left_right(cells, center_line_3d=5.1)
    eq_(len(left), 6)
    eq_(len(right), 4)


def test_separate_source_and_targets():
    cells = pd.DataFrame(np.arange(10) + .1, columns=['z', ])
    cells['x'] = 1
    cells['y'] = 2
    sgids = cells.index.values
    left_cells, right_cells = micro.partition_cells_left_right(cells, center_line_3d=5.1)

    ret = micro.separate_source_and_targets(left_cells, right_cells, sgids,
                                            hemisphere='ipsi', side='left')
    eq_(len(ret), 6)

    ret = micro.separate_source_and_targets(left_cells, right_cells, sgids,
                                            hemisphere='contra', side='left')
    eq_(len(ret), 4)

    ret = micro.separate_source_and_targets(left_cells, right_cells, sgids,
                                            hemisphere='ipsi', side='right')
    eq_(len(ret), 4)

    ret = micro.separate_source_and_targets(left_cells, right_cells, sgids,
                                            hemisphere='contra', side='right')
    eq_(len(ret), 6)


def test__assign_groups_worker():
    tgt_flat = np.array([[-10., -10.],
                         [10., 10.],
                         ])

    src_flat = np.array([[0., 0.], ])
    res = micro._assign_groups_worker(src_flat, tgt_flat, sigma=10, closest_count=10, rng=np.random)
    eq_(list(res), [0, 0])

    src_flat = np.array([[-10., -10.],
                         [10., 10.],
                         ])
    res = micro._assign_groups_worker(src_flat, tgt_flat, sigma=10, closest_count=10, rng=np.random)
    eq_(list(res), [0, 1])


def test_assign_groups():
    tgt_flat = np.array([[-10., -10.],
                         [10., 10.],
                         ])
    src_flat = pd.DataFrame([[-10., -10.],
                             [10., 10.],
                             ], index=(10, 20))
    res = micro.assign_groups(src_flat, tgt_flat, sigma=10, closest_count=10, n_jobs=1, seed=0)
    eq_(list(res), [10, 20])


def test__calculate_delay_streamline():
    np.random.seed(42)
    src_cells = pd.DataFrame([(55., 0., 0.,),
                              (65., 0., 0.,),
                              (75., 0., 0.,),
                              (85., 0., 0.,), ],
                             columns=wmp_utils.XYZ,
                             index=[1, 2, 3, 4])
    syns = pd.DataFrame([(1, 1., 0., 0.),
                         (2, 2., 0., 0.),
                         (2, 3., 0., 0.),
                         (4, 4., 0., 0.), ],
                        columns=['sgid', ] + wmp_utils.XYZ)
    columns = ['path_row', 'length', 'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z', ]
    streamline_metadata = pd.DataFrame([(3, 3000., 0., 0., 0., 3., 0., 0.),
                                        (4, 4000., 1., 0., 0., 4., 0., 0.),
                                        (5, 5000., 2., 0., 0., 5., 0., 0.),
                                        ],
                                       columns=columns)
    conduction_velocity = {'inter_region': 10.,
                           'intra_region': 1.
                           }

    delay, gid2row = micro._calculate_delay_streamline(src_cells, syns, streamline_metadata,
                                                       conduction_velocity=conduction_velocity,
                                                       rng=np.random)
    eq_(list(gid2row.row), [5, 3, 5, 5, ])  # based on random picking
    assert_allclose(gid2row.sgid.values, syns.sgid.values)
    eq_(list(delay), [557., 366., 565., 584.])


def test__calculate_delay_direct():
    conduction_velocity = {'inter_region': 10.,
                           'intra_region': 1.
                           }

    src_cells = pd.DataFrame([(55., 0., 0.,),
                              (65., 0., 0.,),
                              (75., 0., 0.,),
                              (85., 0., 0.,), ],
                             columns=wmp_utils.XYZ,
                             index=[1, 2, 3, 4])
    syns = pd.DataFrame([(1, 1., 0., 0.),
                         (2, 2., 0., 0.),
                         (2, 3., 0., 0.),
                         (4, 4., 0., 0.), ],
                        columns=['sgid', ] + wmp_utils.XYZ)

    delay = micro._calculate_delay_direct(src_cells, syns, conduction_velocity)
    assert_allclose(delay, np.array([54, 63, 62, 81, ]) / 10.)


def test__calculate_delay_dive():
    conduction_velocity = {'inter_region': 10.,
                           'intra_region': 1.
                           }

    src_cells = pd.DataFrame([(55., 0., 0.,),
                              (65., 0., 0.,),
                              (75., 0., 0.,),
                              (85., 0., 0.,), ],
                             columns=wmp_utils.XYZ,
                             index=[1, 2, 3, 4])
    syns = pd.DataFrame([(1, 1., 0., 0.),
                         (2, 2., 0., 0.),
                         (2, 3., 0., 0.),
                         (4, 4., 0., 0.), ],
                        columns=['sgid', ] + wmp_utils.XYZ)

    class Atlas():
        @staticmethod
        def load_data(dataset):
            assert dataset == '[PH]y'
            return voxcell.VoxelData(np.ones((10, 10, 10)),
                                     voxel_dimensions=(10., 10., 10.),
                                     offset=(0., 0., 0.))

    delay = micro._calculate_delay_dive(src_cells, syns, conduction_velocity, Atlas())
    # delay is the same as direct delay, except one is added on the way from the synapse
    # to bottom of layer 6, and then another 1 on the way back 'up'
    assert_allclose(delay, np.array([54, 63, 62, 81, ]) / 10. + 1. + 1.)

# def test_assignment():
#    config
#    allocations
#    side
#    with utils.tempdir('test_assignment') as output:
#        micro.assignment(output, config, allocations, side)
