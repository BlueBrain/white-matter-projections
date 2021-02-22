import os
from bluepy.circuit import Circuit
from white_matter_projections import utils, macro, flat_mapping
import numpy as np
import pandas as pd
import voxcell
from voxcell.nexus import voxelbrain
from mock import Mock

from nose.tools import eq_, ok_, assert_raises
import utils as test_utils
from numpy.testing import assert_allclose, assert_array_equal


class TestConfig(object):
    def __init__(self):
        self.config = test_utils.get_config()

    #{ slow - pulls data from internet
    #def test_types(self):
    #    types = (('region_layer_heights', pd.DataFrame),
    #             ('hierarchy', voxcell.hierarchy.Hierarchy),
    #             ('atlas', voxelbrain.Atlas),
    #             ('recipe', macro.MacroConnections),
    #             ('regions', list),
    #             #('circuit', Circuit),  # need a valid circuit
    #             ('config', dict),
    #             ('config_path', str),
    #             #('flat_map', flat_mapping.FlatMap),  # hits AIBS API, reduce load
    #             )

    #    for attr_, type_ in types:
    #        ok_(isinstance(getattr(self.config, attr_), type_))

    #    #ok_(isinstance(self.config.voxel_to_flat(), voxcell.VoxelData))  # too slow
    #    #ok_(isinstance(self.config.get_cells()), pd.DataFrame)  # need a valid circuit

    #def test_region_layer_heights(self):
    #    ret = self.config.region_layer_heights
    #    eq_(ret.loc['FRP'].sum(), 6 * 100)
    #    eq_(ret.loc['FRP']['l6'], 100)
    #    eq_(tuple(ret.loc['MOs']), tuple(ret.loc['ACAd']))
    #} slow

    def test__relative_to_config(self):
        assert_raises(Exception, utils.Config._relative_to_config, 'asdfasdfas', 'asdfasdfas')


#{ slow - pulls data from internet
#def test_normalize_layer_profiles():
#    config_path = os.path.join(test_utils.DATADIR, 'config.yaml')
#    layer_heights = utils.Config(config_path).region_layer_heights
#    profiles = pd.DataFrame([['profile_1', 'l1', 5.],
#                             ['profile_1', 'l2', 4.],
#                             ['profile_1', 'l4', 4.],
#                             ['profile_1', 'l5', 2.],
#                             ['profile_1', 'l6', 1.],
#                             ['profile_1', 'l3', 0.5],
#                             ['profile_2', 'l1', 0.],
#                             ['profile_2', 'l2', 1.],
#                             ['profile_2', 'l4', 2.],
#                             ['profile_2', 'l5', 3.],
#                             ['profile_2', 'l6', 4.],
#                             ['profile_2', 'l3', 5.],
#                             ],
#                            columns=['name', 'layer', 'relative_density']
#                            )
#
#    ret = utils.normalize_layer_profiles(layer_heights, profiles)
#    eq_(tuple(ret.loc['MOs']), tuple(ret.loc['ACAd']))
#    w = [100, 100, 100, 100, 100, 100]
#    eq_(ret.loc['FRP']['profile_1'],
#        sum(w) / sum(w_ * p for w_, p in zip(w, [5., 4., 4.,  2., 1., 0.5])))
#}

def test_perform_module_grouping():
    module_grouping = (('Group0', ('Region0', 'Region2', 'Region4', )),
                       ('Group1', ('Region1', 'Region3', 'Region5', )),
                       ('Group3', ('Region10', ))
                       )

    df = pd.DataFrame()
    ret = utils.perform_module_grouping(df, module_grouping)
    eq_(ret.shape, (7, 7))
    eq_(np.count_nonzero(ret.isna()), np.product(ret.shape))

    index = columns = ('Region1', 'Region3', 'Region5', 'Region10', 'Region0', 'Region2', 'Region4', )
    df = pd.DataFrame(np.arange(49).reshape((7, 7)), index=index, columns=columns)
    ret = utils.perform_module_grouping(df, module_grouping)
    eq_(ret.loc['Group1', 'Region1']['Group1', 'Region1'], 0)
    eq_(ret.loc['Group0', 'Region0']['Group0', 'Region0'], 32)


def test_get_region_layer_to_id():
    ret = utils.get_region_layer_to_id(test_utils.REGION_MAP,
                                       'ECT',
                                       [2, 3, 4],
                                       '{region};{subregion}'
                                       )
    eq_(ret, {2: 426, 3: 427, 4: 428})



def test_region_subregion_to_id():
    eq_(utils.region_subregion_to_id(test_utils.REGION_MAP, 'FA', 'KE', '{region}_{subregion}'),
        -1)
    eq_(utils.region_subregion_to_id(test_utils.REGION_MAP, 'FRP', '1', '{region}_l{subregion}'),
        68)
    eq_(utils.region_subregion_to_id(test_utils.REGION_MAP, 'ECT', '1', '{region};{subregion}'),
        836)


def test_mirror_vertices_y():
    vertices = np.array([(0, 100.), (1, 100.), (0., 101)])
    center_line = 0.
    ret = utils.mirror_vertices_y(vertices, center_line)
    assert_allclose(ret, np.array([(0, -100.), (1, -100.), (0., -101)]))


def test_in_2dtriangle():
    points = np.array([[0.1, 0.1],
                       [10., 10.],
                       [0., 0.],
                       [0., 5.],
                       [0., -.00001],
                       ],)

    vertices = np.array([(0., 0), (10., 0), (0., 10.)])
    ret = utils.in_2dtriangle(vertices, points)
    assert_array_equal(ret, (True, False, True, True, False))

    # wind the opposite way
    vertices = np.array([(0., 0), (0., 10.), (10., 0)])
    ret = utils.in_2dtriangle(vertices, points)
    assert_array_equal(ret, (True, False, True, True, False))


def test_raster_triangle():
    vertices = np.array([(0., 0), (2., 0), (0., 2.)])
    ret = utils.raster_triangle(vertices)
    assert_array_equal(ret, np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0]]))


def test_population2region():
    populations = pd.DataFrame([('Foo', 'FOO_REGION', 'l1', ),
                                ('Foo', 'FOO_REGION', 'l2', ),
                                ('Bar', 'BAR_REGION', 'l1', ),
                                ('Bar', 'BAR_REGION', 'l2', )],
                                columns=['population', 'region', 'layer', ])
    population_name = 'Foo'
    ret = utils.population2region(populations, population_name)
    eq_(ret, 'FOO_REGION')


#def test_is_mirror():
#    ok_(False)


def test_find_executable():
    ok_('bin/sh' in utils.find_executable('sh'))
    eq_(None, utils.find_executable('a_fake_executable_which_does_not_exist'))


def test_read_write_frame():
    data = np.arange(10)
    df = pd.DataFrame({'a': data, 'b': data + 25})
    df.index += 10

    with test_utils.tempdir('test_read_write_frame') as tmp:
        path = os.path.join(tmp, 'test.feather')

        #no need to care about index
        utils.write_frame(path, df.copy())
        res = utils.read_frame(path)
        assert_array_equal(res.index, df.index - 10)
        ok_('a' in res)
        ok_('b' in res)

        res = utils.read_frame(path, columns=['a'])
        ok_('b' not in res)

        #care about index
        utils.write_frame(path, df.copy(), reset_index=False)
        res = utils.read_frame(path)
        assert_array_equal(res.index, df.index)

        res = utils.read_frame(path, columns=['a', 'index', ])
        assert_array_equal(res.index, df.index)
        ok_('b' not in res)

        res = utils.read_frame(path, columns=['a', ])
        assert_array_equal(res.index, df.index)
        ok_('b' not in res)

def test_partition_left_right():
    df = pd.DataFrame(np.arange(10) + .1, columns=['z', ])
    left = utils.partition_left_right(df, side='left', center_line_3d=5.1)
    eq_(len(left), 6)

    right = utils.partition_left_right(df, side='right', center_line_3d=5.1)
    eq_(len(right), 4)


def test_normalize_layer_profiles():
    layer_heights = pd.DataFrame([['S1DZ', 665., 180., 499., 141., 335., 156.],
                                  ['S1DZO', 653., 176., 489., 138., 329., 153.],
                                  ],
                                 columns=['name', '6a', '4', '5', '2', '3', '1']).set_index('name')
    profiles = pd.DataFrame([['profile_1', '1', 2.63],
                             ['profile_1', '2', 1.65],
                             ['profile_1', '3', 1.65],
                             ['profile_1', '4', 0.39],
                             ['profile_1', '5', 0.35],
                             ['profile_1', '6a', 0.35],
                             ['profile_2', '1', 2.79],
                             ['profile_2', '2', 0.42],
                             ['profile_2', '3', 0.42],
                             ['profile_2', '4', 0.32],
                             ['profile_2', '5', 1.19],
                             ['profile_2', '6a', 0.44],
                             ['profile_3', '1', 0.36],
                             ['profile_3', '2', 1.59],
                             ['profile_3', '3', 1.59],
                             ['profile_3', '4', 1.37],
                             ['profile_3', '5', 1.29],
                             ['profile_3', '6a', 0.41],
                             ['profile_4', '1', 1.89],
                             ['profile_4', '2', 0.36],
                             ['profile_4', '3', 0.36],
                             ['profile_4', '4', 0.05],
                             ['profile_4', '5', 0.32],
                             ['profile_4', '6a', 2.79],
                             ['profile_5', '1', 0.54],
                             ['profile_5', '2', 1.26],
                             ['profile_5', '3', 1.26],
                             ['profile_5', '4', 0.55],
                             ['profile_5', '5', 0.47],
                             ['profile_5', '6a', 2.12],
                             ['profile_6', '1', 0.18],
                             ['profile_6', '2', 0.18],
                             ['profile_6', '3', 0.18],
                             ['profile_6', '4', 0.15],
                             ['profile_6', '5', 1.11],
                             ['profile_6', '6a', 4.18],
                             ],
                            columns=['name', 'subregion', 'relative_density']
        )
    ret = utils.normalize_layer_profiles(layer_heights, profiles)
    expected = pd.DataFrame.from_dict({'profile_1': {'S1DZ': 1.180914, 'S1DZO': 1.180785},
                                       'profile_2': {'S1DZ': 1.251290, 'S1DZO': 1.251485},
                                       'profile_3': {'S1DZ': 1.000020, 'S1DZO': 1.000376},
                                       'profile_4': {'S1DZ': 0.793501, 'S1DZO': 0.792819},
                                       'profile_5': {'S1DZ': 0.814063, 'S1DZO': 0.813591},
                                       'profile_6': {'S1DZ': 0.568739, 'S1DZO': 0.568273},
                                       })
    expected.index.name = 'region'
    pd.testing.assert_frame_equal(ret, expected)


def test_calculate_region_layer_heights():
    atlas = Mock()
    brain_regions, region_map = test_utils.fake_brain_regions()
    raw = np.zeros((5, 5, 5, 2))
    raw[:, :, :, 1] = 1.
    ph = voxcell.VoxelData(raw, np.ones(3), offset=np.zeros(3))
    def load_atlas(name):
        if name == 'brain_regions':
            return brain_regions
        elif name == '[PH]1':
            return ph
        raise Exception('Unknown atlas: %s' % name)
    atlas.load_data = load_atlas

    regions = ['one', ]
    layers = ['1', ]
    ret = utils.calculate_region_layer_heights(atlas, region_map, regions, layers, layer_splits={})
    eq_(ret, {'one': {'1': 1.0,}})

    layers = ['1a', '1b', ]
    layer_splits = {'1': [('1a', 0.25), ('1b', 0.75)]}
    ret = utils.calculate_region_layer_heights(atlas, region_map, regions, layers, layer_splits)
    eq_(ret, {'one': {'1a': 0.25, '1b': 0.75}})
