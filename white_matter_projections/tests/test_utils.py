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


def test_hierarchy_2_df():
    ret = utils.hierarchy_2_df(test_utils.HIER_DICT)
    eq_(ret.index.name, 'id')
    eq_(np.count_nonzero(ret.acronym.str.contains('ECT')), 7) # 6 layers and the parent region
    eq_(set(ret.columns), {'acronym', 'name', 'parent_id'})


def test_get_acronym_volumes():
    brain_regions, region_map = test_utils.fake_brain_regions()

    ret = utils.get_acronym_volumes(['one'], brain_regions, region_map, 4, 'left')
    eq_(ret.loc['one'].volume, 14.)

    ret = utils.get_acronym_volumes(['one'], brain_regions, region_map, 2, 'left')
    eq_(ret.loc['one'].volume, 10.)

    ret = utils.get_acronym_volumes(['one'], brain_regions, region_map, 2, 'right')
    eq_(ret.loc['one'].volume, 4.)

    ret = utils.get_acronym_volumes(['one', 'two', 'twenty', 'thirty'],
                                    brain_regions,
                                    region_map,
                                    4,
                                    'left')
    eq_(len(ret), 4)
    eq_(ret.loc['one'].volume, 14.)
    eq_(ret.loc['two'].volume, 10.)
    eq_(ret.loc['twenty'].volume, 4.)
    eq_(ret.loc['thirty'].volume, 4.)


class TestRegionSubregionTranslation(object):
    def __init__(self):
        pass

    def test_translate_subregion(self):
        ''' '''
        rst = utils.RegionSubregionTranslation(region_subregion_separation_format='(?P<region>.*)(?P<subregion>\d+)')
        eq_(('MOs', '1'), rst.extract_region_subregion_from_acronym('MOs1'))

        rst = utils.RegionSubregionTranslation(region_subregion_separation_format=test_utils.REGION_SUBREGION_SEPARATION_FORMAT)
        eq_(('MOs', '1'), rst.extract_region_subregion_from_acronym('MOs;1'))
        eq_(('MOs', '1'), rst.extract_region_subregion_from_acronym('MOs_l1'))


    def test_get_region_layer_to_id(self):
        rst = utils.RegionSubregionTranslation(region_subregion_format='{region};{subregion}')
        ret = rst.get_region_layer_to_id(test_utils.REGION_MAP, 'ECT', [2, 3, 4])
        eq_(ret, {2: 426, 3: 427, 4: 428})

    def test_region_subregion_to_id(self):
        rst = utils.RegionSubregionTranslation(region_subregion_format='{region}_{subregion}')
        eq_(rst.region_subregion_to_id(test_utils.REGION_MAP, 'FA', 'KE'), -1)

        rst = utils.RegionSubregionTranslation(region_subregion_format='{region}_l{subregion}')
        eq_(rst.region_subregion_to_id(test_utils.REGION_MAP, 'FRP', '1', ), 68)

        rst = utils.RegionSubregionTranslation(region_subregion_format='{region};{subregion}')
        eq_(rst.region_subregion_to_id(test_utils.REGION_MAP, 'ECT', '1'), 836)


def test_choice():
    np.random.seed(0)
    indices = utils.choice(np.array([[1., 2, 3, 4],
                                     [0, 0, 1, 0],
                                     [6, 5, 4, 0]]),
                           np.random)
    assert_array_equal(indices, [2, 2, 1])


def test_normalize_probability():
    p = np.array([1, 0])
    ret = utils.normalize_probability(p)
    assert_array_equal(p, ret)


def test_normalize_probability_raises():
    p = np.array([1e-10, -2e-12])
    assert_raises(utils.ErrorCloseToZero, utils.normalize_probability, p)
