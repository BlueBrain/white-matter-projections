from mock import Mock
from nose.tools import ok_, eq_
import voxcell

from white_matter_projections import macro, region_densities

import numpy as np
import pandas as pd
import utils
from pandas.testing import assert_series_equal


class TestSamplingRegionDensities(object):
    def __init__(self):
        self.recipe = macro.MacroConnections.load_recipe(
            utils.RECIPE_TXT,
            utils.REGION_MAP,
            cache_dir=None,
            region_subregion_translation=utils.get_region_subregion_translation(),
            flat_map_names=utils.FLAT_MAP_NAMES
        )

    def test__get_target_needed_subregions(self):
        #  also tested by test__get_all_target_relative_densities
        srd = region_densities.SamplingRegionDensities(self.recipe, cache_dir=None)

        ret = srd._get_target_needed_subregions(target_population='POP2_ALL_LAYERS')
        eq_(len(ret), 6)
        eq_(set(ret.region), {'MOs'})
        eq_(list(ret.subregion), ['1', '2', '3', '4', '5', '6', ])
        eq_(list(ret.layer_group), [0, 1, 1, 2, 3, 4, ])

        ret = srd._get_target_needed_subregions(target_population='SUB_POP4_L23')
        # profile_3 includes '1', '2', '3', but since '1' doesn't exist in SUB_POP4_L23,
        # only 2 rows, and subregion is 2 & 3
        eq_(len(ret), 2)
        eq_(set(ret.region), {'FRP'})
        eq_(list(ret.subregion), ['2', '3', ])
        eq_(list(ret.layer_group), [1, 2,])

        # not a target of any projections
        ret = srd._get_target_needed_subregions(target_population='POP1_ALL_LAYERS')
        eq_(len(ret), 0)

    def test_get_region_layer_weights(self):
        atlas = utils.make_mock_atlas(brain_regions=utils.recipe_brain_regions(),
                                      region_map=None,
                                      have_ph=False)

        srd = region_densities.SamplingRegionDensities(self.recipe, cache_dir=None)
        res = srd.get_region_layer_weights(atlas, 'POP2_ALL_LAYERS')
        expected = pd.DataFrame(
            {
            'region': ['MOs', 'MOs', 'MOs', 'MOs', 'MOs', 'MOs'],
            'subregion': ['1', '2', '3', '4', '5', '6'],
            'weight': [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
            }
            )
        pd.testing.assert_frame_equal(expected, res)

        # region doesn't materialize, get empty DataFrame
        res = srd.get_region_layer_weights(atlas, 'POP_DOES_EXIST')
        pd.testing.assert_frame_equal(pd.DataFrame(columns=['region', 'subregion', 'weight']),
                                      res)

    def test_get_region_layer_weights_caching(self):
        atlas = utils.make_mock_atlas(brain_regions=utils.recipe_brain_regions(),
                                      region_map=None,
                                      have_ph=True)  # needed for _calculate_region_layer_heights

        for use_volume in (True, False):
            srd = region_densities.SamplingRegionDensities(self.recipe,
                                                           cache_dir=None,
                                                           use_volume=use_volume)
            no_cache = srd.get_region_layer_weights(atlas, 'POP2_ALL_LAYERS')

            with utils.tempdir('test_get_region_layer_weights') as tmp:
                srd = region_densities.SamplingRegionDensities(self.recipe,
                                                               cache_dir=tmp,
                                                               use_volume=use_volume)
                empty_cache = srd.get_region_layer_weights(atlas, 'POP2_ALL_LAYERS')

                cached = srd.get_region_layer_weights(atlas, 'POP2_ALL_LAYERS')

            pd.testing.assert_frame_equal(no_cache, empty_cache)
            pd.testing.assert_frame_equal(no_cache, cached)

    def test_get_densities_by_target_population(self):
        atlas = utils.make_mock_atlas(brain_regions=utils.recipe_brain_regions(), region_map=None)
        srd = region_densities.SamplingRegionDensities(self.recipe, cache_dir=None)
        ret = srd.get_sample_densities_by_target_population(atlas, target_population='POP2_ALL_LAYERS')
        eq_(len(ret), 6)
        eq_(set(ret.region), {'MOs'})
        eq_(set(ret.target_population), {'POP2_ALL_LAYERS'})


def test__get_all_target_needed_regions():
    populations = pd.DataFrame([('Foo', 'FOO_REGION', '1', 1,),
                                ('Foo', 'FOO_REGION', '2', 2,),
                                ('Bar', 'BAR_REGION', '1', 3,),
                                ('Bar', 'BAR_REGION', '2', 4,)],
                                columns=['population', 'region', 'subregion', 'id', ])

    projection_columns = ['projection_name',
                          'source_population',
                          'target_population',
                          'target_density',
                          'target_layer_profile_name',
                          'hemisphere']
    projections = pd.DataFrame([['Proj0', 'SRC', 'Foo', 2.63, 'profile_1', 'right', ],
                                ],
                               columns=projection_columns
                               )

    layer_profiles = pd.DataFrame([['profile_1', '1', 2.63],
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
                                   ['profile_2', '6a', 0.44],],
                                   columns=['name', 'subregion', 'relative_density']
                                  )

    ret = region_densities._get_target_needed_subregions(
        populations, projections, layer_profiles, target_population='Foo')
    eq_(len(ret), 2)
    eq_(set(ret.region), {'FOO_REGION'})
    eq_(set(ret.subregion), {'1', '2', })

    # duplicate except in target_layer_profile_name
    projections = pd.DataFrame([['Proj0', 'SRC', 'Foo', 2.63, 'profile_1', 'right'],
                                ['Proj1', 'SRC', 'Foo', 2.63, 'profile_2', 'right']
                                ],
                               columns=projection_columns
                               )
    ret = region_densities._get_target_needed_subregions(
        populations, projections, layer_profiles, target_population='Foo')
    eq_(len(ret), 4) # 'Foo' only has 2 layers, but uses 2 profiles
    eq_(set(ret.region), {'FOO_REGION'})
    eq_(set(ret.subregion), {'1', '2', })

    # from layer_profiles
    ret = ret.set_index(['projection_name', 'subregion'])
    eq_(ret.loc['Proj0', '1'].relative_density, 2.63)
    eq_(ret.loc['Proj0', '2'].relative_density, 1.65)
    eq_(ret.loc['Proj1', '1'].relative_density, 2.79)
    eq_(ret.loc['Proj1', '2'].relative_density, 0.42)

    populations = pd.DataFrame([('Foo', 'FOO_REGION', '1', 1,),
                                ('Foo', 'FOO_REGION', '2', 2,),
                                ('Foo', 'FOO_REGION', '3', 2,),
                                ('Foo', 'FOO_REGION', '4', 2,),
                                ('Foo', 'FOO_REGION', '5', 2,),
                                ('Foo', 'FOO_REGION', '6a', 2,),
                                ],
                                columns=['population', 'region', 'subregion', 'id', ])
    projections = pd.DataFrame([['Proj0', 'SRC', 'Foo', 2.63, 'profile_1', 'right'],
                                ['Proj1', 'SRC', 'Foo', 2.63, 'profile_2', 'right']
                                ],
                               columns=projection_columns
                               )
    ret = region_densities._get_target_needed_subregions(
        populations, projections, layer_profiles, target_population='Foo')
    eq_(len(ret), 2*6)
    eq_(set(ret.region), {'FOO_REGION'})
    eq_(set(ret.subregion), {'1', '2', '3', '4', '5', '6a'})


def test__calculate_region_layer_heights():
    brain_regions, region_map = utils.fake_brain_regions()
    atlas = utils.make_mock_atlas(brain_regions, region_map, have_ph=True)
    needed_subregions = pd.DataFrame([['MOs', '2', 2],
                                      ['MOs', '2', 2],
                                      ],
                                     columns=['region', 'subregion', 'id'])
    ret = region_densities._calculate_region_layer_heights(atlas, needed_subregions)
    eq_(len(ret), 1)
    eq_(ret.iloc[0].subregion, '2')
    eq_(ret.iloc[0].weight, 1.0)


def test__calculate_region_layer_volume():
    brain_regions, region_map = utils.fake_brain_regions()
    atlas = utils.make_mock_atlas(brain_regions, region_map, have_ph=True)
    needed_subregions = pd.DataFrame([['MOs', '2', 2],
                                      ['MOs', '2', 2],
                                      ],
                                     columns=['region', 'subregion', 'id'])
    ret = region_densities._calculate_region_layer_volume(atlas, needed_subregions)
    eq_(len(ret), 1)
    eq_(ret.iloc[0].subregion, '2')
    eq_(ret.iloc[0].weight, 10.0) # 10 voxels of volume 1.


def test__get_sample_densities_by_target_population_equal_relative_density():
    # `relative_density` are equal, should just get the same target densities
    region_layer_weights = pd.DataFrame([['R1', 'S1', 10.],
                                         ['R1', 'S2', 10.]],
                                        columns=['region', 'subregion', 'weight'])

    relative_densities = pd.DataFrame([['Proj1', 'R1', 'S1', 1., 0.25],
                                       ['Proj1', 'R1', 'S2', 1., 0.5]
                                       ],
                                      columns=['projection_name',
                                               'region',
                                               'subregion',
                                               'relative_density',
                                               'target_density'])
    ret = (region_densities._get_sample_densities_by_target_population(region_layer_weights,
                                                                      relative_densities)
           .set_index(['region', 'subregion'])
           )
    assert_series_equal(ret.density,
                        relative_densities.set_index(['region', 'subregion']).target_density,
                        check_names=False)

    # this continues to be the case when the weights are changed
    region_layer_weights = pd.DataFrame([['R1', 'S1', 100.],
                                         ['R1', 'S2', 10.]],
                                        columns=['region', 'subregion', 'weight'])
    ret = (region_densities._get_sample_densities_by_target_population(region_layer_weights,
                                                                      relative_densities)
           .set_index(['region', 'subregion'])
           )
    assert_series_equal(ret.density,
                        relative_densities.set_index(['region', 'subregion']).target_density,
                        check_names=False)

    region_layer_weights = pd.DataFrame([['R1', 'S1', 10.],
                                         ['R1', 'S2', 20.],
                                         ['R1', 'S3', 30.],
                                         ],
                                        columns=['region', 'subregion', 'weight'])

    relative_densities = pd.DataFrame([['Proj1', 'R1', 'S1', 1., 0.25],
                                       ['Proj1', 'R1', 'S2', 1., 0.5],
                                       ['Proj1', 'R1', 'S3', 1., 0.75],
                                       ],
                                      columns=['projection_name',
                                               'region',
                                               'subregion',
                                               'relative_density',
                                               'target_density'])
    ret = (region_densities._get_sample_densities_by_target_population(region_layer_weights,
                                                                      relative_densities)
           .set_index(['region', 'subregion'])
           )
    assert_series_equal(ret.density,
                        relative_densities.set_index(['region', 'subregion']).target_density,
                        check_names=False)


def test__get_sample_densities_by_target_population_different_relative_density():
    # all in same region, all subregions have same size
    region_layer_weights = pd.DataFrame([['R1', 'S1', 10.],
                                         ['R1', 'S2', 10.],
                                         ['R1', 'S3', 10.],
                                         ['R1', 'S4', 10.],
                                         ],
                                        columns=['region', 'subregion', 'weight'])

    relative_densities = pd.DataFrame([['Proj1', 'R1', 'S1', 1., 0.5],
                                       ['Proj1', 'R1', 'S2', 2., 0.5],
                                       ['Proj1', 'R1', 'S3', 4., 0.5],
                                       ['Proj1', 'R1', 'S4', 8., 0.5],
                                       ],
                                      columns=['projection_name',
                                               'region',
                                               'subregion',
                                               'relative_density',
                                               'target_density'])
    ret = (region_densities._get_sample_densities_by_target_population(region_layer_weights,
                                                                       relative_densities)
           .set_index(['region', 'subregion'])
           )
    # S4 has 8 times the relative_density of S1
    np.testing.assert_almost_equal(ret.loc['R1', 'S4'].density, ret.loc['R1', 'S1'].density * 8)


def test__get_sample_densities_by_target_population_projections_have_multiple_regions():
    # need to test that the `groupby` is working correctly

    # support target_population is composed of R1_S{1,2,3}, and R2_S{1,2,3}:

    # normalized weight over *all* the regions
    region_layer_weights = pd.DataFrame([['R1', 'S1', 10.],
                                         ['R1', 'S2', 10.],
                                         ['R1', 'S3', 10.],
                                         ['R2', 'S1', 20.], # R2 twice as big as R1
                                         ['R2', 'S2', 20.],
                                         ['R2', 'S3', 20.],
                                         ],
                                        columns=['region', 'subregion', 'weight'])

    relative_densities = pd.DataFrame([['Proj1', 'R1', 'S1', 1., 0.5],
                                       ['Proj1', 'R2', 'S1', 2., 0.5],
                                       ['Proj2', 'R1', 'S2', 3., 0.5],
                                       ['Proj2', 'R2', 'S2', 4., 0.5],
                                       ['Proj3', 'R1', 'S3', 5., 0.5],
                                       ['Proj3', 'R2', 'S3', 6., 0.5],
                                       ],
                                      columns=['projection_name',
                                               'region',
                                               'subregion',
                                               'relative_density',
                                               'target_density'])
    ret = (region_densities._get_sample_densities_by_target_population(region_layer_weights,
                                                                       relative_densities)
           .set_index(['region', 'subregion'])
           )
    np.testing.assert_almost_equal(ret.loc['R1', 'S1'].density * 2. / 1., ret.loc['R2', 'S1'].density)
    np.testing.assert_almost_equal(ret.loc['R1', 'S2'].density * 4. / 3., ret.loc['R2', 'S2'].density)
    np.testing.assert_almost_equal(ret.loc['R1', 'S3'].density * 6. / 5., ret.loc['R2', 'S3'].density)
