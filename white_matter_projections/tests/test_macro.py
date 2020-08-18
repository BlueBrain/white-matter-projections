import pandas as pd
from nose.tools import ok_, eq_
from white_matter_projections import macro, utils
from utils import (POP_CAT, RECIPE, RECIPE_TXT, REGION_MAP,
                   REGION_SUBREGION_FORMAT,
                   REGION_SUBREGION_SEPARATION_FORMAT,
                   SUBREGION_TRANSLATION, tempdir,
                   get_region_subregion_translation
                   )
from numpy.testing import assert_allclose, assert_raises
from pandas.testing import assert_frame_equal


FLAT_MAP_NAMES = ['Allen Dorsal Flatmap', ]


def test__parse_populations():
    region_subregion_translation = get_region_subregion_translation()
    pop_cat, populations = macro._parse_populations(RECIPE['populations'],
                                                    REGION_MAP,
                                                    region_subregion_translation)

    # check categories
    ok_('POP1_ALL_LAYERS' in pop_cat.categories)
    ok_('SUB_POP4_L23' in pop_cat.categories)
    ok_('POP4_ALL_LAYERS' in pop_cat.categories)

    # check populations
    eq_(len(populations), 26)
    eq_(len(populations.query('population == "POP1_ALL_LAYERS"')), 6)
    SUB_POP4_L23 = populations.query('population == "SUB_POP4_L23"')
    eq_(tuple(SUB_POP4_L23[['region', 'subregion', 'id']].values[0]),
        ('FRP', '2', 666))
    eq_(populations.population_filter.nunique(), 3)

    eq_(dict(populations.population_filter.value_counts()),
        {'intratelencephalic': 6, 'EXC': 2, 'Empty': 18})

    populations = [{'atlas_region': {'name': 'ECT' },
                   'filters': {},
                   'name': 'ECT_ALL_LAYERS',
                    }]

    # utils.REGION_SUBREGION_FORMAT has empty middle non-capture group, work around this
    region_subregion_translation.region_subregion_format = '@{region}(?:_l|;){subregion}'

    pop_cat, populations = macro._parse_populations(populations,
                                                    REGION_MAP,
                                                    region_subregion_translation)
    eq_(len(populations.region.unique()), 1)
    eq_(populations.region.unique()[0], 'ECT')
    eq_(set(populations.subregion), {'1', '2', '3', '4', '5', '6'})

    populations = [{'atlas_region': {'name': 'RH' },
                   'filters': {},
                   'name': 'RH_ALL_LAYERS',
                   }]

    pop_cat, populations = macro._parse_populations(populations,
                                                    REGION_MAP,
                                                    region_subregion_translation)
    eq_(len(populations.region.unique()), 1)
    eq_(populations.region.unique()[0], 'RH')
    eq_(set(populations.subregion), {'RH'})

    populations = [{'atlas_region': {'name': 'SSp-bfd'},
                   'filters': {},
                   'name': 'SSp-bfd_ALL_LAYERS',
                   }]

    region_subregion_translation.region_subregion_separation_format = '(?P<region>[^\d]+)(?P<subregion>\d.*)'

    pop_cat, populations = macro._parse_populations(populations,
                                                    REGION_MAP,
                                                    region_subregion_translation)
    eq_(len(populations), 14)
    eq_(set(populations.hier_region), {'VISrll', 'SSp-bfd'})


def test__parse_projections():
    projections, projections_mapping = macro._parse_projections(RECIPE['projections'],
                                                                POP_CAT,
                                                                {'Allen Dorsal Flatmap'})
    eq_(len(projections.query('hemisphere == "ipsi"')), 3)
    eq_(len(projections.query('target_layer_profile_name == "profile_1"')), 1)
    eq_(len(projections.query('target_layer_profile_name == "profile_2"')), 4)


def test__parse_ptypes():
    ptypes, interaction_matrices = macro._parse_ptypes(RECIPE['p-types'], POP_CAT)

    eq_(len(ptypes.query('source_population == "SUB_POP4_L23"')), 3)
    assert_allclose(ptypes.query('source_population == "POP3_ALL_LAYERS"').fraction.sum(),
                    0.1 + 0.3 + 0.2 + 0.1)


def test__parse_layer_profiles():
    subregion_translation = {'l1': '1',
                             'l23': '23',
                             'l4': '4',
                             'l5': '5',
                             # note: 6a/b missing, not replaced
                             }
    region_subregion_translation = utils.RegionSubregionTranslation(
       subregion_translation=subregion_translation)

    layer_profiles = macro._parse_layer_profiles(RECIPE['layer_profiles'],
                                                 region_subregion_translation)
    eq_(len(layer_profiles.name.unique()), 2)
    eq_(len(layer_profiles.subregion.unique()), 6)
    eq_(sorted(layer_profiles.subregion.unique()),
        ['1', '23', '4', '5', 'l6a', 'l6b'])


def test_MacroConnections():
    region_subregion_translation = get_region_subregion_translation()
    recipe = macro.MacroConnections.load_recipe(
        RECIPE_TXT,
        REGION_MAP,
        cache_dir=None,
        region_subregion_translation=region_subregion_translation,
        flat_map_names=FLAT_MAP_NAMES
    )
    ipsi = recipe.get_connection_density_map('ipsi')
    assert_allclose(ipsi.loc['ECT']['ACAd'], 0.26407104)
    eq_(ipsi.loc['MOs']['ACAd'], 0.)

    layer_heights = {
        'FRP':  {'l6a': 50,  'l6b': 200, '5': 300, '4': 50,  'l23': 400, '1': 500, },
        'MOs':  {'l6a': 300, 'l6b': 700, '5': 800, '4': 300, 'l23': 900, '1': 1000, },
        'ACAd': {'l6a': 450, 'l6b': 800, '5': 700, '4': 450, 'l23': 600, '1': 500, },
    }

    layer_heights = utils.region_layer_heights(layer_heights)

    norm_layer_profiles = utils.normalize_layer_profiles(layer_heights, recipe.layer_profiles)

    ipsi = recipe.get_target_region_density(norm_layer_profiles, 'ipsi')
    assert_allclose(ipsi.loc['ACAd']['1'], 0.13528667)

    ret = recipe.get_target_region_density_sources(norm_layer_profiles, 'FRP')
    assert_allclose(ret.loc['1']['POP2_ALL_LAYERS'], 0.08996559)

    modules = [('TopLevel', ['FRP', 'MOs', ]),
               ]
    ret = recipe.get_target_region_density_modules(norm_layer_profiles, 'FRP', modules)
    ok_(isinstance(ret, pd.DataFrame))
    assert_allclose(ret.loc['1', 'TopLevel'], 0.08996559651848632)


def test__parse_synapse_types():
    fake_params_gaussian = {'name': 'truncated_gaussian',
                            'params': {'mean': 0.46, 'std': 0.26}}
    fake_params_uniform_int = {'name': 'uniform_int',
                               'params': {'min': 0, 'max': 10}}
    fake_params_fixed_value = {'name': 'fixed_value',
                               'params': {'value': 42}}

    synapse_types_matching_phys_parameters = {
        'type_1': {
            'physiology': {
                'U': fake_params_gaussian,
                'D': fake_params_uniform_int,
                'F': fake_params_fixed_value,
                },
            },
        'type_2': {
            'physiology': {
                'U': fake_params_gaussian,
                'D': fake_params_uniform_int,
                'F': fake_params_fixed_value,
                },
            },
        'type_3': {
            'physiology': {
                'U': fake_params_gaussian,
                'D': fake_params_uniform_int,
                'F': fake_params_fixed_value,
                },
            },
        }

    res = macro._parse_synapse_types(synapse_types_matching_phys_parameters)
    eq_(res, synapse_types_matching_phys_parameters)

    synapse_types_mismatching_phys_parameters = synapse_types_matching_phys_parameters

    del synapse_types_mismatching_phys_parameters['type_2']['physiology']['U']

    with assert_raises(AssertionError):
        macro._parse_synapse_types(synapse_types_mismatching_phys_parameters)

    '''  Validate recipe while parsing?
    parameters = { 'type_1': { 'physiology': { 'U': fake_params_gaussian.copy(), }, }, }
    del parameters['type_1']['physiology']['U']['params']['mean']
    with assert_raises(AssertionError):
        macro._parse_synapse_types(synapse_types_mismatching_phys_parameters)

    parameters = { 'type_1': { 'physiology': { 'U': fake_params_gaussian.copy(), }, }, }
    del parameters['type_1']['physiology']['U']['params']['std']
    with assert_raises(AssertionError):
        macro._parse_synapse_types(synapse_types_mismatching_phys_parameters)
    '''


def test_MacroConnections_repr():
    region_subregion_translation = get_region_subregion_translation()
    recipe = macro.MacroConnections.load_recipe(RECIPE_TXT,
                                                REGION_MAP,
                                                cache_dir=None,
                                                region_subregion_translation=region_subregion_translation,
                                                flat_map_names=FLAT_MAP_NAMES
                                                )
    out = str(recipe)
    ok_('MacroConnections' in out)
    ok_('populations: 26' in out)


def test_MacroConnections_serialization():
    region_subregion_translation = get_region_subregion_translation()

    with tempdir('test_MacroConnections_serialization') as tmp:
        recipe = macro.MacroConnections.load_recipe(
            RECIPE_TXT,
            REGION_MAP,
            cache_dir=tmp,
            region_subregion_translation=region_subregion_translation,
            flat_map_names=FLAT_MAP_NAMES
            )


        recipe_cached = macro.MacroConnections.load_recipe(
            RECIPE_TXT,
            REGION_MAP,
            cache_dir=tmp,
            region_subregion_translation=region_subregion_translation,
            flat_map_names=FLAT_MAP_NAMES
            )

        assert_frame_equal(recipe.populations, recipe_cached.populations)
        assert_frame_equal(recipe.projections, recipe_cached.projections)
        #self.projections_mapping = projections_mapping
        assert_frame_equal(recipe.ptypes, recipe_cached.ptypes)
        #ptypes_interaction_matrix
        assert_frame_equal(recipe.layer_profiles, recipe_cached.layer_profiles)
        eq_(recipe.synapse_types, recipe_cached.synapse_types)

#TODO:
#_get_projections
#_get_connection_map
#_calculate_densities
#get_target_region_density_modules
