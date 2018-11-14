import pandas as pd
from nose.tools import ok_, eq_
from white_matter_projections import macro, utils
from utils import POP_CAT, RECIPE, HIER
from numpy.testing import assert_allclose


def test__parse_populations():
    pop_cat, populations = macro._parse_populations(RECIPE['populations'], HIER)
    ok_('POP1_ALL_LAYERS' in pop_cat)
    ok_('SUB_POP4_L23' in pop_cat)
    ok_('POP4_ALL_LAYERS' in pop_cat)
    eq_(len(populations), 26)
    eq_(len(populations.query('population == "POP1_ALL_LAYERS"')), 6)
    SUB_POP4_L23 = populations.query('population == "SUB_POP4_L23"')
    eq_(tuple(SUB_POP4_L23[['region', 'layer', 'id']].values[0]),
        ('FRP', 'l2', 666))


def test__parse_projections():
    projections, projections_mapping = macro._parse_projections(RECIPE['projections'], POP_CAT)
    eq_(len(projections.query('hemisphere == "ipsi"')), 3)
    eq_(len(projections.query('target_layer_profile_name == "profile_1"')), 1)
    eq_(len(projections.query('target_layer_profile_name == "profile_2"')), 4)


def test__parse_ptypes():
    ptypes, interaction_matrices = macro._parse_ptypes(RECIPE['p-types'], POP_CAT)

    eq_(len(ptypes.query('source_population == "SUB_POP4_L23"')), 3)
    assert_allclose(ptypes.query('source_population == "POP3_ALL_LAYERS"').fraction.sum(),
                    0.1 + 0.3 + 0.2 + 0.1)


def test__parse_layer_profiles():
    layer_profiles = macro._parse_layer_profiles(RECIPE['layer_profiles'])
    eq_(len(layer_profiles.name.unique()), 2)
    eq_(len(layer_profiles.layer.unique()), 6)


def test_MacroConnections():
    recipe = macro.MacroConnections.load_recipe(RECIPE, HIER)
    ipsi = recipe.get_connection_density_map('ipsi')
    assert_allclose(ipsi.loc['ECT']['ACAd'], 0.26407104)
    eq_(ipsi.loc['MOs']['ACAd'], 0.)

    layer_heights = {
        'FRP': {'l6': 50,   'l5': 200, 'l4': 300, 'l3': 50,  'l2': 400, 'l1': 500, },
        'MOs': {'l6': 300,  'l5': 700, 'l4': 800, 'l3': 300, 'l2': 900, 'l1': 1000, },
        'ACAd': {'l6': 450, 'l5': 800, 'l4': 700, 'l3': 450, 'l2': 600, 'l1': 500, },
    }

    layer_heights = utils.region_layer_heights(layer_heights)
    norm_layer_profiles = utils.normalize_layer_profiles(layer_heights, recipe.layer_profiles)

    ipsi = recipe.get_target_region_density(norm_layer_profiles, 'ipsi')
    assert_allclose(ipsi.loc['ACAd']['l1'], 0.13528667)

    ret = recipe.get_target_region_density_sources(norm_layer_profiles, 'FRP')
    assert_allclose(ret.loc['l1']['POP2_ALL_LAYERS'], 0.08996559)

    modules = [('TopLevel', ['FRP', 'MOs', ]),
               ]
    ret = recipe.get_target_region_density_modules(norm_layer_profiles, 'FRP', modules)
    ok_(isinstance(ret, pd.DataFrame))
    assert_allclose(ret.loc['l1', 'TopLevel'], 0.08996559651848632)

    #_get_projections
    #_get_connection_map
    #_calculate_densities
    #get_target_region_density_modules
