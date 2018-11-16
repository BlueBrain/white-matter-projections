import os
from bluepy.v2.circuit import Circuit
from white_matter_projections import utils, macro
import numpy as np
import pandas as pd
import voxcell
from voxcell.nexus import voxelbrain

from nose.tools import eq_, ok_, assert_raises
import utils as test_utils


class TestConfig(object):
    def __init__(self):
        self.config = test_utils.get_config()

    def test_types(self):
        types = (('region_layer_heights', pd.DataFrame),
                 ('hierarchy', voxcell.hierarchy.Hierarchy),
                 ('atlas', voxelbrain.Atlas),
                 ('recipe', macro.MacroConnections),
                 ('regions', list),
                 #('circuit', Circuit),  # need a valid circuit
                 ('config', dict),
                 ('config_path', str)
                 )

        for attr_, type_ in types:
            ok_(isinstance(getattr(self.config, attr_), type_))

    def test_region_layer_heights(self):
        ret = self.config.region_layer_heights
        eq_(ret.loc['FRP'].sum(), 6 * 100)
        eq_(ret.loc['FRP']['l6'], 100)
        eq_(tuple(ret.loc['MOs']), tuple(ret.loc['ACAd']))

    def test__relative_to_config(self):
        assert_raises(Exception, utils.Config._relative_to_config, 'asdfasdfas', 'asdfasdfas')


def test_normalize_layer_profiles():
    config_path = os.path.join(test_utils.DATADIR, 'config.yaml')
    layer_heights = utils.Config(config_path).region_layer_heights
    profiles = pd.DataFrame([['profile_1', 'l1', 5.],
                             ['profile_1', 'l2', 4.],
                             ['profile_1', 'l4', 4.],
                             ['profile_1', 'l5', 2.],
                             ['profile_1', 'l6', 1.],
                             ['profile_1', 'l3', 0.5],
                             ['profile_2', 'l1', 0.],
                             ['profile_2', 'l2', 1.],
                             ['profile_2', 'l4', 2.],
                             ['profile_2', 'l5', 3.],
                             ['profile_2', 'l6', 4.],
                             ['profile_2', 'l3', 5.],
                             ],
                            columns=['name', 'layer', 'relative_density']
                            )

    ret = utils.normalize_layer_profiles(layer_heights, profiles)
    eq_(tuple(ret.loc['MOs']), tuple(ret.loc['ACAd']))
    w = [100, 100, 100, 100, 100, 100]
    eq_(ret.loc['FRP']['profile_1'],
        sum(w) / sum(w_ * p for w_, p in zip(w, [5., 4., 4.,  2., 1., 0.5])))


def test_perform_module_grouping():
    module_grouping = (('Group0', ('Region0', 'Region2', 'Region4', )),
                       ('Group1', ('Region1', 'Region3', 'Region5', )),
                       ('Group3', ('Region10', ))
                       )

    df = pd.DataFrame()
    ret = utils.perform_module_grouping(df, module_grouping)
    eq_(ret.shape, (7, 7))
    eq_(np.count_nonzero(ret.isna()),  np.product(ret.shape))

    index = columns = 'Region1', 'Region3', 'Region5', 'Region10',  'Region0', 'Region2', 'Region4',
    df = pd.DataFrame(np.arange(49).reshape((7, 7)), index=index, columns=columns)
    ret = utils.perform_module_grouping(df, module_grouping)
    eq_(ret.loc['Group1', 'Region1']['Group1', 'Region1'], 0)
    eq_(ret.loc['Group0', 'Region0']['Group0', 'Region0'], 32)


def test_get_region_layer_to_id():
    ret = utils.get_region_layer_to_id(test_utils.HIER, 'ECT', [2, 3, 4])
    eq_(ret, {2: 426, 3: 427, 4: 428})
