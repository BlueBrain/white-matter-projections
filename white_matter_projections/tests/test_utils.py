import numpy as np
import pandas as pd
from white_matter_projections import utils

from nose.tools import eq_


def test_region_layer_heights():
    layer_heights = {
        'Region1': {'l6': 100, 'l5': 100, 'l4': 100, 'l3': 100, 'l2': 100, 'l1': 100, },
        'Region2': {'l6': 300, 'l5': 700, 'l4': 800, 'l3': 900, 'l2': 300, 'l1': 1000, },
        'Region3': {'l6': 300, 'l5': 700, 'l4': 800, 'l3': 900, 'l2': 300, 'l1': 1000, },
    }
    ret = utils.region_layer_heights(layer_heights)
    eq_(ret.loc['Region1'].sum(), 6 * 100)
    eq_(ret.loc['Region1']['l6'], 100)
    eq_(tuple(ret.loc['Region2']), tuple(ret.loc['Region3']))


def test_normalize_layer_profiles():
    layer_heights = {
        'Region1': {'l6': 100, 'l5': 100, 'l4': 100, 'l3': 100, 'l2': 100, 'l1': 100, },
        'Region2': {'l6': 300, 'l5': 700, 'l4': 800, 'l3': 900, 'l2': 300, 'l1': 1000, },
        'Region3': {'l6': 300, 'l5': 700, 'l4': 800, 'l3': 900, 'l2': 300, 'l1': 1000, },
    }
    layer_heights = utils.region_layer_heights(layer_heights)
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
    eq_(tuple(ret.loc['Region2']), tuple(ret.loc['Region3']))
    w = [100, 100, 100, 100, 100, 100]
    eq_(ret.loc['Region1']['profile_1'],
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


#def test_draw_connectivity():
#    #utils.draw_connectivity(fig, df_ipsi, df_contra)
#    pass


def test_relative_to_config():
    config_path, path = __file__.strip('c'), 'test_utils.py'

    # if file doesn't exist, it uses relative
    ret = utils.relative_to_config(config_path, path)
    eq_(ret, config_path)

    # if file does exist, it returns it
    ret = utils.relative_to_config(config_path, ret)
    eq_(ret, config_path)
