import pandas as pd
from nose.tools import eq_
from utils import HIER
from white_matter_projections import hierarchy


def test_populate_brain_region_ids():
    columns = ('acronym', 'layer')
    df = pd.DataFrame([('ECT', 'l6'),
                       ('ECT', 'l5'),
                       ('FRP', 'l1'),
                       ('FRP', 'l2'),
                       ('FAKE', 'l2')], columns=columns)

    ids, removed = hierarchy.populate_brain_region_ids(df, HIER)
    eq_(ids, [977, 988, 68, 666, -1])
    eq_(removed, ['FAKE2', ])


def test_get_id_by_region_layer():
    region, layer = 'ECT', 'l6'
    ret = hierarchy.get_id_by_region_layer(HIER, region, layer)
    eq_(ret, 977)


def test_get_full_target_acronym():
    region, layer = 'ECT', 'l6'
    ret = hierarchy.get_full_target_acronym(region, layer)
    eq_(ret, 'ECT6')
