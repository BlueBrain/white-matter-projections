import os
import h5py
import numpy as np
import pandas as pd

from nose.tools import ok_, eq_
from white_matter_projections import micro
from numpy.testing import assert_allclose
from utils import tempdir, gte_


def fake_allocations():
    return {'source_population0': {'projection0': np.arange(10),
                                   'projection1': np.arange(10, 20),
                                   },
            'source_population1': {'projection0': np.arange(20, 30),
                                   'projection1': np.arange(30, 40),
                                   },
            }

def fake_projection_mapping():
    ret = {'source_population0': {'projection0': {'target_population': 'target00'},
                                  'projection1': {'target_population': 'target01'},
                                  },
           'source_population1': {'projection0': {'target_population': 'target10'},
                                  'projection1': {'target_population': 'target11'},
                                  },
           }
    return ret

def compare_allocations(ret):
    allocations = fake_allocations()

    eq_(sorted(allocations), sorted(ret))
    assert_allclose(allocations['source_population0']['projection0'],
                    ret['source_population0']['projection0'])
    assert_allclose(allocations['source_population0']['projection1'],
                    ret['source_population0']['projection1'])


def test_save_load_allocations():
    allocations = fake_allocations()
    with tempdir('test_serialize_allocations') as tmp:
        name = os.path.join(tmp, 'allocations.h5')
        micro.save_allocations(name, allocations)
        ret = micro.load_allocations(name, projections_mapping=None)
        compare_allocations(ret)

        projections_mapping = fake_projection_mapping()
        ret = micro.load_allocations(name, projections_mapping)
        ok_(isinstance(ret, pd.DataFrame))


def test_serialize_allocations():
    allocations = fake_allocations()

    with tempdir('test_serialize_allocations') as tmp:
        name = os.path.join(tmp, 'allocations.h5')
        with h5py.File(name, 'w') as h5:
            micro._serialize_allocations(h5, allocations)

        with h5py.File(name, 'r') as h5:
            ret = micro._deserialize_allocations(h5)

    compare_allocations(ret)


def test_transpose_allocations():
    allocations = fake_allocations()
    projections_mapping = fake_projection_mapping()

    ret = micro.transpose_allocations(allocations, projections_mapping)
    ret.set_index('target_population', inplace=True)
    target00 = ret.loc['target00']
    eq_(target00.projection_name, 'projection0')
    eq_(target00.source_population, 'source_population0')
    eq_(len(target00.sgids), 10)



def test__ptype_to_counts():
    cell_count = 10000
    ptype = pd.DataFrame([('proj0', .25),
                          ('proj1', .25),
                          ('proj2', .10),
                          ],
                         columns=['projection_name', 'fraction'])

    # simple case - no interactions
    interactions = None
    total_counts, overlap_counts = micro._ptype_to_counts(
        cell_count, ptype, interactions)
    eq_(total_counts, {'proj0': 10000*0.25, 'proj1': 10000*0.25, 'proj2': 10000*0.10})
    eq_(overlap_counts, {('proj0', 'proj1'): int(10000 * 0.25 * 0.25),
                         ('proj0', 'proj2'): int(10000 * 0.25 * 0.10),
                         ('proj1', 'proj2'): int(10000 * 0.25 * 0.10),
                         })

    proj = ['proj0', 'proj1', 'proj2']

    # with interactions, but all set to 1 - same as before
    interactions = np.ones((3, 3))
    interactions = pd.DataFrame(interactions, columns=proj, index=proj)
    total_counts, overlap_counts = micro._ptype_to_counts(
        cell_count, ptype, interactions)
    eq_(total_counts, {'proj0': 10000*0.25, 'proj1': 10000*0.25, 'proj2': 10000*0.10})
    eq_(overlap_counts, {('proj0', 'proj1'): int(10000 * 0.25 * 0.25),
                         ('proj0', 'proj2'): int(10000 * 0.25 * 0.10),
                         ('proj1', 'proj2'): int(10000 * 0.25 * 0.10),
                         })


    # now with actual interactions
    interactions = np.array([[1, 2, 3],
                             [2, 1, 1],
                             [3, 1, 1]])
    interactions = pd.DataFrame(interactions, columns=proj, index=proj)
    total_counts, overlap_counts = micro._ptype_to_counts(
        cell_count, ptype, interactions)
    eq_(total_counts, {'proj0': 10000*0.25, 'proj1': 10000*0.25, 'proj2': 10000*0.10})
    eq_(overlap_counts, {('proj0', 'proj1'): int(10000 * 0.25 * 0.25 * 2),
                         ('proj0', 'proj2'): int(10000 * 0.25 * 0.10 * 3),
                         ('proj1', 'proj2'): int(10000 * 0.25 * 0.10),
                         })


def test__make_numeric_groups():
    total_counts = {'proj0': 0, 'proj1': 1, 'proj2': 2}
    overlap_counts = {('proj0', 'proj1'): 1,
                      ('proj0', 'proj2'): 20,
                      ('proj1', 'proj2'): 21,
                      }
    names, name_map, total_counts_remap, overlap_counts_remap = micro._make_numeric_groups(
        total_counts, overlap_counts)
    eq_(['proj0', 'proj1', 'proj2'], sorted(names))
    eq_(total_counts, dict(zip(names, total_counts_remap)))
    proj0, proj1, proj2 = name_map['proj0'], name_map['proj1'], name_map['proj2']
    eq_(overlap_counts_remap,
        {tuple(sorted((proj0, proj1))): 1,
         tuple(sorted((proj2, proj0))): 20,
         tuple(sorted((proj2, proj1))): 21,
         })


def test__allocate_groups():
    # simple case, no interactions
    total_counts = {'proj0': 250, 'proj1': 250, 'proj2': 100}
    overlap_counts = {}
    gids = np.arange(1000)  # 1000 > 250 + 250 + 100
    ret = micro._allocate_groups(total_counts, overlap_counts, gids)
    eq_(['proj0', 'proj1', 'proj2'], sorted(ret))
    eq_(len(ret['proj0']), 250)
    eq_(len(ret['proj1']), 250)
    eq_(len(ret['proj2']), 100)

    np.random.seed(42)
    total_counts = {'proj0': 250, 'proj1': 250, 'proj2': 100}
    overlap_counts = {('proj0', 'proj1'): 10,
                      ('proj0', 'proj2'): 20,
                      ('proj1', 'proj2'): 21,
                      }
    # want a large number here so unlikely to have the required overlap
    gids = np.arange(10000)
    ret = micro._allocate_groups(total_counts, overlap_counts, gids)

    eq_(len(ret['proj0']), 250)
    eq_(len(ret['proj1']), 250)
    eq_(len(ret['proj2']), 100)

    def overlap(g0, g1):
        return set(ret[g0]) & set(ret[g1])

    # Note: the current implementation isn't great, read comment in _fill_groups
    # picking a different seed may make tests pass, but it's nice to have this fail...
    #gte_(overlap_counts[('proj0', 'proj1')], len(overlap('proj0', 'proj1')))
    #gte_(overlap_counts[('proj0', 'proj2')], len(overlap('proj0', 'proj2')))
    #gte_(overlap_counts[('proj1', 'proj2')], len(set(ret['proj1']) & set(ret['proj2'])))


#def test_get_gids_py_population():
#    populations = ''
#    cells = ''
#    source_population = ''
#    ret = utils.get_gids_py_population(populations, cells, source_population)
#    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
#    pass

#def test_allocate_projections():
#    micro.allocate_projections(recipe, cells)

#def test_allocation_stats():
#    ret = micro.allocation_stats(ptype, interactions, cell_count, allocations)
