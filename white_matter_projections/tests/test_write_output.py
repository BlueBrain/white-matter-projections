import os
import h5py
import numpy as np
import pandas as pd
from white_matter_projections import write_output as wo
from white_matter_projections import utils
from utils import tempdir

from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_allclose


def test_create_synapse_data():
    for i, (syn2_property, prop) in enumerate((('connected_neurons_pre', 'sgid'),
                                               ('connected_neurons_post', 'tgid'),
                                               ('morpho_section_id_post', 'section_id'),
                                               ('morpho_segment_id_post', 'segment_id'),
                                               ('morpho_offset_segment_post', 'segment_offset'),
                                               ('delay', 'delay'),
                                               ('syn_type_id', 'syn_type_id'),
                                               )):
        i += 1
        df = pd.DataFrame(np.random.random((i, 1)), columns=[prop])
        ret = wo.create_synapse_data(syn2_property, '', df, synapse_data={})
        eq_(len(ret), i)

    #'position', df[utils.XYZ].values  # not used at the moment

    # truncated_gaussian
    np.random.seed(42)
    synapse_data = {'type_1': {'physiology': {'U':
                                              {'distribution':
                                               {'name': 'truncated_gaussian',
                                                'params': {'mean': 0.46,
                                                           'std': 0.26,
                                                           }
                                                }
                                               }
                                              }}}
    dataset = wo.DataSet(1, np.float)
    df = pd.DataFrame(np.ones(500),  # force at least one value to be rejected by 'truncated'
                      columns=['synapse_type_name', ])
    ret = wo.create_synapse_data('u_syn', dataset, df, synapse_data)
    eq_(len(ret), 500)
    assert_allclose(ret[0], [0.5891456797829205])

    # uniform_int
    synapse_data = {'type_1': {'physiology': {'U':
                                              {'distribution':
                                               {'name': 'uniform_int',
                                                'params': {'min': 1,
                                                           'max': 4,
                                                           }
                                                }
                                               }
                                              }}}
    df = pd.DataFrame([[1, ]], columns=['synapse_type_name', ])
    ret = wo.create_synapse_data('u_syn', dataset, df, synapse_data)


def test__create_syn2_properties():
    needed_datasets = {'single_row': wo.DataSet(1, np.float32),
                       'multi_row': wo.DataSet(2, np.int),
                       }
    with tempdir('test__create_syn2_properties') as tmp:
        path = os.path.join(tmp, 'test.syn2')
        with h5py.File(path, 'w') as h5:
            ret = wo._create_syn2_properties(h5, needed_datasets, chunk_size=100)
            ok_(os.path.exists(path))
            eq_(ret['single_row'].shape, (0, ))
            eq_(ret['single_row'].dtype, np.float32)
            eq_(ret['multi_row'].shape, (0, 2))
            eq_(ret['multi_row'].dtype, np.int)


def test_write_syn2():
    count = 5
    df = pd.DataFrame(np.random.random((count, 1)), columns=['delay'])
    df['tgid'] = np.random.randint(10, size=(count, ))
    df['sgid'] = np.random.randint(10, size=(count, ))
    needed_datasets = {'delay': wo.DataSet(1, np.float32), }
    synapse_data = {}
    extra_properties = {'foo': 3}

    def _fake_create_synapse_data(_, __, df, ___):
        return [1.] * len(df)

    with tempdir('test_write_syn2') as tmp:
        output_path = os.path.join(tmp, 'fake.syn2')

        path0 = os.path.join(tmp, '0.feather')
        utils.write_frame(path0, df)

        wo.write_syn2(output_path,
                      [(path0, extra_properties, )],
                      _fake_create_synapse_data,
                      synapse_data,
                      needed_datasets)
        ok_(os.path.exists(output_path))

        with h5py.File(output_path) as h5:
            props = h5[wo.DEFAULT_GROUP]
            ok_('delay' in props)
            eq_(list(props['delay']), [1.] * 5)


def test_chunk_feathers():
    count = 5
    df = pd.DataFrame(np.random.random((count, 3)), columns=utils.XYZ)
    df['tgid'] = np.random.randint(10, size=(count, ))
    df['sgid'] = np.random.randint(10, size=(count, ))

    with tempdir('test_chunk_feathers') as tmp:
        path0 = os.path.join(tmp, '0.feather')
        path1 = os.path.join(tmp, '1.feather')
        utils.write_frame(path0, df)
        utils.write_frame(path1, df)

        extra_properties = {'foo': 3}
        ret = wo.chunk_feathers([(path0, extra_properties, ),
                                 (path1, extra_properties, ), ],
                                chunk_size=3)
        eq_(next(ret).shape, (3, 6))
        eq_(next(ret).shape, (3, 6))  # first consumed, part of second
        eq_(next(ret).shape, (3, 6))
        eq_(next(ret).shape, (1, 6))
        assert_raises(StopIteration, next, ret)
