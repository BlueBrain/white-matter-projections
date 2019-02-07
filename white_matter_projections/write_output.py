'''write output data, currently just syn2'''
import collections
import logging

import pandas as pd
import numpy as np
import h5py
from white_matter_projections import utils


L = logging.getLogger(__name__)

DEFAULT_GROUP = '/synapses/default/properties'

# Note: syn2 is 'transposed': a single 'column' dataset is actually a vector
#       and a 'multi-column' has multiple rows
DataSet = collections.namedtuple('DataSet', 'row_count, dtype')
DEFAULT_CHUNK_SIZE = 100000
# from https://github.com/adevress/syn2_spec/blob/master/spec_synapse_v2.md
DATASETS = {'connected_neurons_pre': DataSet(1, 'i8'),
            'connected_neurons_post': DataSet(1, 'i8'),

            # 'position': DataSet(3, 'd'),  # not used

            'delay': DataSet(1, 'f'),
            'conductance': DataSet(1, 'f'),
            'u_syn': DataSet(1, 'f'),
            'depression_time': DataSet(1, 'i8'),
            'facilitation_time': DataSet(1, 'i8'),
            'decay_time': DataSet(1, 'i8'),

            # guessing on this, not in syn2 spec
            'n_rrp_vesicles': DataSet(1, 'f'),

            'syn_type_id': DataSet(1, 'i8'),

            'morpho_section_id_post': DataSet(1, 'i8'),
            # guessing on this, not in syn2 spec
            'morpho_segment_id_post': DataSet(1, 'i8'),
            'morpho_offset_segment_post': DataSet(1, 'f'),
            }

DATASET_PHYSIOLOGY = {
    'conductance': 'gsyn',
    'u_syn': 'U',
    'depression_time': 'D',
    'facilitation_time': 'F',
    'decay_time': 'dtc',
    'n_rrp_vesicles': 'nrrp',
}


def create_synapse_data(syn2_property_name, dataset, df, synapse_data):
    '''create concrete values for syn2_property_name for all proto-synapses in `df`

    Args:
        syn2_property_name(str): syn2 property name to be populated
        dataset(DataSet): information about dataset to be returned
        df(pd.DataFrame): with columns sgid, tgid, x, y, z, section_id,
        segment_id, segment_offset
        synapse_data(dict): parameters used to populate synapses

    Returns:
        np.array with len(df)
    '''
    def distribute_param(prop):
        '''paramters with distributions: create random numbers'''
        ret = np.empty(len(df), dtype=dataset.dtype)
        for synapse_type_name, frame in df.reset_index().groupby('synapse_type_name'):
            dist = synapse_data['type_' + str(int(synapse_type_name))]
            dist = dist['physiology'][DATASET_PHYSIOLOGY[prop]]['distribution']
            assert dist['name'] in ('uniform_int', 'truncated_gaussian'), 'unknown distribution'

            if dist['name'] == 'uniform_int':
                low, high = dist['params']['min'], dist['params']['max']
                ret[frame.index] = np.random.random_integers(low, high, size=len(frame))
            elif dist['name'] == 'truncated_gaussian':
                # definition of truncated gaussian
                #  https://bbpteam.epfl.ch/project/issues/browse/NCX-169?focusedCommentId=82081
                #  &page=com.atlassian.jira.plugin.system.issuetabpanels%3Acomment-tabpanel
                #  #comment-82081
                mean, std = dist['params']['mean'], dist['params']['std']
                data = np.random.normal(mean, std, size=len(frame))
                rejected = np.nonzero(np.abs(data - mean) > 2. * std)
                while len(rejected[0]):
                    data[rejected] = np.random.normal(mean, std, size=len(rejected))
                    rejected = np.nonzero(np.abs(data - mean) > 2. * std)
                ret[frame.index] = data
        return ret

    if syn2_property_name == 'connected_neurons_pre':
        ret = df.sgid.values - 1  # Note: syn2 is 0 indexed
    elif syn2_property_name == 'connected_neurons_post':
        ret = df.tgid.values - 1  # Note: syn2 is 0 indexed
    elif syn2_property_name == 'position':
        ret = df[utils.XYZ].values
    elif syn2_property_name == 'morpho_section_id_post':
        ret = df.section_id.values
    elif syn2_property_name == 'morpho_segment_id_post':
        ret = df.segment_id.values
    elif syn2_property_name == 'morpho_offset_segment_post':
        ret = df.segment_offset.values
    elif syn2_property_name in DATASET_PHYSIOLOGY:
        ret = distribute_param(syn2_property_name)
    elif syn2_property_name == 'delay':
        ret = 100 * np.ones(len(df))  # TODO: calculate real delay
    elif syn2_property_name == 'syn_type_id':
        ret = 120 * np.ones(len(df))  # TODO: get real syn type?

    return ret


def _create_syn2_properties(h5, needed_datasets, chunk_size):
    '''create datasets in h5 needed for syn2'''
    properties = h5.create_group(DEFAULT_GROUP)

    datasets = {}
    for name, props in needed_datasets.items():
        if props.row_count == 1:
            shape = (0, )
            chunks = (chunk_size, )
            maxshape = (None, )
        else:
            shape = (0, props.row_count)
            chunks = (chunk_size, props.row_count)
            maxshape = (None, props.row_count)

        datasets[name] = properties.create_dataset(name,
                                                   shape,
                                                   chunks=chunks,
                                                   dtype=props.dtype,
                                                   maxshape=maxshape)
    return datasets


def write_syn2(output_path, feather_paths, synapse_data_creator, synapse_data,
               needed_datasets=None, chunk_size=DEFAULT_CHUNK_SIZE):
    '''create a syn2 connectome file

    Args:
        output_path(path): path where syn2 file will be written
        feather_paths(list): paths to feather files containing synapses
        synapse_data_creator(callable): function to population synapse data
        synapse_data: data passed to synapse_data_creator
        chunk_size(int): number of rows in a chunk
        needed_datasets(dict): name ->

    '''
    if needed_datasets is None:
        needed_datasets = DATASETS

    with h5py.File(output_path, 'w') as h5:
        datasets = _create_syn2_properties(h5, needed_datasets, chunk_size)

        for df in chunk_feathers(feather_paths, chunk_size):
            for name, ds in datasets.items():
                data = synapse_data_creator(name, needed_datasets[name], df, synapse_data)
                start = len(ds)
                end = start + len(data)
                ds.resize(end, axis=0)

                row_count = needed_datasets[name].row_count
                if row_count == 1:
                    ds[start:end] = data
                else:
                    ds[:row_count, start:end] = data.T


def chunk_feathers(feather_paths, chunk_size):
    '''load feathers in feather_paths, yield frames of chunk_size'''
    frame = None
    for feather in feather_paths:
        df = utils.read_frame(feather)

        L.debug('Frame: %s -> %d', feather, len(df))
        start = 0

        while start < len(df):
            end = min(start + chunk_size, len(df))
            if frame is None:
                frame = df.iloc[start:end]
            else:
                end = chunk_size - len(frame)
                frame = pd.concat((frame, df.iloc[start:end]),
                                  sort=False, ignore_index=True)
            start = end

            if len(frame) == chunk_size:
                yield frame
                frame = None

    yield frame
