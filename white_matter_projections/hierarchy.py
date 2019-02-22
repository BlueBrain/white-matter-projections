'''Atlas/Hierarchy related routines'''
import logging

L = logging.getLogger(__name__)


# translate short forms used in the recipe to longer names used in the hierarchy
LAYER_NAMES_SHORT = {'l1': '1',
                     'l2': '2',
                     'l3': '3',
                     'l4': '4',
                     'l5': '5',
                     'l6': '6',
                     'l6a': '6a',
                     'l6b': '6b',
                     }


def get_full_target_acronym(region, layer):
    '''helper to translate recipe format into hierarchy format'''
    return region + LAYER_NAMES_SHORT[layer]


def populate_brain_region_ids(df, hierarchy):
    '''Populate ids for ('acronym', 'layer') columns in df

    Args:
        df: datafram with ('acronym', 'layer') columns
        hier(voxcell.hierarchy.Hierarchy): hierarchy to verify population acronyms against

    Returns:
        tuple of array of ids corresponding to rows in df and a dataframe w/ the removed rows

    '''
    # XXX: this is slow, speed it up
    ret, removed = [], []
    for _, acronym, layer in df.itertuples():
        id_ = get_id_by_region_layer(hierarchy, acronym, layer)
        if id_ == -1:
            removed.append(get_full_target_acronym(acronym, layer))
        ret.append(id_)

    return ret, removed


def get_id_by_region_layer(hierarchy, region, layer):
    '''Get the id(s) of the region and layer from the hierarchy

    Args:
        hierarchy(voxcell.Hierarchy): the hierarchy
        region(str): region acronym
        layer(str): ex: 'l1', if None, gets all the ids associated with region
    '''
    if layer is None:
        id_ = hierarchy.collect('acronym', region, 'id')
    else:
        name = get_full_target_acronym(region, layer)
        id_ = hierarchy.collect('acronym', name, 'id')

        if len(id_) != 1:
            L.warning('Missing region %s, or too many', name)
            return -1

        id_ = next(iter(id_))
    return id_
