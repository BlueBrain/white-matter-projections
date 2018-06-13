'''utils'''
import os

import itertools as it
import numpy as np
import pandas as pd


def perform_module_grouping(df, module_grouping):
    '''group regions in df, a DataFrame into a multiindex based on `module_grouping`

    Args:
        df(DataFrame): dataframe to reindex
        module_grouping(list of [Module, [list of regions]]): provides ordering and grouping

    Note: it seems natural to have `module_grouping` be a dictionary, but then
    the ordering is lost
    '''
    tuples = tuple(it.chain.from_iterable(
        tuple(it.product([k], v)) for k, v in module_grouping))
    midx = pd.MultiIndex.from_tuples(tuples, names=['Module', 'Region'])

    ret = df.copy().reindex(index=midx, level=1)
    ret = ret.reindex(columns=midx, level=1)

    ret.index.name = 'Source Population'
    ret.columns.name = 'Target Population Density'
    return ret


def region_layer_heights(layer_heights, columns=('l1', 'l2', 'l3', 'l4', 'l5', 'l6')):
    '''convert `layer_heights` dictionary to DataFrame '''
    return pd.DataFrame.from_dict(layer_heights, orient='index', columns=columns)


def normalize_layer_profiles(layer_heights, profiles):
    '''

    Args:
        layer_heights:
        profiles:

    As per Michael Reiman in NCX-121:
        Show overall density in each layer of the target region. Method: let w be the vector of
        layer widths of the target region, p the layer profile of a projection:
           x  = sum(w) / sum(w * p)
    '''
    ret = pd.DataFrame(index=layer_heights.index, columns=profiles.name.unique())
    ret.index.name = 'region'
    for profile_name, profile in profiles.groupby('name'):
        for region in layer_heights.index:
            w = layer_heights.loc[region].values
            p = profile['relative_density'].values
            ret.loc[region][profile_name] = np.sum(w) / np.dot(p, w)

    return ret


def draw_connectivity(fig, df, title, module_grouping_color):
    '''create figure of connectivity densities

    Args:
        fig: matplotlib figure
        df(dataframe): ipsilateral densities

    '''
    # pylint: disable=too-many-locals

    import seaborn as sns
    fig.suptitle(title)
    ax = fig.add_subplot(1, 1, 1)

    cmap = sns.light_palette('blue', as_cmap=True)
    cmap.set_bad(color='white')
    df = df.replace(0., np.NaN)
    ax = sns.heatmap(df, ax=ax, cmap=cmap,
                     square=True, xticklabels=1, yticklabels=1, linewidth=0.5)
    ax.set_xlabel('Target')
    ax.set_ylabel('Source')

    xtl = ax.set_xticklabels([v[1] for v in df.index.values])
    ytl = ax.set_yticklabels([v[1] for v in df.columns.values],
                             rotation='horizontal')

    x_colors = [module_grouping_color[k] for k in df.columns.get_level_values(0)]
    y_colors = [module_grouping_color[k] for k in df.index.get_level_values(0)]

    for x, c in zip(xtl, x_colors):
        x.set_backgroundcolor(c)

    for y, c in zip(ytl, y_colors):
        y.set_backgroundcolor(c)


def relative_to_config(config_path, path):
    '''given path to config, open `path`'''
    if not os.path.exists(path):
        relative = os.path.join(os.path.dirname(config_path), path)
        if os.path.exists(relative):
            path = relative
        else:
            raise Exception('Cannot find path: %s' % path)
    return path
