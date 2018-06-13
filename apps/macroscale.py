#!/usr/bin/env python
'''macroscale application'''
import argparse
import logging
import os
import yaml
from voxcell.nexus import voxelbrain

#import matplotlib
#matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt

from white_matter_projections import hierarchy, macro, utils

L = logging.getLogger(__name__)


def target_density(recipe, norm_layer_profiles):
    # Overall target density
    df_ipsi, df_contra = recipe.get_target_region_density(norm_layer_profiles)
    import numpy as np
    import seaborn as sns
    #cmap = sns.light_palette('blue', as_cmap=True)
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True, center='light')
    cmap.set_bad(color='white')
    df = df_ipsi.replace(0., np.NaN)
    sns.heatmap(df, cmap=cmap, square=True, xticklabels=1, yticklabels=1, linewidth=0.5)
    plt.show()

def connectivity_matrix(recipe, module_grouping):
    df_ipsi, df_contra = recipe.get_connection_density_map()
    df_ipsi = utils.perform_module_grouping(df_ipsi, module_grouping)

    fig = plt.figure()
    utils.draw_connectivity(fig, df_ipsi, 'Connection synapse density - Ipsilateral')
    ax = fig.axes[1]
    ax.set_title('syns/um^3')
    plt.show()
    fig = plt.figure()

    #utils.draw_connectivity(fig, df_contra, 'Connection synapse density - Contralateral')
    #ax = fig.axes[1]
    #ax.set_title('syns/um^3')
    #plt.show()

    #TODO: is this really something useful? realistic?
    #df_ipsi, df_contra = recipe.get_connection_synapse_count_map(module_grouping)
    #fig = plt.figure()
    #utils.draw_connectivity(fig, df_ipsi, 'Connection synapse count - Ipsilateral')
    #ax = fig.axes[1]
    #ax.set_title('count')
    #ax = fig.axes[0]
    #plt.show()


def stacked_target_density(recipe, norm_layer_profiles, target='FRP'):
    '''
    Colored by source:
        1. MODULE
        2. LAYER (or subregion, i.e. 5it and 5pt are separate)
        3. REGION
    '''
    stacked_density = recipe.get_target_region_density_sources(norm_layer_profiles, target)
    #import seaborn as sns
    #cmap = sns.light_palette('blue', as_cmap=True)
    #cmap.set_bad(color='white')
    #sns.set(cmap)
    ax = stacked_density.plot(kind='bar', stacked=True)
    ax.set_title('Vertical Profile of incoming regions for target: FRP')
    plt.show()


def stacked_target_density_bokeh(recipe, norm_layer_profiles, target='FRP'):
    from bokeh.core.properties import value
    from bokeh.io import output_file, show
    from bokeh.models import HoverTool, ColumnDataSource, CustomJS, FactorRange
    from bokeh.plotting import figure
    from bokeh.palettes import Spectral8

    stacked_density = recipe.get_target_region_density_sources(norm_layer_profiles, target)
    #stacked_density.index.name = 'layer'
    stacked_density.columns.name = 'region'
    df = stacked_density

    output_file('stacked_density1.html')

    import itertools as it
    colors = list(it.islice(it.cycle(['blue', 'red', 'green']), len(df.columns)))


    p = figure(plot_width=800, x_range=tuple(df.index))
    p.vbar_stack(list(df.columns), x='layer', width=0.8, fill_color=colors,
                 line_color=None, source=df, legend=[value(x) for x in df.columns])
    show(p)

    sd = (stacked_density
          .reset_index()
          .melt('index', )
          )

    output_file('stacked_density1.html')

    import itertools as it
    layers =  list(stacked_density.index)
    source_regions = list(stacked_density.columns)
    colors = list(it.islice(it.cycle(['blue', 'red', 'green']), len(source_regions)))

    p = figure(x_range=FactorRange(*sd.index), plot_height=250, title="Fruit Counts by Year",
               toolbar_location=None, tools="")
    p.vbar_stack(sd.columns, x='Layer', width=10, source=sd, fill_color=Spectral8)
    show(p)
    #p = figure(title='Vertical Profile of incoming regions for target: %s' % target,
    #           plot_height=250,
    #           match_aspect=True,
    #           tools='wheel_zoom,reset',
    #           background_fill_color='#CCCCCC')
    p.grid.visible = False
    sd = stacked_density

    idx = idx[:3]
    colors = ['blue', 'red', 'green']

    colors = ['blue', 'red', 'green']
    idx = ['2016', '2017',]
    p = figure(plot_height=250, title="Fruit Counts by Year",
               x_range=idx,
               toolbar_location=None, tools="")
    source = ColumnDataSource({'2016': [1, 2, 3], '2017':[4, 5, 6]})
    p.vbar_stack(idx, width=10, source=source, color=colors)
    show(p)


def stacked_target_density_altair(recipe, norm_layer_profiles, target='FRP', modules=None):
    import altair as alt
    # stacked target density

    stacked_density = recipe.get_target_region_density_sources(norm_layer_profiles, target)
    #stacked_density = recipe.get_target_region_density_modules(norm_layer_profiles, target, modules)

    stacked_density = (stacked_density
                       .T
                       .reset_index()
                       .melt('Source')
                       )
    order = alt.Order('Source', sort='ascending')
    chart = (alt.Chart(stacked_density)
             .properties(height=1000, width=400, title='FRP Layer by Source Population')
             .mark_bar()
             .encode(
                 x=alt.X('Target:N', axis=alt.Axis(title='Layer')),
                 y=alt.Y('sum(value):Q', axis=alt.Axis(title='Density')),
                 color=alt.Color('Source'),
                 order=order,
                 tooltip=['Source', 'Target', 'value']
             )
             .save('stacked_density.html')
             )
    os.system('/usr/bin/google-chrome ' 'stacked_density.html')


def get_parser():
    '''return the argument parser'''
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output',
                        help='Output directory')
    parser.add_argument('-c', '--config',
                        help='Config file')
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='-v for INFO, -vv for DEBUG')

    return parser


def main(args):
    '''main'''
    logging.basicConfig(level=(logging.WARNING,
                               logging.INFO,
                               logging.DEBUG)[min(args.verbose, 2)])

    config_path = args.config
    with open(config_path) as fd:
        config = yaml.load(fd)

    atlas = voxelbrain.VoxelBrainAtlas(config['atlas_url'], cache_dir=config['cache_dir'])
    hier = atlas.load_hierarchy()

    recipe_path = config['projections_recipe']
    if not os.path.exists(recipe_path):
        relative = os.path.join(os.path.dirname(config_path), recipe_path)
        if os.path.exists(relative):
            recipe_path = relative
        else:
            raise Exception('Cannot find recipe: %s', recipe_path)

    with open(recipe_path) as fd:
        recipe = yaml.load(fd)
    recipe = macro.MacroConnections.load_recipe(recipe, hier)

    layer_heights = utils.region_layer_heights(config['region_layer_heights'])
    norm_layer_profiles = utils.normalize_layer_profiles(layer_heights, recipe.layer_profiles)

    #target_density(recipe, norm_layer_profiles)
    #stacked_target_density(recipe, norm_layer_profiles, target='FRP')
    stacked_target_density_altair(recipe, norm_layer_profiles, target='FRP', modules=config['module_grouping'])
    #connectivity_matrix(recipe, config['module_grouping'])

    #from white_matter_projections import micro
    #sample_output = os.path.join(args.output, 'sample')
    #if not os.path.exists(sample_output):
    #    os.makedirs(sample_output)
    #micro.make_region_counts(sample_output, atlas, norm_layer_profiles, recipe)


if __name__ == '__main__':
    PARSER = get_parser()
    main(PARSER.parse_args())
