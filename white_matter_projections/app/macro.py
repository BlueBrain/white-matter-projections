'''Macro connectivity: based on recipe, but not concretized with circuit details'''
from __future__ import print_function
import click

import numpy as np
from white_matter_projections import utils
from white_matter_projections.app.utils import print_color


@click.group()
@click.option('-c', '--config', type=click.Path(exists=True), required=True)
@click.pass_context
def cmd(ctx, config):
    '''Macro connectivity: based on recipe, but not concretized with circuit details'''
    ctx.obj['config'] = utils.Config(config)


@cmd.command()
@click.option('-t', '--target', default='FRP')
@click.pass_context
def stacked_target_density(ctx, target):
    '''Stacked barchart of incoming regions synapse density for the selected target '''
    config, recipe = ctx.obj['config'], ctx.obj['config'].recipe

    norm_layer_profiles = utils.normalize_layer_profiles(config.region_layer_heights,
                                                         recipe.layer_profiles)

    name = 'stacked_region_%s' % target
    with ctx.obj['figure'](name) as fig:
        ax = fig.gca()
        stacked_density = recipe.get_target_region_density_sources(norm_layer_profiles, target)
        stacked_density.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Vertical Profile of incoming regions for target: ' + target)


@cmd.command()
@click.option('-t', '--target', default='FRP')
@click.pass_context
def stacked_target_density_module(ctx, target):
    '''create stacked density per module as altair interactive graph in html'''
    import altair as alt  # pylint: disable=import-error
    config, recipe = ctx.obj['config'], ctx.obj['config'].recipe

    norm_layer_profiles = utils.normalize_layer_profiles(config.region_layer_heights,
                                                         recipe.layer_profiles)

    stacked_density = recipe.get_target_region_density_modules(
        norm_layer_profiles, target, config.config['module_grouping'])

    stacked_density = (stacked_density
                       .T
                       .reset_index()
                       .melt('Source')
                       )
    order = alt.Order('Source', sort='ascending')
    _ = (alt.Chart(stacked_density)
         .properties(height=1000, width=400,
                     title='%s Layer by Source Population' % target)
         .mark_bar()
         .encode(x=alt.X('Target:N', axis=alt.Axis(title='Layer')),
                 y=alt.Y('sum(value):Q', axis=alt.Axis(title='Density')),
                 color=alt.Color('Source'),
                 order=order,
                 tooltip=['Source', 'Target', 'value']
                 )
         .save('stacked_density_%s.html' % target)
         )


@cmd.command()
@click.option('--hemisphere', type=click.Choice('contra', 'ipsi'), required=True)
@click.pass_context
def connectivity(ctx, hemisphere):
    '''Plot connectivity matrix of of synapse densities per region'''
    from white_matter_projections import display
    config = ctx.obj['config']

    df = config.recipe.get_connection_density_map(hemisphere)
    df = utils.perform_module_grouping(df, config.config['module_grouping'])

    name = 'connectivity_%s' % hemisphere
    with ctx.obj['figure'](name) as fig:
        title = 'Connection synapse density - %s' % hemisphere
        display.draw_connectivity(fig, df, title, config.config['module_grouping_color'])


@cmd.command()
@click.option('--hemisphere', type=click.Choice('contra', 'ipsi'), required=True)
@click.pass_context
def target_density(ctx, hemisphere):
    '''Plot synapse density in target regions, by layer'''
    import seaborn as sns
    config = ctx.obj['config']

    norm_layer_profiles = utils.normalize_layer_profiles(config.region_layer_heights,
                                                         config.recipe.layer_profiles)

    df = config.recipe.get_target_region_density(norm_layer_profiles, hemisphere)

    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True, center='light')
    cmap.set_bad(color='white')
    df = df.replace(0., np.NaN)

    name = 'target_density_%s' % hemisphere
    with ctx.obj['figure'](name) as fig:
        ax = fig.gca()
        sns.heatmap(df, cmap=cmap, square=True, xticklabels=1, yticklabels=1, linewidth=0.5, ax=ax)


@cmd.command()
@click.pass_context
def source_mapping_triangles(ctx):
    '''Plot all triangles of the source regions on the flat-map'''
    from white_matter_projections import display
    config = ctx.obj['config']

    name = 'src_mapping_triangles'
    with ctx.obj['figure'](name) as fig:
        ax = fig.gca()
        display.plot_allen_coloured_flat_map(ax, config, regions='all', only_right=True)
        display.plot_source_region_triangles(ax, config, regions='all', only_right=True)


@cmd.command()
@click.option('-p', '--population', required=True)
@click.pass_context
def stats(ctx, population):
    '''Calculate stats per population'''
    config = ctx.obj['config']

    region_names = set(config.recipe.populations.query('population == @population').subregion)
    population, region_names = population, region_names  # trick pylint

    sources = config.recipe.projections.query('source_population == @population')
    if len(sources):
        region_cells = config.get_cells().query('region in @region_names')
        print_color('Potential source cells in population: %d', len(region_cells))

        print_color('Projections with population as source:')
        out = sources.sort_values('target_density').to_string(
            max_rows=None,
            index=False,
            columns=['target_population', 'projection_name', 'target_density'])
        print(out)

    targets = config.recipe.projections.query('target_population == @population')
    if len(targets):
        region_cells = config.get_cells().query('region in @region_names')
        print_color('Potential target cells in population: %d', len(region_cells))

        print_color('Projections with population as target:')
        out = targets.sort_values('target_density').to_string(
            max_rows=None,
            index=False,
            columns=['source_population', 'projection_name', 'target_density'])
        print(out)
