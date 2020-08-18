'''Macro connectivity: based on recipe, but not concretized with circuit details'''
import click

from white_matter_projections import utils
from white_matter_projections.app.utils import print_color, REQUIRED_PATH


@click.group()
@click.option('-c', '--config', type=REQUIRED_PATH, required=True)
@click.pass_context
def cmd(ctx, config):
    '''Macro connectivity: based on recipe, but not concretized with circuit details'''
    ctx.obj['config'] = utils.Config(config)


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
