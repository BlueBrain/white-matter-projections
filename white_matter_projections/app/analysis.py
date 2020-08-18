'''general analysis'''
from __future__ import print_function
import json
import logging
import os

import click
import numpy as np

from white_matter_projections import utils
from white_matter_projections.app.utils import print_color


L = logging.getLogger(__name__)


@click.group()
@click.option('-c', '--config', type=click.Path(exists=True), required=True)
@click.option('-o', '--output', required=True)
@click.pass_context
def cmd(ctx, config, output):
    '''general analysis'''
    ctx.obj['config'] = utils.Config(config)
    assert os.path.exists(output)
    ctx.obj['output'] = output


@cmd.command()
@click.argument('regions', nargs=-1)
@click.pass_context
def flat_map(ctx, regions):
    '''Plot flat map with source triangles'''
    from white_matter_projections import display
    config, output = ctx.obj['config'], ctx.obj['output']

    name = os.path.join(output, 'flat_map')
    with ctx.obj['figure'](name) as fig:
        painter = display.FlatmapPainter(config, fig)

        regions = regions if regions else 'all'
        painter.draw_flat_map_in_colour(regions='all')
        painter.plot_source_region_triangles(regions)


@cmd.command()
@click.option('-n', '--name', 'projection_name')
@click.option('-s', '--side', type=click.Choice(utils.SIDES))
@click.pass_context
def calculate_compensation(ctx, projection_name, side):
    '''Display the result of using the 'compensation' method'''
    from white_matter_projections import display

    name = os.path.join(ctx.obj['output'],
                        'calculate_compensation_%s_%s' % (projection_name, side))
    with ctx.obj['figure'](name) as fig:
        painter = display.FlatmapPainter(ctx.obj['config'], fig)
        painter.draw_flat_map_in_colour(regions='all')
        painter.plot_compensation(projection_name, side)

    print_color('Green triangle: Source region\n'
                '   White: source sampled positions in flat map\n'
                'Yellow triangle: Target region\n'
                '   Yellow: source sampled in flat mapped to target region\n'
                '   Green: source sampled in flat mapped to target region, '
                'within the cutoff range of target region mapped cells\n'
                )


@cmd.command()
@click.option('-n', '--name', 'projection_name')
@click.option('-s', '--side', type=click.Choice(utils.SIDES))
@click.pass_context
def projection(ctx, projection_name, side):
    '''plot all and used source locations for `projection_name` and `side`'''
    from white_matter_projections import display, micro
    config, output = ctx.obj['config'], ctx.obj['output']

    allocations_path = os.path.join(output, 'allocations.h5')
    allocations = (micro.load_allocations(allocations_path, config.recipe.projections_mapping)
                   .set_index('projection_name'))

    path = os.path.join(output, micro.ASSIGNMENT_PATH, side, projection_name + '.feather')
    if not os.path.exists(path):
        L.info('Missing path for loading projections: %s', path)
        return

    syns = utils.read_frame(path, columns=['sgid', 'z'])

    name = str(os.path.join(output, 'projection_%s_%s' % (projection_name, side)))
    with ctx.obj['figure'](name) as fig:
        painter = display.FlatmapPainter(config, fig)
        painter.draw_flat_map_in_colour(regions='all')
        painter.draw_projection(allocations, syns, projection_name, side)

    print_color('Green triangle: Source region\n'
                '   Grey: source cell positions in flat map\n'
                '   Green: used source cell positions in flat map\n'
                'Yellow triangle: Target region'
                '   Red: source cell positions in flat mapped to target region\n'
                '   Blue: used source cell positions in flat mapped to target region\n'
                )


@cmd.command()
@click.option('-p', '--population', required=True, help='Source Population')
@click.pass_context
def allocation_stats(ctx, population):
    '''Based on the allocations created by 'micro allocate', display stats'''
    from white_matter_projections import micro
    config, recipe, output = ctx.obj['config'], ctx.obj['config'].recipe, ctx.obj['output']

    allocations_path = os.path.join(output, 'allocations.h5')
    allocations = micro.load_allocations(allocations_path, recipe.projections_mapping)

    fractions, interactions = micro.allocation_stats(
        recipe, config.get_cells, allocations, population)

    print_color('Population Fractions')
    print(fractions.to_string(max_rows=None))

    print('\n')

    print_color('Population Interactions')
    print(interactions
          .sort_values('absolute_difference', ascending=False)
          .to_string(max_rows=None))
    print_color('Mean absolute difference: %0.2f', interactions['absolute_difference'].abs().mean())


@cmd.command()
@click.option('-n', '--name', 'projection_name')
@click.option('-s', '--side', type=click.Choice(utils.SIDES))
@click.pass_context
def triangle_map(ctx, projection_name, side):
    '''plot projections from `projection_name` and `side`'''
    from white_matter_projections import display
    config, output = ctx.obj['config'], ctx.obj['output']

    name = str(os.path.join(output, 'triangle_map_%s_%s' % (projection_name, side)))
    with ctx.obj['figure'](name) as fig:
        painter = display.FlatmapPainter(config, fig)
        painter.draw_flat_map_in_colour(regions='all')
        painter.draw_triangle_map(projection_name, side)

    print_color('Green triangle: Source region\n'
                'Yellow triangle: Target region'
                )


@cmd.command()
@click.option('-n', '--name', 'projection_name')
@click.option('-s', '--side', type=click.Choice(utils.SIDES))
@click.pass_context
def assignment_validation(ctx, projection_name, side):
    '''Compare the achieved density post assignment to the desired density'''
    from white_matter_projections import micro
    config, output = ctx.obj['config'], ctx.obj['output']

    with open(config.atlas.fetch_hierarchy(), 'r') as fd:
        df = utils.hierarchy_2_df(json.load(fd))

    brain_regions = config.atlas.load_data('brain_regions')

    path = os.path.join(output, micro.ASSIGNMENT_PATH, side, '%s.feather' % projection_name)
    new = utils.read_frame(path)
    new['region_id'] = brain_regions.lookup(new[utils.XYZ].to_numpy())
    new = new.join(df.acronym, on='region_id')

    densities = config.recipe.calculate_densities(
        utils.normalize_layer_profiles(config.region_layer_heights,
                                       config.recipe.layer_profiles))

    region_format = config.region_subregion_translation.region_subregion_format
    densities = densities.query('projection_name == @projection_name')
    densities = densities[['region_tgt', 'subregion_tgt', 'density']].copy()
    densities['acronym'] = densities.apply(
        lambda r: region_format.format(region=r.region_tgt, subregion=r.subregion_tgt).lstrip('@'),
        axis=1)

    region_volumes = utils.get_acronym_volumes(list(densities.acronym.unique()),
                                               brain_regions,
                                               config.atlas.load_region_map())

    volumes = (new.acronym.value_counts() / region_volumes.volume)
    volumes.name = 'desired_density'
    densities = densities.join(volumes, on='acronym')
    densities['absolute_difference'] = np.abs(
        (densities.density - densities.desired_density) / densities.desired_density)

    print_color('Densities')
    print(densities
          .sort_values('absolute_difference', ascending=False)
          .to_string(max_rows=None))
