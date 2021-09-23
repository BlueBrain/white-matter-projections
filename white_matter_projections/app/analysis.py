'''general analysis'''
from __future__ import print_function
import json
import logging
import os

import click
import numpy as np
import pandas as pd

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
@click.option('-s', '--side', type=click.Choice(utils.SIDES), required=True)
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
@click.option('-s', '--side', type=click.Choice(utils.SIDES), required=True)
@click.pass_context
def source_locations(ctx, projection_name, side):
    '''plot all and used source locations for `projection_name` and `side`'''
    from white_matter_projections import allocations, display, micro
    config, output = ctx.obj['config'], ctx.obj['output']

    allocations_path = os.path.join(output, 'allocations.h5')
    alloc = (allocations.load_allocations(allocations_path, config.recipe.projections_mapping)
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
        painter.draw_projection(alloc, syns, projection_name, side)

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
    from white_matter_projections import allocations
    config, recipe, output = ctx.obj['config'], ctx.obj['config'].recipe, ctx.obj['output']

    allocations_path = os.path.join(output, 'allocations.h5')
    alloc = allocations.load_allocations(allocations_path, recipe.projections_mapping)

    fractions, interactions = allocations.allocation_stats(
        recipe, config.get_cells, alloc, population)

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
@click.option('-s', '--side', type=click.Choice(utils.SIDES), required=True)
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

    print_color('Green triangle: source region\n'
                '   Blue points: source sampled points\n'
                'Yellow triangle: Target region\n'
                '   Yellow points: source points mapped to target region\n'
                )


@cmd.command()
@click.option('-n', '--name', 'projection_name')
@click.option('-s', '--side', type=click.Choice(utils.SIDES), required=True)
@click.option('--sort', type=bool, is_flag=True, default=False)
@click.pass_context
def assignment_validation(ctx, projection_name, side, sort):
    '''Compare the achieved density post assignment to the desired density'''
    from white_matter_projections import mapping, micro, region_densities
    config = ctx.obj['config']

    projection = config.recipe.get_projection(projection_name)
    flatmap = config.flat_map(
        mapping.base_system_from_projection(config,
                                            projection.source_population,
                                            projection_name))

    densities = (region_densities
                 .SamplingRegionDensities(recipe=config.recipe, cache_dir=config.cache_dir)
                 .get_sample_densities_by_target_population(config.atlas,
                                                            projection.target_population)
                 )

    syns = os.path.join(ctx.obj['output'],
                        micro.ASSIGNMENT_PATH,
                        side,
                        '%s.feather' % projection_name)
    syns = utils.read_frame(syns)
    syns['region_id'] = flatmap.brain_regions.lookup(syns[utils.XYZ].to_numpy())
    with open(flatmap.hierarchy_path, 'r', encoding='utf8') as fd:
        syns = syns.join(utils.hierarchy_2_df(json.load(fd)).acronym, on='region_id')

    region_format = config.region_subregion_translation.region_subregion_format
    densities = densities.query('projection_name == @projection_name')
    densities = densities[['region', 'subregion', 'density']].drop_duplicates().copy()
    densities['acronym'] = densities.apply(
        lambda r: region_format.format(region=r.region, subregion=r.subregion).lstrip('@'),
        axis=1)

    volumes = utils.get_acronym_volumes(list(densities.acronym.unique()),
                                        flatmap.brain_regions,
                                        flatmap.region_map,
                                        flatmap.center_line_3d,
                                        side)

    volumes = (syns.acronym.value_counts() / volumes.volume)
    volumes.name = 'achieved_density'
    densities = densities.join(volumes, on='acronym')
    densities['difference'] = densities.achieved_density - densities.density
    densities['percentage_difference'] = densities.difference / densities.density
    densities['abs_percentage_difference'] = np.abs(densities.percentage_difference)

    if sort:
        densities.sort_values('abs_percentage_difference', ascending=False, inplace=True)

    print_color('Densities')
    print(densities.to_string(max_rows=None))


@cmd.command()
@click.pass_context
def density_weights(ctx):
    '''calculate the density weights'''
    from white_matter_projections import region_densities

    config = ctx.obj['config']

    srd = region_densities.SamplingRegionDensities(recipe=config.recipe,
                                                   cache_dir=config.cache_dir,
                                                   use_volume=True)

    for target_population in config.recipe.projections.target_population.unique():
        print(target_population)
        df = srd.get_sample_densities_by_target_population(config.atlas, target_population)
        df = df[['projection_name', 'density']]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
        print()
