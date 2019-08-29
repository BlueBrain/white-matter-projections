'''general analysis'''
from __future__ import print_function
import logging
import os
from glob import glob

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
@click.pass_context
def flat_map(ctx):
    '''Plot the source locations, after allocation, but before used'''
    from white_matter_projections import display
    config, output = ctx.obj['config'], ctx.obj['output']

    name = os.path.join(output, 'flat_map.png')
    with ctx.obj['figure'](name) as fig:
        ax = fig.gca()
        ax.set_aspect('equal')

        display.plot_allen_coloured_flat_map(ax, config, regions='all')


@cmd.command()
@click.option('-p', '--population', 'source_population', required=False, default=None)
@click.pass_context
def source_locations(ctx, source_population):
    '''Plot the source locations, after allocation, but before used'''
    from white_matter_projections import display, micro, mapping
    config, output = ctx.obj['config'], ctx.obj['output']

    allocations = micro.load_allocations(os.path.join(output, 'allocations.h5'),
                                         config.recipe.projections_mapping)

    if source_population is not None:
        source_populations = allocations.source_population.unique()
        allocations = allocations.query('source_population == @source_population')
        if allocations.empty:
            raise Exception('No allocations for source_population == "%s", try one of %s' %
                            (source_population, sorted(source_populations)))

    sgids = np.hstack(allocations.sgids.values).ravel()
    src_cells = config.get_cells()
    mapper = mapping.CommonMapper.load_default(config)

    name = 'src_positions'
    with ctx.obj['figure'](name) as fig:
        ax = fig.gca()
        ax.set_aspect('equal')

        display.plot_allen_coloured_flat_map(ax, config, regions='all')
        display.plot_flat_cells(ax, src_cells, sgids, mapper)


@cmd.command()
@click.option('-p', '--population', 'target_population', required=True)
@click.option('-s', '--side', type=click.Choice(utils.SIDES))
@click.pass_context
def used_locations(ctx, target_population, side):
    '''Plot the source locations for particular target region, but only ones used'''
    from white_matter_projections import display, micro, mapping
    config = ctx.obj['config']

    files = glob(os.path.join(ctx.obj['output'], micro.ASSIGNMENT_PATH, side, '*.feather'))

    sgids = [utils.read_frame(f, columns=['sgid']) for f in files]
    sgids = pd.concat(sgids, ignore_index=True, sort=False)

    name = os.path.join(ctx.obj['output'],
                        'used_src_positions_%s_%s' % (target_population, side))
    with ctx.obj['figure'](name) as fig:
        ax = fig.gca()
        ax.set_aspect('equal')

        display.plot_allen_coloured_flat_map(ax, config, regions='all')
        display.plot_flat_cells(ax,
                                config.get_cells(),
                                sgids,
                                mapping.CommonMapper.load_default(config))

        regions = set(config.regions)
        regions.discard(
            utils.population2region(config.recipe.populations, target_population).split('_')[0])
        display.plot_source_region_triangles(ax, config, regions)


@cmd.command()
@click.option('-n', '--name', 'projection_name')
@click.option('-s', '--side', type=click.Choice(utils.SIDES))
@click.pass_context
def projection(ctx, projection_name, side):
    '''plot projections from `projection_name` and `side`'''
    from white_matter_projections import display, micro
    config, output = ctx.obj['config'], ctx.obj['output']

    allocations_path = os.path.join(output, 'allocations.h5')
    allocations = (micro.load_allocations(allocations_path, config.recipe.projections_mapping)
                   .set_index('projection_name'))

    name = str(os.path.join(output, 'projection_%s_%s' % (projection_name, side)))
    with ctx.obj['figure'](name) as fig:
        ax = fig.gca()
        ax.set_aspect('equal')

        path = os.path.join(output, micro.ASSIGNMENT_PATH, side, projection_name + '.feather')
        if not os.path.exists(path):
            L.info('Missing path for loading projections: %s', path)
            return

        syns = utils.read_frame(path, columns=['sgid', 'z'])

        display.plot_allen_coloured_flat_map(ax, config, regions='all')
        display.draw_projection(ax, config, allocations, syns, projection_name, side)


@cmd.command()
@click.option('-p', '--population', required=True)
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
