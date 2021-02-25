'''general analysis'''
from __future__ import print_function
import json
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
    '''Plot flat map'''
    from white_matter_projections import display
    config, output = ctx.obj['config'], ctx.obj['output']

    name = os.path.join(output, 'flat_map')
    with ctx.obj['figure'](name) as fig:
        ax = fig.gca()
        ax.set_aspect('equal')

        display.plot_allen_coloured_flat_map(ax, config, regions='all')


@cmd.command()
@click.option('-p', '--population', 'source_population', required=False, default=None)
@click.pass_context
def source_locations(ctx, source_population):
    '''Plot the source locations, after allocation, but before used'''
    # pylint: disable=too-many-locals
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

    name = os.path.join(output, 'src_positions_' + source_population)
    with ctx.obj['figure'](name) as fig:
        ax = fig.gca()
        ax.set_aspect('equal')

        display.plot_allen_coloured_flat_map(ax, config, regions='all')
        display.plot_flat_cells(ax, src_cells, sgids, mapper)

        vertices = config.recipe.projections_mapping[source_population]['vertices']
        display.draw_triangle(ax, vertices)


@cmd.command()
@click.option('-p', '--population', 'target_population', required=True)
@click.option('-s', '--side', type=click.Choice(utils.SIDES), required=True)
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
def calculate_compensation(ctx, projection_name, side):
    '''Display the result of using the 'compensation' method'''
    # pylint: disable=too-many-locals
    from white_matter_projections import sampling, display
    config = ctx.obj['config']

    src_uvs, src_uvs_mapped, tgt_uvs, wi_cutoff = sampling.calculate_compensation(
        config, projection_name, side)

    comp = np.count_nonzero(wi_cutoff) / float(len(wi_cutoff) + 1)

    source_population, hemisphere = config.recipe.get_projection(
        projection_name)[['source_population', 'hemisphere']]

    src_vertices = config.recipe.projections_mapping[source_population]['vertices']
    tgt_vertices = config.recipe.projections_mapping[source_population][projection_name]['vertices']

    name = os.path.join(ctx.obj['output'],
                        'calculate_compensation_%s_%s.png' % (projection_name, side))
    with ctx.obj['figure'](name) as fig:
        ax = fig.gca()
        ax.set_aspect('equal')

        display.plot_allen_coloured_flat_map(ax, config, regions='all')
        size = 10

        # src points
        color, alpha = 'white', 0.5
        ax.scatter(src_uvs[:, utils.Y], src_uvs[:, utils.X],
                   marker='.', s=size, alpha=alpha, color=color)

        # all
        color, alpha = 'yellow', 1
        ax.scatter(src_uvs_mapped[:, utils.Y], src_uvs_mapped[:, utils.X],
                   marker='.', s=size, alpha=alpha, color=color)

        # used
        color, alpha = 'green', 1
        ax.scatter(tgt_uvs[wi_cutoff, utils.Y], tgt_uvs[wi_cutoff, utils.X],
                   marker='.', s=size, alpha=alpha, color=color)

        if utils.is_mirror(side, hemisphere):
            src_vertices = utils.mirror_vertices_y(src_vertices, config.flat_map.center_line_2d)
            tgt_vertices = utils.mirror_vertices_y(tgt_vertices, config.flat_map.center_line_2d)

        display.draw_triangle(ax, src_vertices, color='green')
        display.draw_triangle(ax, tgt_vertices, color='yellow')

        ax.set_title('Projection: %s, %.4f times compensation' % (projection_name, 1. / comp))

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
    '''plot projections from `projection_name` and `side`'''
    from white_matter_projections import display, micro
    config, output = ctx.obj['config'], ctx.obj['output']

    allocations_path = os.path.join(output, 'allocations.h5')
    allocations = (micro.load_allocations(allocations_path, config.recipe.projections_mapping)
                   .set_index('projection_name'))

    name = str(os.path.join(output, 'projection_%s_%s.png' % (projection_name, side)))
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

    print_color('Green triangle: Source region\n'
                '   White: source cell positions in flat map\n'
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
        ax = fig.gca()
        ax.set_aspect('equal')

        display.plot_allen_coloured_flat_map(ax, config, regions='all')
        display.draw_triangle_map(ax, config, projection_name, side)

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
    densities = densities.query('projection_name == @projection_name').copy()
    densities = densities[['region_tgt', 'subregion_tgt', 'density']]
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
