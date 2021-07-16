'''Micro connectivity: based on recipe, including circuit details'''
from glob import glob
import logging
import os
import sys

import click
import pandas as pd
import numpy as np

from white_matter_projections import macro, micro, sampling, utils, write_output
from white_matter_projections.app.utils import print_color, REQUIRED_PATH


L = logging.getLogger(__name__)


@click.group()
@click.option('-c', '--config', type=REQUIRED_PATH, required=True)
@click.option('-o', '--output', required=True)
@click.pass_context
def cmd(ctx, config, output):
    '''Micro connectivity: based on recipe, including circuit details'''
    ctx.obj['config'] = utils.Config(config)
    ctx.obj['output'] = output
    utils.ensure_path(output)

    with open(ctx.obj['config'].recipe_path) as fd:
        path = macro.MacroConnections.cached_recipe_path(fd.read(),
                                                         ctx.obj['config'].cache_dir)
        if os.path.exists(path):
            L.debug('Using serialized recipe from %s', path)


@cmd.command()
@click.pass_context
def allocate(ctx):
    '''Allocate source cell GIDS by region, and store them for use when assigning target GIDS'''
    config, output = ctx.obj['config'], ctx.obj['output']

    allocations_path = os.path.join(output, 'allocations.h5')
    if os.path.exists(allocations_path):
        print_color('Already have created %s, delete it if it needs to be recreated',
                    allocations_path, color='red')
        return

    allocations = micro.allocate_projections(config.recipe, config.get_cells, config.rng)

    micro.save_allocations(allocations_path, allocations)


@cmd.command()
@click.option('-s', '--side', type=click.Choice(utils.SIDES), required=True)
@click.pass_context
def calculate_compensation(ctx, side):
    '''Calculate density compensation for all projections'''
    config, output = ctx.obj['config'], ctx.obj['output']

    compensation_path = sampling.get_compensation_path(output, side)
    if os.path.exists(compensation_path):
        print_color('Already have created %s, delete it if it needs to be recreated',
                    compensation_path, color='red')
        return

    utils.ensure_path(os.path.join(output, sampling.COMPENSATION_PATH))

    projection_names = config.recipe.projections.projection_name.to_list()

    compensation = []
    for projection_name in projection_names:
        L.debug('Doing compensation for %s[%s]', projection_name, side)
        _, src_uvs_mapped, _, within_cutoff = sampling.calculate_compensation(config,
                                                                              projection_name,
                                                                              side)
        compensation.append((side, np.count_nonzero(within_cutoff), len(within_cutoff)))

        df = pd.DataFrame(src_uvs_mapped, columns=['u', 'v'])
        df.to_csv(sampling.get_compensation_src_uvs_path(output, side, projection_name),
                  index=False)

    compensation = pd.DataFrame(compensation,
                                columns=['side', 'within_cutoff', 'total'],
                                index=projection_names)

    compensation.index.name = 'projection_name'
    compensation.to_csv(compensation_path)


@cmd.command()
@click.option('-p', '--population', 'target_population', required=True)
@click.option('-s', '--side', type=click.Choice(utils.SIDES), required=True)
@click.pass_context
def sample_all(ctx, target_population, side):
    '''create and save segment sample regions for target_population'''
    config, output = ctx.obj['config'], ctx.obj['output']
    index_base = config.config['indices']
    target_population = target_population  # keep linter happy
    population = config.recipe.populations.query('population == @target_population')

    brain_regions = config.atlas.load_data('brain_regions')

    sampling.sample_all(output,
                        config,
                        index_base,
                        population,
                        brain_regions,
                        side,
                        dilation_size=config.volume_dilatation)


@cmd.command()
@click.option('-p', '--population', 'target_population', required=True)
@click.option('-s', '--side', type=click.Choice(utils.SIDES))
@click.option('--use-compensation', is_flag=True, default=False,
              help='Perform compensation of density')
@click.option('--rank', type=click.INT,
              help='Only do rows modulo rank')
@click.option('--max-rank', type=click.INT,
              help='Only do rows modulo rank')
@click.pass_context
def subsample(ctx, target_population, side, use_compensation, rank, max_rank):
    '''create candidate synapses from full set of segments created by sample_all'''
    config, output = ctx.obj['config'], ctx.obj['output']

    sampling.subsample_per_target(
        output, config, target_population, side, use_compensation, rank, max_rank)


@cmd.command()
@click.option('-p', '--population', 'target_population', required=True)
@click.option('-s', '--side', type=click.Choice(utils.SIDES))
@click.option('-r', '--reverse', is_flag=True,
              help='Perform assignment of projections in reverse order')
@click.pass_context
def assignment(ctx, target_population, side, reverse):
    '''assign sgids created in allocations with candidate synapses from subsample'''
    config, output = ctx.obj['config'], ctx.obj['output']

    join_cols = ['projection_name', 'source_population']
    hemisphere = config.recipe.projections.set_index(join_cols).hemisphere

    allocations_path = os.path.join(output, 'allocations.h5')
    target_population = target_population  # trick pylint
    allocations = (micro.load_allocations(allocations_path, config.recipe.projections_mapping)
                   .query('target_population == @target_population')
                   .join(hemisphere, on=join_cols)
                   )

    projections_mapping = config.recipe.projections_mapping
    closest_count = config.config['assignment']['closest_count']

    micro.assignment(output,
                     config,
                     allocations,
                     projections_mapping,
                     side,
                     closest_count,
                     reverse)


@cmd.command()
@click.option('-p', '--population', 'target_population', required=True)
@click.option('-s', '--side', type=click.Choice(utils.SIDES))
@click.pass_context
def concat_assignments(ctx, target_population, side):
    '''concatenate assignments'''
    projections = ctx.obj['config'].recipe.projections

    if target_population == 'all':
        target_populations = projections.target_population.unique()
        target_populations = set(target_populations)
        target_populations.remove('ORBvl_ALL_LAYERS')
        target_populations = list(target_populations)
    else:
        target_populations = (target_population, )

    base_path = os.path.join(ctx.obj['output'], micro.ASSIGNMENT_PATH, side)
    for target_population in target_populations:  # pylint: disable=redefined-argument-from-local
        output_path = os.path.join(ctx.obj['output'],
                                   '%s_%s_proj_nrn.feather' % (target_population, side))
        if os.path.exists(output_path):
            L.warning('Already have %s, delete it if it needs to be recreated', output_path)
            continue

        L.info('Concat target_population: %s', target_population)
        projection_names = projections.query('target_population == @target_population')
        syns, missing = [], []
        for name, synapse_type in projection_names[['projection_name', 'synapse_type_name']].values:
            path = os.path.join(base_path, name + '.feather')
            if not os.path.exists(path):
                missing.append(name)
                continue
            L.debug('Loading: %s', path)
            new_syns = utils.read_frame(path)
            new_syns['synapse_type_name'] = int(synapse_type[-1])
            syns.append(new_syns)

        L.debug('Missing feathers for %s', missing)

        L.debug('Concat %d frames', len(syns))
        syns = pd.concat(syns, ignore_index=True, sort=False)

        syns['sgid'] = pd.to_numeric(syns['sgid'], downcast='unsigned')

        L.debug('Sorting %s', output_path)
        syns.sort_values(['tgid', 'sgid'], inplace=True)
        syns.reset_index(drop=True, inplace=True)
        L.debug('Writing %s', output_path)
        utils.write_frame(output_path, syns)


@cmd.command()
@click.option('-p', '--population', 'target_population', required=True)
@click.option('-s', '--side', type=click.Choice(utils.SIDES))
@click.pass_context
def write_syn2(ctx, target_population, side):
    'write out syn2 synapse file for target_population and side'''
    config, output = ctx.obj['config'], ctx.obj['output']

    base_path = os.path.join(output, micro.ASSIGNMENT_PATH, side)

    output = os.path.join(output, 'SYN2')
    utils.ensure_path(output)
    output_path = os.path.join(output, 'white_matter_%s_%s.syn2' % (target_population, side))
    if os.path.exists(output_path):
        L.warning('Already have %s, delete it if it needs to be recreated', output_path)
        return
    L.info('Write syn2 for: %s@%s', target_population, side)

    names = (config.recipe.ptypes
             .merge(config.recipe.projections, on='projection_name')
             .query('target_population == @target_population')
             )

    feather_paths, missing = [], []
    for name, synapse_type in names[['projection_name', 'synapse_type_name']].values:
        path = os.path.join(base_path, name + '.feather')
        if not os.path.exists(path):
            missing.append(path)
            continue
        feather_paths.append((path, {'synapse_type_name': int(synapse_type[-1])}))

    if missing:
        L.warning('Missing feathers for: %s', missing)

    write_output.write_syn2(output_path,
                            feather_paths,
                            write_output.create_synapse_data,
                            config.recipe.synapse_types)


@cmd.command()
@click.pass_context
def write_streamline_mapping(ctx):
    '''write the streamline mapping for the viz team'''
    from white_matter_projections import streamlines
    output = ctx.obj['output']
    files = (glob(os.path.join(output, micro.ASSIGNMENT_PATH, 'left', '*_gid2row.feather')) +
             glob(os.path.join(output, micro.ASSIGNMENT_PATH, 'right', '*_gid2row.feather')))

    sgid2row = pd.concat([utils.read_frame(path) for path in files])
    streamlines.write_mapping(output, sgid2row)


@cmd.command()
@click.option('--working-dir', default=os.path.abspath('.'), help='')
@click.pass_context
def create_sbatch(ctx, working_dir):
    '''create sbatch files to run the whole white-matter pipeline'''
    from white_matter_projections import slurm
    config, output = ctx.obj['config'], ctx.obj['output']

    output = os.path.abspath(output)

    sbatch_output = os.path.join(output, 'slurm')
    utils.ensure_path(sbatch_output)

    log_location = os.path.join(output, 'logs')
    utils.ensure_path(log_location)

    # TODO: this should come from the config
    slurm_conf = dict(account='proj68',
                      partition='prod',
                      constraint='uc3|uc4',
                      memory='0G',
                      cpus=72,
                      log_location=log_location
                      )
    app_path = os.path.abspath(sys.argv[0])

    for stage in slurm.STAGES:
        slurm.write_sbatch_stage(
            slurm_conf,
            app_path,
            config.config_path,
            working_dir,
            output,
            sbatch_output,
            stage,
            config.regions)
