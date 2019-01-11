'''helper to deal with slurm'''
import logging
import os
import pkg_resources

from jinja2 import Template

from white_matter_projections import utils

L = logging.getLogger(__name__)


class Stage(object):
    '''Stage'''
    def __init__(self, name, cmd, next_):
        self.name = name
        self.cmd = cmd
        self.next = next_


STAGES = (Stage('sample_all', 'sample_all --population {region}_ALL_LAYERS',
                ['subsample_left', 'subsample_right', ]),
          Stage('subsample_left', 'subsample --population {region}_ALL_LAYERS --side left',
                ['assignment_left', ]),
          Stage('subsample_right', 'subsample --population {region}_ALL_LAYERS --side right',
                ['assignment_right', ]),
          Stage('assignment_left', 'assignment --population {region}_ALL_LAYERS --side left',
                ['syn2_left', ]),
          Stage('assignment_right', 'assignment --population {region}_ALL_LAYERS --side right',
                ['syn2_right', ]),
          Stage('syn2_left', 'write_syn2 --population {region}_ALL_LAYERS --side left', []),
          Stage('syn2_right', 'write_syn2 --population {region}_ALL_LAYERS --side right', []),
          )
STAGES = {s.name: s for s in STAGES}


def write_sbatch_stage(slurm_conf, app_path, config_path, working_directory,
                       output_path, sbatch_output, stage, regions):
    '''
    Args:
        slurm_conf(dict): dictionary of parameters needed for slurm template
        app_path(str): path to executable to run
        config_path(str): path to white-matter.yaml recipe
        working_directory(str): working directory
        output_path(str): output directory
        sbatch_output(str): where .sbatch files are written
        stage(str): stage name
        regions(list of str): regions to create sbatch files for
    '''
    template = load_file('sbatch.jinja2')

    stage = STAGES[stage]
    launch_cmd = ('{app_path} '
                  '-vv micro '
                  '-c {config_path} '
                  '-o {output_path} '
                  '{cmd}'
                  ).format(app_path=app_path,
                           config_path=config_path,
                           output_path=output_path,
                           cmd=stage.cmd)

    utils.ensure_path(os.path.join(sbatch_output, stage.name))

    for region in regions:
        cmd = launch_cmd.format(region=region) + ' || exit; '

        for stage_next in stage.next:
            cmd += '\nsbatch ' + os.path.join(sbatch_output, stage_next, region + '.sbatch')

        params = {'name': '%s_%s' % (stage.name, region),
                  'cmd': cmd,
                  'sbatch_path': os.path.join(sbatch_output, stage.name, region + '.sbatch'),
                  'working_directory': working_directory,
                  }
        params.update(slurm_conf)

        L.debug('Creating %s', params['sbatch_path'])
        with open(params['sbatch_path'], 'w') as fd:
            fd.write(template.render(params))


def load_file(name):
    '''load a file from the white_matter_projections directory'''
    path = pkg_resources.resource_filename('white_matter_projections', 'data')
    path = os.path.join(path, name)
    with open(path, 'r') as fd:
        contents = fd.read()
    return Template(contents)
