import os
from utils import tempdir
from white_matter_projections import slurm

from nose.tools import ok_, eq_


SLURM_CONF = dict(
    account='proj68',
    partition='prod',
    constraint='uc3',
    memory='100G',
    cpus=4,
    log_location='./',
    )


def test_write_sbatch_stage():
    app_path = '/foo/bar/white-matter'
    config_path = 'config_path'

    with tempdir('write_sbatch_stage') as tmp:
        slurm.write_sbatch_stage(
                slurm_conf=SLURM_CONF,
                app_path=app_path,
                config_path=config_path,
                working_directory='.',
                output_path=os.path.join(tmp, 'output'),
                sbatch_output=os.path.join(tmp, 'sbatch_output'),
                stage='sample_all',
                regions=['foo', 'bar', 'baz', ])
        ok_(os.path.isdir(os.path.join(tmp, 'sbatch_output')))
        ok_(os.path.isdir(os.path.join(tmp, 'sbatch_output', 'sample_all')))

        path = os.path.join(tmp, 'sbatch_output', 'sample_all', 'foo.sbatch')
        ok_(os.path.exists(path))
        with open(path) as fd:
            contents = fd.read()
            ok_(app_path in contents)
            ok_(config_path in contents)


def test_load_file():
    template = slurm.load_file('sbatch.jinja2')
    ok_('SBATCH' in template.render({}))
