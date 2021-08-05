'''white-matter application'''
from contextlib import contextmanager

import logging

import click
from white_matter_projections import utils
from white_matter_projections.app.utils import print_color, REQUIRED_PATH
from white_matter_projections.app import (aibs,
                                          analysis,
                                          macro,
                                          micro,
                                          )

L = logging.getLogger(__name__)
# Allow for matplotlib to be interactive or writing to file
plt = None


@click.group()
@click.version_option()
@click.option('-v', '--verbose', count=True)
@click.option('-m', '--interactive-matplotlib', is_flag=True)
@click.option('--plot-format', default='png', required=False)
@click.pass_context
def app(ctx, verbose, interactive_matplotlib, plot_format):
    '''White Matter Generation'''
    global plt  # pylint: disable=global-statement
    if not interactive_matplotlib:
        import matplotlib
        matplotlib.use('Agg')  # noqa

    import matplotlib.pyplot
    plt = matplotlib.pyplot

    if interactive_matplotlib:
        @contextmanager
        def figure(name):  # pylint: disable=unused-argument
            '''get a figure'''
            yield plt.figure()
            plt.show()
    else:
        @contextmanager
        def figure(name):
            '''get a figure'''
            logging.getLogger('matplotlib.font_manager').disabled = True
            plt.close('all')
            fig = plt.gcf()
            fig.set_size_inches(20, 20)
            fig.tight_layout()

            yield fig

            if not name.endswith(plot_format):
                name += '.' + plot_format

            plt.savefig(name)
            print_color('Wrote: %s', name)

    ctx.obj['figure'] = figure

    logging.basicConfig(level=(logging.WARNING,
                               logging.INFO,
                               logging.DEBUG)[min(verbose, 2)],
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


@app.command()
@click.option('-c', '--config', type=REQUIRED_PATH, required=True)
@click.pass_context
def repl(ctx, config):  # pylint: disable=unused-argument
    '''Start ipython REPL'''
    # pylint: disable=import-error,unused-variable
    import IPython
    config = utils.Config(config)
    norm_layer_profiles = utils.normalize_layer_profiles(config.region_layer_heights,
                                                         config.recipe.layer_profiles)
    IPython.embed(banner1='You have access to: config, norm_layer_profiles')


app.add_command(name='aibs', cmd=aibs.cmd)
app.add_command(name='analysis', cmd=analysis.cmd)
app.add_command(name='macro', cmd=macro.cmd)
app.add_command(name='micro', cmd=micro.cmd)


def main():
    '''main entry point'''
    app(obj={})  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg


if __name__ == '__main__':
    main()
