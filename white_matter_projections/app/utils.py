'''app utils'''

import click


REQUIRED_PATH = click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True)


def print_color(s, *args, **kwargs):
    '''output text in green'''
    if args:
        s = s % args
    color = kwargs.get('color', 'green')
    click.echo(click.style(s, fg=color))
