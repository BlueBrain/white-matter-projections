'''app utils'''

import click


def print_color(s, *args, **kwargs):
    '''output text in green'''
    if args:
        s = s % args
    color = kwargs.get('color', 'green')
    click.echo(click.style(s, fg=color))
