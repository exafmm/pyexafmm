import subprocess

import click


HELP_TEXT = """
    Command Line Interface for PyExaFMM
"""


@click.group(help=HELP_TEXT)
def cli():
    pass

@click.command(
    help='Build dev version in current python env'
)
def build():
    click.echo('Building and installing')
    subprocess.run(['conda', 'develop', '.'])


@click.command(
    help='Run test suite'
)
def test():
    click.echo('Running test suite')    
    subprocess.run(['pytest', 'fmm'])

@click.command(
    help='Run linter'
)
def lint():
    click.echo('Running linter')
    subprocess.run(['pylint', '--rcfile=tox.ini', 'fmm'])

cli.add_command(build)
cli.add_command(test)
cli.add_command(lint)
