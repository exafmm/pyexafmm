import os
import subprocess

import click

HERE = os.path.abspath(os.path.dirname(__file__))
TOX_FILEPATH = os.path.join(HERE, 'tox.ini')
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
    subprocess.run(['pip', 'install', '-e.[dev]'])


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
    subprocess.run(['pylint', 'fmm', TOX_FILEPATH])

cli.add_command(build)
cli.add_command(test)
cli.add_command(lint)
