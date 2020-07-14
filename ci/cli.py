import os
import pathlib
import subprocess

import click

HELP_TEXT = """
    Command Line Interface for PyExaFMM
"""

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))

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
    subprocess.run(['pytest', 'scripts'])
    subprocess.run(['pytest', 'utils'])


@click.command(
    help='Run linter'
)
def lint():
    click.echo('Running linter')
    subprocess.run(['pylint', '--rcfile=tox.ini', 'fmm'])


@click.command(
    help='Precompute operators using defualt config'
)
def compute_operators():
    click.echo('Computing operators')
    subprocess.run([
        'python',
        'scripts/precompute_operators.py',
        HERE.parent / 'config.json'
    ])


cli.add_command(build)
cli.add_command(test)
cli.add_command(lint)
cli.add_command(compute_operators)


