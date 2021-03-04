import os
import pathlib
import subprocess

import click

from utils.data import load_json

HELP_TEXT = """
    Command Line Interface for PyExaFMM
"""

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG = load_json(HERE.parent / 'config.json')


@click.group(help=HELP_TEXT)
def cli():
    pass

@click.command(
    help='Build dev version in current python env'
)
def build():
    click.echo('Building and installing')
    os.chdir(HERE.parent)
    subprocess.run(['conda', 'build', 'conda.recipe'])
    os.chdir(HERE)


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
@click.option(
    '--config', '-c',
    default='config',
    help="""JSON configuration filename e.g. `experiment-1`,
            file in source root directory."""
    )
def compute_operators(config):
    click.echo('Computing operators')
    subprocess.run([
        'python',
        HERE.parent / 'scripts/precompute_operators.py',
        HERE.parent / f'{config}.json'
    ])


@click.command(
    help='Generate targets and sources with unit density'
)
@click.option(
    '--config', '-c',
    default='config',
    help="""JSON configuration filename e.g. `experiment-1`,
            file in source root directory."""
    )
def generate_test_data(config):
    data_type = CONFIG['data_type']
    click.echo(f'Generating {data_type} sources & targets')
    subprocess.run([
        'python',
        HERE.parent/ 'scripts/generate_test_data.py',
        HERE.parent / f'{config}.json',
        data_type
    ])


cli.add_command(build)
cli.add_command(test)
cli.add_command(lint)
cli.add_command(compute_operators)
cli.add_command(generate_test_data)
