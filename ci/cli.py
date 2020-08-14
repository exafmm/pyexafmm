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
        HERE.parent / 'scripts/precompute_operators.py',
        HERE.parent / 'config.json'
    ])


@click.command(
    help='Recompute operators for current configuration'
)
def recompute_operators():
    click.echo('Deleting operators at this configuration')
    order = CONFIG['order']

    subprocess.call([
        'rm',
        '-rf',
        HERE.parent / CONFIG['operator_dirname'],
    ])

    click.echo('Recomputing operators')
    subprocess.run([
        'python',
        HERE.parent / 'scripts/precompute_operators.py',
        HERE.parent / 'config.json'
    ])


@click.command(
    help='Generate random targets and sources with unit density'
)
@click.argument('npoints')
@click.argument('dtype')
def generate_test_data(npoints, dtype):
    click.echo(f'Generating {npoints} {dtype} sources & targets')
    subprocess.run([
        'python',
        HERE.parent/ 'scripts/generate_test_data.py',
        HERE.parent / 'config.json',
        npoints,
        dtype
    ])



@click.command(
    help='Compress pre-computed M2L operators using defualt config'
)
def compress_m2l():
    click.echo('Compressing M2L operators')
    subprocess.run([
        'python',
        HERE.parent / 'scripts/compress_m2l_operators.py',
        HERE.parent / 'config.json'
    ])


@click.command(
    help='Re-compress pre-computed M2L operators using defualt config'
)
def recompress_m2l():
    click.echo('Re-compressing M2L operators')

    subprocess.call([
        'rm',
        HERE.parent / CONFIG['operator_dirname'] / f"{CONFIG['m2l_compressed_filename']}.pkl",
    ])

    subprocess.run([
        'python',
        HERE.parent / 'scripts/compress_m2l_operators.py',
        HERE.parent / 'config.json'
    ])


cli.add_command(build)
cli.add_command(test)
cli.add_command(lint)
cli.add_command(compute_operators)
cli.add_command(generate_test_data)
cli.add_command(recompute_operators)
cli.add_command(compress_m2l)
cli.add_command(recompress_m2l)
