import os

from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))
PATH_VERSION = os.path.join(HERE, 'fmm', '__version__.py')

ABOUT = {}
with open(PATH_VERSION, mode='r', encoding='utf-8') as f:
    exec(f.read(), ABOUT)


setup(
    name=ABOUT['__title__'],
    version=ABOUT['__version__'],
    description=ABOUT['__description__'],
    packages=find_packages(
        exclude=['*.test']
    ),
    install_requires=[
        'llvmlite==0.30.0',
        'numba==0.46.0',
        'numpy==1.17.3'

    ],
    extras_require={
        'dev': [
            'pytest==3.6.4',
            'pytest-cov==2.6.0',
            'pylint==2.4.3'
        ]
    },
    setup_requires=[
        'pytest-runner'
    ]
)