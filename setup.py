import os

from setuptools import setup, find_packages


HERE = os.path.abspath(os.path.dirname(__file__))
PATH_VERSION = os.path.join(HERE, 'fmm', '__version__.py')

ABOUT = {}
with open(PATH_VERSION, mode='r', encoding='utf-8') as f:
    exec(f.read(), ABOUT)

requirements = [
    "numpy",
    "numba",
    "pytest",
    "click",
    "scipy",
    "h5py",
    "pylint"

]

setup(
    name=ABOUT['__title__'],
    version=ABOUT['__version__'],
    description=ABOUT['__description__'],
    license="BSD3",
    author="Srinath Kailasa",
    author_email="srinathkailasa@gmail.com",
    url="https://github.com/exafmm/pyexafmm",
    packages=find_packages(
        exclude=['*.test']
    ),
    zip_safe=False,
    install_requires=requirements,
    keywords='PyExaFMM',
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
    entry_points={
        'console_scripts': [
            'fmm=ci.cli:cli'
        ]
    },
)
