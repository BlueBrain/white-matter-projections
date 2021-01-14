#!/usr/bin/env python

import imp

from setuptools import setup, find_packages


VERSION = imp.load_source('', 'white_matter_projections/version.py').__version__

setup(
    name='white-matter-projections',
    author='BlueBrain NSE',
    author_email='bbp-ou-nse@groupes.epfl.ch',
    version=VERSION,
    description='White Matter Projections',
    url='https://bbpteam.epfl.ch/project/issues/projects/NSETM/issues',
    download_url='ssh://bbpcode.epfl.ch/nse/white-matter-projections',
    license='BBP-internal-confidential',
    install_requires=[
        'jinja2>=2.10',
        'bluepy>=0.14.3,<2.0.0',
        'click>=6.0',
        'lazy>=1.0',
        'joblib>=0.13.1',
        'matplotlib>=2.0.0,<3.0.0',
        'pandas>=0.23.0',
        'projectionizer>=1.2.0.dev1',
        'pyarrow>=0.11.1',
        'pyrsistent==0.16.1',  # newer versions don't work with py2.7
        'pyyaml>=3.12',
        'requests>=2.19.1',
        'seaborn>=0.8.1',
        'voxcell>=2.4.0',
        'networkx==2.2',
        'scipy>=1.2.2'
    ],
    packages=find_packages(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    entry_points={
        'console_scripts': [
            'white-matter=white_matter_projections.app.__main__:main'
        ]
    },
)
