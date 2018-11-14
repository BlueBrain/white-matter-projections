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
        'bluepy>=0.12.5',
        'click>=6.0',
        'lazy>=1.0',
        'matplotlib>=2.0.0',
        'pandas>=0.23.0',
        'pyyaml>=3.12',
        'seaborn>=0.8.1',
        'voxcell>=2.4.0',
    ],
    packages=find_packages(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    scripts=[
        'apps/white-matter',
    ],
)
