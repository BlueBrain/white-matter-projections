'''commands related to Allen Institute'''
from __future__ import print_function
import json
import os

import click
from white_matter_projections.app.utils import print_color, REQUIRED_PATH


# This should be in nexus...
# https://bbpteam.epfl.ch/project/issues/browse/NCX-194
# https://bbpteam.epfl.ch/project/issues/browse/BBPP82-90
CORTICAL_URL = ('https://github.com/AllenInstitute/mouse_connectivity_models/raw/'
                'master/mcmodels/core/cortical_coordinates/dorsal_flatmap_paths_100.h5')
BRAIN_REGIONS_URL = ('http://download.alleninstitute.org/informatics-archive/current-release/'
                     'mouse_ccf/annotation/ccf_2017/annotation_100.nrrd')
# http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies near the bottom
HIERARCHY_URL = 'http://api.brain-map.org/api/v2/structure_graph_download/1.json'


@click.group()
def cmd():
    '''commands related to Allen Institute'''


@cmd.command()
@click.option('-o', '--output', required=True,
              help='Output path')
@click.option('--cortical-url', default=CORTICAL_URL,
              help='Cortical coordinates URL')
@click.option('--brain-regions-url', default=BRAIN_REGIONS_URL,
              help='Brain Regions Atlas URL')
@click.option('--hierarchy-url', default=HIERARCHY_URL,
              help='Brain Regions Hierarchy URL')
def get_cortical_map(output, cortical_url, brain_regions_url, hierarchy_url):
    '''download the cortical map files from the AIBS'''
    import requests

    def get_file(url, path):
        '''download `url` and save it to `path`'''
        resp = requests.get(url)
        resp.raise_for_status()
        path = os.path.join(output, path)
        with open(path, 'wb') as fd:
            fd.write(resp.content)
        print_color('Downloaded %s -> %s', url, path)

    get_file(brain_regions_url, 'brain_regions.nrrd')
    get_file(cortical_url, 'dorsal_flatmap_paths_100.h5')

    resp = requests.get(hierarchy_url)
    resp.raise_for_status()

    # The Allen Institute adds an extra wrapper around the contents need to strip that
    resp = resp.json()['msg'][0]
    path = os.path.join(output, 'hierarchy.json')
    with open(path, 'wb') as fd:
        json.dump(resp, fd, indent=2)
    print_color('Downloaded %s -> %s', hierarchy_url, path)


@cmd.command()
@click.option('--cortical-map', type=REQUIRED_PATH, required=True,
              help='Path to cortical map downloaded from AIBS')
@click.option('--brain-regions', type=REQUIRED_PATH, required=True,
              help='Path to brain regions NRRD from AIBS.  *must* correspond to the cortical map')
@click.option('--hierarchy', type=REQUIRED_PATH, required=True,
              help='JSON file of labels for the brain-regions dataset')
@click.option('--center-line-2d', type=int, required=True,
              help='X position dividing the left/right hemisphere on the cortical map')
@click.option('--center-line-3d', type=int, required=True,
              help='Z position dividing the left/right hemisphere on the brain regions dataset')
@click.option('--regions', required=True,
              help='')
@click.option('-o', '--output', required=True)
def cortical2flatmap(cortical_map,
                     brain_regions, hierarchy,
                     center_line_2d, center_line_3d,
                     regions,
                     output):
    '''convert a cortical map to a flatmap'''
    from white_matter_projections import cortical_mapping
    assert output.endswith('.nrrd'), 'Should produce an nrrd file'

    regions = map(str, regions.split(','))

    cortical_map_paths = cortical_mapping.CorticalMapParameters(
        cortical_map, brain_regions, hierarchy, center_line_2d, center_line_3d)
    ret, flat_map_shape = cortical_mapping.create_cortical_to_flatmap(cortical_map_paths, regions)
    ret.save_nrrd(output)
    print_color('The flat_map shape is (%d, %d)', *flat_map_shape)
