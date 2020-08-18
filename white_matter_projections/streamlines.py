'''download flow/stream lines from AIBS

For each projection from source to target

Note: Currently the hemisphere is ignored, as the API for AIBS doesn't seem
to have an option to consider it (??)
'''
import itertools as it
import logging
import os
import re
import time

import numpy as np
import pandas as pd
import requests

from white_matter_projections import utils
from white_matter_projections.utils import X, Y, Z, XYZ

L = logging.getLogger(__name__)
QUERY_STR = ('http://connectivity.brain-map.org/projection/csv?'
             'criteria=service::mouse_connectivity_target_spatial'
             '[injection_structures$eq{source_region_id}]'
             '[seed_point$eq{seed_x},{seed_y},{seed_z}]'
             '[primary_structure_only$eqfalse]'
             )
STREAMLINES_NAME = 'streamlines'


def get_region_centroid(flat_map, region, side):
    '''using flat_map.{brain_regions, region_map}, find the centroid of `region` in `side`'''
    ids = list(flat_map.region_map.find(region, 'acronym', with_descendants=True))
    center_line = int(flat_map.center_line_3d / flat_map.brain_regions.shape[Z])

    if side == 'right':
        locations = flat_map.brain_regions.raw[:, :, center_line:]
        locations = np.isin(locations, ids)
        locations = (0, 0, center_line) + np.array(np.nonzero(locations)).T
    else:
        assert side == 'left'
        locations = flat_map.brain_regions.raw[:, :, :center_line]
        locations = np.isin(locations, ids)
        locations = np.array(np.nonzero(locations)).T

    ret = None
    if len(locations):
        mean_locations = np.array([locations.mean(axis=0), ]).astype(int)
        ret = flat_map.brain_regions.indices_to_positions(mean_locations)[0].astype(int)

    return ret


def get_all_region_centroids(flat_map, regions, layer='6b'):
    '''get *all* centroids of regions used, default to layer 6b'''
    ret = []
    for region, side in it.product(regions, ('left', 'right')):
        centroid = get_region_centroid(flat_map, region + layer, side)
        if centroid is not None:
            ret.append((region, side) + tuple(centroid))

    columns = ['region', 'side', 'x', 'y', 'z']
    ret = pd.DataFrame(ret, columns=columns)
    ret = ret.set_index(['region', 'side'])
    return ret


def get_connected_regions(recipe):
    '''based on recipe, find the regions that are actually connected'''
    population_regions = (recipe
                          .populations[['population', 'region']]
                          .drop_duplicates()
                          .set_index('population'))

    target_populations = recipe.projections.set_index('projection_name')
    target_populations = target_populations[['target_population', 'hemisphere']]

    regions = (recipe.ptypes
               .join(target_populations, on='projection_name')
               .join(population_regions, on='source_population')
               .rename({'region': 'source_region'}, axis='columns')
               .join(population_regions, on='target_population')
               .rename({'region': 'target_region'}, axis='columns')
               )[['source_region', 'target_region', 'hemisphere']].drop_duplicates()

    return regions


def get_connected_centroids(flat_map, recipe):
    '''for all used projections in the recipe, find the centroids'''
    regions = get_connected_regions(recipe)
    needed_regions = set(regions.source_region) | set(regions.target_region)
    centroids = get_all_region_centroids(flat_map, needed_regions)

    ret = []
    for source_region, target_region in regions[['source_region', 'target_region']].values:
        for source_side, target_side in it.product(utils.SIDES, utils.SIDES):
            ret.append((source_region, target_region, source_side, target_side))

    columns = ['source_region', 'target_region', 'source_side', 'target_side']
    ret = pd.DataFrame(ret, columns=columns).drop_duplicates()

    ret = ret.join(centroids, on=('target_region', 'target_side'))
    ret.rename({ax: 'target_' + ax for ax in list('xyz')}, axis='columns', inplace=True)

    ret = ret.join(centroids, on=('source_region', 'source_side'))
    ret.rename({ax: 'source_' + ax for ax in list('xyz')}, axis='columns', inplace=True)

    return ret


def get_streamline_per_region_connection(connected_regions, metadata, center_line_3d):
    '''for all the connected regions, find an appropriate streamline'''
    metadata = metadata.set_index(['source', 'target', ])

    ret, missing = [], []
    for source, target, hemisphere in connected_regions.values:
        try:
            df = metadata.loc[source, target]
        except KeyError:
            ret.append((source, target, 'contra', 'left', -1))
            ret.append((source, target, 'contra', 'right', -1))
            ret.append((source, target, 'ipsi', 'left', -1))
            ret.append((source, target, 'ipsi', 'right', -1))

            missing.append((source, target))
            continue

        if hemisphere == 'ipsi':
            left_mask = (df.start_z < center_line_3d) & (df.end_z < center_line_3d)
            right_mask = (df.start_z > center_line_3d) & (df.end_z > center_line_3d)
        else:
            assert hemisphere == 'contra'
            left_mask = (df.start_z > center_line_3d) & (df.end_z < center_line_3d)
            right_mask = (df.start_z < center_line_3d) & (df.end_z > center_line_3d)

        if np.any(left_mask):
            row = df[left_mask].iloc[0]['path_row']
        else:
            row = -1

        ret.append((source, target, hemisphere, 'left', row))

        if np.any(right_mask):
            row = df[right_mask].iloc[0]['path_row']
        else:
            row = -1

        ret.append((source, target, hemisphere, 'right', row))

    ret = pd.DataFrame(ret, columns=['source', 'target', 'hemisphere', 'side', 'path_row'])
    ret['hemisphere'] = ret.hemisphere.astype(utils.HEMISPHERE)
    ret['side'] = ret.side.astype(utils.SIDE)
    return ret, missing


def download_streamline(source_region, source_region_id, target_region, seed):
    '''download a set of streamslines from AIBS based on source and target

    Note: need a source_region_id, not just the acronym since the AIBS query
    doesn't allow for '-' in the region_id, so things like 'SSp-tr' don't work
    '''
    seed_x, seed_y, seed_z = seed
    url = QUERY_STR.format(
        source_region_id=source_region_id, seed_x=seed_x, seed_y=seed_y, seed_z=seed_z)

    req = requests.get(url)

    if req.status_code == 500:
        L.debug('Failed(500) for %s -> %s(%s, %s, %s): %s',
                source_region, target_region, seed_x, seed_y, seed_z, url)
        return None

    if not len(req.text):
        L.debug('Failed (empty body) for %s -> %s(%s, %s, %s): %s',
                source_region, target_region, seed_x, seed_y, seed_z, url)
        return None

    try:
        req.raise_for_status()
    except requests.exceptions.HTTPError:
        L.exception('Failed (exception) for %s -> %s(%s, %s, %s): %s',
                    source_region, target_region, seed_x, seed_y, seed_z, url)
        return None

    return req.text


def download_streamlines(centroids, region_map, output_path, sleep_time=0.5):
    '''download streamlines from AIBS'''
    missing = []
    columns = ['source_region', 'target_region',
               'source_x', 'source_y', 'source_z',
               'target_x', 'target_y', 'target_z', ]
    # Note: spatial query doesn't specify which hemisphere, afaik, so only do one query
    for keys in centroids[columns].drop_duplicates().values:
        source_region, target_region = keys[:2]
        seed = keys[-3:]  # seeds are from the target regions

        # Note: expect SOURCENAME_TARGETNAME_.... for get_source_target_from_path
        name = '{source_region}_{target_region}_{seed_x}_{seed_y}_{seed_z}.csv'.format(
            source_region=source_region, target_region=target_region,
            seed_x=seed[X], seed_y=seed[Y], seed_z=seed[Z])
        path = os.path.join(output_path, name)

        if os.path.exists(path):
            L.debug('Already have: %s', path)
            continue

        # throttle the request rate, to be a good citizen
        time.sleep(sleep_time)

        source_region_id = next(iter(region_map.find(source_region, 'acronym')))

        text = download_streamline(source_region, source_region_id, target_region, seed)
        if text is not None:
            with open(path, 'w') as fd:
                fd.write(text)
            L.debug('Wrote: %s', path)
        else:
            missing.append(keys)

    missing = pd.DataFrame(missing, columns=columns)
    return missing


COORD_RE = r'''
    \[                   # start
    (?P<x>\d+(?:.\d+)?)  # x number
    ,\s+                 # , separator, w/ whitespace
    (?P<y>\d+(?:.\d+)?)  # y number
    ,\s+                 # , separator, w/ whitespace
    (?P<z>\d+(?:.\d+)?)  # z number
    \]                   # end
'''
COORD_RE = re.compile(COORD_RE, re.VERBOSE | re.MULTILINE)


def extract_coords(coords):
    '''given str `coords`, return all the coordinates in the string as a list of triples

    Note:
        coordinates are encoded like (repeating this pattern):
           {""coord""=>[9466.0, 2711.0, 9811.0], ""density""=>0.0, ""intensity""=>0.0}
       in the AIBS return values
    '''
    ret = []
    for m in COORD_RE.finditer(coords):
        ret.append(tuple(map(float, m.groups())))
    return ret


def get_source_target_from_path(csv_path):
    '''extract source and target from saved CSV file

    Note: depends on how download_streamlines saves file
    '''
    csv_path = os.path.basename(csv_path)
    csv_path = csv_path.split('_')
    source, target = csv_path[0], csv_path[1]
    return source, target


def _correct_metadata_dtypes(metadata):
    # floats to 32bit
    for name, axis in it.product(('injection', 'start', 'end'), XYZ):
        name = name + '_' + axis
        metadata[name] = metadata[name].astype(np.float32)
    metadata['length'] = metadata['length'].astype(np.float32)

    # integers
    for name in ('region_id', 'path_row', ):
        metadata[name] = pd.to_numeric(metadata[name], downcast='signed')


def _convert_csv(csv_path):
    '''extrace metadata and path coordinates from from `csv_path`'''
    # pylint doesn't handle pd.read_csv right
    # pylint: disable=unsubscriptable-object
    df = pd.read_csv(csv_path)
    metadata = pd.DataFrame(index=df.id.astype(int))
    metadata.index.name = 'id'

    metadata['region_id'] = df['structure-id'].to_numpy(dtype=int)
    metadata['region_acronym'] = df['structure-abbrev'].to_numpy()
    x, y, z = zip(*extract_coords(' '.join(df['injection-coordinates'])))

    metadata['injection_x'] = x
    metadata['injection_y'] = y
    metadata['injection_z'] = z

    paths = [extract_coords(p) for p in df['path']]

    metadata['length'] = [np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
                          for p in paths]
    metadata['start_x'] = [path[0][X] for path in paths]
    metadata['start_y'] = [path[0][Y] for path in paths]
    metadata['start_z'] = [path[0][Z] for path in paths]

    metadata['end_x'] = [path[-1][X] for path in paths]
    metadata['end_y'] = [path[-1][Y] for path in paths]
    metadata['end_z'] = [path[-1][Z] for path in paths]

    source, target = get_source_target_from_path(csv_path)

    metadata['source'] = source
    metadata['target'] = target

    metadata = metadata.reset_index(drop=True)

    return metadata, paths


def _assign_side_and_hemisphere(metadata, center_line_3d):
    sides = (metadata.end_z > center_line_3d).astype(int)
    metadata['target_side'] = pd.Categorical.from_codes(sides, dtype=utils.SIDE)

    contras = (((metadata.start_z < center_line_3d) & (metadata.end_z > center_line_3d)) |
               ((metadata.start_z > center_line_3d) & (metadata.end_z < center_line_3d))
               ).astype(int)
    metadata['hemisphere'] = pd.Categorical.from_codes(contras, dtype=utils.HEMISPHERE)


def _create_missing(metadata, connected_centroids, center_line_3d):
    '''add centroids direct connections for centroids in connected_centroids but not in metadata'''
    centroids = connected_centroids.rename({'target_region': 'target', 'source_region': 'source'},
                                           axis='columns')
    centroids.rename({'source_' + ax: 'start_' + ax for ax in list('xyz')}, axis='columns',
                     inplace=True)
    centroids.rename({'target_' + ax: 'end_' + ax for ax in list('xyz')}, axis='columns',
                     inplace=True)
    centroids['length'] = np.linalg.norm(centroids[['start_x', 'start_y', 'start_z']].values -
                                         centroids[['end_x', 'end_y', 'end_z']].values,
                                         axis=1)
    centroids['path_row'] = -1
    del centroids['source_side']

    _assign_side_and_hemisphere(centroids, center_line_3d)

    # remove any that already exist; want to use streamlines when they are available
    keys = ['source', 'target', 'target_side', 'hemisphere']
    metadata = metadata[keys].drop_duplicates()
    keep = centroids.merge(metadata, how='left', on=keys, indicator='merged')
    keep = (keep['merged'].values == 'left_only')

    return centroids[keep]


def convert_csvs(csv_paths, connected_centroids, center_line_3d, create_mirrors=True):
    '''convert and fill in missing streamlines'''
    metadata, streamlines = [], []
    for csv_path in csv_paths:
        if csv_path.endswith('missing_streamlines.csv'):
            continue

        metadata_, paths_ = _convert_csv(csv_path)

        metadata.append(metadata_)
        streamlines.extend(paths_)

        if create_mirrors:
            metadata_, paths_ = _mirror_streamlines(metadata_, paths_, center_line_3d)
            metadata.append(metadata_)
            streamlines.extend(paths_)

    metadata = pd.concat(metadata, ignore_index=True, sort=False)
    metadata['path_row'] = np.arange(len(streamlines), dtype=int)
    _assign_side_and_hemisphere(metadata, center_line_3d)

    missing = _create_missing(metadata, connected_centroids, center_line_3d)
    metadata = pd.concat((metadata, missing), ignore_index=True, sort=False)

    # remove ipsi connections w/ same source and target - those are local
    # connectivity, not white-matter
    mask = (metadata.source == metadata.target) & (metadata.hemisphere == 'ipsi')
    metadata = metadata[np.invert(mask)]

    _correct_metadata_dtypes(metadata)

    return metadata, streamlines


def _mirror_streamlines(metadata, streamlines, center_line_3d):
    '''mirror all the streamlines over the center_line_3d

    Most of the injections are done only in one hemisphere, so the data
    is quite sparse for the opposite hemisphere
    '''
    metadata = metadata.copy()

    ret = []
    for row, stream in enumerate(streamlines):
        stream = [[x, y, 2 * center_line_3d - z] for x, y, z in stream]
        ret.append(stream)

        metadata.loc[row, 'start_z'] = stream[0][Z]
        metadata.loc[row, 'end_z'] = stream[-1][Z]

    return metadata, ret


def load(path, only_metadata=False):
    '''load_streamlines'''
    streamlines_path = os.path.join(path, STREAMLINES_NAME + '.csv')
    streamlines_rows = os.path.join(path, STREAMLINES_NAME + '.rows')

    if not os.path.exists(streamlines_path):
        raise Exception('Missing streamlines: %s' % streamlines_path)

    if not os.path.exists(streamlines_rows):
        raise Exception('Missing streamlines: %s' % streamlines_rows)

    metadata = pd.read_csv(streamlines_path,
                           dtype={'target_side': utils.SIDE,
                                  'hemisphere': utils.HEMISPHERE,
                                  })
    _correct_metadata_dtypes(metadata)

    if only_metadata:
        return metadata

    streamlines = _read_streamlines_rows(streamlines_rows)

    return metadata, streamlines


def save(output, metadata, streamlines):
    '''save_streamlines'''
    metadata.to_csv(os.path.join(output, STREAMLINES_NAME + '.csv'), index=False)
    path = os.path.join(output, STREAMLINES_NAME + '.rows')
    _write_streamlines_rows(path, streamlines)


def _read_streamlines_rows(path):
    '''read serialized streamlines'''
    streamlines = []
    with open(path) as fd:
        for line in fd:
            line = line.split()
            assert int(line[0]) == (len(line) - 1) / 3, 'incorrect point count'
            xyz = map(float, line[1:])  # pylint: disable=no-member
            streamlines.append(zip(xyz, xyz, xyz))
    return streamlines


def _write_streamlines_rows(path, streamlines):
    '''write the streamlines, one per line

    in the form:
    n_points x0 y0 z0 x1 y1 z1 ...

    as per:

    https://bbpteam.epfl.ch/project/issues/browse/NCX-54?focusedCommentId=81835&page=com.atlassian.jira.plugin.system.issuetabpanels:comment-tabpanel#comment-81835
    '''
    with open(path, 'w') as fd:
        for stream in streamlines:
            points = ' '.join(map(str, it.chain.from_iterable(stream)))
            fd.write('{count} {points}\n'.format(count=len(stream),
                                                 points=points))


def write_mapping(output_path, sgid2row):
    '''write the streamline mapping to the output directory, for the VIZ team'''
    sgid2row = sgid2row[sgid2row.row >= 0].copy()
    sgid2row['row'] = sgid2row.row.astype(int)
    sgid2row['sgid'] = sgid2row.sgid.astype(int)
    sgid2row.sort_values(list(sgid2row.columns), inplace=True)
    path = os.path.join(output_path, 'sgid2path_row.mapping')
    sgid2row.to_csv(path, sep=' ', index=False, header=False)
