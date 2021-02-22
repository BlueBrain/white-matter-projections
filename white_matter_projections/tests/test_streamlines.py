import numpy as np
import os
import pandas as pd
import utils
import white_matter_projections.streamlines as sl

from mock import patch, Mock
from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from white_matter_projections import macro

RECIPE = macro.MacroConnections.load_recipe(utils.RECIPE_TXT,
                                            utils.REGION_MAP,
                                            subregion_translation=utils.SUBREGION_TRANSLATION,
                                            region_subregion_format=utils.REGION_SUBREGION_FORMAT
                                            )
EMPTY_CONNECTED_CENTROIDS = pd.DataFrame(columns=['start_x', 'start_y', 'start_z',
                                                  'end_x', 'end_y', 'end_z',
                                                  'source_side', 'source_region', 'target_region'])


def test_get_region_centroid():
    flat_map = utils.fake_flat_map()
    region = 'two'
    ret = sl.get_region_centroid(flat_map, region, side='right')
    eq_(list(ret), [1, 2, 0])


def test_get_all_region_centroids():
    flat_map = utils.fake_flat_map()
    ret = sl.get_all_region_centroids(flat_map, regions=['two'], layer='')
    eq_(len(ret), 1)
    eq_(list(ret.loc['two', 'right']), [1, 2, 0])


def test_get_connected_regions():
    ret = sl.get_connected_regions(RECIPE)
    expected = pd.DataFrame([('FRP', 'FRP', 'ipsi'),    # SUB_POP4_L23 -> PROJ2a_ipsi
                             ('FRP', 'FRP', 'contra'),  # SUB_POP4_L23 -> PROJ1b_contra
                             ('FRP', 'MOs', 'ipsi'),    # SUB_POP4_L23 -> PROJ2a_ipsi
                             ('MOs', 'MOs', 'ipsi'),    # POP2_ALL_LAYERS -> PROJ1a_ipsi
                             ('ACAd', 'MOs', 'ipsi'),   # POP3_ALL_LAYERS -> PROJ1a_ipsi
                             ('ACAd', 'ACAd', 'ipsi'),  # POP3_ALL_LAYERS -> PROJ1c_ipsi
                             ('ACAd', 'FRP', 'contra'), # POP3_ALL_LAYERS -> PROJ1b_contra
                             ],
                            columns=['source_region', 'target_region', 'hemisphere'])
    expected['hemisphere'] = expected.hemisphere.astype(sl.utils.HEMISPHERE)
    columns = ['source_region', 'target_region', 'hemisphere']
    assert_frame_equal(ret.sort_values(columns).reset_index(drop=True),
                       expected.sort_values(columns).reset_index(drop=True))


def test_get_streamline_per_region_connection():
    connected_regions = sl.get_connected_regions(RECIPE)
    columns = ['source', 'target', 'start_z', 'end_z', 'path_row']
    metadata = pd.DataFrame([('FRP', 'FRP', 25, 125, 5),
                             ('FRP', 'FRP', 125, 25, 6)
                             ],
                            columns=columns)
    center_line_3d = 100
    ret, missing = sl.get_streamline_per_region_connection(connected_regions, metadata, center_line_3d)

    expected_missing = {('FRP', 'MOs'),
                        ('MOs', 'MOs'),
                        ('ACAd', 'MOs'),
                        ('ACAd', 'ACAd'),
                        ('ACAd', 'FRP')}
    eq_(set(missing), expected_missing)
    eq_(np.count_nonzero(ret.path_row == -1), 22)
    eq_(np.count_nonzero(ret.path_row > -1), 2)  # FRP to FRP, contraa


def test_download_streamline():
    with patch('white_matter_projections.streamlines.requests') as patch_req:
        src_region = 'src_region'
        src_region_id = 123
        tgt_region = 'tgt_region'
        seed = 1, 2, 3

        patch_req.get.return_value = mock_req = Mock()
        mock_req.text = ''
        ok_(not sl.download_streamline(src_region, src_region_id, tgt_region, seed))

        mock_req.status_code = 500
        ok_(not sl.download_streamline(src_region, src_region_id, tgt_region, seed))

        mock_req.text = 'some csv values'
        mock_req.status_code = 200
        eq_(mock_req.text,
            sl.download_streamline(src_region, src_region_id, tgt_region, seed))


def test_download_streamlines():
    columns = ['source_region', 'target_region',
               'source_x', 'source_y', 'source_z',
               'target_x', 'target_y', 'target_z', ]
    centroids = pd.DataFrame([('FRP', 'FRP', 0, 0, 0, 25, 125, 5),
                              ('FRP', 'MOs', 0, 0, 0, 125, 25, 6),
                              ],
                             columns=columns)

    with patch('white_matter_projections.streamlines.download_streamline') as mock_download:
        fake_contents = 'fake_contents'
        with utils.tempdir('test_download_streamlines') as tmp:
            mock_download.side_effect = [fake_contents, None]
            missing = sl.download_streamlines(centroids, utils.REGION_MAP, tmp, sleep_time=.1)

            eq_(mock_download.call_count, 2)
            eq_(tuple(missing.iloc[0]), ('FRP', 'MOs', 0, 0, 0, 125, 25, 6))

            path = os.path.join(tmp, '_'.join(('FRP', 'FRP', '25', '125', '5')) + '.csv')
            with open(path) as fd:
                eq_(fd.read(), fake_contents)


def test_extract_coords():
    coords = '{""coord""=>[2900.5, 3800, 7200.0], ""density""=>0.0, ""intensity""=>0.0}'
    ret = sl.extract_coords(coords)
    eq_(ret, [(2900.5, 3800.0, 7200.0)])

    ret = sl.extract_coords(coords + coords)
    eq_(ret, [(2900.5, 3800.0, 7200.0),
              (2900.5, 3800.0, 7200.0)
              ])


def test_get_source_target_from_path():
    csv_path = '/asdf/asdf_asdf/source_target_some_other_data.csv'
    source, target = sl.get_source_target_from_path(csv_path)
    eq_(source, 'source')
    eq_(target, 'target')


def test__assign_side_and_hemisphere():
    metadata = pd.DataFrame([(0, 10), (10, 0), (10, 10), (0, 0)],
        columns=['start_z', 'end_z'])
    sl._assign_side_and_hemisphere(metadata, center_line_3d=5)
    eq_(list(metadata.target_side), ['right', 'left', 'right', 'left'])
    eq_(list(metadata.hemisphere), ['contra', 'contra', 'ipsi', 'ipsi'])


def test__convert_csv():
    csv_path = os.path.join(utils.DATADIR, 'VISrl_VISpor_9500_2700_9800.csv')
    metadata, paths = sl._convert_csv(csv_path)
    eq_(len(metadata), 2)
    eq_(tuple(metadata.iloc[1][['start_x', 'start_y', 'start_z']].values), paths[1][0])
    eq_(tuple(metadata.iloc[1][['end_x', 'end_y', 'end_z']].values), paths[1][-1])

    eq_(len(paths), 2)
    eq_(len(paths[0]), 35)
    eq_(paths[0][:5], [(9500.0, 2700.0, 9800.0),
                       (9466.0, 2700.0, 9733.0),
                       (9422.0, 2700.0, 9677.0),
                       (9377.0, 2688.0, 9633.0),
                       (9333.0, 2666.0, 9600.0), ])

def test_convert_csvs():
    csv_path = os.path.join(utils.DATADIR, 'VISrl_VISpor_9500_2700_9800.csv')
    csv_paths = [csv_path, csv_path]
    metadata, streamlines = sl.convert_csvs(csv_paths, EMPTY_CONNECTED_CENTROIDS,
                                            center_line_3d=5700, create_mirrors=False)

    eq_(len(metadata), 4)
    eq_(len(streamlines), 4)
    eq_(len(streamlines[0]), 35)


def test__mirror_streamlines():
    csv_path = os.path.join(utils.DATADIR, 'VISrl_VISpor_9500_2700_9800.csv')
    metadata, streamlines = sl.convert_csvs([csv_path, ], EMPTY_CONNECTED_CENTROIDS,
                                            center_line_3d=5700, create_mirrors=False)
    metadata, streamlines = sl._mirror_streamlines(metadata, streamlines, center_line_3d=5700)

    eq_(len(streamlines), 2)
    eq_(len(streamlines[0]), 35)
    eq_(streamlines[0][:5], [[9500.0, 2700.0, 1600.0],
                             [9466.0, 2700.0, 1667.0],
                             [9422.0, 2700.0, 1723.0],
                             [9377.0, 2688.0, 1767.0],
                             [9333.0, 2666.0, 1800.0], ])
    eq_(list(metadata.iloc[1][['start_x', 'start_y', 'start_z']].values), streamlines[1][0])
    eq_(list(metadata.iloc[1][['end_x', 'end_y', 'end_z']].values), streamlines[1][-1])


def test_load_save():
    csv_path = os.path.join(utils.DATADIR, 'VISrl_VISpor_9500_2700_9800.csv')
    metadata, streamlines = sl.convert_csvs([csv_path, ], EMPTY_CONNECTED_CENTROIDS,
                                            center_line_3d=5700, create_mirrors=True)

    with utils.tempdir('test_load_save_streamlines') as tmp:
        sl.save(tmp, metadata, streamlines)
        new_metadata, new_streamlines = sl.load(tmp)
        assert_frame_equal(metadata, new_metadata)
        eq_(list(map(tuple, new_streamlines[0])), streamlines[0])


def test_write_output():
    columns = ['row', 'sgid']
    sgid2path_row = pd.DataFrame([[10, 100],
                                  [20, 200],
                                  [-1, 200],
                                  ],
                                 columns=columns)

    with utils.tempdir('test_write_output') as tmp:
        sl.write_mapping(tmp, sgid2path_row)

        path = os.path.join(tmp, 'sgid2path_row.mapping')
        ok_(os.path.exists(path))
        with open(path) as fd:
            lines = fd.readlines()
            eq_(lines[0].split(), ['10', '100'])
            eq_(lines[1].split(), ['20', '200'])


#def test_assign_streamlines():
#    sl.assign_streamlines(cells, streamline_per_region, target_region, feather_path, center_line_3d)
#
#def test_get_centroids():
#    flat_map = utils.fake_flat_map()
#    ret = sl.get_connected_centroids(flat_map, RECIPE)
#    pass
