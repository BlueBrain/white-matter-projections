'''utilities for displaying pertinent information'''
import itertools as it
import logging
import numpy as np
import pandas as pd

from white_matter_projections import utils, mapping, micro


L = logging.getLogger(__name__)


def draw_connectivity(fig, df, title, module_grouping_color):
    '''create figure of connectivity densities

    Args:
        fig: matplotlib figure
        df(dataframe): ipsilateral densities

    '''
    # pylint: disable=too-many-locals
    import seaborn as sns
    fig.suptitle(title)
    ax = fig.add_subplot(1, 1, 1)

    cmap = sns.light_palette('blue', as_cmap=True)
    cmap.set_bad(color='white')
    df = df.replace(0., np.NaN)
    ax = sns.heatmap(
        df, ax=ax, cmap=cmap,
        square=True, xticklabels=1, yticklabels=1, linewidth=0.5
    )
    ax.set_xlabel('Target')
    ax.set_ylabel('Source')
    ax.set_title('syns/um^3')

    xtl = ax.set_xticklabels([v[1] for v in df.index.values])
    ytl = ax.set_yticklabels([v[1] for v in df.columns.values],
                             rotation='horizontal')

    x_colors = [module_grouping_color[k] for k in df.columns.get_level_values(0)]
    y_colors = [module_grouping_color[k] for k in df.index.get_level_values(0)]

    for x, c in zip(xtl, x_colors):
        x.set_backgroundcolor(c)

    for y, c in zip(ytl, y_colors):
        y.set_backgroundcolor(c)


def create_module_color_mapping(module_grouping, module_grouping_color, colors, region_map):
    '''assign each region id a color based on the module it is in

    Args:
        module_grouping(list of list): [[module, [region...], ]
        module_grouping_color(dict): module -> color_name
        colors: color_name -> tuple(R, G, B)
        region_map(voxcell.region_map)

    Return:
        dict(region id -> color)
    '''
    id2color = {}
    for module, regions in module_grouping:
        color = colors[module_grouping_color[module]]
        for region in regions:
            id2color[next(iter(region_map.find(region, 'acronym')))] = color

    return id2color


def plot_allen_coloured_flat_map(ax, config, regions='all', only_right=False):
    '''Plot flatmap using colors from AIBS'''
    id2color = create_module_color_mapping(config.config['module_grouping'],
                                           config.config['module_grouping_color'],
                                           config.config['colors'],
                                           config.flat_map.region_map)
    if regions == 'all':
        regions = config.regions

    return draw_module_flat_map(
        ax, id2color, config.flat_map, regions=regions, only_right=only_right)


def plot_flat_cells(ax, cells, gids, mapper, color='black', alpha=0.025):
    '''plot the source cell locations in the flat space

    Args:
        ax: axis to plot to
        cells(DataFrame): cell dataframe, must have x/y/z columns
        gids(array of ints): gids to be plotted
        mapper: map from voxels to flata
        color: matplotlib color
        alpha: matplotlib alpha
    '''

    gids = np.unique(gids)
    gids = pd.DataFrame(index=gids).sort_index()
    cells = cells[utils.XYZ]
    xyz = cells.join(gids, how='right').values
    plot_xyz_to_flat(ax, mapper, xyz, color=color, alpha=alpha)


def plot_xyz_to_flat(ax, mapper, xyz, color='black', alpha=0.05):
    '''Map `xyz` positions to flat space, using `mapper`

    Args:
        ax: axis to plot to
        mapper: map from voxels to flata
        xyz(array Nx3): positions
        color: matplotlib color
        alpha: matplotlib alpha
    '''
    uvs = mapper.map_points_to_flat(xyz)
    uvs = uvs[uvs[:, utils.Y] > 0.]

    # Note: Y/X are reversed to keep the same perspective as plt.imshow
    ax.scatter(uvs[:, utils.Y], uvs[:, utils.X], marker='.', s=1, alpha=alpha, color=color)
    return uvs


def draw_region_outlines(ax, region_map, flat_id, regions, midline, labels='all', only_right=False):
    '''draws the outlines of the specified regions

    Args:
        ax: matplotlib axis to which the outlines will be drawn
        region_map: voxcell.region_map
        flat_id(np.array): 2D array with the all ids to be potentially outlined
        regions(list of region names or 'all'): which regions are outlined
        labels(list of region names or 'all'): which regions have their names drawn
        only_right('bool'): only draw the right hemisphere
    '''
    if labels == 'all':
        labels = set(regions)
    elif labels is None:
        labels = set()
    else:
        labels = set(labels)

    region2ids = {region: list(region_map.find(region, 'acronym', with_descendants=True))
                  for region in regions}

    if only_right:
        flat_id = flat_id[:, midline:]
        midline = 0

    for region, ids in region2ids.items():
        mask = np.isin(flat_id, list(ids))
        ax.contour(mask, levels=[0.5], colors=['black'], linewidths=2, alpha=0.5)

        if region in labels:
            idx = np.array(np.nonzero(mask)).T
            idx = idx[midline < idx[:, 1]]
            x, y = np.mean(idx, axis=0)

            ax.text(y, x, region,  # note inverse index
                    horizontalalignment='center', verticalalignment='center',
                    color='white')


def draw_module_flat_map(ax, id2color, flat_map, regions, only_right=False):
    '''draw flat map to `ax` colored by module, with outlines

    Args:
        ax: matplotlib axis to which the outlines will be drawn
        id2color: dict of region id to color
        flat_map: flat_mapping.FlatMap object
        regions: iterable of regions to be drawn
        only_right(bool): if true, only the right side of the flatmap is drawn
    '''
    flat_region_id = flat_map.make_flat_id_region_map(regions)

    # assign RGB colors to equivalently shaped flat region
    flat_id = np.zeros(flat_region_id.shape + (3, ), dtype=int)
    for id_, color in id2color.items():
        idx = np.nonzero(flat_region_id == id_)
        flat_id[idx] = color

    midline = 0
    if only_right:
        midline = flat_map.center_line_2d

    ax.imshow(flat_id[:, midline:])

    draw_region_outlines(ax, flat_map.region_map, flat_region_id, regions,
                         flat_map.center_line_2d, only_right=only_right)

    return midline


def draw_triangle(ax, vertices, color='white'):
    '''draw `vertices` to `ax`'''
    v = (0, 1, 2, 0)
    ax.plot(vertices[v, 1], vertices[v, 0], c=color)
    ax.scatter(vertices[v[:3], 1], vertices[v[:3], 0], c=('red', 'green', 'blue'), s=40)
    for s in v[:-1]:
        ax.text(x=vertices[s, 1], y=vertices[s, 0], s=str(s))


def plot_source_region_triangles(ax, config, regions='all', only_right=False):
    '''draw all source barycentric triangles

    Args:
        ax: matplotlib axis to which the outlines will be drawn
        config: utils.Config object
        regions: iterable of regions to be drawn, 'all' if all in config
        only_right(bool): if true, only the right side of the flatmap is drawn
    '''
    midline = 0
    if only_right:
        midline = config.flat_map.center_line_2d

    if regions == 'all':
        wanted_regions = set(config.regions)
    else:
        wanted_regions = set(regions)

    for region_name, values in config.recipe.projections_mapping.items():
        region_name = region_name.split('_')[0]
        if region_name not in wanted_regions:
            continue
        vertices = values['vertices'] - (0, midline)
        draw_triangle(ax, vertices, color='white')


def draw_triangle_map(ax, config, projection_name, side):
    '''for `projection_name`, plot the target synapse locations'''
    # pylint: disable=too-many-locals
    mapper = mapping.CommonMapper.load_default(config)
    projection = config.recipe.get_projection(projection_name)

    source_populations = (config.recipe.populations.set_index('population')
                          .loc[projection.source_population])

    mirror = utils.is_mirror(side, projection.hemisphere)

    brain_regions = config.flat_map.brain_regions

    colors = it.cycle((('red', 'green',), ('blue', 'yellow',)))

    for (src_color, dst_color), src_population in zip(colors, source_populations.itertuples()):
        xyz = np.nonzero(brain_regions.raw == src_population.id)
        xyz = brain_regions.indices_to_positions(np.array(xyz).T)
        uvs = plot_xyz_to_flat(ax, mapper, xyz, color=src_color, alpha=1)

        uvs = mapper.map_flat_to_flat(projection.source_population, projection_name, uvs, mirror)
        uvs = uvs[uvs[:, utils.Y] > 0.]
        # Note: Y/X are reversed to keep the same perspective as plt.imshow
        ax.scatter(uvs[:, utils.Y], uvs[:, utils.X], marker='.', s=2, alpha=1, color=dst_color)

    # draw source vertices
    projections_mapping = config.recipe.projections_mapping[projection.source_population]
    src_verts = projections_mapping['vertices']
    tgt_verts = projections_mapping[projection_name]['vertices']

    if mirror:
        src_verts = utils.mirror_vertices_y(src_verts, config.flat_map.center_line_2d)
        tgt_verts = utils.mirror_vertices_y(tgt_verts, config.flat_map.center_line_2d)

    draw_triangle(ax, src_verts, color='green')
    draw_triangle(ax, tgt_verts, color='yellow')


def draw_projection(ax, config, allocations, syns, projection_name, side):
    '''for `projection_name`, plot the target synapse locations'''
    # pylint: disable=too-many-locals
    mapper = mapping.CommonMapper.load_default(config)
    hemisphere = config.recipe.get_projection(projection_name).hemisphere

    left_cells, right_cells = micro.partition_cells_left_right(config.get_cells(),
                                                               config.flat_map.center_line_3d)

    source_population, all_sgids = allocations.loc[projection_name][['source_population', 'sgids']]
    all_sgids = np.unique(all_sgids)
    used_sgids = syns.sgid.unique()
    mirror = utils.is_mirror(side, hemisphere)

    L.info('%d unique sgids, %d used unique sgids', len(all_sgids), len(used_sgids))

    for sgids, src_color, dst_color, alpha in ((all_sgids, 'white', 'red', 1.),
                                               (used_sgids, 'green', 'blue', 1.),
                                               ):
        src_cell_positions = micro.separate_source_and_targets(
            left_cells, right_cells, sgids, hemisphere, side)

        xyz = src_cell_positions[utils.XYZ].values
        uvs = plot_xyz_to_flat(ax, mapper, xyz, color=src_color, alpha=alpha)

        uvs = mapper.map_flat_to_flat(source_population, projection_name, uvs, mirror)
        uvs = uvs[uvs[:, utils.Y] > 0.]
        # Note: Y/X are reversed to keep the same perspective as plt.imshow
        ax.scatter(uvs[:, utils.Y], uvs[:, utils.X], marker='.', s=2, alpha=alpha, color=dst_color)

    # draw source vertices
    projections_mapping = config.recipe.projections_mapping[source_population]
    src_verts = projections_mapping['vertices']
    tgt_verts = projections_mapping[projection_name]['vertices']
    if mirror:
        src_verts = utils.mirror_vertices_y(src_verts, config.flat_map.center_line_2d)
        tgt_verts = utils.mirror_vertices_y(tgt_verts, config.flat_map.center_line_2d)
    draw_triangle(ax, src_verts, color='green')
    draw_triangle(ax, tgt_verts, color='yellow')


def draw_voxel_to_flat_mapping(ax, config, regions, mapper, alpha=0.5, color='limegreen'):
    '''Helper to draw the voxel to mapping, for debug purposes

    >>> for region in config.regions:
    >>>     fig, ax = get_fig()
    >>>     display.draw_voxel_to_flat_mapping(ax, config, regions, mapper)
    >>>     ax.set_title(region)
    >>>     fig.savefig(os.path.join(output, region + '.png'))
    '''
    plot_allen_coloured_flat_map(ax, config, regions='all', only_right=False)

    for region in regions:
        ids = config.flat_map.region_map.find(region, 'acronym', 'id', with_descendants=True)
        mask = np.isin(config.flat_map.brain_regions.raw, list(ids))
        idx = np.array(np.nonzero(mask)).T
        uvs, _ = mapper.voxel_to_flat(idx, np.array([[0, 0, 0]]))

        ax.scatter(uvs[:, utils.Y], uvs[:, utils.X], marker='.', s=5, alpha=alpha, color=color)
