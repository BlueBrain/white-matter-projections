'''utilities for displaying pertinent information'''
import logging
import numpy as np


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


def create_module_color_mapping(module_grouping, module_grouping_color, colors, hier):
    '''assign each region id a color based on the module it is in

    Args:
        module_grouping(list of list): [[module, [region...], ]
        module_grouping_color(dict): module -> color_name
        colors: color_name -> tuple(R, G, B)
        hier(Hierarchy):

    Return:
        dict(region id -> color)
    '''
    id2color = {}
    for module, regions in module_grouping:
        color = colors[module_grouping_color[module]]
        for region in regions:
            id_ = hier.find('acronym', region)
            if len(id_) > 1:
                L.warning('Too many (%d) regions for %s', len(id_), region)
                continue
            elif len(id_) == 0:
                L.warning('Not enough regions for %s', region)
                continue
            id_ = id_[0].data['id']
            id2color[id_] = color

    return id2color


def draw_region_outlines(ax, hier, flat_id, regions, labels='all', only_right=False):
    '''draws the outlines of the specified regions

    Args:
        ax: matplotlib axis to which the outlines will be drawn
        hier: voxcell.hierarchy
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

    region2ids = {region: list(hier.collect('acronym', region, 'id'))
                  for region in regions}

    midline = flat_id.shape[1] // 2
    if only_right:
        flat_id = flat_id[:, midline:]
        midline = 0
    else:
        midline = flat_id.shape[1] // 2

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


def draw_module_flat_map(ax, id2color, flat_map, regions):
    '''draw flat map to `ax` colored by module, with outlines

    Args:
        ax: matplotlib axis to which the outlines will be drawn
        id2color: dict of region id to color
        flat_map: flat_mapping.FlatMap object
        regions: iterable of regions to be drawn
    '''
    flat_region_id = flat_map.make_flat_id_region_map(regions)

    # assign RGB colors to equivalently shaped flat region
    flat_id = np.zeros(flat_region_id.shape + (3, ), dtype=int)
    for id_, color in id2color.items():
        flat_id[np.nonzero(flat_id == id_)] = color

    midline = flat_id.shape[1] // 2
    ax.imshow(flat_id[:, midline:])

    draw_region_outlines(ax, flat_map.hierarchy, flat_region_id, regions, only_right=True)

    return midline


def plot_source_region_triangles(ax, config):
    '''draw all source barycentric triangles on colored flat map

    Args:
        ax: matplotlib axis to which the outlines will be drawn
        config: utils.Config object
    '''
    flat_map = config.flat_map
    c = config.config
    id2color = create_module_color_mapping(c['module_grouping'],
                                           c['module_grouping_color'],
                                           c['colors'],
                                           flat_map.hierarchy)
    midline = draw_module_flat_map(ax, id2color, flat_map, config.regions)
    v = (0, 1, 2, 0)
    seen = set()
    for region_name, values in config.recipe.projections_mapping.items():
        region_name = region_name.split('_')[0]
        if region_name in seen:
            continue
        seen.add(region_name)
        verts = values['vertices']
        ax.plot(verts[v, 1] - midline, verts[v, 0], c='white')
