'''utilities for displaying pertinent information'''
import itertools as it
import logging
import numpy as np

from lazy import lazy

from white_matter_projections import flat_mapping, mapping, micro, utils


L = logging.getLogger(__name__)
X = utils.X
Y = utils.Y


class FlatmapPainter:
    '''work with multiple flatmaps to paint onto a figure various debugging/visualization'''
    # Note: imshow draws 'funny': for an array of shape (M, N), the first
    # index runs along the vertical, the second index runs along the
    # horizontal.
    def __init__(self, config, fig):
        self.config = config
        self.fig = fig
        self.flatmaps = dict()
        self.axs = dict()

        base_systems = config.config['flat_mapping']
        for base_system, ax in zip(base_systems,
                                   fig.subplots(nrows=len(base_systems), ncols=1, sharex=True)):
            self.flatmaps[base_system] = config.flat_map(base_system)
            L.debug('Loading %s[%s]', base_system, self.flatmaps[base_system].shape)

            ax.set_aspect('equal')
            ax.set_title(base_system)
            self.axs[base_system] = ax

    @lazy
    def _id_to_rgb_color(self):
        '''based on the config, create a mapping of ids to colours

        colours are based on the module the region belongs to
        '''
        return create_module_color_mapping(self.config.config['module_grouping'],
                                           self.config.config['module_grouping_color'],
                                           self.config.config['colors'],
                                           self.config.region_map)

    def get_axis(self, *base_systems):
        '''get the axis used to display `base_system`'''
        return [self.axs[bs] for bs in base_systems]

    def draw_flat_map_in_colour(self, regions='all'):
        '''draw flat map to `ax` colored by module, with outlines

        Args:
            regions: iterable of regions to be drawn
        '''
        if regions == 'all':
            regions = self.config.regions

        for base_system, flat_map in self.flatmaps.items():
            flat_region_id = flat_map.make_flat_id_region_map(regions)

            flat_id = np.zeros(shape=(flat_region_id.shape + (3, )), dtype=int)
            for id_, color in self._id_to_rgb_color.items():
                idx = np.nonzero(flat_region_id == id_)
                flat_id[idx] = color

            ax = self.axs[base_system]
            ax.imshow(flat_id)

            _draw_region_outlines(ax,
                                  flat_map.region_map,
                                  flat_region_id,
                                  regions)

    def plot_source_region_triangles(self, regions='all'):
        '''draw all source barycentric triangles

        Args:
            regions: iterable of regions to be drawn
        '''
        if regions == 'all':
            wanted_regions = set(r.split('_')[0]
                                 for r in self.config.recipe.projections_mapping)
        else:
            wanted_regions = set(regions)

        for region_name, values in self.config.recipe.projections_mapping.items():
            region_name = region_name.split('_')[0]
            if region_name not in wanted_regions:
                continue

            draw_triangle(self.axs[values['base_system']],
                          vertices=values['vertices'],
                          color='white')

    def draw_projection(self, allocations, syns, projection_name, side):
        '''for `projection_name`, plot the target synapse locations'''
        # pylint: disable=too-many-locals
        hemisphere = self.config.recipe.get_projection(projection_name).hemisphere
        mirror = utils.is_mirror(side, hemisphere)

        source_population, all_sgids = \
            allocations.loc[projection_name][['source_population', 'sgids']]

        src_base_system = mapping.base_system_from_projection(self.config, source_population)
        tgt_base_system = mapping.base_system_from_projection(self.config,
                                                              source_population,
                                                              projection_name)
        left_cells, right_cells = micro.partition_cells_left_right(
            self.config.get_cells(),
            self.config.flat_map(src_base_system).center_line_3d)

        src_mapper = mapping.CommonMapper.load(self.config, src_base_system)

        src_ax, tgt_ax = self.get_axis(src_base_system, tgt_base_system)

        all_sgids = np.unique(all_sgids)
        used_sgids = syns.sgid.unique()

        L.info('%d unique sgids, %d used unique sgids', len(all_sgids), len(used_sgids))

        for sgids, src_color, dst_color, alpha in ((all_sgids, 'white', 'red', 1.),
                                                   (used_sgids, 'green', 'blue', 1.),
                                                   ):
            src_cell_positions = micro.separate_source_and_targets(
                left_cells, right_cells, sgids, hemisphere, side)

            uvs = plot_xyz_to_flat(src_ax,
                                   src_mapper,
                                   src_cell_positions[utils.XYZ].to_numpy(),
                                   color=src_color,
                                   alpha=alpha)

            uvs = src_mapper.map_flat_to_flat(source_population, projection_name, uvs, mirror)
            uvs = uvs[flat_mapping.FlatMap.mask_in_2d_flatmap(uvs), :]

            # Note: Y/X are reversed to keep the same perspective as plt.imshow
            tgt_ax.scatter(
                uvs[:, utils.Y], uvs[:, utils.X], marker='.', s=2, alpha=alpha, color=dst_color)

        self.draw_mapping_triangles(mirror, source_population, projection_name)

    def draw_mapping_triangles(self, mirror, source_population, projection_name):
        '''draw the source and target mapping triangles'''
        src_base_system = mapping.base_system_from_projection(self.config, source_population)
        tgt_base_system = mapping.base_system_from_projection(self.config,
                                                              source_population,
                                                              projection_name)

        src_ax, tgt_ax = self.get_axis(src_base_system, tgt_base_system)

        projections_mapping = self.config.recipe.projections_mapping[source_population]
        src_verts = projections_mapping['vertices']
        tgt_verts = projections_mapping[projection_name]['vertices']
        if mirror:
            src_flatmap = self.config.flat_map(src_base_system)
            tgt_flatmap = self.config.flat_map(tgt_base_system)
            src_verts = utils.mirror_vertices_y(src_verts, src_flatmap.center_line_2d)
            tgt_verts = utils.mirror_vertices_y(tgt_verts, tgt_flatmap.center_line_2d)

        draw_triangle(src_ax, src_verts, color='green')
        draw_triangle(tgt_ax, tgt_verts, color='yellow')

    def draw_triangle_map(self, projection_name, side):
        '''for `projection_name`, plot the target synapse locations'''
        # pylint: disable=too-many-locals
        projection = self.config.recipe.get_projection(projection_name)
        mirror = utils.is_mirror(side, projection.hemisphere)

        # XXX try w/ one that has multiple source pops!
        colors = it.cycle((('red', 'green',), ('blue', 'yellow',)))
        source_populations = self.config.recipe.get_population(projection.source_population)
        for (src_color, dst_color), source_population in zip(colors,
                                                             source_populations.itertuples()):
            src_id, source_population = source_population.id, source_population.Index
            src_base_system = mapping.base_system_from_projection(self.config, source_population)
            src_mapper = mapping.CommonMapper.load(self.config, src_base_system)
            src_ax, tgt_ax = self.get_axis(
                mapping.base_system_from_projection(self.config,
                                                    source_population),
                mapping.base_system_from_projection(self.config,
                                                    source_population,
                                                    projection_name))

            xyz = np.nonzero(src_mapper.flat_map.brain_regions.raw == src_id)
            xyz = src_mapper.flat_map.brain_regions.indices_to_positions(np.array(xyz).T)
            uvs = plot_xyz_to_flat(src_ax, src_mapper, xyz, color=src_color, alpha=1)

            uvs = src_mapper.map_flat_to_flat(
                projection.source_population, projection_name, uvs, mirror)
            uvs = uvs[flat_mapping.FlatMap.mask_in_2d_flatmap(uvs), :]

            # Note: Y/X are reversed to keep the same perspective as plt.imshow
            tgt_ax.scatter(
                uvs[:, utils.Y], uvs[:, utils.X], marker='.', s=2, alpha=1, color=dst_color)

            # XXX: needed? if the multiple source pops exist???
            self.draw_mapping_triangles(mirror, source_population, projection_name)

    def plot_compensation(self, projection_name, side, size=10):
        '''Display the result of using the 'compensation' method'''
        from white_matter_projections import sampling

        src_uvs, src_uvs_mapped, tgt_uvs, wi_cutoff = sampling.calculate_compensation(
            self.config, projection_name, side)

        source_population, hemisphere = \
            self.config.recipe.get_projection(projection_name)[['source_population', 'hemisphere']]

        src_ax, tgt_ax = self.get_axis(mapping.base_system_from_projection(self.config,
                                                                           source_population),
                                       mapping.base_system_from_projection(self.config,
                                                                           source_population,
                                                                           projection_name))

        # src points
        src_ax.scatter(src_uvs[:, utils.Y], src_uvs[:, utils.X],
                       marker='.', s=size, alpha=0.5, color='grey')

        # all
        tgt_ax.scatter(src_uvs_mapped[:, utils.Y], src_uvs_mapped[:, utils.X],
                       marker='.', s=size, alpha=1., color='yellow')

        # used
        tgt_ax.scatter(tgt_uvs[wi_cutoff, utils.Y], tgt_uvs[wi_cutoff, utils.X],
                       marker='.', s=size, alpha=1., color='green')

        self.draw_mapping_triangles(
            utils.is_mirror(side, hemisphere), source_population, projection_name)

        comp = np.count_nonzero(wi_cutoff) / float(len(wi_cutoff) + 1)
        self.fig.suptitle('Projection: %s, %.4f times compensation' %
                          (projection_name, 1. / comp))

    def draw_voxel_to_flat_mapping(self, regions, base_system, alpha=0.5, color='limegreen'):
        '''Helper to draw the voxel to mapping, for debug purposes'''
        mapper = mapping.CommonMapper.load(self.config, base_system)
        ax = self.get_axis(base_system)

        self.draw_flat_map_in_colour(regions='all')

        for region in regions:
            ids = mapper.flat_map.region_map.find(region, 'acronym', 'id', with_descendants=True)
            mask = np.isin(mapper.flat_map.brain_regions.raw, list(ids))
            idx = np.array(np.nonzero(mask)).T
            uvs, _ = mapper.voxel_to_flat(idx, np.array([[0, 0, 0]]))

            ax.scatter(uvs[:, utils.Y], uvs[:, utils.X], marker='.', s=5, alpha=alpha, color=color)


def draw_connectivity(fig, df, title, module_grouping_color):
    '''create figure of connectivity densities

    Args:
        fig: matplotlib figure
        df(dataframe): ipsilateral densities
        title(str): figure title
        module_grouping_color(dict): module -> color_name

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
    uvs = uvs[flat_mapping.FlatMap.mask_in_2d_flatmap(uvs), :]

    # Note: Y/X are reversed to keep the same perspective as plt.imshow
    ax.scatter(uvs[:, utils.Y], uvs[:, utils.X], marker='.', s=1, alpha=alpha, color=color)

    return uvs


def _draw_region_outlines(ax, region_map, flat_id, regions, labels='all'):
    '''draws the outlines of the specified regions

    Args:
        ax: matplotlib axis to which the outlines will be drawn
        region_map: voxcell.region_map
        flat_id(np.array): 2D array with the all ids to be potentially outlined
        regions(list of region names or 'all'): which regions are outlined
        labels(list of region names or 'all'): which regions have their names drawn
    '''
    if labels == 'all':
        labels = set(regions)
    elif labels is None:
        labels = set()
    else:
        labels = set(labels)

    region2ids = {region: list(region_map.find(region, 'acronym', with_descendants=True))
                  for region in regions}

    for region, ids in region2ids.items():
        mask = np.isin(flat_id, list(ids))
        ax.contour(mask, levels=[0.5], colors=['black'], linewidths=2, alpha=0.5)

        if region in labels:
            idx = np.array(np.nonzero(mask)).T
            x, y = np.mean(idx, axis=0)

            ax.text(y, x, region,  # note inverse index
                    horizontalalignment='center', verticalalignment='center',
                    color='white')


def draw_triangle(ax, vertices, color='white'):
    '''draw `vertices` to `ax`'''
    v = (0, 1, 2, 0)
    ax.plot(vertices[v, 1], vertices[v, 0], c=color)
    ax.scatter(vertices[v[:3], 1], vertices[v[:3], 0], c=('red', 'green', 'blue'), s=40)
    for s in v[:-1]:
        ax.text(x=vertices[s, 1], y=vertices[s, 0], s=str(s))
