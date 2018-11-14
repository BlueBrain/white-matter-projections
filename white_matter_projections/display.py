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
