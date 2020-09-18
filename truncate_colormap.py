import numpy as np
import matplotlib.colors as colors

# Todo: add to plot_utils


def truncate_colormap(colmap, minval=0.0, maxval=1.0, n=100):
    """ This function truncates a colormap

    Taken from: https://stackoverflow.com/questions/18926031/
    how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib?
    utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

    :param colmap:          colormap to be truncated
    :param minval:          minimum value
    :param maxval:          maximum value
    :param n:               number of values
    :return: cmap_trunc:    truncated color map
    """

    cmap_trunc = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=colmap.name, a=minval, b=maxval),
        colmap(np.linspace(minval, maxval, n)))

    return cmap_trunc
