"""plotting_help.py

Useful functions for matplolib plots

"""

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox

redorange = (.96, .59, .2)
dorange = (0.85, 0.54, 0.24)
orange = (.96, .59, .2)
purple = (0.5, 0.2, .6)
dpurple = (0.2, 0.2, .6)
nblue = (.11, .27, .49)
blue = (0.31, 0.47, 0.69)
bblue = (0.51, 0.67, 0.89)
green = (0.5, 0.75, 0.42)
dgreen = (0.3, 0.55, 0.22)
ddgreen = (0.1, 0.35, 0.02)
bgreen = (.34, .63, .56)
red = (0.82, 0.32, 0.22)
dred = (.52, .12, 0)

celltype_cmaps = {'EPG': plt.cm.Blues, 'PEN': plt.cm.Oranges, 'PFL1': plt.cm.Purples, 'PFL3': plt.cm.Purples,
                  'IBSPSP': plt.cm.Oranges, 'FR1': plt.cm.Greens}
celltype_color = {'EPG': blue, 'PEN': orange, 'PFL1': purple, 'PFL3': purple, 'IBSPSP': orange, 'FR1': purple}


def color_to_facecolor(input_color, alpha=0.2):
    facecolor = to_rgba(c=input_color)
    facecolor = list(facecolor)
    facecolor[-1] = alpha
    facecolor = list(facecolor)
    return facecolor


def adjust_spines(ax, spines, pad=None, lw=None, ticklen=None):
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    if 'left' in spines:
        ax.spines['left'].set_visible(True)
        ax.axes.get_yaxis().set_visible(True)
    if 'right' in spines:
        ax.spines['right'].set_visible(True)
        ax.axes.get_yaxis().set_visible(True)
    if 'bottom' in spines:
        ax.spines['bottom'].set_visible(True)
        ax.axes.get_xaxis().set_visible(True)
    if 'top' in spines:
        ax.spines['top'].set_visible(True)
        ax.axes.get_xaxis().set_visible(True)

    for loc, spine in ax.spines.items():
        if loc in spines:
            if pad is not None:
                spine.set_position(('outward', pad))
            if lw is not None:
                spine.set_linewidth(lw)

    if (ticklen is not None) & (lw is not None):
        ax.tick_params('both', length=ticklen, width=lw, which='major')
        ax.tick_params('both', length=ticklen / 2., width=lw, which='minor')


def despine_axes(axes, style=['bottom', 'left'], adjust_spines_kws=None):
    """
       removes non-border spines of seaborn facetgrid
       doesn't work if col_wrap is used

       axes: array of axes with shape of grid (e.g. FacetGrid.axes)
       adjust_spines_kws: dict of keyword args for adjust_spines


   """

    if adjust_spines_kws is None:
        adjust_spines_kws = {}
    shape = np.shape(axes)
    if len(shape) == 1:
        # assumes you want columns...
        axes = np.reshape(axes, (1, shape[0]))
        shape = np.shape(axes)
    for irow, row in enumerate(axes):
        for icol, ax in enumerate(row):
            spines = []
            if irow == shape[0] - 1:
                if 'bottom' in style:
                    spines.append('bottom')
            if irow == 0:
                if 'top' in style:
                    spines.append('top')
            if icol == 0:
                if 'left' in style:
                    spines.append('left')

            adjust_spines(ax, spines, **adjust_spines_kws)


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(x, y, z=None, ax=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), lw=1, **kwargs):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=lw, **kwargs)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


def circplot(x, y, circ='x', color='black', lw=1, ls='-',
             alpha=1, period=360, zorder=None, ax=None, args=None):
    if args is None:
        args = {}
    if circ == 'x':
        jumps = np.abs(np.diff(x)) > period / 2.
    elif circ == 'y':
        jumps = np.abs(np.diff(y)) > period / 2.

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segments = segments[jumps == False]
    lc = LineCollection(segments, colors=color, linewidth=lw, linestyles=ls, alpha=alpha, zorder=zorder, **args)
    if not ax:
        ax = plt.gca()
    ax.add_collection(lc)


class FacetGrid(object):
    def __init__(self, data, col=None, row=None, col_wrap=None, unit=None,
                 fig_kws=None, gridspec_kws=None, subplot_kws=None,
                 col_order=None, row_order=None, observed=True
                 # sharex=False,
                 # sharey=False
                 ):
        # col_wrap should work but have not tested extensively
        self.data = data
        self.col = col
        self.row = row
        self.unit = unit
        self.groupby_list = []
        self.col_wrap = col_wrap

        if col is None:
            self.ncol = 1
            self.col_names = ['None']
        else:
            self.col_names = data[col].dropna().unique()
            if col_order is not None:
                self.col_names = col_order
            self.ncol = len(self.col_names)

        self.col_names_to_num = dict(zip(self.col_names, np.arange(self.ncol)))

        if row is None:
            self.nrow = 1
            self.row_names = ['None']
        else:
            self.row_names = data[row].dropna().unique()
            if row_order is not None:
                self.row_names = row_order
        self.nrow = len(self.row_names)
        self.row_names_to_num = dict(zip(self.row_names, np.arange(self.nrow)))

        if col_wrap is not None:
            if row is not None:
                err = "Cannot use `row` and `col_wrap` together."
                raise ValueError(err)

            self.nrow = int(np.ceil(self.ncol / col_wrap))
            self.ncol = col_wrap
            # self.col_names_to_num = dict(zip(self.col_names,np.arange(self.ncol)))

            # self.col_names_to_num
            # self.row_names_to_num

        for i in [row, col, unit]:
            if i is not None:
                self.groupby_list.append(i)

        if gridspec_kws is None:
            gridspec_kws = {}

        if fig_kws is None:
            fig_kws = {}

        if subplot_kws is None:
            subplot_kws = {}
        self.figure = plt.figure(**fig_kws)
        self.gs = gridspec.GridSpec(figure=self.figure, nrows=self.nrow, ncols=self.ncol, **gridspec_kws)
        self.axes = np.empty((self.nrow, self.ncol), dtype=object)

        for i in range(self.nrow):
            for j in range(self.ncol):
                ax = plt.subplot(self.gs[i, j], **subplot_kws)
                self.axes[i, j] = ax

        # # not working atm...
        # # should be able to do this in creation of subplots, but was having difficutly..
        # if sharex is True:
        #     self.axes.ravel()[0].get_shared_x_axes().join(*self.axes.ravel()[1:])
        # elif sharex=='row':
        #     for i in range(self.nrow):
        #         self.axes[i,:].ravel()[0].get_shared_x_axes().join(* self.axes[i,:].ravel()[1:])
        # elif sharex=='col':
        #     for j in range(self.ncol):
        #         self.axes[:,j].ravel()[0].get_shared_x_axes().join(* self.axes[:,j].ravel()[1:])
        #
        # if sharey is True:
        #     self.axes.ravel()[0].get_shared_y_axes().join(*self.axes.ravel()[1:])
        # elif sharey=='row':
        #     for i in range(self.nrow):
        #         self.axes[i,:].ravel()[0].get_shared_y_axes().join(* self.axes[i,:].ravel()[1:])
        # elif sharey=='col':
        #     for j in range(self.ncol):
        #         self.axes[:,j].ravel()[0].get_shared_y_axes().join(* self.axes[:,j].ravel()[1:])

    def map_dataframe(self, func, *args, **kwargs):
        grouped = self.data.groupby(self.groupby_list, observed=True)
        if self.col_wrap is not None:
            # counter = 0
            for group, data in grouped:
                # careful! when using col_wrap first group should always the col
                if type(group) is not tuple:
                    counter = self.col_names_to_num[group]
                else:
                    counter = self.col_names_to_num[group[0]]
                icol = counter % self.ncol
                irow = int(np.floor(counter / (self.ncol)))
                ax = self.axes[irow, icol]
                plt.sca(ax)
                func(data, *args, **kwargs)
                # counter+=1
        else:
            for group, data in grouped:
                ax = self.get_axis(group)
                plt.sca(ax)
                func(data, *args, **kwargs)

    def get_axis(self, group):

        if type(group) is not tuple:
            group = [group]

        if self.row is None:
            i = 0
        else:
            i = self.row_names_to_num[group[0]]

        if self.col is None:
            j = 0
        else:
            if self.row is None:
                j = self.col_names_to_num[group[0]]
            else:
                j = self.col_names_to_num[group[1]]

        ax = self.axes[i, j]
        return ax


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, barcolor="black", barwidth=None,
                 **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0, 0), sizex, 0, ec=barcolor, lw=barwidth, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0, 0), 0, sizey, ec=barcolor, lw=barwidth, fc="none"))

        if sizex and labelx:
            self.xlabel = TextArea(labelx)
            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
        if sizey and labely:
            self.ylabel = TextArea(labely)
            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)


def add_scalebar(ax, **kwargs):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns created scalebar object
    """

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    return sb


def error_bar_clip_on(error_bars, clip_on=False):
    for error_bar in error_bars:
        for e in error_bar:
            if isinstance(e, tuple):
                for b in e:
                    b.set_clip_on(clip_on)
            else:
                e.set_clip_on(clip_on)
