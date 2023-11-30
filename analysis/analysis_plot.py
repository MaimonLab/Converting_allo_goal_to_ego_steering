"""analysis_plot.py

Common analyses and plotting functions

"""

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from simplification.cutil import simplify_coords_idx
from scipy import stats, ndimage
from scipy.signal import find_peaks
import sqlite3

import functions as fc
import plotting_help as ph


# FOR DEBUGGING
# from pdb import set_trace
# set_trace()

def plot(recs, rec_name,
         im_colormesh_signals=["pb_c1_roi.*.dF/F"],
         bh_overlay_signals=['xstim'],
         im_overlay_signals=[],
         bh_signals=['heading', 'forw'],
         im_signals=[],
         cmaps=None,
         tlim=None,
         vlims=None,
         fig_kws=None,
         plot_kws=None,
         phase_color=None,
         cb_kws=None,
         gridspec_kws=None,
         adjust_spines_kws=None,
         return_gs=False,
         bh_signals_kws=None,
         scatter=False,
         dark=False,
         sb=False,
         sb_kws=None,
         xlabel=True
         ):
    if fig_kws is None:
        fig_kws = {'figsize': [8, 40], 'dpi': None}

    if plot_kws is None:
        plot_kws = {}

    if gridspec_kws is None:
        gridspec_kws = {'height_ratios': [0.99, 0.01],
                        'wspace': 0.25,
                        'hspace': 0}

    if adjust_spines_kws is None:
        adjust_spines_kws = {}

    plt.figure(1, **fig_kws)

    abf_df = recs.merged_abf_df.query("rec_name=='" + rec_name + "'")
    if hasattr(recs, 'merged_im_df'):
        im_df = recs.merged_im_df.query("rec_name=='" + rec_name + "'")
    else:
        # if no imaging, im_df is just used for tlim
        # confusing fix
        im_df = abf_df
    rec = recs[rec_name]

    n_signals = len(im_colormesh_signals) + len(bh_signals) + len(im_signals)
    if (bh_overlay_signals + im_overlay_signals):
        n_signals = n_signals + 1

    gs = gridspec.GridSpec(nrows=2, ncols=n_signals, **gridspec_kws)

    if not tlim:
        start = im_df['t'].iloc[0]
        end = im_df['t'].iloc[-1]
    else:
        if tlim[0] > tlim[1]:
            print('tlim bad!')
            return
        else:
            start = tlim[0]
            end = tlim[1]

    i_col = 0
    # Plot imaging
    axes = []
    for i_colormesh_signal, im_colormesh_signal in enumerate(im_colormesh_signals):
        if 'ax' in locals():
            ax = plt.subplot(gs[0, i_col], sharey=ax)
        else:
            ax = plt.subplot(gs[0, i_col])
        cbar_ax = plt.subplot(gs[1, i_col])

        im_signal = im_df.filter(regex=(im_colormesh_signal)).values

        # ugly, but only way to get celltype
        if cmaps is None:
            structure = im_colormesh_signal.split('_')[0]
            channel = im_colormesh_signal.split('_')[1]
            celltypes = getattr(rec.im, structure).celltypes
            celltype = celltypes[channel]
            cmap = ph.celltype_cmaps[celltype]
        else:
            cmap = cmaps[i_colormesh_signal]

        # defines edges for y axis of colormesh
        t_colormesh = rec.im.df['t'].values - (rec.im.volume_period / 2.)
        t_colormesh = np.append(t_colormesh, t_colormesh[-1] + rec.im.volume_period)

        img = ax.pcolormesh(np.arange(im_signal.shape[1] + 1), t_colormesh,
                            im_signal, cmap=cmap)
        # if photostim is not None:
        #     photostim_signal = im_df.filter(regex=(photostim)).values
        #     # print(photostim_signal)
        #     plt.pcolormesh(np.arange(im_signal.shape[1]+1), rec.im.df['t'],
        #                    photostim_signal, alpha=1,edgecolors='k',cmap=cmap)
        #     # ax.add_patch(Rectangle((3, 4), 1, 1, fill=False, edgecolor='blue', lw=1))

        if vlims is not None:
            if vlims[i_colormesh_signal] is not None:
                img.set_clim(vmin=vlims[i_colormesh_signal][0], vmax=vlims[i_colormesh_signal][1])

        ax.set_ylim(start, end)
        ax.invert_yaxis()
        ax.set_facecolor(cmap(0))
        if (i_col == 0):
            ph.adjust_spines(ax, spines=['left'], **adjust_spines_kws)
        else:
            ph.adjust_spines(ax, spines=[])

        cbar_ax.axis('off')
        if cb_kws is None:
            cb_kws = dict(orientation='horizontal',
                          fraction=1,
                          aspect=5,
                          shrink=0.5,
                          pad=0,
                          anchor=(0, -1),
                          use_gridspec=False)

        cb = plt.colorbar(img, ax=cbar_ax, **cb_kws)
        cb.ax.tick_params(size=2.5, width=0.4)
        cb.outline.set_visible(False)

        if vlims is None:
            cmin = np.ceil(np.nanmin(im_signal))
            cmax = np.nanmax(im_signal)
            if (cmax > 1):
                cmax = np.floor(cmax)
            cb.set_ticks([np.round(cmin, 2), np.round(cmax, 2)])
        elif vlims[i_colormesh_signal] is not None:
            cb.set_ticks(vlims[i_colormesh_signal])

        if 'photostim' in im_colormesh_signal:
            ax.grid(True, color="k", lw=5, zorder=1)

        axes.append(ax)
        i_col += 1

    # Plot imaging and behaviour overlay
    if (bh_overlay_signals + im_overlay_signals):
        if 'ax' in locals():
            ax = plt.subplot(gs[0, i_col], sharey=ax)
        else:
            ax = plt.subplot(gs[0, i_col])

        if dark:
            ax.axvspan(-180, 180, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0)

        else:
            ax.axvspan(-180, -135, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0)
            ax.axvspan(135, 180, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0)

        for bh_overlay_signal in bh_overlay_signals:
            ph.circplot(abf_df[bh_overlay_signal], abf_df['t'], zorder=1, **plot_kws)

        for i, im_overlay_signal in enumerate(im_overlay_signals):

            if phase_color is None:
                structure = im_overlay_signal.split('_')[0]
                channel = im_overlay_signal.split('_')[1]
                celltypes = getattr(rec.im, structure).celltypes
                celltype = celltypes[channel]
                color = ph.celltype_color[celltype]
            else:
                color = phase_color[i]
            if scatter:
                plt.scatter(im_df[im_overlay_signal], im_df['t'], color=color, zorder=1, s=0.5, **plot_kws)
                ph.circplot(im_df[im_overlay_signal], im_df['t'], color=color, zorder=1, **plot_kws)
            else:
                ph.circplot(im_df[im_overlay_signal], im_df['t'], color=color, zorder=1, **plot_kws)

        ax.set_ylim(start, end)
        ax.set_xlim(-180, 180)
        # ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))

        ax.set_xticks([-180, 0, 180])
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        if (i_col == 0):
            ph.adjust_spines(ax, spines=['left', 'top'], **adjust_spines_kws)
        else:
            ph.adjust_spines(ax, spines=['top'], **adjust_spines_kws)

        axes.append(ax)
        i_col += 1

    # Plot behaviour
    for bh_signal in bh_signals:
        if bh_signals_kws is not None:
            bh_signal_kws = bh_signals_kws[bh_signal]
        else:
            bh_signal_kws = {'c': 'k', 'lw': plot_kws['lw']}

        if 'ax' in locals():
            ax = plt.subplot(gs[0, i_col], sharey=ax)
        else:
            ax = plt.subplot(gs[0, i_col])
        if rec.abf.is_circ(bh_signal):
            ph.circplot(abf_df[bh_signal], abf_df['t'], **plot_kws)
            ax.set_xlim(-180, 180)
            ax.set_xticks([-180, 0, 180])

        else:
            if 'xlim' in bh_signal_kws.keys():
                xlim = bh_signal_kws.pop('xlim')
                ax.set_xlim(xlim)
            plt.plot(abf_df[bh_signal], abf_df['t'], **bh_signal_kws)
            if ((bh_signal == 'dforw_boxcar_average_0.5_s') | (bh_signal == 'dheading_boxcar_average_0.5_s')):
                ax.fill_betweenx(y=abf_df['t'],
                                 x1=np.zeros(len(abf_df['t'])),
                                 x2=abf_df[bh_signal],
                                 color=ph.color_to_facecolor(bh_signal_kws.pop('c')), lw=0)

        if xlabel:
            ax.set_xlabel(bh_signal, labelpad=20)
        ax.set_ylim(start, end)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        if (i_col == 0):
            ph.adjust_spines(ax, spines=['left', 'top'], **adjust_spines_kws)
        else:
            ph.adjust_spines(ax, spines=['top'], **adjust_spines_kws)

        axes.append(ax)
        i_col += 1

    for im_signal in im_signals:

        if 'ax' in locals():
            ax = plt.subplot(gs[0, i_col], sharey=ax)
        else:
            ax = plt.subplot(gs[0, i_col])
        # make is circ!
        if rec.im.is_circ(im_signal):
            ph.circplot(im_df[im_signal], im_df['t'], **plot_kws)
            ax.set_xlim(-180, 180)
            ax.set_xticks([-180, 0, 180])
        else:
            plt.plot(im_df[im_signal], im_df['t'], c='k', **plot_kws)

        if im_signal == 'rml':
            ax.fill_betweenx(y=im_df['t'],
                             x1=np.zeros(len(im_df['t'])),
                             x2=im_df[im_signal],
                             color=ph.color_to_facecolor('k'), lw=0)

        if xlabel:
            ax.set_xlabel(im_signal, labelpad=20)
        ax.set_ylim(start, end)

        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        if (i_col == 0):
            ph.adjust_spines(ax, spines=['left', 'top'], **adjust_spines_kws)
        else:
            ph.adjust_spines(ax, spines=['top'], **adjust_spines_kws)

        axes.append(ax)
        i_col += 1

    if sb:
        ax = axes[0]
        ph.adjust_spines(ax, [])

        if sb_kws is None:
            sb_kws = {'sizex': 0, 'sizey': 5, 'barwidth': 0.8, 'barcolor': 'k', 'loc': 'center left', 'pad': -0.5}

        sb = ph.add_scalebar(ax, bbox_transform=ax.transAxes, **sb_kws)

    if return_gs:
        return axes, gs
    else:
        return axes


def plot_patch(recs, rec_name, tlim, pd_df, fig_kws=None,
               adjust_spines_kws=None, vm_ylim=None,
               dforw_ylim=None, sp_rate_ylim=None, tline=None,
               density=False, height_ratios=None, width_ratios=None, hspace=None, dheading_ylim=None):
    if fig_kws is None:
        fig_kws = {}
    if vm_ylim is None:
        vm_ylim = [-70, -15]
    if dforw_ylim is None:
        dforw_ylim = [-1, 11]
    if dheading_ylim is None:
        dheading_ylim = [-200, 200]
    if sp_rate_ylim is None:
        sp_rate_ylim = [0, 55]
    if height_ratios is None:
        height_ratios = [0.1, 0.4, 0.2, 0.3]
    if width_ratios is None:
        width_ratios = [0.9, 0.1]
    if hspace is None:
        hspace = 0.25
    rec = recs[rec_name]
    lw2 = 0.5
    lw = 0.25

    spike_color = 'k'
    s = '#5e5e5e'

    # pad=1

    idx_orig = (rec.abf.df_orig['t'] > tlim[0]) & (rec.abf.df_orig['t'] < tlim[1])
    idx = (rec.abf.df['t'] > tlim[0]) & (rec.abf.df['t'] < tlim[1])

    fig = plt.figure(1, **fig_kws)
    gs = gridspec.GridSpec(figure=fig, nrows=4, ncols=2,
                           wspace=0,
                           hspace=hspace,
                           height_ratios=height_ratios,
                           width_ratios=width_ratios)

    # dforw
    # dforw_color='#9c5916'
    dforw_color = '#5e5e5e'
    ax4 = plt.subplot(gs[0, 0])
    ax4.plot(rec.abf.df['t'][idx], rec.abf.df['dforw_boxcar_average_0.5_s'][idx], color=dforw_color, lw=lw2)
    ax4.fill_between(x=rec.abf.df['t'][idx],
                     y1=np.zeros(len(rec.abf.df['t'][idx])),
                     y2=rec.abf.df['dforw_boxcar_average_0.5_s'][idx],
                     color=ph.color_to_facecolor(dforw_color), lw=0)
    ax4.set_xlim(tlim)
    ax4.set_ylim(dforw_ylim)
    ax4.set_yticks([0, dforw_ylim[1]])
    ph.despine_axes([ax4], style=['left'], adjust_spines_kws=adjust_spines_kws)

    # # dheading
    # # dheading_color=ph.ddgreen
    # dheading_color='#5e5e5e'
    # ax6 = plt.subplot(gs[1,0],sharex=ax4)
    # ax6.plot(rec.abf.df['t'][idx],rec.abf.df['dheading_boxcar_average_0.5_s'][idx],color=dheading_color,lw=lw2)
    # ax6.fill_between(x=rec.abf.df['t'][idx],
    #                  y1=np.zeros(len(rec.abf.df['t'][idx])),
    #                  y2=rec.abf.df['dheading_boxcar_average_0.5_s'][idx],
    #                  color=ph.color_to_facecolor(dheading_color),lw=0)
    # ax6.set_ylim(dheading_ylim)
    # ax6.set_yticks([dheading_ylim[0],0,dheading_ylim[1]])
    # ph.despine_axes([ax6],style=['left'],adjust_spines_kws=adjust_spines_kws)

    # xstim
    ax2 = plt.subplot(gs[1, 0], sharex=ax4)
    ph.circplot(rec.abf.df['t'][idx], rec.abf.df['vh'][idx], circ='y', color='k', zorder=1, lw=lw2)
    ax2.axhspan(-180, -135, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0)
    ax2.axhspan(135, 180, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0)
    ax2.set_yticks([-180, 0, 180])
    ax2.set_ylim([-180, 180])
    ph.despine_axes([ax2], style=['left'], adjust_spines_kws=adjust_spines_kws)

    # spike rate
    ax5 = plt.subplot(gs[2, 0], sharex=ax4)
    ax5.plot(rec.abf.df['t'][idx], rec.abf.df['sp_rate_boxcar_average_1_s'][idx], color=spike_color, lw=lw2)
    ax5.fill_between(x=rec.abf.df['t'][idx],
                     y1=np.zeros(len(rec.abf.df['t'][idx])),
                     y2=rec.abf.df['sp_rate_boxcar_average_1_s'][idx],
                     color=ph.color_to_facecolor(spike_color), lw=0)
    ax5.set_ylim(sp_rate_ylim)
    ax5.set_yticks(sp_rate_ylim)
    ph.despine_axes([ax5], style=['left'], adjust_spines_kws=adjust_spines_kws)

    # Vm
    ax1 = plt.subplot(gs[3, 0], sharex=ax4)
    ax1.plot(rec.abf.df_orig['t'][idx_orig], rec.abf.df_orig['patch_1'][idx_orig], color='k', lw=lw)
    if tline is not None:
        ax1.hlines(y=vm_ylim[0] + 5, xmin=tline[0], xmax=tline[1], color='k', lw=0.5)
    # ax1.scatter(rec.abf.df_orig['t'][idx_orig],rec.abf.df_orig['sp'][idx_orig]*-20,s=5,color='r')
    ax1.set_ylim(vm_ylim)
    ax1.set_yticks(vm_ylim)
    ph.despine_axes([ax1], style=['left'], adjust_spines_kws=adjust_spines_kws)

    # pd hist or hist of xzstim
    if density == False:
        ax3 = plt.subplot(gs[2, 1], sharey=ax2)
        temp_df = pd_df.query(f'rec_name=="{rec_name}"')
        temp_df.sort_values(by=['vh_bin_center'], inplace=True)
        ax3.plot(temp_df['mean_value'], temp_df['vh_bin_center'], color='k', lw=lw2, ls=':')
        ph.despine_axes([ax3], style=['bottom'], adjust_spines_kws=adjust_spines_kws)
    elif density == True:
        ax3 = plt.subplot(gs[2, 1], sharey=ax2)

        plt.hist(rec.abf.df['vh'][idx].values, color='k', bins=np.linspace(-180, 180, 37), density=True,
                 orientation='horizontal')
        ph.despine_axes([ax3], style=['bottom'], adjust_spines_kws=adjust_spines_kws)
    else:
        pass


def plot_patch3(recs, rec_name, tlim, fig_kws=None,
                adjust_spines_kws=None, vm_ylim=None, tline_s=0.2):
    # just Vm with spikes

    if fig_kws is None:
        fig_kws = {}
    if vm_ylim is None:
        vm_ylim = [-70, -15]

    rec = recs[rec_name]
    lw = 0.25
    idx_orig = (rec.abf.df_orig['t'] > tlim[0]) & (rec.abf.df_orig['t'] < tlim[1])

    fig = plt.figure(1, **fig_kws)
    # ax1 = plt.subplot(gs[3,0],sharex=ax4)
    ax1 = plt.gca()
    ax1.plot(rec.abf.df_orig['t'][idx_orig], rec.abf.df_orig['patch_1'][idx_orig], color='k', lw=lw)
    ax1.scatter(rec.abf.df_orig['t'][idx_orig], rec.abf.df_orig['sp'][idx_orig] * (vm_ylim[1] - 2), s=1.5, color='k',
                lw=0,
                clip_on=False)
    ax1.set_ylim(vm_ylim)
    ph.despine_axes([ax1], style=['left'], adjust_spines_kws=adjust_spines_kws)

    sb = ph.add_scalebar(ax1, sizex=tline_s, sizey=0, barwidth=0.4, barcolor='k', loc='lower center',
                         pad=0,
                         bbox_transform=ax1.transAxes,
                         )


def plot_trajectory(x, y, idx=None, color='#a8a8a8', color_idx='k'):
    plt.plot(x, y, color=color, clip_on=False)
    plt.scatter(0, 0, s=1, color='r', zorder=3, clip_on=False)

    if idx is not None:
        plt.plot(x[idx], y[idx], color=color_idx, clip_on=False)

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    max_range = max(x_range, y_range)
    x_center = (x.max() + x.min()) / 2
    y_center = (y.max() + y.min()) / 2
    plt.xlim(x_center - max_range / 2, x_center + max_range / 2)
    plt.ylim(y_center - max_range / 2, y_center + max_range / 2)

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.axis('off')


def subtract_offset(recs, signal, idx='(xstim > -135) & (xstim < 135)'):
    # recs.merged_im_df.groupby('rec_name')
    def subtract(data, signal, idx):
        xstim = data['xstim']
        if idx is not None:
            idx = data.query(idx).index
            offset = fc.circmean(data[signal][idx] - xstim[idx])
        else:
            offset = fc.circmean(data[signal] - xstim)
        data[signal + '_offset_subtracted'] = fc.wrap(data[signal] - offset)
        return data

    recs.merged_im_df = recs.merged_im_df.groupby('rec_name').apply(subtract, signal, idx)
    # return is unnecessary since recs gets modified anyways, but might keep
    return recs


def get_hist_df(df, value, query=None, id_vars=None, bins=None, density=True):
    if query is not None:
        df = df.query(query).copy()
    else:
        df = df.copy()

    if id_vars is None:
        id_vars = ['fly_id', 'rec_name']
    if bins is None:
        if value in ['xstim', 'distance_to_goal']:
            bins = np.linspace(-180, 180, 37)
        else:
            print('need to set bins')

    def get_hist(data):
        hist, bin_edges = np.histogram(data[value], bins, density=density)
        bin_diff = np.diff(bin_edges)
        bin_centers = bin_edges[:-1] + bin_diff / 2.
        # call it norm_counts for historical reasons...
        return pd.DataFrame({value + '_bin_center': bin_centers, 'norm_counts': hist})

    hist_df = df.groupby(id_vars, observed=True).apply(get_hist).reset_index()

    return hist_df


def get_binned_df(df, bin_values, labels, query=None, id_vars=None, metric=None, observed=True):
    """
        Bins specified columns in df using pd.cut
        Then, metric is applied to values of labels that fall within each bin
        (similar to scipy binned_statistic_2d)

       :param df: Pandas DataFrame
       :param bin_values: dictionary mapping column name  to bins e.g. {'xstim':np.linspace(-180,180,37)}
       :param labels: list of column names to apply metric
       :param metric: function to apply to labels
       :param query :string to query df prior to binning
       :param id_vars: list of id columns for pd.melt


       :returns binned_df: Pandas DataFrame

   """

    df = df.copy()
    if query is not None:
        df = df.query(query)
    if id_vars is None:
        id_vars = ['fly_id', 'rec_name']
    if metric is None:
        metric = np.nanmean

    id_vars_melt = id_vars.copy()
    id_vars_melt = id_vars_melt + list(bin_values.keys())
    melted_df = pd.melt(df, id_vars=id_vars_melt,
                        value_vars=labels)

    # get binned data
    groupby_vars = id_vars.copy()
    groupby_vars.append('variable')
    for bin_label, bins in bin_values.items():
        if bins is None:
            if bin_label in ['xstim', 'distance_to_goal']:
                bins = np.linspace(-180, 180, 37)
            else:
                print('need to set bins')

        melted_df[bin_label + '_bin'] = pd.cut(melted_df[bin_label], bins=bins)
        groupby_vars.append(bin_label + '_bin')

    binned_df = melted_df.groupby(groupby_vars, observed=observed).agg(mean_value=('value', metric))
    binned_df.reset_index(inplace=True)

    # gets bin center
    for bin_label, bins in bin_values.items():
        binned_df[bin_label + '_bin_center'] = binned_df[bin_label + '_bin'].apply(lambda x: np.round(x.mid, 2))

    # drops multi index, note that "xstim" and "value" refter to mean, maybe way to add it to col name?
    # binned_df=binned_df.droplevel(axis=1,level=1)

    return binned_df


def get_fixation_events_df(recs, detection_parameters, inplace=False, im=True):
    """
    adds is_fixating and fixation_event_id to both merged_abf_df and merged_im_df
    based on RDP based fixation event algorithm
    if inplace is True modifies recs df,
    else returns new dfs (which currently drops non fixating rows)

	"""

    # removes columns if already exists (allows re-run)
    for col in ['is_fixating', 'fixation_event_id']:
        if col in recs.merged_abf_df.columns:
            recs.merged_abf_df.drop(col, axis=1, inplace=True)
    if im:
        for col in ['is_fixating', 'fixation_event_id']:
            if col in recs.merged_im_df.columns:
                recs.merged_im_df.drop(col, axis=1, inplace=True)

    abf_df = recs.merged_abf_df.copy()
    if im:
        im_df = recs.merged_im_df.copy()

    def get_fixation_events_timepoints(data):
        x = data['x'].values
        y = data['y'].values
        t = data['t'].values
        coords = list(zip(x, y))
        simplified_coords_idx = simplify_coords_idx(coords, detection_parameters['RDP_epsilon'])
        simplified_coords = np.array(coords)[simplified_coords_idx]
        simplified_coords = np.array(list(zip(*simplified_coords)))

        # computes length of each segment
        x_diff = np.diff(simplified_coords[0])
        y_diff = np.diff(simplified_coords[1])
        distance, goal = fc.cart2polar(x_diff, y_diff)

        # gets indices  of segments greater than threshold
        start = np.where(distance > detection_parameters['min_length'])[0]
        start_indices = np.array(simplified_coords_idx)[start]
        end_indices = np.array(simplified_coords_idx)[start + 1]

        start_time = t[start_indices]
        end_time = t[end_indices]
        timepoints = list(zip(start_time, end_time))

        return timepoints

    start_end = abf_df.groupby('rec_name').apply(get_fixation_events_timepoints)
    start_end = pd.DataFrame({'start_end': start_end})
    start_end.reset_index(inplace=True)
    start_end = start_end.explode('start_end', ignore_index=True)
    start_end.dropna(axis=0, inplace=True)
    start_end['start'] = start_end['start_end'].apply(lambda x: x[0])
    start_end['end'] = start_end['start_end'].apply(lambda x: x[1])
    start_end.drop(['start_end'], axis=1, inplace=True)
    start_end['is_fixating'] = True
    start_end['fixation_event_id'] = start_end.index + 1

    # SQL based creation of data frame
    conn = sqlite3.connect(':memory:')
    # write the tables
    abf_df.to_sql('abf_df', conn, index=False)
    start_end.to_sql('start_end', conn, index=False)

    qry = '''
          select 
              abf_df.*,
              start_end.is_fixating,
              start_end.fixation_event_id
    
          from abf_df
          left join start_end
             on
                abf_df.t > start_end.start and abf_df.t < start_end.end 
                and
                abf_df.rec_name=start_end.rec_name
              
          '''
    abf_df = pd.read_sql_query(qry, conn)

    abf_df['is_fixating'] = abf_df['is_fixating'].fillna(0)
    abf_df['is_fixating'] = abf_df['is_fixating'].astype(bool)

    def get_rec_fixation_event_id(data):
        data['rec_fixation_event_id'] = data['fixation_event_id'] - np.min(data['fixation_event_id']) + 1
        return data

    abf_df = abf_df.groupby('rec_name').apply(get_rec_fixation_event_id)
    abf_df.loc[abf_df['fixation_event_id'].isna(), 'fixation_event_id'] = -1
    abf_df.loc[abf_df['rec_fixation_event_id'].isna(), 'rec_fixation_event_id'] = -1
    abf_df['unique_fixation_event_id'] = abf_df['rec_name'] + '_' + abf_df['rec_fixation_event_id'].astype(str)

    # note that the non-fixation events all have fixation_event_id == -1, so the goal
    # will be nonsense
    def get_goal(data):
        idx = data['dforw_boxcar_average_0.5_s'].values > 1
        goal = fc.circmean(data['xstim'].values[idx])
        data['goal'] = goal
        # data['distance_to_goal'] = fc.wrap(data['xstim']-data['goal'])
        data['distance_to_goal'] = fc.wrap(data['goal'] - data['xstim'])
        return data

    abf_df = abf_df.groupby(['fixation_event_id']).apply(get_goal)

    if im:
        # repeat for imaging dataframe
        # SQL based creation of data frame
        conn = sqlite3.connect(':memory:')
        # write the tables
        im_df.to_sql('im_df', conn, index=False)
        start_end.to_sql('start_end', conn, index=False)

        qry = '''
              select 
                  im_df.*,
                  start_end.is_fixating,
                  start_end.fixation_event_id
        
              from im_df
              left join start_end
                 on
                    im_df.t > start_end.start and im_df.t < start_end.end 
                    and
                    im_df.rec_name=start_end.rec_name
                  
              '''
        im_df = pd.read_sql_query(qry, conn)

        im_df['is_fixating'] = im_df['is_fixating'].fillna(0)
        im_df['is_fixating'] = im_df['is_fixating'].astype(bool)
        im_df['fixation_event_id'] = im_df['fixation_event_id'].fillna(-1)

        # Goal is re-calculated on downsampled behaviour (not ideal, but in practice the two are nearly identical)
        im_df = im_df.groupby(['fixation_event_id']).apply(get_goal)

    if inplace:
        recs.merged_abf_df = abf_df
        if im:
            recs.merged_im_df = im_df
            return abf_df, im_df
        else:
            return abf_df

    else:
        if im:
            return abf_df, im_df
        else:
            return abf_df


def get_peaks_df(recs, pad_s, label='ps_c1_rml_z', detection_parameters=None, align_to_start=False):
    """
        Finds transient increases and decreases in imaging signal (using sc.find_peaks)
        align_to_start: if True, signal will be aligned to the first index before the peak where the signal is positive
        (instead of aligned to peak)

   """

    abf_df = recs.merged_abf_df.copy()
    im_df = recs.merged_im_df.copy()

    # if this number is different for different recordings, you won't be able to take average across trials! which
    # is the whole point of creating turn_time
    sampling_periods = [np.round(1. / rec.rec.abf.subsampling_rate, 3) for rec in recs]
    im_sampling_periods = [np.round(rec.rec.im.volume_period, 3) for rec in recs]
    if (len(set(sampling_periods)) != 1) | (len(set(im_sampling_periods)) != 1):
        print('samplig periods are different!')
    else:
        sampling_period = sampling_periods[0]
        sampling_rate = 1. / sampling_period
        im_sampling_period = im_sampling_periods[0]
        im_sampling_rate = 1. / im_sampling_period

    if detection_parameters is None:
        detection_parameters = {
            # 'edge':'left',
            'gaussfilt_s': 0.2,
            'height': 1,
            'distance': 3,
            'width': 0.5,
        }

    # convert detection_parameters from seconds to frames
    detection_parameters['width'] *= im_sampling_rate
    detection_parameters['distance'] *= im_sampling_rate
    if 'gaussfilt_s' in detection_parameters.keys():
        sigma = detection_parameters['gaussfilt_s'] * im_sampling_rate

    removed_keys = ['gaussfilt_s']
    find_peaks_parameters = {x: detection_parameters[x] for x in detection_parameters if x not in removed_keys}

    def get_peaks(data):
        data.reset_index(inplace=True, drop=True)
        signal = data[label].values
        if 'gaussfilt_s' in detection_parameters.keys():
            signal = ndimage.filters.gaussian_filter1d(signal, sigma)

        up_peak_indices, up_peak_properties = find_peaks(signal, **find_peaks_parameters)
        if align_to_start:
            all_inds = np.arange(len(signal))
            up_peak_indices = [np.where((all_inds < ind) & (signal > 0))[0][-1] for ind in up_peak_indices]
        up_peak_t = data['t_abf'].values[up_peak_indices]
        down_peak_indices, down_peak_properties = find_peaks(-1 * signal, **find_peaks_parameters)
        if align_to_start:
            all_inds = np.arange(len(signal))
            down_peak_indices = [np.where((all_inds < ind) & (signal < 0))[0][-1] for ind in down_peak_indices]
        down_peak_t = data['t_abf'].values[down_peak_indices]
        up_peak_df = pd.DataFrame({'t_0': up_peak_t, 'trial_type': 'up'})
        down_peak_df = pd.DataFrame({'t_0': down_peak_t, 'trial_type': 'down'})
        df = pd.concat([up_peak_df, down_peak_df])
        return df

    t0_df = im_df.groupby('rec_name').apply(get_peaks).reset_index().drop('level_1', axis=1)
    t0_df['start'] = t0_df['t_0'] - pad_s
    t0_df['end'] = t0_df['t_0'] + pad_s
    t0_df['trial_id'] = t0_df.index.values + 1

    # SQL based creation of data frame
    conn = sqlite3.connect(':memory:')
    # write the tables
    im_df.to_sql('im_df', conn, index=False)
    t0_df.to_sql('t0_df', conn, index=False)

    qry = '''
          select  
              im_df.*,
              t0_df.trial_id,
              t0_df.trial_type,
              t0_df.t_0,
              t0_df.start,
              t0_df.end
    
          from
              im_df join t0_df on
              t_abf >= t0_df.start and t_abf <= t0_df.end 
              and
              im_df.rec_name=t0_df.rec_name
          '''
    im_peaks_df = pd.read_sql_query(qry, conn)
    im_peaks_df['trial_type'] = im_peaks_df['trial_type'].astype('category')

    # SQL based creation of data frame, faster than other methods tried
    conn = sqlite3.connect(':memory:')
    # write the tables
    abf_df.to_sql('abf_df', conn, index=False)
    t0_df.to_sql('t0_df', conn, index=False)

    qry = '''
        select
            abf_df.*,
            t0_df.trial_id,
            t0_df.trial_type,
            t0_df.t_0,
            t0_df.start,
            t0_df.end

        from
            abf_df join t0_df on
            t >= t0_df.start and t <= t0_df.end
            and
            abf_df.rec_name=t0_df.rec_name
        '''
    abf_peaks_df = pd.read_sql_query(qry, conn)
    abf_peaks_df['trial_type'] = abf_peaks_df['trial_type'].astype('category')
    abf_peaks_df['trial_time'] = abf_peaks_df['t'] - abf_peaks_df['t_0']
    im_peaks_df['trial_time'] = im_peaks_df['t_abf'] - im_peaks_df['t_0']

    return abf_peaks_df, im_peaks_df


def get_trials_df(recs, pad_s=5, im=True, stimid_map=None, stimid_label='stimid'):
    """
       Returns trials data frame
       Merged dfs will be modified as well:
       Adds columns "is_OL", "trial_id", "trial_type" and "trial_time"
       Each row is a timepoint belonging to a trial (where trial is a stimid event)
       padded by pad_s

       # TODO:
       -Break up stuff to re-use for fixation events?
    """

    # if the sampling rate is different for different recordings, you won't be able to take average across trials
    sampling_periods = [np.round(1. / rec.rec.abf.subsampling_rate, 3) for rec in recs]
    if (len(set(sampling_periods)) != 1):
        print('samplig periods are different!')
        return
    sampling_period = sampling_periods[0]
    sampling_rate = 1. / sampling_period

    if im:
        im_sampling_periods = [np.round(rec.rec.im.volume_period, 3) for rec in recs]
        if (len(set(im_sampling_periods)) != 1):
            print('samplig periods are different!')
            return
        im_sampling_period = im_sampling_periods[0]

    if stimid_map is None:
        stimid_map = {}
        stimid_map['-90 jump'] = [5, 8]
        stimid_map['+90 jump'] = [8, 11]

    # get open-loop trials
    def get_trials(data):

        data['trial_type'] = np.nan
        for trial_type in stimid_map.keys():
            stimid_range = stimid_map[trial_type]
            data.loc[(data[stimid_label] > stimid_range[0]) & (
                    data[stimid_label] < stimid_range[1]), 'trial_type'] = trial_type

        data['is_OL'] = ~data['trial_type'].isin(['wind_off', 'CL_bar'])
        data['trial_id'] = fc.number_islands(data['is_OL'].values)

        return data

    # this modifies actual merged df
    recs.merged_abf_df = recs.merged_abf_df.groupby('rec_name').apply(get_trials)
    if im:
        recs.downsample_merged(['is_OL', 'trial_id', 'trial_type'], lag_ms=0, metric=fc.mode)

    def get_padded_trial_time(t, pad_s):
        t = t.values
        t0 = t[0] - pad_s
        t1 = t[-1] + pad_s
        return (t0, t1)

    start_end = recs.merged_abf_df.query('trial_id!=0').groupby(['rec_name', 'trial_id', 'trial_type']).t.apply(
        get_padded_trial_time, pad_s).reset_index(name='t')
    t = start_end.t.apply(pd.Series)
    t.columns = ['start', 'end']
    start_end.drop(['t'], axis=1, inplace=True)
    start_end = pd.concat([start_end, t], axis=1)

    # below modify recs dfs
    abf_df = recs.merged_abf_df.copy()

    # these will be assigned based on start_end
    abf_df.drop(['trial_id', 'trial_type'], axis=1, inplace=True)

    # SQL based creation of data frame, faster than other methods tried
    conn = sqlite3.connect(':memory:')
    # write the tables
    abf_df.to_sql('abf_df', conn, index=False)
    start_end.to_sql('start_end', conn, index=False)

    qry = '''
           select  
               abf_df.*,
               start_end.trial_id,
               start_end.trial_type,
               start_end.start,
               start_end.end

           from
               abf_df join start_end on
               t >= start_end.start and t <= start_end.end 
               and
               abf_df.rec_name=start_end.rec_name
           '''

    abf_trials_df = pd.read_sql_query(qry, conn)
    abf_trials_df['trial_type'] = abf_trials_df['trial_type'].astype('category')

    # very slow!
    def get_trial_time(data):
        t0 = data['start'].values[0] + pad_s
        idx = np.where(data['t'].values == t0)[0]
        trial_time = (np.arange(len(data.index)) * sampling_period) - pad_s
        # round bc of float imprecision
        trial_time = np.round(trial_time - trial_time[idx], 6)
        data['trial_time'] = trial_time
        return data

    abf_trials_df = abf_trials_df.groupby(['rec_name', 'trial_id']).apply(get_trial_time)
    abf_trials_df['unique_trial_id'] = abf_trials_df['rec_name'].astype(str) + '_' + abf_trials_df['trial_id'].astype(
        str)
    abf_trials_df['unique_trial_id'] = abf_trials_df['unique_trial_id'].astype('category')

    if im:

        # Repeats fom im_df
        im_df = recs.merged_im_df.copy()
        im_df.drop(['trial_id', 'trial_type'], axis=1, inplace=True)

        # SQL based creation of data frame
        conn = sqlite3.connect(':memory:')
        # write the tables
        im_df.to_sql('im_df', conn, index=False)
        start_end.to_sql('start_end', conn, index=False)

        qry = '''
                 select  
                     im_df.*,
                     start_end.trial_id,
                     start_end.trial_type,
                     start_end.start,
                     start_end.end
    
                 from
                     im_df join start_end on
                     t_abf >= start_end.start and t_abf <= start_end.end 
                     and
                     im_df.rec_name=start_end.rec_name
                 '''
        im_trials_df = pd.read_sql_query(qry, conn)
        im_trials_df['trial_type'] = im_trials_df['trial_type'].astype('category')

        # very slow!
        def get_trial_time(data):
            data['trial_time'] = (np.arange(len(data.index)) * im_sampling_period) - pad_s
            return data

        im_trials_df = im_trials_df.groupby(['rec_name', 'trial_id']).apply(get_trial_time)
        im_trials_df['unique_trial_id'] = im_trials_df['rec_name'].astype(str) + '_' + im_trials_df['trial_id'].astype(
            str)
        im_trials_df['unique_trial_id'] = im_trials_df['unique_trial_id'].astype('category')

        return abf_trials_df, im_trials_df

    else:
        return abf_trials_df


def get_photostim_trials_df(recs, pad_s=5):
    """
        COPY PASTED FROM get_trials_df()


       Returns trials data frame (based on merged_im_df, which will be modified as well)
       Adds columns "is_OL", "trial_id", "trial_type" and "trial_time"
       Each row is a timepoint belonging to a trial (where trial is a stimid event)
       padded by pad_s

       #TO DO:
       -dictionary mapping stimid to trial_type (the stimid_map)
       -Break up stuff to re-use for fixation events..
       -Fix time, now assumes everything is at same rate and is aligned to start
    """

    # if this number is different for different recordings, you won't be able to take average across trials! which
    # is the whole point of creating turn_time
    sampling_periods = [np.round(1. / rec.rec.abf.subsampling_rate, 3) for rec in recs]
    im_sampling_periods = [np.round(rec.rec.im.volume_period, 3) for rec in recs]
    if (len(set(sampling_periods)) != 1) | (len(set(sampling_periods)) != 1):
        print('samplig periods are different!')
        return
    else:
        sampling_period = sampling_periods[0]
        sampling_rate = 1. / sampling_period
        im_sampling_period = im_sampling_periods[0]

    def get_padded_trial_time(t, pad_s):
        t = t.values
        t0 = t[0] - pad_s
        t1 = t[-1] + pad_s
        return (t0, t1)

    start_end = recs.merged_im_df.query('scanfield_config_name!="control"').groupby(
        ['rec_name', 'scanfield_config_name', 'photostim_trial_id']).t_abf.apply(
        get_padded_trial_time, pad_s).reset_index(name='t')
    t = start_end.t.apply(pd.Series)
    t.columns = ['start', 'end']
    start_end.drop(['t'], axis=1, inplace=True)
    start_end = pd.concat([start_end, t], axis=1)

    # below doesnt actually modify recs dfs
    abf_df = recs.merged_abf_df.copy()
    im_df = recs.merged_im_df.copy()

    # these will be assigned based on start_end
    im_df.drop(['scanfield_config_name', 'photostim_trial_id'], axis=1, inplace=True)

    # SQL based creation of data frame, faster than other methods tried
    conn = sqlite3.connect(':memory:')
    # write the tables
    im_df.to_sql('im_df', conn, index=False)
    start_end.to_sql('start_end', conn, index=False)

    qry = '''
             select  
                 im_df.*,
                 start_end.scanfield_config_name,
                 start_end.photostim_trial_id,
                 start_end.start,
                 start_end.end

             from
                 im_df join start_end on
                 t_abf >= start_end.start and t_abf <= start_end.end 
                 and
                 im_df.rec_name=start_end.rec_name
             '''

    im_trials_df = pd.read_sql_query(qry, conn)
    im_trials_df['scanfield_config_name'] = im_trials_df['scanfield_config_name'].astype('category')

    # very slow!
    def get_trial_time(data):
        t0 = data['start'].values[0] + pad_s
        idx = np.where(data['t_abf'].values == t0)[0]
        trial_time = (np.arange(len(data.index)) * im_sampling_period) - pad_s
        trial_time = trial_time - trial_time[idx]

        data['trial_time'] = trial_time

        return data

    im_trials_df = im_trials_df.groupby(['rec_name', 'photostim_trial_id']).apply(get_trial_time)
    im_trials_df['unique_trial_id'] = im_trials_df['rec_name'].astype(str) + '_' + im_trials_df[
        'photostim_trial_id'].astype(str)
    im_trials_df['unique_trial_id'] = im_trials_df['unique_trial_id'].astype('category')

    # SQL based creation of data frame
    conn = sqlite3.connect(':memory:')
    # write the tables
    abf_df.to_sql('abf_df', conn, index=False)
    start_end.to_sql('start_end', conn, index=False)

    qry = '''
           select  
               abf_df.*,
               start_end.scanfield_config_name,
               start_end.photostim_trial_id,
               start_end.start,
               start_end.end

           from
               abf_df join start_end on
               t >= start_end.start and t <= start_end.end 
               and
               abf_df.rec_name=start_end.rec_name
           '''

    abf_trials_df = pd.read_sql_query(qry, conn)
    abf_trials_df['scanfield_config_name'] = abf_trials_df['scanfield_config_name'].astype('category')

    # very slow!
    def get_trial_time(data):
        # trial time isnt calculated differently, I think this is OK because sampling rate is higher
        data['trial_time'] = (np.arange(len(data.index)) * sampling_period) - pad_s
        return data

    abf_trials_df = abf_trials_df.groupby(['rec_name', 'photostim_trial_id']).apply(get_trial_time)
    abf_trials_df['unique_trial_id'] = abf_trials_df['rec_name'].astype(str) + '_' + abf_trials_df[
        'photostim_trial_id'].astype(str)
    abf_trials_df['unique_trial_id'] = abf_trials_df['unique_trial_id'].astype('category')

    return abf_trials_df, im_trials_df


def get_sliding_mean_vector(recs, window, im=True):
    # Adds r and mean_vector to recs.merged_abf_df and recs.merged_im_df

    def sliding_window(data, window_s, sampling_rate):

        data[label_r] = np.nan
        data[label_mean] = np.nan

        idx = (data['dforw_boxcar_average_0.5_s'].values > 1)
        signal = data['xstim'].values[idx]
        inds = np.arange(len(signal))
        half_window_s = window_s / 2.
        # bc an integer number of indices is used, the window is not necessarily exact
        half_window_ind = int(np.round(half_window_s * sampling_rate))
        start = inds - half_window_ind
        end = inds + half_window_ind + 1
        r = np.zeros(len(signal))
        theta = np.zeros(len(signal))
        # prevents wrapping, but means that some windows are shorter
        start[start < 0] = 0
        end[end > len(signal)] = len(signal)
        for i in inds:
            t0 = start[i]
            t1 = end[i]
            r[i], theta[i] = fc.mean_vector(np.deg2rad(signal[t0:t1]))
        data[label_r][idx] = r
        data[label_mean][idx] = np.rad2deg(theta)

        return data

    sampling_rates = [np.round(rec.rec.abf.subsampling_rate) for rec in recs]
    if (len(set(sampling_rates)) != 1):
        print('samplig rates are different!')
        return
    sampling_rate = sampling_rates[0]
    label_r = 'r_' + str(window)
    label_mean = 'mean_vector_' + str(window)
    recs.merged_abf_df = recs.merged_abf_df.groupby(['rec_name'],
                                                    as_index=False).apply(
        lambda x: sliding_window(x, window, sampling_rate))

    if im:
        recs.downsample_merged([label_r], 0, np.nanmean)
        recs.downsample_merged([label_mean], 0, fc.circmean)
