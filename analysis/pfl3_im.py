"""pfl3_im.py

Analysis and plotting functions for PFL3_LAL_imaging.ipynb

"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy as sc
from scipy.interpolate import interp1d
import analysis_plot as ap
import functions as fc
import plotting_help as ph
import read_data as rd

# ------------------ Plotting parameters ----------------------- #
dpi = 300
axis_lw = 0.4
axis_ticklen = 2.5
pad = 2
font = {'family': 'arial',
        'weight': 'normal',
        'size': 5}
mpl.rc('font', **font)


# ---------------------------- Load data ---------------------------- #
def load_data(DATA_PATH, reprocess=False):
    rec_names = [
        (
            '2022_07_01_0002',
            '2022_07_01_0003',
        ),

        (
            '2022_07_01_0004',
            '2022_07_01_0005',
        ),

        (
            '2022_07_01_0006',
            '2022_07_01_0007',
        ),

        (
            '2022_07_01_0009',
            '2022_07_01_0010',
        ),

        (
            '2022_07_01_0011',
        ),

        (
            '2022_07_03_0001',
            '2022_07_03_0002',
        ),

        (
            '2022_07_03_0003',
            '2022_07_03_0004',
        ),

        (
            '2022_07_03_0005',
            '2022_07_03_0006',
        ),

        (
            '2022_07_03_0007',
            '2022_07_03_0008',
        ),

        (
            '2022_07_03_0009',
        ),
    ]

    parent_folder = DATA_PATH + 'PFL3_LAL_imaging' + os.path.sep + '57C10-AD-VT037220-DBD' + os.path.sep
    recs = rd.quickload_experiment(parent_folder + 'nature_57C10_AD_VT037220_DBD.pkl',
                                   rec_names=rec_names,
                                   exp_kws={
                                       'rec_type': rd.ImagingRec,
                                       'parent_folder': parent_folder,
                                       'merge_df': True,
                                       'genotype': '57C10-AD-VT037220-DBD',
                                   },
                                   reprocess=reprocess,
                                   roi_types=[rd.PairedStructure],
                                   roi_kws={'celltypes': {'c1': 'PFL3', 'c2': 'PFL3'}},
                                   bh_kws={'angle_offset': 86, 'boxcar_average': {'dforw': 0.5, 'dheading': 0.5, }}
                                   )
    return recs


# ------------------ Process & analyze data ----------------------- #

def get_mean_peaks_dheading(abf_peaks_df, im_peaks_df):
    # upsamples dheading and rml to a common 100 Hz timebase
    # and gets mean across trials

    def get_upsampled(data, label):
        trial_time = data['trial_time'].values
        signal = data[label].values
        f = interp1d(trial_time, signal, kind='linear', bounds_error=False, fill_value=np.nan)
        t = np.linspace(-12, 12, 24 * 100)
        signal_upsampled = f(t)
        df = pd.DataFrame({'t': t, label: signal_upsampled})
        return df

    upsampled_abf_peaks_df = abf_peaks_df.groupby(['trial_id', 'trial_type']).apply(
        lambda x: get_upsampled(x, 'dheading')).reset_index().drop('level_2', axis=1)
    upsampled_im_peaks_df = im_peaks_df.groupby(['trial_id', 'trial_type']).apply(
        lambda x: get_upsampled(x, 'ps_c1_rml_dF/F')).reset_index().drop('level_2', axis=1)

    peaks = pd.merge(left=upsampled_abf_peaks_df,
                     right=upsampled_im_peaks_df,
                     on=['trial_id', 'trial_type', 't'])

    peaks_melted = pd.melt(peaks, id_vars=['trial_id', 'trial_type', 't'],
                           value_vars=['ps_c1_rml_dF/F', 'dheading']
                           )

    mean_peaks_df = peaks_melted.groupby(['trial_type',
                                          't', 'variable']).agg(mean=('value', np.nanmean), sem=('value',
                                                                                                 lambda x: sc.stats.sem(
                                                                                                     x,
                                                                                                     nan_policy='omit')))
    mean_peaks_df.reset_index(inplace=True)
    return mean_peaks_df


def get_mean_peaks_distance_to_goal(abf_peaks_df, im_peaks_df):
    # try to consolidate with above function
    def get_upsampled_abf(data):
        trial_time = data['trial_time'].values
        dheading = data['dheading'].values
        goal = data['mean_vector_60'].interpolate(method='nearest').values
        # distance_to_goal = fc.wrap(data['xstim'].values-goal)

        distance_to_goal = fc.wrap(goal - data['xstim'].values)
        distance_to_goal_x, distance_to_goal_y = fc.polar2cart(np.ones(len(distance_to_goal)),
                                                               np.deg2rad(distance_to_goal))
        t = np.linspace(-12, 12, 24 * 100)
        # upsampled imaging
        f = interp1d(trial_time, dheading, kind='linear', bounds_error=False, fill_value=np.nan)
        dheading_upsampled = f(t)
        f = interp1d(trial_time, distance_to_goal_x, kind='linear', bounds_error=False, fill_value=np.nan)
        x_upsampled = f(t)
        f = interp1d(trial_time, distance_to_goal_y, kind='linear', bounds_error=False, fill_value=np.nan)
        y_upsampled = f(t)
        r, distace_to_goal_upsampled = fc.cart2polar(x_upsampled, y_upsampled)
        distace_to_goal_upsampled = np.rad2deg(distace_to_goal_upsampled)

        df = pd.DataFrame({'t': t, 'dheading': dheading_upsampled, 'distance_to_goal': distace_to_goal_upsampled})
        return df

    upsampled_abf_peaks_df = abf_peaks_df.groupby(['trial_id', 'trial_type']).apply(
        get_upsampled_abf).reset_index().drop('level_2', axis=1)

    def get_upsampled(data):
        trial_time = data['trial_time'].values
        rml = data['ps_c1_rml_dF/F'].values
        # upsampled imaging
        f = interp1d(trial_time, rml, kind='linear', bounds_error=False, fill_value=np.nan)
        t = np.linspace(-12, 12, 24 * 100)
        rml_upsampled = f(t)
        df = pd.DataFrame({'t': t, 'ps_c1_rml_dF/F': rml_upsampled})

        return df

    upsampled_im_peaks_df = im_peaks_df.groupby(['trial_id', 'trial_type']).apply(get_upsampled).reset_index().drop(
        'level_2', axis=1)

    peaks = pd.merge(left=upsampled_abf_peaks_df,
                     right=upsampled_im_peaks_df,
                     on=['trial_id', 'trial_type', 't'])

    peaks_melted = pd.melt(peaks, id_vars=['trial_id', 'trial_type', 't'],
                           value_vars=['ps_c1_rml_dF/F', 'dheading', 'distance_to_goal']
                           )

    mean_peaks_df_dist = peaks_melted.query('variable=="distance_to_goal"').groupby(['trial_type',
                                                                                     't', 'variable']).agg(
        mean=('value', fc.circmean), sem=('value', fc.circ_stderror))
    mean_peaks_df_dist.reset_index(inplace=True)

    mean_peaks_df = peaks_melted.query('variable!="distance_to_goal"').groupby(['trial_type',
                                                                                't', 'variable']).agg(
        mean=('value', np.nanmean), sem=('value',
                                         lambda x: sc.stats.sem(x, nan_policy='omit')))
    mean_peaks_df.reset_index(inplace=True)

    mean_peaks_df = pd.concat([mean_peaks_df_dist, mean_peaks_df])

    return mean_peaks_df


def get_shifted_signals(recs, im_fixation_df):
    def shift_signal(data, signal):
        shift_n_rows = -2
        data['shifted_' + signal] = data[signal].shift(periods=shift_n_rows)
        return data

    labels = ['ps_c1_rml_dF/F', 'ps_c1_roi_1_dF/F', 'ps_c1_roi_2_dF/F']
    for label in labels:
        im_fixation_df = im_fixation_df.groupby('fixation_event_id').apply(shift_signal, signal=label)

    print(2 / np.round(recs['2022_07_01_0002'].im.volume_rate, 4))

    return im_fixation_df


def get_lal_vs_distance_to_goal(im_fixation_df, save=False, savepath=None, fname=None):
    binned_df = ap.get_binned_df(im_fixation_df,
                                 bin_values={'distance_to_goal': np.linspace(-180, 180, 37),
                                             },
                                 labels=['shifted_ps_c1_rml_dF/F',
                                         'shifted_ps_c1_roi_1_dF/F',
                                         'shifted_ps_c1_roi_2_dF/F'],

                                 id_vars=['fly_id', 'rec_name', 'fixation_event_id'],
                                 query='`dforw_boxcar_average_0.5_s`>1 & is_fixating==True',
                                 )

    flies_df = binned_df.groupby(['fly_id', 'variable', 'distance_to_goal_bin_center']).agg(
        mean_value=('mean_value', np.nanmean))
    flies_df = flies_df.reset_index()

    mean_df = flies_df.groupby(['variable', 'distance_to_goal_bin_center']).agg(
        mean_value=('mean_value', np.nanmean),
        std_err_value=('mean_value', lambda x: sc.stats.sem(x, nan_policy='omit')))
    mean_df = mean_df.reset_index()
    if save:
        mean_df_save = mean_df.copy()
        mean_df_save['variable'] = mean_df_save['variable'].map({'shifted_ps_c1_roi_1_dF/F': 'Left',
                                                                 'shifted_ps_c1_roi_2_dF/F': 'Right',
                                                                 'shifted_ps_c1_rml_dF/F': 'RML'})
        mean_df_save.rename({'mean_value': 'mean', 'std_err_value': 'sem'}, axis=1, inplace=True)
        flies_df_save = flies_df.copy()
        flies_df_save['variable'] = flies_df_save['variable'].map({'shifted_ps_c1_roi_1_dF/F': 'Left',
                                                                   'shifted_ps_c1_roi_2_dF/F': 'Right',
                                                                   'shifted_ps_c1_rml_dF/F': 'RML'})
        flies_df_save.rename({'mean_value': 'value'}, axis=1, inplace=True)
        pd.concat([flies_df_save, mean_df_save]).to_csv(savepath + fname)

    return mean_df, flies_df


def get_menotaxis_bouts_raster_and_goals(recs, save=False, savepath=None, fname=None):
    menotaxis_bouts_raster_df = recs.merged_im_df.filter(
        ['genotype', 'fly_id', 'rec_name', 't', 'is_fixating', 'goal', 'fixation_event_id']).copy()
    t_max = np.round(menotaxis_bouts_raster_df['t'].max())
    menotaxis_bouts_raster_df['t_bin'] = pd.cut(menotaxis_bouts_raster_df['t'], bins=np.arange(-1, t_max, 1)).apply(
        lambda x: np.round(x.mid, 2))
    menotaxis_bouts_raster_df['unique_fly_id'] = menotaxis_bouts_raster_df['genotype'] + '_' + \
                                                 menotaxis_bouts_raster_df['fly_id'].astype(str)
    menotaxis_bouts_raster_df['unique_fly_id'] = menotaxis_bouts_raster_df['unique_fly_id'].astype("category")
    menotaxis_bouts_raster_df['unique_fly_id'] = menotaxis_bouts_raster_df['unique_fly_id'].cat.set_categories(
        menotaxis_bouts_raster_df.sort_values(['genotype', 'fly_id'])['unique_fly_id'].unique().tolist(), ordered=True)
    summary_menotaxis_df = menotaxis_bouts_raster_df.query('is_fixating==True').groupby(['genotype'],
                                                                                        as_index=False).apply(
        lambda x: x.drop_duplicates('fixation_event_id')).reset_index().copy()
    menotaxis_bouts_raster_df = menotaxis_bouts_raster_df.pivot_table(
        index=[
            'genotype',
            'fly_id',
            'rec_name',
        ],
        columns=[
            't_bin',
        ],
        values='is_fixating').reindex(index=np.arange(0, 100), level='fly_id')

    # goal from fixation events is based on bar position, but here we want heading
    # not ideal place to do this...
    summary_menotaxis_df['goal'] = summary_menotaxis_df['goal'] * -1
    if save:
        summary_menotaxis_df.filter(['fly_id', 'goal']).to_csv(savepath + fname)
    return menotaxis_bouts_raster_df, summary_menotaxis_df


def get_bar_jump_trials(recs):
    stimid_map = {'+90 jump': [9, 11],
                  '-90 jump': [5, 8],
                  'CL_bar': [-1, 4]
                  }
    pad_s = 30
    bar_jump_duration_s = 2

    abf_trials_df, im_trials_df = ap.get_trials_df(recs, pad_s=pad_s, im=True, stimid_map=stimid_map)

    def get_jump_pos(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values <= bar_jump_duration_s)
        data['jump_pos'] = fc.circmean(data['xstim'].values[idx])
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_jump_pos)

    def get_goal(data):
        idx = (data['trial_time'].values >= -5) & (data['trial_time'].values < 0)
        goal = fc.circmean(data['xstim'].values[idx])
        data['goal'] = goal
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_goal)

    def mean_xstim_after(data):
        idx = (data['trial_time'].values >= 5) & (data['trial_time'].values < (10))
        mean_xstim_after = fc.circmean(data['xstim'].values[idx])
        data['mean_xstim_after'] = mean_xstim_after
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(mean_xstim_after)
    abf_trials_df['abs_goal_diff'] = np.abs(fc.wrap(abf_trials_df['mean_xstim_after'] - abf_trials_df['goal']))

    def get_xstim_zeroed(data):
        idx = (data['trial_time'].values < 0)
        signal_zero = data['xstim'].values[idx][-1]
        signal_zero = fc.wrap(data['xstim'].values - signal_zero)
        data['xstim_zeroed'] = signal_zero
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_xstim_zeroed)

    def get_mean_turning_during(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values <= bar_jump_duration_s)
        data['mean_turning_during'] = np.mean(data['dheading'].values[idx])
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_mean_turning_during)

    def get_jump_in_fix(data):
        idx = np.argmin(np.abs(data['trial_time']))
        data['jump_in_fix'] = data['is_fixating'].values[idx]
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_jump_in_fix)

    def get_mean_rml_during(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values <= bar_jump_duration_s)
        data['mean_rml_during'] = np.mean(data['ps_c1_rml_z'].values[idx])
        return data

    im_trials_df = im_trials_df.groupby(['unique_trial_id']).apply(get_mean_rml_during)

    def get_jump_in_fix(data):
        idx = np.argmin(np.abs(data['trial_time']))
        data['jump_in_fix'] = data['is_fixating'].values[idx]
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_jump_in_fix)

    abf_trials_df['mean_rml_during'] = abf_trials_df['unique_trial_id'].map(
        dict(zip(im_trials_df['unique_trial_id'], im_trials_df['mean_rml_during'])))

    return abf_trials_df, im_trials_df


def get_rml_heading_during_bar_jumps(abf_trials_df, save=False, savepath=None, fname=None):
    summary_abf_trials_df = abf_trials_df.drop_duplicates('unique_trial_id')
    rml_dheading_during = summary_abf_trials_df.copy()
    query = '(jump_pos<=135) & (jump_pos >=-135) & (abs_goal_diff<45)  & jump_in_fix==True'
    valid_trials = rml_dheading_during.query(query)['unique_trial_id'].tolist()
    rml_dheading_during['selected'] = rml_dheading_during['unique_trial_id'].isin(valid_trials)
    rml_dheading_during = rml_dheading_during.filter(
        ['unique_trial_id', 'mean_rml_during', 'mean_turning_during', 'selected'])
    if save:
        rml_dheading_during.drop('unique_trial_id', axis=1).to_csv(savepath + fname)
    return rml_dheading_during


def plot_aligned_peaks(mean_peaks_df, rml_ylim_up, rml_ylim_down, turning_ylim,
                       save=False, savepath=None, fname=None):
    def plot(data):
        lal_color = 'k'
        dheading_color = 'grey'
        sub_data = data.query('variable=="ps_c1_rml_dF/F"')
        t = sub_data['t'].values
        mean = sub_data['mean'].values
        sem = sub_data['sem'].values
        ax = plt.gca()

        ax.plot(t, mean, color=lal_color, lw=0.5)
        ax.fill_between(t, (mean - sem), (mean + sem),
                        facecolor=ph.color_to_facecolor(lal_color), edgecolor='none')
        ax.tick_params(axis='y', colors=lal_color)
        ax.spines['left'].set_color(lal_color)
        ax.set_xlim([-3, 3])

        # manually chose y axis such that turning and rml peaks are roughly aligned
        if data['trial_type'].values[0] == "down":
            ax.set_ylim(rml_ylim_down)
        else:
            ax.set_ylim(rml_ylim_up)

        sub_data = data.query('variable=="dheading"')
        t = sub_data['t'].values
        mean = sub_data['mean'].values
        sem = sub_data['sem'].values
        ax2 = ax.twinx()
        ax2.plot(t, mean, color=dheading_color, lw=0.5)
        ax2.tick_params(axis='y', colors=dheading_color)
        ax2.spines['right'].set_color(dheading_color)
        ax2.fill_between(t, (mean - sem), (mean + sem),
                         facecolor=ph.color_to_facecolor(dheading_color), edgecolor='none')
        ax2.set_ylim(turning_ylim)
        ph.adjust_spines(ax2, ['right'], lw=axis_lw, ticklen=axis_ticklen, pad=pad)
        ph.adjust_spines(ax, ['left'], lw=axis_lw, ticklen=axis_ticklen, pad=pad)

        ax2.spines['right'].set_position(('outward', 25))

    g = ph.FacetGrid(mean_peaks_df, row='trial_type',
                     fig_kws={'figsize': [1.25, 2], 'dpi': dpi},
                     gridspec_kws={
                         'wspace': 0,
                         'hspace': 0.2},
                     row_order=['up', 'down', ]
                     )

    g.map_dataframe(plot)
    ax = g.axes[1, 0]
    ph.adjust_spines(ax, ['left', 'bottom'], lw=axis_lw, ticklen=axis_ticklen, pad=pad)
    ax.spines['bottom'].set_position(('outward', 25))

    # plt.show()
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_aligned_peaks_inset_up(mean_peaks_df, rml_ylim_up, turning_ylim, save=False, savepath=None, fname=None):
    rml_ylim_up_inset = [0.6, rml_ylim_up[1]]
    scale = (rml_ylim_up_inset[1] - rml_ylim_up_inset[0]) / (rml_ylim_up[1] - rml_ylim_up[0])
    turning_ylim_up_inset = [turning_ylim[1] - ((turning_ylim[1] - turning_ylim[0]) * scale), turning_ylim[1]]

    def plot(data):

        lal_color = 'k'
        dheading_color = 'grey'

        sub_data = data.query('variable=="ps_c1_rml_dF/F"')

        t = sub_data['t'].values
        mean = sub_data['mean'].values
        sem = sub_data['sem'].values
        ax = plt.gca()

        ax.plot(t, mean, color=lal_color, lw=0.5, )
        ax.fill_between(t, (mean - sem), (mean + sem),
                        facecolor=ph.color_to_facecolor(lal_color), edgecolor='none')
        ax.tick_params(axis='y', colors=lal_color)
        ax.spines['left'].set_color(lal_color)
        ax.spines['left'].set_position(("axes", -0.1))

        sub_data = data.query('variable=="dheading"')
        t = sub_data['t'].values
        mean = sub_data['mean'].values
        sem = sub_data['sem'].values
        ax2 = ax.twinx()
        ax2.yaxis.tick_left()
        ax2.spines['left'].set_position(("axes", -0.7))
        ax2.spines['left'].set_color(dheading_color)
        ax2.tick_params(axis='y', colors=dheading_color)

        ax2.plot(t, mean, color=dheading_color, lw=0.5)
        ax2.fill_between(t, (mean - sem), (mean + sem),
                         facecolor=ph.color_to_facecolor(dheading_color), edgecolor='none')
        ax2.set_xlim([-0.3, 0.3])

        trial_type = data['trial_type'].values[0]
        if trial_type == "down":
            pass

        else:
            ax.set_ylim(rml_ylim_up_inset)
            ax2.set_ylim(turning_ylim_up_inset)

        ph.adjust_spines(ax2, [], lw=axis_lw, ticklen=axis_ticklen)
        ph.adjust_spines(ax, [], lw=axis_lw, ticklen=axis_ticklen)

    g = ph.FacetGrid(mean_peaks_df.query('trial_type=="up"'),
                     unit='trial_type',
                     fig_kws={'figsize': [0.5, 0.5], 'dpi': dpi},
                     )

    g.map_dataframe(plot)
    ax = g.axes[0, 0]
    ph.adjust_spines(ax, [], lw=axis_lw, ticklen=axis_ticklen, pad=pad)

    ax.hlines(y=0.6, xmin=0, xmax=0.1, color='k', lw=1, clip_on=False)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_aligned_peaks_inset_down(mean_peaks_df, rml_ylim_down, turning_ylim, save=False, savepath=None, fname=None):
    rml_ylim_down_inset = [rml_ylim_down[0], -0.65, ]
    scale = (rml_ylim_down_inset[1] - rml_ylim_down_inset[0]) / (rml_ylim_down[1] - rml_ylim_down[0])
    turning_ylim_down_inset = [turning_ylim[0], turning_ylim[0] + ((turning_ylim[1] - turning_ylim[0]) * scale)]

    def plot(data):

        lal_color = 'k'
        dheading_color = 'grey'

        sub_data = data.query('variable=="ps_c1_rml_dF/F"')

        t = sub_data['t'].values
        mean = sub_data['mean'].values
        sem = sub_data['sem'].values
        ax = plt.gca()
        ax.plot(t, mean, color=lal_color, lw=0.5, )
        ax.fill_between(t, (mean - sem), (mean + sem),
                        facecolor=ph.color_to_facecolor(lal_color), edgecolor='none')
        ax.tick_params(axis='y', colors=lal_color)
        ax.spines['left'].set_color(lal_color)
        ax.spines['left'].set_position(("axes", -0.1))
        sub_data = data.query('variable=="dheading"')
        t = sub_data['t'].values
        mean = sub_data['mean'].values
        sem = sub_data['sem'].values
        ax2 = ax.twinx()
        ax2.yaxis.tick_left()
        ax2.spines['left'].set_position(("axes", -0.7))
        ax2.spines['left'].set_color(dheading_color)
        ax2.tick_params(axis='y', colors=dheading_color)

        ax2.plot(t, mean, color=dheading_color, lw=0.5)

        ax2.fill_between(t, (mean - sem), (mean + sem),
                         facecolor=ph.color_to_facecolor(dheading_color), edgecolor='none')
        ax2.set_xlim([-0.3, 0.3])

        trial_type = data['trial_type'].values[0]
        if trial_type == "down":
            ax.set_ylim(rml_ylim_down_inset)
            ax2.set_ylim(turning_ylim_down_inset)
        else:
            pass

        ph.adjust_spines(ax2, [], lw=axis_lw, ticklen=axis_ticklen)
        ph.adjust_spines(ax, [], lw=axis_lw, ticklen=axis_ticklen)

    g = ph.FacetGrid(mean_peaks_df.query('trial_type=="down"'),
                     unit='trial_type',
                     fig_kws={'figsize': [0.5, 0.5], 'dpi': dpi},
                     )

    g.map_dataframe(plot)
    ax = g.axes[0, 0]
    ph.adjust_spines(ax, [], lw=axis_lw, ticklen=axis_ticklen, pad=pad)
    ax.hlines(y=-0.8, xmin=0, xmax=0.1, color='k', lw=1, clip_on=False)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')

    # plt.show()


def plot_aligned_peaks_dist(mean_peaks_df, rml_ylim_up, rml_ylim_down, save=False, savepath=None, fname=None):
    # rml_ylim_up=[-0.1,0.8]
    # rml_ylim_down = [-0.8,0.1]
    dist_ylim_up = [-10, 60]
    dist_ylim_down = [-60, 10]

    def plot(data):

        lal_color = 'k'
        dist_color = '#008080'
        sub_data = data.query('variable=="ps_c1_rml_dF/F"')

        t = sub_data['t'].values
        mean = sub_data['mean'].values
        sem = sub_data['sem'].values
        ax = plt.gca()

        ax.plot(t, mean, color=lal_color, lw=0.5)
        ax.fill_between(t, (mean - sem), (mean + sem),
                        facecolor=ph.color_to_facecolor(lal_color), edgecolor='none')
        ax.tick_params(axis='y', colors=lal_color)
        ax.spines['left'].set_color(lal_color)
        ax.set_xlim([-3, 3])

        # flip for plotting purposes
        sub_data = data.query('variable=="distance_to_goal"')
        t = sub_data['t'].values
        mean = sub_data['mean'].values * -1
        sem = sub_data['sem'].values
        ax2 = ax.twinx()
        # manually chose y axis such that turning and rml peaks are roughly aligned
        if data['trial_type'].values[0] == "down":
            ax.set_ylim(rml_ylim_down)
            ax2.set_ylim(dist_ylim_down)
            ax2_yticks = ax2.get_yticks()
            ax2.set_yticklabels(ax2_yticks * -1)

        else:
            ax.set_ylim(rml_ylim_up)
            ax2.set_ylim(dist_ylim_up)
            ax2_yticks = ax2.get_yticks()
            ax2.set_yticklabels(ax2_yticks * -1)

        ph.circplot(t, mean, circ='y', color=dist_color, lw=0.5, ax=ax2)

        ax2.tick_params(axis='y', colors=dist_color)
        ax2.spines['right'].set_color(dist_color)

        ax2.fill_between(t, (mean - sem), (mean + sem),
                         facecolor=ph.color_to_facecolor(dist_color), edgecolor='none')

        ph.adjust_spines(ax2, ['right'], lw=axis_lw, ticklen=axis_ticklen, pad=pad)
        ph.adjust_spines(ax, ['left'], lw=axis_lw, ticklen=axis_ticklen, pad=pad)
        ax2.spines['right'].set_position(('outward', 25))

    g = ph.FacetGrid(mean_peaks_df, row='trial_type',
                     fig_kws={'figsize': [1.25, 2], 'dpi': dpi},
                     gridspec_kws={
                         'wspace': 0,
                         'hspace': 0.2},
                     row_order=['up', 'down', ]
                     )

    g.map_dataframe(plot)
    ax = g.axes[1, 0]
    ph.adjust_spines(ax, ['left', 'bottom'], lw=axis_lw, ticklen=axis_ticklen, pad=pad)
    ax.spines['bottom'].set_position(('outward', 25))
    # ax.spines['bottom'].set_position(('outward', 25))

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_aligned_peaks_dist_inset_down(mean_peaks_df, rml_ylim_down, save=False, savepath=None, fname=None):
    dist_ylim = [-60.5, 0]

    rml_ylim_down_inset = [rml_ylim_down[0], -0.65, ]
    scale = (rml_ylim_down_inset[1] - rml_ylim_down_inset[0]) / (rml_ylim_down[1] - rml_ylim_down[0])
    dist_ylim_down_inset = [dist_ylim[0], dist_ylim[0] + ((dist_ylim[1] - dist_ylim[0]) * scale)]

    def plot(data):

        lal_color = 'k'
        dist_color = '#008080'

        sub_data = data.query('variable=="ps_c1_rml_dF/F"')

        t = sub_data['t'].values
        mean = sub_data['mean'].values
        sem = sub_data['sem'].values
        ax = plt.gca()
        ax.plot(t, mean, color=lal_color, lw=0.5, )
        ax.fill_between(t, (mean - sem), (mean + sem),
                        facecolor=ph.color_to_facecolor(lal_color), edgecolor='none')
        ax.tick_params(axis='y', colors=lal_color)
        ax.spines['left'].set_color(lal_color)
        ax.spines['left'].set_position(("axes", -0.1))
        sub_data = data.query('variable=="distance_to_goal"')
        t = sub_data['t'].values
        mean = sub_data['mean'].values * -1  # flip for plotting
        sem = sub_data['sem'].values
        ax2 = ax.twinx()
        ax2.yaxis.tick_left()
        ax2.spines['left'].set_position(("axes", -0.7))
        ax2.spines['left'].set_color(dist_color)
        ax2.tick_params(axis='y', colors=dist_color)

        ph.circplot(t, mean, circ='y', color=dist_color, lw=0.5, ax=ax2)

        ax2.fill_between(t, (mean - sem), (mean + sem),
                         facecolor=ph.color_to_facecolor(dist_color), edgecolor='none')
        ax2.set_xlim([-0.6, 0.6])

        trial_type = data['trial_type'].values[0]
        if trial_type == "down":
            ax.set_ylim(rml_ylim_down_inset)
            ax2.set_ylim(dist_ylim_down_inset)

        else:
            pass

        ph.adjust_spines(ax2, [], lw=axis_lw, ticklen=axis_ticklen)
        ph.adjust_spines(ax, [], lw=axis_lw, ticklen=axis_ticklen)

    g = ph.FacetGrid(mean_peaks_df.query('trial_type=="down"'),
                     unit='trial_type',
                     fig_kws={'figsize': [1, 0.5], 'dpi': dpi},
                     )

    g.map_dataframe(plot)
    ax = g.axes[0, 0]
    ph.adjust_spines(ax, [], lw=axis_lw, ticklen=axis_ticklen, pad=pad)
    ax.hlines(y=-0.8, xmin=0, xmax=0.2, color='k', lw=1, clip_on=False)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_aligned_peaks_dist_inset_up(mean_peaks_df, rml_ylim_up, save=False, savepath=None, fname=None):
    dist_ylim = [0, 60.5]

    rml_ylim_up_inset = [0.6, rml_ylim_up[1]]
    scale = (rml_ylim_up_inset[1] - rml_ylim_up_inset[0]) / (rml_ylim_up[1] - rml_ylim_up[0])
    dist_ylim_up_inset = [dist_ylim[1] - ((dist_ylim[1] - dist_ylim[0]) * scale), dist_ylim[1]]

    def plot(data):

        lal_color = 'k'
        dist_color = '#008080'

        sub_data = data.query('variable=="ps_c1_rml_dF/F"')

        t = sub_data['t'].values
        mean = sub_data['mean'].values
        sem = sub_data['sem'].values
        ax = plt.gca()
        ax.plot(t, mean, color=lal_color, lw=0.5, )
        ax.fill_between(t, (mean - sem), (mean + sem),
                        facecolor=ph.color_to_facecolor(lal_color), edgecolor='none')
        ax.tick_params(axis='y', colors=lal_color)
        ax.spines['left'].set_color(lal_color)
        ax.spines['left'].set_position(("axes", -0.1))
        sub_data = data.query('variable=="distance_to_goal"')
        t = sub_data['t'].values
        mean = sub_data['mean'].values * -1  # flip for plotting
        sem = sub_data['sem'].values
        ax2 = ax.twinx()
        ax2.yaxis.tick_left()
        ax2.spines['left'].set_position(("axes", -0.7))
        ax2.spines['left'].set_color(dist_color)
        ax2.tick_params(axis='y', colors=dist_color)

        ph.circplot(t, mean, circ='y', color=dist_color, lw=0.5, ax=ax2)

        ax2.fill_between(t, (mean - sem), (mean + sem),
                         facecolor=ph.color_to_facecolor(dist_color), edgecolor='none')
        ax2.set_xlim([-0.6, 0.6])

        trial_type = data['trial_type'].values[0]
        if trial_type == "up":
            ax.set_ylim(rml_ylim_up_inset)
            ax2.set_ylim(dist_ylim_up_inset)

        else:
            pass

        ph.adjust_spines(ax2, [], lw=axis_lw, ticklen=axis_ticklen)
        ph.adjust_spines(ax, [], lw=axis_lw, ticklen=axis_ticklen)

    g = ph.FacetGrid(mean_peaks_df.query('trial_type=="up"'),
                     unit='trial_type',
                     fig_kws={'figsize': [1, 0.5], 'dpi': dpi},
                     )

    g.map_dataframe(plot)
    ax = g.axes[0, 0]
    ph.adjust_spines(ax, [], lw=axis_lw, ticklen=axis_ticklen, pad=pad)
    ax.hlines(y=0.59, xmin=0, xmax=0.2, color='k', lw=1, clip_on=False)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_peak(im_peaks_df, abf_peaks_df, trial_id, fig_kws=None, adjust_spines_kws=None, tline_s=5,
                      save=False, savepath=None, fname=None):
    if fig_kws is None:
        fig_kws = {'figsize': [1.25, 2.25], 'dpi': dpi}
    if adjust_spines_kws is None:
        adjust_spines_kws = {'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2}

    fig = plt.figure(1, **fig_kws)
    gs = gridspec.GridSpec(figure=fig, nrows=5, ncols=1,
                           hspace=0.4,
                           height_ratios=[0.225, 0.225, 0.225, 0.1, 0.225],
                           )

    sub_im_df = im_peaks_df.query(f'trial_id=={trial_id}')
    sub_abf_df = abf_peaks_df.query(f'trial_id=={trial_id}')

    ax1 = plt.subplot(gs[2, 0])
    ax1.set_ylim([-1, 1])
    ax1.set_yticks([-1, 0, 1])
    ax1.plot(sub_im_df['t_abf'], sub_im_df['ps_c1_rml_dF/F'], lw=0.5, color='k', clip_on=False)

    rec_name = im_peaks_df.query(f'trial_id=={trial_id}')['rec_name'].values[0]
    start = im_peaks_df.query(f'trial_id=={trial_id}')['start'].values[0]
    end = im_peaks_df.query(f'trial_id=={trial_id}')['end'].values[0]
    rec_df = im_peaks_df.query(f'rec_name=="{rec_name}" & t_abf>={start} & t_abf <={end}')
    rec_df = rec_df[rec_df['t_0'] == rec_df['t_abf']]
    plt.xlim([start, end])
    up = rec_df.query('trial_type=="up"')
    down = rec_df.query('trial_type=="down"')
    plt.scatter(up['t_0'], up['ps_c1_rml_dF/F'] + 0.25, s=0.5, marker='o', color='#b51700', zorder=2.5, clip_on=False)
    plt.scatter(down['t_0'], down['ps_c1_rml_dF/F'] - 0.25, s=0.5, marker='o', color='#1c75bc', zorder=2.5,
                clip_on=False)

    ph.despine_axes([ax1], style=['left'], adjust_spines_kws=adjust_spines_kws)

    ax2 = plt.subplot(gs[0, 0], sharex=ax1)
    ax2.set_ylim([0, 1])
    ax2.set_yticks([0, 1])
    ax2.plot(sub_im_df['t_abf'], sub_im_df['ps_c1_roi_2_dF/F'], lw=0.5, color='#b51700', clip_on=False)
    ph.despine_axes([ax2], style=['left'], adjust_spines_kws=adjust_spines_kws)

    ax3 = plt.subplot(gs[1, 0], sharex=ax1)
    ax3.set_ylim([0, 1])
    ax3.set_yticks([0, 1])
    ax3.plot(sub_im_df['t_abf'], sub_im_df['ps_c1_roi_1_dF/F'], lw=0.5, color='#1c75bc', clip_on=False)
    ph.despine_axes([ax3], style=['left'], adjust_spines_kws=adjust_spines_kws)

    # dheading
    ax4 = plt.subplot(gs[3, 0], sharex=ax1)
    ax4.plot(sub_abf_df['t'], sub_abf_df['dheading_boxcar_average_0.5_s'], color='#5e5e5e', lw=0.5, clip_on=False)
    ax4.fill_between(x=sub_abf_df['t'],
                     y1=np.zeros(len(sub_abf_df['t'])),
                     y2=sub_abf_df['dheading_boxcar_average_0.5_s'],
                     color=ph.color_to_facecolor('#5e5e5e'), lw=0)
    ax4.set_ylim([-200, 200])
    ax4.set_yticks([-200, 0, 200])
    ph.despine_axes([ax4], style=['left'], adjust_spines_kws=adjust_spines_kws)

    ax5 = plt.subplot(gs[4, 0], sharex=ax1)
    # negative bc we want virtual heading
    ph.circplot(sub_abf_df['t'], -sub_abf_df['xstim'], circ='y', color='k', zorder=1, lw=0.5)
    ax5.axhspan(-180, -135, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0)
    ax5.axhspan(135, 180, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0)
    ax5.set_yticks([-180, 0, 180])
    ax5.set_ylim([-180, 180])
    ph.despine_axes([ax5], style=['left'], adjust_spines_kws=adjust_spines_kws)

    sb = ph.add_scalebar(ax5, sizex=tline_s, sizey=0, barwidth=0.4, barcolor='k', loc='lower center',
                         pad=-1,
                         bbox_transform=ax5.transAxes,
                         )
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_lal_vs_distance_to_goal(mean_df, flies_df, save=False, savepath=None, fname=None):
    print(len(flies_df['fly_id'].unique()), 'flies')

    color_dic = {'shifted_ps_c1_roi_1_dF/F': "#1c75bc",
                 'shifted_ps_c1_roi_2_dF/F': "#b51700",
                 'shifted_ps_c1_rml_dF/F': 'k'
                 }

    def plot(data):
        x = data['distance_to_goal_bin_center'].values
        y = data['mean_value'].values
        err = data['std_err_value'].values
        variable = data['variable'].iloc[0]
        color = color_dic[variable]
        plt.plot(x, y, color=color, lw=1, clip_on=False)

        ax = plt.gca()

        ax.fill_between(x, y1=y - err, y2=y + err, color=ph.color_to_facecolor(color),
                        edgecolor=None, clip_on=False, )

        ax.axvline(x=0, ls=':', color='k', lw=0.5, zorder=0)

        if variable == "shifted_ps_c1_rml_dF/F":
            ax.set_ylim([-0.5, 0.5])
            ax.set_yticks([-0.5, 0, 0.5])
            ax.axhline(y=0, ls=':', color='k', lw=0.5)
        else:
            ax.set_ylim([0, 0.75])
            ax.set_yticks([0, 0.75])

        ax.set_xlim([-180, 180])
        ax.set_xticks([-180, -90, 0, 90, 180])

    g = ph.FacetGrid(mean_df, row='variable',

                     #                unit='variable',
                     fig_kws={'figsize': [1, 2.5], 'dpi': dpi},
                     gridspec_kws={
                         #                       'height_ratios':[0.5,0.25,0.25],
                         'wspace': 0,
                         'hspace': 0.25},
                     row_order=['shifted_ps_c1_roi_2_dF/F', 'shifted_ps_c1_roi_1_dF/F', 'shifted_ps_c1_rml_dF/F', ]
                     )

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2})
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_lal_vs_distance_to_goal_model(mean_df, fit_path, save=False, savepath=None, fname=None):
    right_LAL = pd.read_csv(fit_path + 'RightLALFit.csv', names=['right']).values.flatten().tolist()
    left_LAL = pd.read_csv(fit_path + 'LeftLALFit.csv', names=['left']).values.flatten().tolist()
    rml_LAL = pd.read_csv(fit_path + 'TurningFit.csv', names=['rml']).values.flatten().tolist()
    LAL_results = pd.DataFrame({'right': right_LAL, 'left': left_LAL, 'rml': rml_LAL})

    color_dic = {'shifted_ps_c1_roi_1_dF/F': "#1c75bc",
                 'shifted_ps_c1_roi_2_dF/F': "#b51700",
                 'shifted_ps_c1_rml_dF/F': 'k'
                 }

    def plot(data):
        x = data['distance_to_goal_bin_center'].values
        y = data['mean_value'].values
        variable = data['variable'].iloc[0]
        color = color_dic[variable]
        plt.scatter(x, y, s=5,
                    facecolors='none', edgecolors=color, clip_on=False, lw=0.5)

        ax = plt.gca()

        #     ax.fill_between(x, y1=y-err, y2=y+err, color=ph.color_to_facecolor(color),
        #                     edgecolor=None,clip_on=False,)

        ax.axvline(x=0, ls=':', color='k', lw=0.5, zorder=0)
        if variable == "shifted_ps_c1_rml_dF/F":
            plt.plot(np.arange(-180, 180, 1), LAL_results['rml'], color=color, lw=1)
            ax.set_ylim([-0.5, 0.5])
            ax.set_yticks([-0.5, 0, 0.5])
            ax.axhline(y=0, ls=':', color='k', lw=0.5)
            ph.adjust_spines(ax, ['left', 'bottom'], lw=axis_lw, ticklen=axis_ticklen, pad=pad)

        elif variable == "shifted_ps_c1_roi_1_dF/F":
            ax.set_ylim([0, 0.75])
            ax.set_yticks([0, 0.75])
            plt.plot(np.arange(-180, 180, 1), LAL_results['left'], color=color, lw=1)
            ph.adjust_spines(ax, ['left', 'bottom'], lw=axis_lw, ticklen=axis_ticklen, pad=pad)


        else:
            ax.set_ylim([0, 0.75])
            ax.set_yticks([0, 0.75])
            plt.plot(np.arange(-180, 180, 1), LAL_results['right'], color=color, lw=1)
            ph.adjust_spines(ax, ['left', 'bottom'], lw=axis_lw, ticklen=axis_ticklen, pad=pad)
        ax.set_xlim([-180, 180])
        ax.set_xticks([-180, -90, 0, 90, 180])

    g = ph.FacetGrid(mean_df, col='variable',

                     #                unit='variable',
                     fig_kws={'figsize': [3.25, 0.8], 'dpi': dpi},
                     gridspec_kws={
                         #                       'height_ratios':[0.5,0.25,0.25],
                         'wspace': 0.5,
                         'hspace': 0},
                     col_order=['shifted_ps_c1_roi_1_dF/F', 'shifted_ps_c1_rml_dF/F', 'shifted_ps_c1_roi_2_dF/F']
                     )
    g.map_dataframe(plot)
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_rml_vs_turning_bar_jumps(rml_dheading_during, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(rml_dheading_during,
                     unit='unique_trial_id',
                     fig_kws={'figsize': [1, 1], 'dpi': dpi}
                     )

    def plot(data):
        x = data['mean_rml_during'].values
        y = data['mean_turning_during'].values
        selected = data['selected'].values[0]

        if selected:
            #         color='#009c46'
            color = 'k'

            plt.scatter(x, y, s=4, color=color, linewidths=0, clip_on=False, zorder=3)

        else:
            color = 'grey'
            plt.scatter(x, y, s=4, color=color, linewidths=0, clip_on=False, alpha=0.75)

        plt.ylim([-150, 150])
        plt.yticks([-150, 0, 150])
        plt.xlim([-4, 4])
        plt.xticks([-4, 0, 4])

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['left', 'bottom'],
                    adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen})

    ax = plt.gca()

    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(4))
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(3))
    ax.grid(b=True, which='major', color='w', linewidth=1.0, alpha=0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5, alpha=0)

    trials = rml_dheading_during.copy()
    x = trials['mean_rml_during'].astype(float)
    y = trials['mean_turning_during']
    print(np.corrcoef(x, y)[0][1])
    trials = rml_dheading_during.query('selected==True').copy()

    x = trials['mean_rml_during'].astype(float)
    y = trials['mean_turning_during']
    print(np.corrcoef(x, y)[0][1])

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


# ---------------------------- Save processed data ---------------------------- #
def save_processed_data(PROCESSED_DATA_PATH, recs):
    def save_abf_as_hd5f(data):
        df = data.copy()
        rec_name = df['rec_name'].values[0]
        genotype = df['genotype'].values[0]
        df = df.filter([
            't',
            'heading',
            'side',
            'forw',
            'xstim',
            'stimid',
            'temp',

        ]).reset_index(drop=True)
        df.to_hdf(PROCESSED_DATA_PATH + genotype + os.path.sep + rec_name + '_abf.h5',
                  key='df', mode='w')

    recs.merged_abf_df.groupby(['rec_name']).apply(save_abf_as_hd5f)

    def save_im_as_hd5f(data):
        df = data.copy()
        rec_name = df['rec_name'].values[0]
        genotype = df['genotype'].values[0]
        df = df.filter([
            't_abf',
            'ps_c1_roi_1_F',
            'ps_c1_roi_2_F',
            'ps_c2_roi_1_F',
            'ps_c2_roi_2_F'
        ]).rename({'t_abf': 't'}, axis=1).reset_index(drop=True)
        df.to_hdf(PROCESSED_DATA_PATH + genotype + os.path.sep + rec_name + '_im.h5',
                  key='df', mode='w')

    recs.merged_im_df.groupby(['rec_name']).apply(save_im_as_hd5f)

    summary_recs = recs.merged_im_df.drop_duplicates('rec_name').copy().sort_values(['fly_id']).filter(
        ['rec_name', 'fly_id']).reset_index(drop=True)
    summary_recs.to_csv(PROCESSED_DATA_PATH + 'summary.csv')
