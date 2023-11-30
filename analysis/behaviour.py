import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from simplification.cutil import simplify_coords

import scipy as sc
import seaborn as sns

import analysis_plot as ap
import functions as fc
import plotting_help as ph

# ------------------ Plotting parameters ----------------------- #
dpi = 300
axis_lw = 0.4
axis_ticklen = 2.5
font = {'family': 'arial',
        'weight': 'normal',
        'size': 5}
mpl.rc('font', **font)


def get_trials(recs, pad_s, stimid_map, bar_jump_duration_s, im=False):
    abf_trials_df = ap.get_trials_df(recs, pad_s=pad_s, im=im, stimid_map=stimid_map)

    abf_trials_df['vh'] = abf_trials_df['xstim'] * -1

    # changed to filter on forward walking!!!
    def get_std_before(data):
        idx = (data['trial_time'].values >= -pad_s) & (data['trial_time'].values < 0) & (
                data['dforw_boxcar_average_0.5_s'].values > 1)
        std_before = fc.circstd(data['xstim'].values[idx])
        data['std_before'] = std_before
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_std_before)

    def get_std_after(data):
        idx = (data['trial_time'].values >= bar_jump_duration_s) & (
                data['trial_time'].values < (pad_s + bar_jump_duration_s)) & (
                      data['dforw_boxcar_average_0.5_s'].values > 1)
        std_before = fc.circstd(data['xstim'].values[idx])
        data['std_after'] = std_before
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_std_after)

    def get_goal(data):
        idx = (data['trial_time'].values >= -pad_s) & (
                data['trial_time'].values < 0 & (data['dforw_boxcar_average_0.5_s'].values > 1))
        goal = fc.circmean(data['vh'].values[idx])
        data['goal'] = goal
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_goal)
    abf_trials_df['distance_to_goal'] = fc.wrap(abf_trials_df['vh'] - abf_trials_df['goal'])

    def get_jump_pos(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values < bar_jump_duration_s)
        jump_pos = fc.circmean(data['xstim'].values[idx])
        data['jump_pos'] = jump_pos
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_jump_pos)

    def mean_vh_after(data):
        idx = (data['trial_time'].values >= bar_jump_duration_s) & (
                data['trial_time'].values < (pad_s + bar_jump_duration_s)) & (
                      data['dforw_boxcar_average_0.5_s'].values > 1)
        mean_vh_after = fc.circmean(data['vh'].values[idx])
        data['mean_vh_after'] = mean_vh_after
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(mean_vh_after)
    abf_trials_df['abs_goal_diff'] = np.abs(fc.wrap(abf_trials_df['mean_vh_after'] - abf_trials_df['goal']))

    abf_trials_df['goal_diff'] = fc.wrap(abf_trials_df['mean_vh_after'] - abf_trials_df['goal'])

    def get_distance_to_goal_after(data):
        idx = (data['trial_time'].values >= bar_jump_duration_s + 30) & (
                data['trial_time'].values < (pad_s + bar_jump_duration_s)) & (
                      data['dforw_boxcar_average_0.5_s'].values > 1)

        data['distance_to_goal_after'] = fc.circmean(data['distance_to_goal'].values[idx])

        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_distance_to_goal_after)

    return abf_trials_df


def get_selected_trials(abf_trials_df, criteria='std_before<45'):
    selected_trials_df = abf_trials_df.query(criteria).copy()
    # we flip +/- 90º because we plot heading relative goal, not bar position
    selected_trials_df['trial_type'] = selected_trials_df['trial_type'].map(
        {'+90 jump': '-90 jump', '-90 jump': '+90 jump'})
    n_valid_trials = len(np.unique(selected_trials_df['unique_trial_id']))
    n_trials = len(np.unique(abf_trials_df['unique_trial_id']))
    print('fraction of valid trials:', n_valid_trials / n_trials)

    return selected_trials_df


def get_mean_goal_diff(trials_df, save=False, savepath=None, fname=None):
    summary_df = trials_df.drop_duplicates(['unique_trial_id'])
    trial_fly_df = summary_df.groupby(['genotype', 'fly_id', 'trial_type'], observed=True).apply(
        lambda x: fc.circmean(x['distance_to_goal_after']))
    trial_fly_df = pd.DataFrame(trial_fly_df).reset_index()
    trial_fly_df = trial_fly_df.rename({0: 'distance_to_goal_after'}, axis=1)
    print('Total flies:\n', trial_fly_df.dropna(axis=0).groupby(['trial_type']).apply(lambda x: len(x)))
    mean_trial_fly_df = trial_fly_df.groupby(['trial_type'], as_index=False).agg(
        mean=('distance_to_goal_after', fc.circmean),
        sem=('distance_to_goal_after', fc.circ_stderror))
    if save:
        trial_fly_df.to_csv(savepath + fname)
    return mean_trial_fly_df, trial_fly_df


# this funciton is different than
# the function with the same name in analysis_plot
# it does not create a value for each timepoint but instead
# uses a step size, might want to consolidate?
def get_sliding_mean_vector(recs, window, step_s=None):
    def sliding_window(data, window_s, sampling_rate, step_s):

        idx = (data['dforw_boxcar_average_0.5_s'].values > 1)
        signal = data['xstim'].values[idx]
        inds = np.arange(len(signal))

        if step_s is not None:
            step_ind = int(np.round(step_s * sampling_rate))
            inds = inds[::step_ind]

        half_window_s = window_s / 2.
        # bc an integer number of indices is used, the window is not necessarily exact
        half_window_ind = int(np.round(half_window_s * sampling_rate))
        start = inds - half_window_ind
        end = inds + half_window_ind + 1

        r = np.zeros(len(inds))
        theta = np.zeros(len(inds))
        # prevents wrapping, but means that some windows are shorter
        start[start < 0] = 0
        end[end > len(signal)] = len(signal)
        for i in range(len(inds)):
            t0 = start[i]
            t1 = end[i]
            r[i], theta[i] = fc.mean_vector(np.deg2rad(signal[t0:t1]))

        return pd.Series({'r': r, 'theta': theta, 'window_s': window_s})

    sampling_rates = [np.round(rec.rec.abf.subsampling_rate) for rec in recs]
    if (len(set(sampling_rates)) != 1):
        print('samplig rates are different!')
        return
    sampling_rate = sampling_rates[0]
    temp_df = recs.merged_abf_df.groupby(['rec_name', 'fly_id']).apply(lambda x: sliding_window(x,
                                                                                                window,
                                                                                                sampling_rate,
                                                                                                step_s))

    def unnesting(df, explode):
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat([
            pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
        df1.index = idx
        return df1.join(df.drop(explode, 1), how='left')

    temp_df = unnesting(temp_df, ['r', 'theta'])
    temp_df.reset_index(inplace=True, drop=False)

    return temp_df


def get_sliding_genotypes(genotypes):
    windows = np.arange(5, 305, 5)
    window_dfs = []

    for genotype, recs in genotypes.items():
        for window in windows:
            window_df = get_sliding_mean_vector(recs, window, step_s=5)
            window_df['genotype'] = genotype
            window_dfs.append(window_df)

    df = pd.concat(window_dfs)
    df['unique_fly_id'] = df['genotype'] + '_' + df['fly_id'].astype(str)
    # do not use unequal bins
    r_bins = np.linspace(0, 1, 21)
    hist_df = ap.get_hist_df(df, 'r', query=None, id_vars=['genotype', 'unique_fly_id', 'window_s'], bins=r_bins)

    # get mean across flies
    geno_hist_df = hist_df.groupby(['r_bin_center', 'window_s'])['norm_counts'].mean()
    geno_hist_df = pd.DataFrame(geno_hist_df).reset_index()

    def get_norm(data):
        data['norm_counts'] = data['norm_counts'] / np.sum(data['norm_counts'])
        dx = np.diff(r_bins)[0]
        data['norm_counts'] = data['norm_counts'] / dx
        return data

    geno_hist_df = geno_hist_df.groupby(['window_s']).apply(lambda x: get_norm(x))

    geno_hist_df['r_bin_center'] = geno_hist_df['r_bin_center'].astype(float)

    # df is raw mean vectors, hist_df is "2D hist" for each fly and  geno_hist_df is "2D hist"
    # data are normalized across column, i.e. r
    return df, hist_df, geno_hist_df


def get_dforw_bar_jump_df(selected_trials_df):
    t_bins = np.linspace(-60, 63, 124 * 50)
    # adds trial_time new
    selected_trials_df['trial_time_new'] = pd.cut(selected_trials_df['trial_time'], t_bins).apply(lambda x: x.mid)

    melted_abf_trials_df = pd.melt(selected_trials_df,
                                   id_vars=['fly_id', 'unique_trial_id', 'trial_time_new'],
                                   value_vars=['dforw']
                                   )

    mean_dforw_df = melted_abf_trials_df.groupby(['fly_id', 'trial_time_new']).agg(mean=('value', np.nanmean), sem=(
        'value', lambda x: sc.stats.sem(x, nan_policy='omit')))
    mean_dforw_df.reset_index(inplace=True)

    print(len(mean_dforw_df['fly_id'].unique()), 'flies')

    fly_mean_dforw_df = mean_dforw_df.groupby(['trial_time_new']).agg(mean=('mean', np.nanmean), sem=(
        'mean', lambda x: sc.stats.sem(x, nan_policy='omit')))
    fly_mean_dforw_df.reset_index(inplace=True)

    return fly_mean_dforw_df


def plot_fixation_bouts_ed(abf_fixation_df, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(abf_fixation_df.query('is_fixating==True'),
                     unit='unique_fixation_event_id',
                     fig_kws={'figsize': [2, 2], 'dpi': dpi}
                     )

    def plot(data):
        x = data['x'].values
        y = data['y'].values
        x = x - x[0]
        y = y - y[0]
        zorder = 1
        plt.plot(x, y, lw=0.5, color='k', alpha=0.2, clip_on=False)
        plt.scatter(0, 0, s=2, color='r', zorder=3)

    g.map_dataframe(plot)
    ax = plt.gca()

    # because clip_on is false, if a point is outside this range it will still be plotted,
    # but if all data is within the x and y lim it will make sure that scale bar are the same length
    # across different plot
    plt.xlim([-900, 900])
    plt.ylim([-900, 900])

    ax.set_aspect('equal')
    ax.axis('off')
    sb = ph.add_scalebar(ax, sizex=200, sizey=0, barwidth=axis_lw * 2, barcolor='k',
                         #                         loc='lower center',
                         pad=0,
                         bbox_transform=ax.transAxes,
                         bbox_to_anchor=[0.65, 0.2]
                         )
    print(abf_fixation_df.query('is_fixating==True ').groupby(['genotype']).apply(lambda x: len(x['fly_id'].unique())))
    print(len(abf_fixation_df.query('is_fixating==True ')['unique_fixation_event_id'].unique()), ' bouts')

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_goal_diff(mean_trial_fly_df, trial_fly_df, save=False, savepath=None, fname=None):
    fig = plt.figure(1, [0.5, 0.5], dpi=dpi)
    ax = sns.swarmplot(x="trial_type", y="distance_to_goal_after", data=trial_fly_df, size=1.5,
                       order=["-90 jump", "+90 jump"],
                       marker='o', alpha=1, color='k', clip_on=False)

    lw = 0.5
    xvals = pd.Categorical(mean_trial_fly_df['trial_type'].values, ["-90 jump", "+90 jump"]).argsort()
    e1 = plt.errorbar(x=xvals + 0.3, y=mean_trial_fly_df['mean'].values,
                      yerr=mean_trial_fly_df['sem'].values,
                      ls='none', elinewidth=lw, ecolor='k', capsize=1.5, capthick=lw, zorder=10)

    e2 = plt.errorbar(x=xvals + 0.3, y=mean_trial_fly_df['mean'].values,
                      xerr=0.2, ls='none', elinewidth=lw, ecolor='k', capsize=None, zorder=10)

    ph.error_bar_clip_on([e1, e2])
    fig.set_size_inches(0.5, 1.5)

    plt.ylim([-180, 180])
    plt.yticks([-180, -90, 0, 90, 180])
    plt.ylabel('')
    ph.adjust_spines(ax, spines=['left'], lw=axis_lw, ticklen=axis_ticklen, pad=2)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_mean_vector_2D_hist(mean_df, name, save=False, savepath=None):
    matrix_df = mean_df.pivot_table(
        index=[
            'r_bin_center',
        ],
        columns=[
            'window_s',
        ],
        values='norm_counts')

    plt.figure(1, [1, 1], dpi=dpi)

    ax = plt.gca()

    x = matrix_df.columns.tolist()
    step = np.diff(x)[0]
    x = x - (step / 2.)
    x = np.append(x, x[-1] + step)

    y = np.linspace(0, 1, 21)

    vmin = 0
    vmax = 5
    cmap = sns.color_palette("rocket", as_cmap=True)
    # cmap=plt.cm.hot

    pc = ax.pcolormesh(
        x, y, matrix_df.values,
        cmap=cmap,
        #                 edgecolors='grey',linewidth=0.1,
        clip_on=False, vmin=vmin, vmax=vmax
    )

    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlim([0, 300])

    ph.adjust_spines(ax, ['left', 'bottom'], pad=1, lw=axis_lw, ticklen=axis_ticklen)
    if save:
        plt.savefig(savepath + name + '_mean_vector_2D_hist.pdf', transparent=True, bbox_inches='tight')
    plt.show()

    fig = plt.figure(1, [0.25, 0.05], dpi=dpi)

    cb = mpl.colorbar.ColorbarBase(plt.gca(), orientation='horizontal',
                                   cmap=cmap
                                   )

    cb.ax.tick_params(size=2.5, width=0.4)
    cb.outline.set_visible(False)
    cb.ax.set_xticklabels([np.round(vmin, 2), np.round(vmax, 2)])

    if save:
        plt.savefig(savepath + name + '_mean_vector_2D_hist_cbar.pdf',
                    transparent=True, bbox_inches='tight')


def plot_menotaxis_bout_anlges(summary_menotaxis_df, save=False, savepath=None, fname=None):
    plt.figure(1, [1, 2], dpi=dpi)
    ax = plt.gca()
    cats = summary_menotaxis_df['unique_fly_id'].cat.categories
    id = np.arange(len(cats))
    cat2id = dict(zip(cats, id))
    summary_menotaxis_df['id'] = summary_menotaxis_df['unique_fly_id'].map(cat2id)
    ax.scatter(summary_menotaxis_df['goal'], summary_menotaxis_df['id'], s=5, alpha=0.5, color='k', linewidths=0,
               clip_on=False)
    ax.set_yticks(id)
    ax.set_ylim([id.min(), id.max()])
    ax.set_xlim([-180, 180])
    ax.set_xticks([-180, -90, 0, 90, 180])
    fly_nb = [cat.split('_')[-1] for cat in cats]
    ax.set_yticklabels(fly_nb)
    ph.adjust_spines(ax, ['left', 'bottom', ], pad=2, lw=axis_lw, ticklen=axis_ticklen)
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_menotaxis_bout_raster(matrix_df, name, save=False, savepath=None):
    columns = matrix_df.columns.get_level_values(0).astype(int)
    ncols = len(columns)
    rows = matrix_df.index.get_level_values(1)
    nrows = len(rows)

    plt.figure(1, [0.0004 * ncols, nrows * 0.07], dpi=dpi)

    # plt.figure(1,[10,10])

    ax = plt.gca()
    cmap = plt.cm.Blues

    cmap.set_bad('#f0f0f0')

    pc = ax.pcolormesh(
        np.round(matrix_df.values),
        cmap=cmap,
    )

    x = np.arange(len(columns)) + 0.5

    ax.set_xticks(x[::500])
    ax.set_xticklabels(np.round(columns[::500]))

    ph.adjust_spines(ax, ['left', 'top', 'right', 'bottom'], pad=0, lw=axis_lw, ticklen=axis_ticklen)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows, fontsize=4)

    plt.xticks(rotation=45)

    if save:
        plt.savefig(savepath + name + '_is_fixating.pdf',
                    transparent=True, bbox_inches='tight')


def plot_example_trajectory_bare(recs, rec_name, save=False, savepath=None, fname=None):
    x = recs[rec_name].abf.df['x']
    y = recs[rec_name].abf.df['y']

    plt.figure(1, [2, 2], dpi=dpi)
    plt.scatter(x, y, s=0.5, color='#a8a8a8', clip_on=False)

    plt.scatter(0, 0, s=2, color='r', zorder=3, clip_on=False)

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

    sb = ph.add_scalebar(ax, sizex=200, sizey=0, barwidth=axis_lw * 2, barcolor='k', loc=3,
                         pad=0,
                         bbox_transform=ax.transAxes,
                         bbox_to_anchor=[0.8, 0.1]
                         )
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_trajectory_RDP(recs, rec_name, save=False, savepath=None, fname=None):
    x = recs[rec_name].abf.df['x']
    y = recs[rec_name].abf.df['y']

    plt.figure(1, [2, 2], dpi=dpi)
    plt.scatter(x, y, s=0.5, color='#a8a8a8', clip_on=False)

    simplified_coords = simplify_coords(list(zip(x, y)), 25)
    simplified_coords = np.array(list(zip(*simplified_coords)))

    plt.scatter(simplified_coords[0], simplified_coords[1], s=2, color='k', clip_on=False)

    plt.plot(simplified_coords[0], simplified_coords[1], lw=0.5, color='k', clip_on=False)
    # plt.scatter(0,0,s=2,color='r',zorder=3)

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

    sb = ph.add_scalebar(ax, sizex=200, sizey=0, barwidth=axis_lw * 2, barcolor='k', loc=3,
                         pad=0,
                         bbox_transform=ax.transAxes,
                         bbox_to_anchor=[0.8, 0.1]
                         )

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_trajectory_menotaxis(abf_fixation_df, rec_name, save=False, savepath=None, fname=None):
    color_palette_dict = {
        50.0: (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
        51.0: (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
        52.0: (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
        53.0: (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),
        54.0: (0.4, 0.7607843137254902, 0.6470588235294118)}

    g = ph.FacetGrid(abf_fixation_df.query(f'rec_name=="{rec_name}"'),
                     unit='fixation_event_id',
                     fig_kws={'figsize': [2, 2], 'dpi': dpi}
                     )

    def plot(data):

        x = data['x'].values
        y = data['y'].values
        #     goal_to_pd_distance_bin_center=data['goal_to_pd_distance_bin_center'].values[0]
        is_fixating = data['is_fixating'].values[0]
        fixation_event_id = data['fixation_event_id'].values[0]
        zorder = 1
        if is_fixating:
            color = color_palette_dict[fixation_event_id]
        else:
            color = '#a8a8a8'
        #         color='k'

        plt.scatter(x, y, color=color, s=0.5, clip_on=False)
        plt.scatter(0, 0, s=2, color='r', zorder=3, clip_on=False)

    g.map_dataframe(plot)

    x = abf_fixation_df.query(f'rec_name=="{rec_name}"')['x'].values
    y = abf_fixation_df.query(f'rec_name=="{rec_name}"')['y'].values

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

    sb = ph.add_scalebar(ax, sizex=200, sizey=0, barwidth=axis_lw * 2, barcolor='k', loc=3,
                         pad=0,
                         bbox_transform=ax.transAxes,
                         bbox_to_anchor=[0.8, 0.1]
                         )

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_single_bout(abf_fixation_df, rec_name, fixation_event_id=52, save=False, savepath=None, fname=None):
    color_palette_dict = {
        50.0: (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
        51.0: (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
        52.0: (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
        53.0: (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),
        54.0: (0.4, 0.7607843137254902, 0.6470588235294118)}
    g = ph.FacetGrid(abf_fixation_df.query(f'rec_name=="{rec_name}" & fixation_event_id=={fixation_event_id}'),
                     unit='fixation_event_id',
                     fig_kws={'figsize': [2, 2], 'dpi': dpi}
                     )

    def plot(data):
        x = data['x'].values
        y = data['y'].values
        is_fixating = data['is_fixating'].values[0]
        fixation_event_id = data['fixation_event_id'].values[0]
        zorder = 1
        if is_fixating:
            color = color_palette_dict[fixation_event_id]
        else:
            color = '#a8a8a8'

        plt.scatter(x, y, color=color, s=0.5)
        plt.plot([x[0], x[-1]], [y[0], y[-1]], lw=1, color='k', clip_on=False)
        plt.scatter([x[0], x[-1]], [y[0], y[-1]], s=2, color='k', clip_on=False)

        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        max_range = max(x_range, y_range)
        x_center = (x.max() + x.min()) / 2
        y_center = (y.max() + y.min()) / 2
        plt.xlim(x_center - max_range / 2, x_center + max_range / 2)
        plt.ylim(y_center - max_range / 2, y_center + max_range / 2)

    g.map_dataframe(plot)

    for ax in g.axes.ravel():
        ax.set_aspect('equal')
        ax.axis('off')

        sb = ph.add_scalebar(ax, sizex=25, sizey=0, barwidth=axis_lw * 2, barcolor='k', loc=3,
                             pad=0,
                             bbox_transform=ax.transAxes,
                             bbox_to_anchor=[0.6, -0.1]
                             )

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_bar_jump_trials(selected_trials_df, save=False, savepath=None, fname=None):
    print(len(selected_trials_df.query('trial_type=="-90 jump"')['unique_trial_id'].unique()), '-90 trials')
    print(len(selected_trials_df.query('trial_type=="+90 jump"')['unique_trial_id'].unique()), '+90 trials')
    g = ph.FacetGrid(selected_trials_df,
                     unit='unique_trial_id',
                     row='trial_type',
                     row_order=['+90 jump', '-90 jump'],
                     fig_kws={'figsize': [2, 2], 'dpi': dpi},
                     gridspec_kws={'hspace': 0.5}
                     )

    def plot(data):
        ph.circplot(data['trial_time'], data['distance_to_goal'], alpha=0.1, circ='y', lw=0.1)

    g.map_dataframe(plot)

    for ax in g.axes.ravel():
        ax.set_ylim([-180, 180])
        ax.set_yticks([-180, 0, 180])
        ax.set_xlim([-60, 62])
        ax.set_xticks([-60, -30, 0, 30, 60])

    ph.despine_axes(g.axes, style=['left', 'bottom'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=1200)


def plot_dforw_bar_jump_trials(fly_mean_dforw_df, save=False, savepath=None, fname=None):
    t = fly_mean_dforw_df['trial_time_new']
    mean = fly_mean_dforw_df['mean']
    sem = fly_mean_dforw_df['sem']

    color = 'k'

    plt.figure(1, [2, 1], dpi=dpi)

    plt.plot(t, mean, color=color, lw=0.5)
    plt.xlim([-10, 20])
    plt.ylim([0, 6])

    ax = plt.gca()
    ax.fill_between(t, (mean - sem), (mean + sem),
                    facecolor=ph.color_to_facecolor(color), edgecolor='none')
    ax.axvspan(0, 2, facecolor=ph.color_to_facecolor('#c4c4c4'), edgecolor='none', zorder=-1)

    ph.adjust_spines(ax, spines=['left', 'bottom'], lw=axis_lw, ticklen=axis_ticklen, pad=2)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')
