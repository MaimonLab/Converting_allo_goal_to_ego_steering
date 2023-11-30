"""pfl3_stim.py

Analysis and plotting functions for PFL3_LAL_stimulation.ipynb

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sc
import matplotlib as mpl

import read_data as rd
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

side_palette = {'right': '#b51700', 'left': '#1c75bc'}
genotype_palette = {'VT000355-AD-VT037220-DBD-Chr-GC7f': 'k', 'SS02239-Chr-GC7f': 'grey',
                    'VT000355-AD-VT037220-DBD-GC7f': '#969696'}
order = ['VT000355-AD-VT037220-DBD-Chr-GC7f', 'VT000355-AD-VT037220-DBD-GC7f', 'SS02239-Chr-GC7f']


# ---------------------------- Load data ---------------------------- #
def load_data(DATA_PATH, reprocess=False):
    pfl3_cschrimson_rec_names = [

        (
            '2021_04_18_0001',
        ),

        (
            '2021_04_18_0003',
        ),

        (
            '2021_04_18_0004',
        ),

        (
            '2021_04_18_0005',
        ),

        (
            '2021_04_18_0006',
        ),

        (
            '2021_04_18_0009',
        ),

        (
            '2021_04_18_0010',
        ),

        (
            '2021_04_19_0002',
        ),

        (
            '2021_04_26_0004',
        ),

        (
            '2021_04_27_0003',
        ),
    ]
    parent_folder = DATA_PATH + 'PFL3_LAL_stimulation' + os.path.sep + 'VT000355-AD-VT037220-DBD-Chr-GC7f' + os.path.sep
    chrimson_pfl3 = rd.quickload_experiment(parent_folder + 'nature_VT000355_AD_VT037220_DBD_GC7f_Chr.pkl',
                                            rec_names=pfl3_cschrimson_rec_names,
                                            exp_kws={
                                                'rec_type': rd.ImagingRec,
                                                'parent_folder': parent_folder,
                                                'merge_df': True,
                                                'genotype': 'VT000355-AD-VT037220-DBD-Chr-GC7f',
                                            },
                                            single_z=True,
                                            reprocess=reprocess,
                                            bh_kws={'angle_offset': 86,
                                                    'boxcar_average': {'dforw': 0.5, 'dheading': 0.5, }},
                                            roi_types=[rd.PairedStructure],
                                            roi_kws={'celltypes': {'c1': 'PFL3', 'c2': 'PFL3'}},
                                            photostim=True,
                                            photostim_kws={'galvo_label': 'mean_x_galvo_pos_slice_0',
                                                           'photostim_scanfield_z': [0]}
                                            )

    pfl3_no_cschrimson_rec_names = [

        (
            '2021_04_19_0001',
        ),

        (
            '2021_04_19_0003',
        ),

        (
            '2021_04_19_0004',
        ),

        (
            '2021_04_19_0005',
        ),

        (
            '2021_04_26_0007',
        ),

        (
            '2021_04_26_0008',
        ),

        (
            '2021_04_27_0005',
        ),
    ]

    parent_folder = DATA_PATH + 'PFL3_LAL_stimulation' + os.path.sep + 'VT000355-AD-VT037220-DBD-GC7f' + os.path.sep
    controls = rd.quickload_experiment(parent_folder + 'nature_VT000355_AD_VT037220_DBD_GC7f_photostim_control.pkl',
                                       rec_names=pfl3_no_cschrimson_rec_names,
                                       exp_kws={
                                           'rec_type': rd.ImagingRec,
                                           'parent_folder': parent_folder,
                                           'merge_df': True,
                                           'genotype': 'VT000355-AD-VT037220-DBD-GC7f',
                                       },
                                       single_z=True,
                                       reprocess=reprocess,
                                       bh_kws={'angle_offset': 86, 'boxcar_average': {'dforw': 0.5, 'dheading': 0.5, }},
                                       roi_types=[rd.PairedStructure],
                                       roi_kws={'celltypes': {'c1': 'PFL3', 'c2': 'PFL3'}},
                                       photostim=True,
                                       photostim_kws={'galvo_label': 'mean_x_galvo_pos_slice_0',
                                                      'photostim_scanfield_z': [0]}
                                       )

    pfl1_cschrimson_rec_names = [

        (
            '2021_04_26_0001',
        ),

        (
            '2021_04_26_0003',
        ),

        (
            '2021_04_26_0006',
        ),

        (
            '2021_04_26_0009',
        ),

        (
            '2021_04_27_0001',
        ),

        (
            '2021_04_27_0002',
        ),

        (
            '2021_04_27_0004',
        ),
    ]

    parent_folder = DATA_PATH + 'PFL3_LAL_stimulation' + os.path.sep + 'SS02239-Chr-GC7f' + os.path.sep
    chrimson_pfl1 = rd.quickload_experiment(parent_folder + 'nature_SS02239_Chr_GC7f.pkl',
                                            rec_names=pfl1_cschrimson_rec_names,
                                            exp_kws={
                                                'rec_type': rd.ImagingRec,
                                                'parent_folder': parent_folder,
                                                'merge_df': True,
                                                'genotype': 'SS02239-Chr-GC7f',
                                            },
                                            single_z=True,
                                            reprocess=reprocess,
                                            bh_kws={'angle_offset': 86,
                                                    'boxcar_average': {'dforw': 0.5, 'dheading': 0.5, }},
                                            roi_types=[rd.PairedStructure],
                                            roi_kws={'celltypes': {'c1': 'PFL1', 'c2': 'PFL1'}},
                                            photostim=True,
                                            photostim_kws={'galvo_label': 'mean_x_galvo_pos_slice_0',
                                                           'photostim_scanfield_z': [0]}
                                            )
    genotypes = {'VT000355-AD-VT037220-DBD-Chr-GC7f': chrimson_pfl3,
                 'SS02239-Chr-GC7f': chrimson_pfl1,
                 'VT000355-AD-VT037220-DBD-GC7f': controls}
    return genotypes


# ------------------ Process & analyze data ----------------------- #

def get_trials(genotypes):
    abf_trials_dfs = []
    im_trials_dfs = []

    for genotype, recs in genotypes.items():
        abf_trials_df, im_trials_df = ap.get_photostim_trials_df(recs, pad_s=5)
        abf_trials_dfs.append(abf_trials_df)
        im_trials_dfs.append(im_trials_df)

    abf_trials_df = pd.concat(abf_trials_dfs)
    im_trials_df = pd.concat(im_trials_dfs)

    def get_unwrapped_zeroed_heading(data):
        t = data['trial_time'].values
        t0_idx = np.where(t == np.min(np.abs(t)))[0][0]
        unwraped_heading = fc.unwrap(data['heading'].values)
        data['unwraped_heading'] = unwraped_heading - unwraped_heading[t0_idx]
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_unwrapped_zeroed_heading)

    def get_mean_dforw_before(data):
        idx = (data['trial_time'].values >= -1) & (data['trial_time'].values <= 0)
        data['mean_dforw_before'] = np.mean(data['dforw'].values[idx])
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_mean_dforw_before)

    def get_is_standing_before(data):
        idx = (data['trial_time'].values >= -1) & (data['trial_time'].values <= 0)
        data['is_standing_before'] = (0 == (np.sum(data['dforw_boxcar_average_0.5_s'].values[idx] > 1)))
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_is_standing_before)

    def get_mean_dheading_before(data):
        idx = (data['trial_time'].values >= -1) & (data['trial_time'].values <= 0)
        data['mean_dheading_before'] = np.mean(data['dheading'].values[idx])
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_mean_dheading_before)

    def get_mean_dheading_during(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values <= 2)
        data['mean_dheading_during'] = np.mean(data['dheading'].values[idx])
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_mean_dheading_during)

    def get_mean_rmlz_during(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values <= 2)
        data['mean_ps_c1_rml_z_during'] = np.mean(data['ps_c1_rml_z'].values[idx])
        return data

    im_trials_df = im_trials_df.groupby(['unique_trial_id']).apply(get_mean_rmlz_during)

    return abf_trials_df, im_trials_df


def get_summary(abf_trials_df, im_trials_df):
    summary_abf_trials_df = abf_trials_df.drop_duplicates('unique_trial_id').reset_index(drop=True)
    summary_im_trials_df = im_trials_df.drop_duplicates('unique_trial_id').reset_index(drop=True)

    # flip dheading of left stim
    summary_abf_trials_df['flip'] = summary_abf_trials_df['scanfield_config_name'].map({'left': -1, 'right': 1}).astype(
        int)
    summary_abf_trials_df['mean_ipsi_dheading_during'] = summary_abf_trials_df['flip'] * summary_abf_trials_df[
        'mean_dheading_during']

    temp_dic = dict(zip(summary_im_trials_df['unique_trial_id'], summary_im_trials_df['mean_ps_c1_rml_z_during']))
    summary_abf_trials_df['mean_ps_c1_rml_z_during'] = summary_abf_trials_df['unique_trial_id'].map(temp_dic).astype(
        'float')

    summary_abf_trials_df['mean_ipsi_rmlz'] = summary_abf_trials_df['flip'] * summary_abf_trials_df[
        'mean_ps_c1_rml_z_during']
    summary_abf_trials_df['mean_ipsi_dheading_before'] = summary_abf_trials_df['flip'] * summary_abf_trials_df[
        'mean_dheading_before']

    # temp_dic=dict(zip(summary_abf_trials_df['unique_trial_id'],summary_abf_trials_df['is_standing_before']))
    # summary_im_trials_df['is_standing_before']=summary_im_trials_df['unique_trial_id'].map(temp_dic).astype('float')
    #
    # temp_dic=dict(zip(summary_abf_trials_df['unique_trial_id'],summary_abf_trials_df['mean_dheading_before']))
    # summary_im_trials_df['mean_dheading_before']=summary_im_trials_df['unique_trial_id'].map(temp_dic).astype('float')

    return summary_abf_trials_df, summary_im_trials_df


def get_mean_signals_df(abf_trials_df, im_trials_df):
    # gets mean for each fly

    melted_abf_trials_df = pd.melt(abf_trials_df, id_vars=['genotype', 'fly_id',
                                                           'scanfield_config_name',
                                                           'trial_time'], value_vars=['unwraped_heading', 'dheading'])

    fly_mean_abf_trials_df = melted_abf_trials_df.groupby(['genotype',
                                                           'scanfield_config_name',
                                                           'fly_id',
                                                           'trial_time',
                                                           'variable']).agg(mean=('value', np.mean))
    fly_mean_abf_trials_df.reset_index(inplace=True)

    fly_mean_abf_trials_df['unique'] = fly_mean_abf_trials_df['scanfield_config_name'].astype(str) + '_' + \
                                       fly_mean_abf_trials_df['fly_id'].astype(str)

    #### behaviour ####

    # gets mean across flies
    mean_abf_trials_df = fly_mean_abf_trials_df.groupby(['genotype',
                                                         'scanfield_config_name',
                                                         'trial_time', 'variable']).agg(mean=('mean', np.mean))
    mean_abf_trials_df.reset_index(inplace=True)

    mean_abf_trials_df['unique'] = mean_abf_trials_df['scanfield_config_name'].astype(str) + '_mean'

    # gets mean for each fly

    melted_im_trials_df = pd.melt(im_trials_df, id_vars=['genotype', 'fly_id',
                                                         'scanfield_config_name',
                                                         'trial_time'],
                                  value_vars=['ps_c1_roi_1_dF/F', 'ps_c1_roi_2_dF/F'])

    #### imaging ####
    fly_mean_im_trials_df = melted_im_trials_df.groupby(
        ['genotype', 'scanfield_config_name', 'fly_id', 'trial_time', 'variable']).agg(mean=('value', np.mean))
    fly_mean_im_trials_df.reset_index(inplace=True)

    fly_mean_im_trials_df['unique'] = fly_mean_im_trials_df['scanfield_config_name'].astype(str) + '_' + \
                                      fly_mean_im_trials_df['fly_id'].astype(str)

    # gets mean across flies
    mean_im_trials_df = fly_mean_im_trials_df.groupby(['genotype',
                                                       'scanfield_config_name',
                                                       'trial_time',
                                                       'variable']).agg(mean=('mean', np.mean))
    mean_im_trials_df.reset_index(inplace=True)

    mean_im_trials_df['unique'] = mean_im_trials_df['scanfield_config_name'].astype(str) + '_mean'

    df = pd.concat([mean_abf_trials_df, fly_mean_abf_trials_df, mean_im_trials_df, fly_mean_im_trials_df])

    return fly_mean_abf_trials_df, df


def get_example_fly_df(abf_trials_df, im_trials_df, rec_name="2021_04_18_0001"):
    example_fly_melted_im_trials_df = pd.melt(im_trials_df.query(f'rec_name=="{rec_name}"'),
                                              id_vars=['genotype', 'unique_trial_id',
                                                       'scanfield_config_name',
                                                       'trial_time'],
                                              value_vars=['ps_c1_roi_1_dF/F', 'ps_c1_roi_2_dF/F']
                                              )

    example_fly_melted_abf_trials_df = pd.melt(abf_trials_df.query(f'rec_name=="{rec_name}"'),
                                               id_vars=['genotype', 'unique_trial_id',
                                                        'scanfield_config_name',
                                                        'trial_time'], value_vars=['unwraped_heading']
                                               )

    example_fly_df = pd.concat([example_fly_melted_im_trials_df, example_fly_melted_abf_trials_df])

    return example_fly_df


def get_mean_ipsi_df(summary_abf_trials_df, save=False, savepath=None, fname=None):
    mean_during = summary_abf_trials_df.groupby(['genotype',
                                                 'fly_id']).agg(mean_ipsi=('mean_ipsi_dheading_during', np.mean))
    mean_during.reset_index(inplace=True)
    mean_during.dropna(inplace=True)

    genotype_mean = mean_during.groupby(['genotype'], as_index=False).agg(mean=('mean_ipsi', np.mean),
                                                                          sem=('mean_ipsi', sc.stats.sem))

    if save:
        mean_during.to_csv(savepath + fname)
    return mean_during, genotype_mean


# ------------------ Plotting functions ----------------------- #

def plot_mean_signals(mean_signals_df, save=False, savepath=None, fname=None):
    col = 'genotype'
    row = 'variable'
    unit = 'unique'
    g = ph.FacetGrid(mean_signals_df.query('variable!="dheading"'), col=col, row=row, unit='unique',
                     fig_kws={'figsize': [1.5, 1.5], 'dpi': dpi},
                     gridspec_kws={'height_ratios': [0.25, 0.25, 0.5],
                                   'wspace': 0.2,
                                   'hspace': 0.2},
                     col_order=order,
                     row_order=['ps_c1_roi_2_dF/F', 'ps_c1_roi_1_dF/F', 'unwraped_heading', ]

                     )

    def plot(data):
        scanfield_config_name = data['scanfield_config_name'].values[0]
        color = side_palette[scanfield_config_name]
        trial_time = data['trial_time'].values
        mean = data['mean'].values

        if 'mean' in data['unique'].values[0]:
            plt.plot(trial_time, mean, color=color, lw=1)
        else:
            plt.plot(trial_time, mean, color=color, lw=0.25, alpha=0.7)

        if data['variable'].values[0] == 'unwraped_heading':
            plt.ylim([-250, 250])
            plt.yticks([-250, 0, 250])

        else:
            plt.ylim([-0.1, 5])

    g.map_dataframe(plot)

    for ax in g.axes.ravel():
        ax.axvspan(0, 2, zorder=0, color='#ffeded')
        ax.set_xlim([-2, 4])
        ax.set_xticks([-2, 4])

    sb = ph.add_scalebar(g.axes[-1, 1], sizex=2, sizey=0, barwidth=axis_lw, barcolor='k', loc='lower center',
                         pad=-1,
                         bbox_transform=ax.transAxes,
                         )

    ph.despine_axes(g.axes, style=['left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_fly(example_fly_df, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(
        example_fly_df.query('(unique_trial_id=="2021_04_18_0001_1")|(unique_trial_id=="2021_04_18_0001_2")'),
        col='scanfield_config_name',
        row='variable', unit='unique_trial_id',
        fig_kws={'figsize': [1.5, 1.5], 'dpi': dpi},
        gridspec_kws={'height_ratios': [0.25, 0.25, 0.5],
                      'wspace': 0.2,
                      'hspace': 0.2},
        col_order=['left', 'right'],
        row_order=['ps_c1_roi_2_dF/F', 'ps_c1_roi_1_dF/F', 'unwraped_heading', ]
    )

    def plot(data):
        scanfield_config_name = data['scanfield_config_name'].values[0]
        color = side_palette[scanfield_config_name]
        trial_time = data['trial_time'].values
        value = data['value'].values

        if data['variable'].values[0] == 'unwraped_heading':
            plt.plot(trial_time, value, color=color, lw=0.5, alpha=1)

            plt.ylim([-250, 250])
            plt.yticks([-250, 0, 250])

        else:
            plt.scatter(trial_time, value, color=color, s=0.1)
            plt.plot(trial_time, value, color=color, lw=0.25, alpha=0.5)
            plt.ylim([-0.1, 5])
            plt.yticks([0, 5])

    g.map_dataframe(plot)

    for ax in g.axes.ravel():
        ax.axvspan(0, 2, zorder=0, color='#ffeded')
        ax.set_xlim([-2, 4])
        ax.set_xticks([-2, 4])

    sb = ph.add_scalebar(g.axes[-1, 1], sizex=2, sizey=0, barwidth=axis_lw, barcolor='k', loc='lower center',
                         pad=-1,
                         bbox_transform=ax.transAxes,
                         )

    ph.despine_axes(g.axes, style=['left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_mean_ipsi(mean_during, genotype_mean, save=False, savepath=None, fname=None):
    fig = plt.figure(1, [0.6, 0.6], dpi=dpi)
    ax = sns.swarmplot(x="genotype", y="mean_ipsi", data=mean_during, size=1.5,
                       order=order,
                       palette=genotype_palette, marker='o', alpha=1)

    lw = 0.5
    xvals = pd.Categorical(genotype_mean['genotype'].values, order).argsort()
    e1 = plt.errorbar(x=xvals + 0.3, y=genotype_mean['mean'].values,
                      yerr=genotype_mean['sem'].values,
                      ls='none', elinewidth=lw, ecolor='k', capsize=1.5, capthick=lw)

    e2 = plt.errorbar(x=xvals + 0.3, y=genotype_mean['mean'].values,
                      xerr=0.2, ls='none', elinewidth=lw, ecolor='k', capsize=None)

    ph.error_bar_clip_on([e1, e2])

    # hack to avoid overlappingg dots
    fig.set_size_inches(0.75, 1.25)
    ax.set_yticks([0, 50, 100])
    ax.set_xticklabels([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ph.despine_axes([ax], style=['left'], adjust_spines_kws={'pad': 2,
                                                             'lw': axis_lw,
                                                             'ticklen': axis_ticklen
                                                             })
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_ipsi_turning_vs_ipsi_rmlz(summary_abf_trials_df,
                                   save_figure=False,
                                   save_figure_path=None,
                                   figure_fname=None,
                                   save_source_data=False,
                                   save_source_data_path=None,
                                   source_data_fname=None):
    plt.figure(1, [1, 1], dpi=dpi)
    plot_df = summary_abf_trials_df.query('genotype=="VT000355-AD-VT037220-DBD-Chr-GC7f"').copy()
    # print(np.corrcoef(plot_df['mean_ipsi_rmlz'],plot_df['mean_ipsi_dheading_during'])[0][1])
    plt.scatter(plot_df['mean_ipsi_rmlz'], plot_df['mean_ipsi_dheading_during'], s=1, color='k', clip_on=False,
                zorder=3)
    plt.axhline(y=0, ls=':', lw=1, color='grey')

    ph.despine_axes([plt.gca()], style=['left', 'bottom'], adjust_spines_kws={'pad': 2,
                                                                              'lw': axis_lw,
                                                                              'ticklen': axis_ticklen
                                                                              })

    plt.xlim([0, 6])
    plt.ylim([-50, 200])
    if save_figure:
        plt.savefig(save_figure_path + figure_fname,
                    transparent=True, bbox_inches='tight')
    if save_source_data:
        plot_df.filter(['mean_ipsi_rmlz', 'mean_ipsi_dheading_during']).to_csv(
            save_source_data_path + source_data_fname)


def plot_ipsi_turning_vs_dforw_before(summary_abf_trials_df,
                                      save_figure=False,
                                      save_figure_path=None,
                                      figure_fname=None,
                                      save_source_data=False,
                                      save_source_data_path=None,
                                      source_data_fname=None):
    plt.figure(1, [1, 1], dpi=dpi)
    plot_df = summary_abf_trials_df.query('genotype=="VT000355-AD-VT037220-DBD-Chr-GC7f"').copy()
    plt.scatter(plot_df['mean_dforw_before'], plot_df['mean_ipsi_dheading_during'], s=1, color='k', zorder=3,
                clip_on=False)
    plt.axhline(y=0, ls=':', lw=1, color='grey')
    plt.ylim([-50, 200])
    ph.despine_axes([plt.gca()], style=['left', 'bottom'], adjust_spines_kws={'pad': 2,
                                                                              'lw': axis_lw,
                                                                              'ticklen': axis_ticklen
                                                                              })
    if save_figure:
        plt.savefig(save_figure_path + figure_fname,
                    transparent=True, bbox_inches='tight')
    if save_source_data:
        plot_df.filter(['mean_dforw_before', 'mean_ipsi_dheading_during']).to_csv(
            save_source_data_path + source_data_fname)


def plot_ipsi_turning_vs_dheading_before(summary_abf_trials_df,
                                         save_figure=False,
                                         save_figure_path=None,
                                         figure_fname=None,
                                         save_source_data=False,
                                         save_source_data_path=None,
                                         source_data_fname=None):
    plt.figure(1, [1, 1], dpi=dpi)
    plot_df = summary_abf_trials_df.query('genotype=="VT000355-AD-VT037220-DBD-Chr-GC7f"')
    plt.scatter(plot_df['mean_ipsi_dheading_before'], plot_df['mean_ipsi_dheading_during'], s=1, color='k', zorder=3,
                clip_on=False)
    plt.axhline(y=0, ls=':', lw=1, color='grey')
    plt.xlim([-100, 100])
    plt.ylim([-50, 200])
    ph.despine_axes([plt.gca()], style=['left', 'bottom'], adjust_spines_kws={'pad': 2,
                                                                              'lw': axis_lw,
                                                                              'ticklen': axis_ticklen
                                                                              })

    if save_figure:
        plt.savefig(save_figure_path + figure_fname,
                    transparent=True, bbox_inches='tight')
    if save_source_data:
        plot_df.filter(['mean_ipsi_dheading_before', 'mean_ipsi_dheading_during']).to_csv(
            save_source_data_path + source_data_fname)


# ---------------------------- Save processed data ---------------------------- #
def save_processed_data(PROCESSED_DATA_PATH, genotypes):
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

    def save_im_as_hd5f(data):
        df = data.copy()
        rec_name = df['rec_name'].values[0]
        genotype = df['genotype'].values[0]
        df = df.filter([
            't_abf',
            'ps_c1_roi_1_F',
            'ps_c1_roi_2_F',
            'ps_c2_roi_1_F',
            'ps_c2_roi_2_F',
            'ps_roi_1_photostim',
            'ps_roi_2_photostim',
            'scanfield_config_name'
        ]).rename({'t_abf': 't', 'scanfield_config_name': 'stim_pos'}, axis=1).reset_index(drop=True)
        df['stim_pos'] = df['stim_pos'].map({'left': 'left', 'right': 'right', 'control': 'inter-trial'})
        df['stim_pos'] = pd.Categorical(df['stim_pos'])
        df.to_hdf(PROCESSED_DATA_PATH + genotype + os.path.sep + rec_name + '_im.h5',
                  key='df', mode='w', format='table')

    summary_recs = []
    for genotype, recs in genotypes.items():
        recs.merged_abf_df.groupby(['rec_name']).apply(save_abf_as_hd5f)
        recs.merged_im_df.groupby(['rec_name']).apply(save_im_as_hd5f)
        summary_recs.append(recs.merged_im_df.drop_duplicates('rec_name').copy().sort_values(['fly_id']).filter(
            ['genotype', 'rec_name', 'fly_id']).reset_index(drop=True))

    pd.concat(summary_recs).reset_index(drop=True).to_csv(PROCESSED_DATA_PATH + 'summary.csv')
