"""epg_fc2.py

Analysis and plotting functions for EPG_FC2.ipynb

"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sc
import seaborn as sns
import copy
from scipy.interpolate import interp1d

import analysis_plot as ap
import functions as fc
import plotting_help as ph
import read_data as rd

# ------------------ Plotting parameters ----------------------- #
dpi = 300
axis_lw = 0.4
axis_ticklen = 2.5
font = {'family': 'arial',
        'weight': 'normal',
        'size': 5}
mpl.rc('font', **font)

genotype_palette = {'VT065306-AD-VT029306-DBD': '#803399',
                    '60D05': '#808080'}


# ------------------ Load data ----------------------- #
def load_data(DATA_PATH, reprocess=False):
    FC2_rec_names = [
        ##### sytGCaMP7f #####

        (
            '2022_06_10_0006',
            '2022_06_10_0009',
        ),

        (
            '2022_06_12_0001',
            '2022_06_12_0002',
        ),

        (
            '2022_06_12_0003',
        ),

        (
            '2022_06_12_0004',
            '2022_06_12_0005',
        ),

        (
            '2022_06_13_0001',
            '2022_06_13_0002',
        ),

        (
            '2022_06_13_0007',
            '2022_06_13_0008',
        ),

        (
            '2022_06_14_0002',
            '2022_06_14_0003',
        ),

        (
            '2023_04_26_0003',
        ),

        (
            '2023_04_26_0005',
            '2023_04_26_0006',
        ),

        ##### GCaMP7f #####

        (
            '2022_08_29_0005',
            '2022_08_29_0006',
        ),

        (
            '2022_08_30_0003',
            '2022_08_30_0004',
        ),

        (
            '2022_08_30_0005',
            '2022_08_30_0006',
        ),

        (
            '2022_08_30_0007',
            '2022_08_30_0008',
        ),

        (
            '2022_09_01_0003',
            '2022_09_01_0004',
        ),

        (
            '2022_09_04_0007',
        ),

    ]

    parent_folder = DATA_PATH + 'EPG_FC2_imaging' + os.path.sep + 'VT065306-AD-VT029306-DBD' + os.path.sep
    VT065306_AD_VT029306_DBD = rd.quickload_experiment(parent_folder + 'nature_VT065306_AD_VT029306_DBD.pkl',
                                                       rec_names=FC2_rec_names,
                                                       reprocess=reprocess,
                                                       exp_kws={
                                                           'rec_type': rd.ImagingRec,
                                                           'parent_folder': parent_folder,
                                                           'merge_df': True,
                                                           'genotype': 'VT065306-AD-VT029306-DBD',
                                                       },
                                                       bh_kws={'angle_offset': 86,
                                                               'boxcar_average': {'dforw': 0.5, 'dheading': 0.5}},
                                                       roi_types=[rd.FanShapedBody],
                                                       roi_kws={'celltypes': {'c1': 'FC2'}},
                                                       )
    EPG_rec_names = [
        (
            '2019_07_22_0002',  # has tomato
        ),

        (
            '2019_07_22_0003',  # has tomato
            '2019_07_22_0004',
        ),

        (
            '2019_07_22_0005',  # has tomato
            '2019_07_22_0006',
        ),

        (
            '2022_08_29_0001',
            '2022_08_29_0002',
        ),

        ('2022_08_29_0003',
         '2022_08_29_0004',
         ),

        ('2022_08_30_0009',
         '2022_08_30_0010',
         ),

        ('2022_09_01_0001',
         '2022_09_01_0002',
         ),

        ('2022_09_01_0007',
         '2022_09_01_0008',
         ),

        ('2022_09_02_0003',  # has tomato
         ),

    ]
    parent_folder = DATA_PATH + 'EPG_FC2_imaging' + os.path.sep + '60D05' + os.path.sep
    R60D05 = rd.quickload_experiment(parent_folder + 'nature_60D05.pkl',
                                     rec_names=EPG_rec_names,
                                     reprocess=reprocess,
                                     exp_kws={
                                         'rec_type': rd.ImagingRec,
                                         'parent_folder': parent_folder,
                                         'merge_df': True,
                                         'genotype': '60D05',
                                     },
                                     bh_kws={'angle_offset': 86, 'boxcar_average': {'dforw': 0.5, 'dheading': 0.5}},
                                     roi_types=[rd.Bridge],
                                     roi_kws={'celltypes': {'c1': 'EPG'}},
                                     )

    genotypes = {'VT065306-AD-VT029306-DBD': VT065306_AD_VT029306_DBD, '60D05': R60D05}
    return genotypes


# ------------------ Process & analyze data ----------------------- #

def extra_pre_processing(genotypes):
    for genotype, recs in genotypes.items():
        if genotype == "VT065306-AD-VT029306-DBD":
            recs.merged_im_df.rename({'fb_c1_pva_amplitude': 'pva_amplitude'}, axis=1, inplace=True)
            recs.merged_im_df.rename({'fb_c1_mean_dF/F': 'mean_dF/F'}, axis=1, inplace=True)
            recs.merged_im_df.rename({'fb_c1_max_min': 'max_min'}, axis=1, inplace=True)
            # phase subtraction
            genotypes[genotype] = ap.subtract_offset(recs, 'fb_c1_phase',
                                                     idx='(xstim > -135) & (xstim < 135)& '
                                                         '`dforw_boxcar_average_0.5_s`>1')
        elif genotype == "60D05":
            recs.merged_im_df.rename({'pb_c1_pva_amplitude': 'pva_amplitude'}, axis=1, inplace=True)
            recs.merged_im_df.rename({'pb_c1_mean_dF/F': 'mean_dF/F'}, axis=1, inplace=True)
            recs.merged_im_df.rename({'pb_c1_max_min': 'max_min'}, axis=1, inplace=True)
            recs.merged_im_df['pb_c1_roi_18_dF/F'] = np.nan
            recs.merged_im_df['pb_c1_roi_1_dF/F'] = np.nan
            genotypes[genotype] = ap.subtract_offset(recs, 'pb_c1_phase',
                                                     idx='(xstim > -135) & (xstim < 135)& '
                                                         '`dforw_boxcar_average_0.5_s`>1')
    get_sliding_r(genotypes, windows=[30, 60, 120])
    return genotypes


def get_corr_df(VT065306_AD_VT029306_DBD, R60D05, save=False, savepath=None, fname=None):
    def get_circ_corr(data, phase_label):
        xstim = data['xstim'].values
        idx = (data['dforw_boxcar_average_0.5_s'].values > 1) & (xstim >= -135) & (xstim <= 135)
        corr = fc.circcorr(xstim[idx], data[phase_label].values[idx])
        return corr

    corr_FC2 = VT065306_AD_VT029306_DBD.merged_im_df.groupby(['genotype', 'fly_id']).apply(get_circ_corr, 'fb_c1_phase')
    corr_EPG = R60D05.merged_im_df.groupby(['genotype', 'fly_id']).apply(get_circ_corr, 'pb_c1_phase')
    fly_corr_df = pd.concat([corr_EPG, corr_FC2]).reset_index().rename({0: 'corr'}, axis=1)
    genotype_corr = fly_corr_df.groupby(['genotype'], as_index=False).agg(mean=('corr', np.mean),
                                                                          sem=('corr', sc.stats.sem))

    if save:
        fly_corr_df.to_csv(savepath + fname)

    return fly_corr_df, genotype_corr


def get_xcorr_df(genotypes):
    mean_xcorr_dfs = []
    for genotype, recs in genotypes.items():

        if genotype == '60D05':
            phase_label = 'pb_c1_dphase'
        elif genotype == 'VT065306-AD-VT029306-DBD':
            phase_label = 'fb_c1_dphase'

        volume_period = [np.round(rec.rec.im.volume_period, 3) for rec in recs]
        if (len(set(volume_period)) != 1):
            print('volume_periods are different!')
        volume_period = volume_period[0]

        # TODO this is now in read_data, can remove and re-process recs
        def get_dphase(data):
            if genotype == '60D05':
                phase_label2 = 'pb_c1_phase'
            elif genotype == 'VT065306-AD-VT029306-DBD':
                phase_label2 = 'fb_c1_phase'
            dphase = fc.circgrad(data[phase_label2]) * volume_period
            data[phase_label] = dphase
            return data

        recs.merged_im_df = recs.merged_im_df.groupby(['rec_name']).apply(get_dphase)

        def get_xcorr(data):
            t, xc1 = fc.xcorr(data[phase_label].values, data['dheading'].values,
                              sampling_period=volume_period, norm=True)

            return pd.Series({'t': t.astype(np.float32), 'xc1': xc1})

        xcorr_df = recs.merged_im_df.groupby(['fly_id', 'rec_name']).apply(get_xcorr)
        xcorr_df.reset_index(inplace=True)

        def unnesting(df, explode):
            idx = df.index.repeat(df[explode[0]].str.len())
            df1 = pd.concat([
                pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
            df1.index = idx
            return df1.join(df.drop(explode, 1), how='left')

        xcorr_df = unnesting(xcorr_df, ['t', 'xc1'])
        xcorr_df.reset_index(inplace=True, drop=True)

        melted_xcorr_df = pd.melt(xcorr_df,
                                  id_vars=['fly_id', 'rec_name', 't'],
                                  value_vars=['xc1'])

        # gets mean for each fly (since mutliple recs for each fly)
        flies_xcorr_df = melted_xcorr_df.groupby(['fly_id', 't']).agg(mean=('value', np.mean),
                                                                      sem=('value', sc.stats.sem))
        flies_xcorr_df.reset_index(inplace=True)

        mean_xcorr_df = flies_xcorr_df.groupby(['t'], as_index=False).agg(mean=('mean', np.mean),
                                                                          sem=('mean', sc.stats.sem))
        mean_xcorr_df['genotype'] = genotype
        mean_xcorr_dfs.append(mean_xcorr_df)

    xcorr_df = pd.concat(mean_xcorr_dfs)
    return xcorr_df


def get_bar_jump_trials(genotypes):
    stimid_map = {'+90 jump': [9, 11],
                  '-90 jump': [5, 8],
                  'CL_bar': [-1, 4]
                  }
    pad_s = 30
    bar_jump_duration_s = 2

    abf_trials_dfs = []
    im_trials_dfs = []

    for genotype, recs in genotypes.items():

        abf_trials_df, im_trials_df = ap.get_trials_df(recs, pad_s=pad_s, im=True, stimid_map=stimid_map)

        if genotype == "60D05":
            im_trials_df = im_trials_df.filter(['genotype', 'fly_id', 'rec_name',
                                                'unique_trial_id', 'trial_type', 'trial_time',
                                                'heading', 'side', 'forw', 'stimid',
                                                'dforw_boxcar_average_0.5_s', 'dheading_boxcar_average_0.5_s',
                                                't', 'pb_c1_phase', 'pb_c1_phase_offset_subtracted', 'pva_amplitude',
                                                'mean_dF/F', 'max_min'])
            im_trials_df.rename({'pb_c1_phase': 'phase',
                                 'pb_c1_phase_offset_subtracted': 'phase_offset_subtracted'
                                 # 'pb_c1_pva_amplitude': 'pva_amplitude'
                                 }, axis=1, inplace=True)

        else:
            im_trials_df = im_trials_df.filter(
                ['genotype', 'fly_id', 'rec_name', 'unique_trial_id', 'trial_type', 'trial_time',
                 'heading', 'side', 'forw', 'stimid',
                 'dforw_boxcar_average_0.5_s', 'dheading_boxcar_average_0.5_s',
                 't', 'fb_c1_phase', 'fb_c1_phase_offset_subtracted', 'pva_amplitude', 'mean_dF/F', 'max_min',
                 'rml'  # rml used for synthetic data
                 ])

            im_trials_df.rename({'fb_c1_phase': 'phase',
                                 'fb_c1_phase_offset_subtracted': 'phase_offset_subtracted'
                                 # 'fb_c1_pva_amplitude': 'pva_amplitude'
                                 }, axis=1, inplace=True)

        abf_trials_dfs.append(abf_trials_df)
        im_trials_dfs.append(im_trials_df)

    abf_trials_df = pd.concat(abf_trials_dfs, ignore_index=True)
    im_trials_df = pd.concat(im_trials_dfs, ignore_index=True)

    def get_jump_pos(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values <= bar_jump_duration_s)
        data['jump_pos'] = fc.circmean(data['xstim'].values[idx])
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_jump_pos)

    def get_jump_in_fix(data):
        idx = np.argmin(np.abs(data['trial_time']))
        data['jump_in_fix'] = data['is_fixating'].values[idx]
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_jump_in_fix)

    # TODO CHECK IF NO FILTER ON FORWARD IS ON PURPOSE HERE
    # It's aslo a bit confusing since goal gets over-written from fixation code
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

    def get_dforw_before(data):
        idx = (data['trial_time'].values >= -5) & (data['trial_time'].values < 0)
        dforw_before = np.mean(data['dforw_boxcar_average_0.5_s'].values[idx])
        data['dforw_before'] = dforw_before
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_dforw_before)

    def get_dforw_after(data):
        idx = (data['trial_time'].values >= 5) & (data['trial_time'].values < 10)
        dforw_after = np.mean(data['dforw_boxcar_average_0.5_s'].values[idx])
        data['dforw_after'] = dforw_after
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_dforw_after)

    def get_xstim_std_before(data):
        idx = (data['trial_time'].values >= -5) & (data['trial_time'].values < 0)
        std = fc.circstd(data['xstim'].values[idx])
        data['std_before'] = std
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_xstim_std_before)

    def get_xstim_std_after(data):
        idx = (data['trial_time'].values >= 5) & (data['trial_time'].values < 10)
        std = fc.circstd(data['xstim'].values[idx])
        data['std_after'] = std
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_xstim_std_after)

    def get_mean_phase_amplitude(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values <= bar_jump_duration_s)
        data['mean_amplitude_during'] = np.mean(data['pva_amplitude'].values[idx])
        return data

    im_trials_df = im_trials_df.groupby(['unique_trial_id']).apply(get_mean_phase_amplitude)
    abf_trials_df['mean_amplitude_during'] = abf_trials_df['unique_trial_id'].map(
        dict(zip(im_trials_df['unique_trial_id'], im_trials_df['mean_amplitude_during'])))

    def get_phase_zeroed(data):
        idx = (data['trial_time'].values >= -1) & (data['trial_time'].values <= 0)
        signal_zero = fc.circmean(data['phase'].values[idx])
        signal_zero = fc.wrap(data['phase'].values - signal_zero)
        data['phase_zeroed'] = signal_zero
        return data

    im_trials_df = im_trials_df.groupby(['unique_trial_id']).apply(get_phase_zeroed)

    def get_xstim_zeroed(data):
        idx = (data['trial_time'].values < 0)
        signal_zero = data['xstim'].values[idx][-1]
        signal_zero = fc.wrap(data['xstim'].values - signal_zero)
        data['xstim_zeroed'] = signal_zero
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_xstim_zeroed)

    # used for reviewers
    def get_offset_subtracted_phase_during(data):
        idx = (data['trial_time'].values >= 1) & (data['trial_time'].values < 2)
        phase_during = fc.circmean(data['phase_offset_subtracted'].values[idx])
        data['offset_subtracted_phase_during'] = phase_during
        return data

    im_trials_df = im_trials_df.groupby(['unique_trial_id']).apply(get_offset_subtracted_phase_during)

    def get_phase_during(data):
        idx = (data['trial_time'].values >= 1) & (data['trial_time'].values < 2)
        phase_during = fc.circmean(data['phase_zeroed'].values[idx])
        data['phase_during'] = phase_during
        return data

    im_trials_df = im_trials_df.groupby(['unique_trial_id']).apply(get_phase_during)

    return abf_trials_df, im_trials_df


# not used...
def get_summary(abf_trials_df, im_trials_df):
    summary_abf_trials = abf_trials_df.drop_duplicates('unique_trial_id')

    summary_im_trials = im_trials_df.drop_duplicates('unique_trial_id')

    summary_im_trials = summary_im_trials.filter(
        ['fly_id', 'genotype', 'unique_trial_id', 'mean_amplitude_during', 'offset_subtracted_phase_during',
         'phase_during'])

    summary_abf_trials = summary_abf_trials.filter(
        ['fly_id', 'genotype', 'trial_type', 'unique_trial_id', 'jump_pos', 'goal_diff', 'abs_goal_diff', 'std_before',
         'std_after', 'jump_in_fix', 'mean_r_during',
         'mean_dheading_during', 'mean_xstim_after', 'dforw_before', 'dforw_after'])

    summary_abf_trials['mean_amplitude_during'] = summary_abf_trials['unique_trial_id'].map(
        dict(zip(summary_im_trials['unique_trial_id'], summary_im_trials['mean_amplitude_during'])))

    summary_abf_trials['offset_subtracted_phase_during'] = summary_abf_trials['unique_trial_id'].map(
        dict(zip(summary_im_trials['unique_trial_id'], summary_im_trials['offset_subtracted_phase_during'])))
    summary_abf_trials['phase_during'] = summary_abf_trials['unique_trial_id'].map(
        dict(zip(summary_im_trials['unique_trial_id'], summary_im_trials['phase_during'])))

    summary_abf_trials['abs_phase_during'] = np.abs(summary_abf_trials['phase_during'])

    summary_abf_trials['jump_goal_after_diff'] = np.abs(
        fc.wrap(summary_abf_trials['jump_pos'] - summary_abf_trials['mean_xstim_after']))

    return summary_abf_trials, summary_im_trials


# TODO make this function not re-compute summary?
def get_selected_trials(abf_trials_df, im_trials_df, query):
    # gets selected trials and computes mean across flies
    # for specific imaging and behaviour signals

    summary_abf = abf_trials_df.drop_duplicates('unique_trial_id')
    valid_trials = summary_abf.query(query)['unique_trial_id'].tolist()

    print(str(len(valid_trials) / len(np.unique(abf_trials_df['unique_trial_id']))) + '% valid trials')

    abf_trials_melted = pd.melt(abf_trials_df[abf_trials_df['unique_trial_id'].isin(valid_trials)],
                                id_vars=['genotype', 'fly_id', 'rec_name', 'unique_trial_id', 'trial_type',
                                         'trial_time'],
                                value_vars=['xstim_zeroed', 'dforw_boxcar_average_0.5_s'])

    # Gets mean across trials for each fly for abf signals
    fly_mean_abf_circ = abf_trials_melted.query('variable=="xstim_zeroed"').groupby(['genotype', 'fly_id',
                                                                                     'trial_type', 'trial_time',
                                                                                     'variable']
                                                                                    , observed=True,
                                                                                    as_index=False).agg(
        mean=('value', fc.circmean))

    fly_mean_abf_noncirc = abf_trials_melted.query('variable=="dforw_boxcar_average_0.5_s"').groupby(
        ['genotype',
         'fly_id',
         'trial_type',
         'trial_time',
         'variable'],
        observed=True,
        as_index=False).agg(mean=('value', np.mean))

    fly_mean_abf = pd.concat([fly_mean_abf_circ, fly_mean_abf_noncirc], ignore_index=True)

    # Gets mean across flies for imaging signals (don't bother with xstim since not plotted)
    abf_mean_df = fly_mean_abf_noncirc.groupby(['genotype',
                                                'trial_type',
                                                'trial_time', 'variable'], observed=True,
                                               as_index=False).agg(mean=('mean', np.mean))

    im_trials_melted = pd.melt(im_trials_df[im_trials_df['unique_trial_id'].isin(valid_trials)],
                               id_vars=['genotype', 'fly_id', 'rec_name', 'unique_trial_id', 'trial_type',
                                        'trial_time'],
                               value_vars=['phase_zeroed', 'pva_amplitude', 'mean_dF/F', 'max_min'])

    # Gets mean across trials for each fly for imaging signals

    fly_mean_im_circ = im_trials_melted.query('variable=="phase_zeroed"').groupby(
        ['genotype', 'fly_id', 'trial_type', 'trial_time', 'variable']
        , observed=True, as_index=False).agg(mean=('value', fc.circmean))

    fly_mean_im_noncirc = im_trials_melted.query(
        'variable=="pva_amplitude"|variable=="mean_dF/F"|variable=="max_min"').groupby(
        ['genotype', 'fly_id', 'trial_type', 'trial_time', 'variable']
        , observed=True, as_index=False).agg(mean=('value', np.mean))

    fly_mean_im = pd.concat([fly_mean_im_circ, fly_mean_im_noncirc], ignore_index=True)

    # Gets mean across flies for imaging signals
    im_mean_df_circ = fly_mean_im_circ.groupby(['genotype', 'trial_type', 'trial_time', 'variable']
                                               , observed=True, as_index=False).agg(mean=('mean', fc.circmean))

    im_mean_df_noncirc = fly_mean_im_noncirc.groupby(['genotype', 'trial_type', 'trial_time', 'variable']
                                                     , observed=True, as_index=False).agg(mean=('mean', np.mean))

    im_mean_df = pd.concat([im_mean_df_circ, im_mean_df_noncirc], ignore_index=True)

    # combines melted dfs
    melted_df = pd.concat([abf_trials_melted, im_trials_melted])

    return melted_df, abf_mean_df, im_mean_df


def get_mean_bar_jump_phase(melted_df, save=False, savepath=None, fname=None):
    df = melted_df.query('variable=="phase_zeroed"').copy()

    df['flip'] = df['trial_type'].map({'+90 jump': 1, '-90 jump': -1}).astype(int)
    df['flipped_value'] = df['flip'] * df['value']

    fly_mean_phase_df = df.query('trial_time>=1 & trial_time<=2 ').groupby(['genotype', 'fly_id'],
                                                                           observed=True,
                                                                           as_index=False).agg(
        mean=('flipped_value', fc.circmean))

    genotype_mean_phase_df = fly_mean_phase_df.groupby(['genotype'], as_index=False).agg(mean=('mean', fc.circmean),
                                                                                         sem=('mean', fc.circ_stderror))
    if save:
        fly_mean_phase_df.to_csv(savepath + fname)

    return fly_mean_phase_df, genotype_mean_phase_df


def get_mean_bar_jump_amplitude_diff(melted_df, label, save=False, savepath=None, fname=None):
    df = melted_df.query(f'variable=="{label}"').copy()

    before_df = df.query('trial_time>=-2 & trial_time<=0 ').groupby(['genotype', 'fly_id'],
                                                                    observed=True,
                                                                    as_index=False).agg(mean=('value', np.mean))

    during_df = df.query('trial_time>=0 & trial_time<=2 ').groupby(['genotype', 'fly_id'],
                                                                   observed=True,
                                                                   as_index=False).agg(mean=('value', np.mean))
    fly_mean_diff = before_df
    fly_mean_diff['mean'] = during_df['mean'] - before_df['mean']
    genotype_mean_diff = fly_mean_diff.groupby(['genotype'], as_index=False).agg(mean=('mean', np.mean),
                                                                                 sem=('mean', sc.stats.sem))
    if save:
        fly_mean_diff.to_csv(savepath + fname)
    return fly_mean_diff, genotype_mean_diff


def get_sliding_r(genotypes, windows=None):
    if windows is None:
        windows = [30, 60, 120]

    for genotype, recs in genotypes.items():
        for window in windows:
            ap.get_sliding_mean_vector(recs, window)


def get_binned_amplitudes(genotypes, save=False, savepath=None, fname=None):
    binned_amplitudes_dfs = []
    for genotype, recs in genotypes.items():
        dforw_binned_df = ap.get_binned_df(recs.merged_im_df,
                                           bin_values={'dforw_boxcar_average_0.5_s': np.linspace(-1, 10, 11)},
                                           labels=['pva_amplitude', 'mean_dF/F', 'max_min'],
                                           id_vars=['fly_id'])

        dforw_mean_df = dforw_binned_df.groupby(['variable',
                                                 'dforw_boxcar_average_0.5_s_bin_center']).agg(
            mean=('mean_value', np.nanmean),
            sem=('mean_value', lambda x: sc.stats.sem(x, nan_policy='omit'))).reset_index()
        turning_binned_df = ap.get_binned_df(recs.merged_im_df,
                                             bin_values={'dheading_boxcar_average_0.5_s': np.linspace(-200, 200, 11)},
                                             labels=['pva_amplitude', 'mean_dF/F', 'max_min'],
                                             id_vars=['fly_id'])

        turning_mean_df = turning_binned_df.groupby(['variable', 'dheading_boxcar_average_0.5_s_bin_center']).agg(
            mean=('mean_value', np.nanmean),
            sem=('mean_value', lambda x: sc.stats.sem(x, nan_policy='omit'))).reset_index()

        turning_mean_df['bin_var'] = 'dheading'
        dforw_mean_df['bin_var'] = 'dforw'

        dforw_mean_df.rename({'dforw_boxcar_average_0.5_s_bin_center': 'bin_center'}, axis=1, inplace=True)
        turning_mean_df.rename({'dheading_boxcar_average_0.5_s_bin_center': 'bin_center'}, axis=1, inplace=True)

        mean_df = pd.concat([dforw_mean_df, turning_mean_df])
        mean_df['genotype'] = genotype
        binned_amplitudes_dfs.append(mean_df)

    binned_amplitudes_dfs = pd.concat(binned_amplitudes_dfs)
    # re-orders columns
    binned_amplitudes_dfs = binned_amplitudes_dfs[['genotype', 'bin_var', 'bin_center', 'variable', 'mean', 'sem']]
    if save:
        binned_amplitudes_dfs.to_csv(savepath + fname)
    return binned_amplitudes_dfs


def get_binned_amplitudes_r(genotypes, windows=None, save=False, savepath=None, fname=None):
    if windows is None:
        windows = [30, 60, 120]

    binned_amplitudes_dfs = []
    for genotype, recs in genotypes.items():
        mean_dfs = []
        for window in windows:
            label = 'r_' + str(window)
            binned_df = ap.get_binned_df(recs.merged_im_df,
                                         bin_values={label: np.linspace(0, 1, 11)},
                                         labels=['pva_amplitude', 'mean_dF/F', 'max_min'],
                                         id_vars=['fly_id'])

            mean_df = binned_df.groupby(['variable', label + '_bin_center']).agg(mean=('mean_value', np.nanmean),
                                                                                 sem=('mean_value',
                                                                                      lambda x: sc.stats.sem(x,
                                                                                                             nan_policy='omit'))).reset_index()

            mean_df['bin_var'] = label
            mean_df.rename({label + '_bin_center': 'bin_center'}, axis=1, inplace=True)
            mean_dfs.append(mean_df)
        mean_dfs = pd.concat(mean_dfs)
        mean_dfs['genotype'] = genotype
        binned_amplitudes_dfs.append(mean_dfs)

    binned_amplitudes_dfs = pd.concat(binned_amplitudes_dfs)
    # re-orders columns
    binned_amplitudes_dfs = binned_amplitudes_dfs[['genotype', 'bin_var', 'bin_center', 'variable', 'mean', 'sem']]
    if save:
        binned_amplitudes_dfs.to_csv(savepath + fname)
    return binned_amplitudes_dfs

def get_phase_transients(recs):
    recs = copy.copy(recs)

    # removes these
    if 'trial_id' in recs.merged_im_df.columns:
        recs.merged_im_df.drop(['trial_id'], axis=1, inplace=True)
    if 'trial_id' in recs.merged_abf_df.columns:
        recs.merged_abf_df.drop(['trial_id'], axis=1, inplace=True)
    if 'trial_type' in recs.merged_im_df.columns:
        recs.merged_im_df.drop(['trial_type'], axis=1, inplace=True)
    if 'trial_type' in recs.merged_abf_df.columns:
        recs.merged_abf_df.drop(['trial_type'], axis=1, inplace=True)

    def get_boxcar_average_dphase(data):
        window_s = 0.5
        signal = data['fb_c1_dphase'].values
        inds = np.arange(len(signal))
        half_window_s = window_s / 2.
        # bc an integer number of indices is used, the window is not necessarily exact
        sampling_rate = [np.round(1. / rec.rec.im.volume_period, 3) for rec in recs]
        if len(set(sampling_rate)) != 1:
            print('samplig rates are different!')
            return
        sampling_rate = sampling_rate[0]

        half_window_ind = int(np.round(half_window_s * sampling_rate))
        start = inds - half_window_ind
        end = inds + half_window_ind + 1
        # prevents wrapping, but means that some windows are shorter
        start[start < 0] = 0
        end[end > len(signal)] = len(signal)
        signal_out = np.zeros(len(signal))
        signal_out[:] = np.nan

        for i in inds:
            t0 = start[i]
            t1 = end[i]
            signal_out[i] = np.mean(signal[t0:t1])

        data['boxcar_average_dphase'] = signal_out

        return data

    recs.merged_im_df = recs.merged_im_df.groupby(['rec_name'], as_index=False).apply(get_boxcar_average_dphase)

    detection_parameters = {

        'height': 5,
        'distance': 3,
        'width': 1,
        'prominence': 1,
    }

    abf_peaks_df, im_peaks_df = ap.get_peaks_df(recs, 10, label="boxcar_average_dphase",
                                                detection_parameters=detection_parameters, align_to_start=True)

    im_peaks_df['unique_trial_id'] = im_peaks_df['rec_name'] + '_' + im_peaks_df['trial_id'].astype(str)

    def get_mean_amp(data):
        idx = (data['trial_time'].values >= -1) & (data['trial_time'].values <= 1)
        data['mean_amp'] = np.mean(data['pva_amplitude'].values[idx])
        return data

    im_peaks_df = im_peaks_df.groupby(['unique_trial_id'], as_index=False).apply(get_mean_amp)

    def get_above_thresh(data):
        idx = (data['trial_time'].values >= -1) & (data['trial_time'].values <= 1)
        data['above_thresh'] = (np.mean(data['pva_amplitude'].values[idx] > 0.15) == 1)
        return data

    im_peaks_df = im_peaks_df.groupby(['unique_trial_id'], as_index=False).apply(get_above_thresh)

    return abf_peaks_df, im_peaks_df


def get_selected_phase_transients(abf_peaks_df, im_peaks_df):
    trial_ids = im_peaks_df.query('(mean_amp>0.25) & (above_thresh==True)')['unique_trial_id'].unique().tolist()

    # "upsampling" is just used to align to traces, so that you can take average across
    # traces (since sampling rates are a bit different)

    def get_upsampled(data):
        trial_time = data['trial_time'].values
        idx = np.where(trial_time == 0)[0]
        phase = data['fb_c1_phase'].values
        phase = fc.wrap(phase - phase[idx])
        phase_x, phase_y = fc.polar2cart(np.ones(len(phase)), np.deg2rad(phase))
        t = np.linspace(-10, 10, 20 * 5)
        # guarantees t=0
        # t=np.concatenate([np.arange(-10,0, (1/4.95)) -(1/ (4.95*2) ),np.arange(0,10, (1/4.95))])
        # upsampled imaging
        f = interp1d(trial_time, phase_x, kind='linear', bounds_error=False, fill_value=np.nan)
        x_upsampled = f(t)
        f = interp1d(trial_time, phase_y, kind='linear', bounds_error=False, fill_value=np.nan)
        y_upsampled = f(t)
        r, phase_upsampled = fc.cart2polar(x_upsampled, y_upsampled)
        phase_upsampled = np.rad2deg(phase_upsampled)

        df = pd.DataFrame({'t': t, 'phase': phase_upsampled})
        return df

    temp_df = im_peaks_df.groupby(['fly_id', 'unique_trial_id', 'trial_type']).apply(get_upsampled).reset_index()

    peaks_melted = pd.melt(temp_df[temp_df['unique_trial_id'].isin(trial_ids)],
                           id_vars=['fly_id', 'unique_trial_id', 'trial_type', 't'],
                           value_vars=['phase']
                           )

    mean_peaks = peaks_melted.groupby(['trial_type',
                                       't', 'variable'], as_index=False).agg(mean=('value', fc.circmean),
                                                                             sem=('value', fc.circ_stderror))

    peaks_melted_flipped = peaks_melted.copy()
    peaks_melted_flipped['flip'] = peaks_melted_flipped['trial_type'].map({'down': 1, 'up': -1}).astype(int)
    peaks_melted_flipped['flipped_value'] = peaks_melted_flipped['value'] * peaks_melted_flipped['flip']
    mean_peaks_flipped = peaks_melted_flipped.groupby([
        't', 'variable'], as_index=False).agg(mean=('flipped_value', fc.circmean),
                                              sem=('flipped_value', fc.circ_stderror))

    ##### might want to re-consolidate, dont you want to use abf_peaks?!
    def get_upsampled_xstim(data):
        trial_time = data['trial_time'].values
        idx = np.where(trial_time == 0)[0]
        xstim = data['xstim'].values
        xstim = fc.wrap(xstim - xstim[idx])
        xstim_x, xstim_y = fc.polar2cart(np.ones(len(xstim)), np.deg2rad(xstim))
        t = np.linspace(-10, 10, 20 * 5)
        # upsampled imaging
        f = interp1d(trial_time, xstim_x, kind='linear', bounds_error=False, fill_value=np.nan)
        x_upsampled = f(t)
        f = interp1d(trial_time, xstim_y, kind='linear', bounds_error=False, fill_value=np.nan)
        y_upsampled = f(t)
        r, xstim_upsampled = fc.cart2polar(x_upsampled, y_upsampled)
        xstim_upsampled = np.rad2deg(xstim_upsampled)

        df = pd.DataFrame({'t': t, 'xstim': xstim_upsampled})
        return df

    temp_df = im_peaks_df.groupby(['fly_id', 'unique_trial_id', 'trial_type']).apply(get_upsampled_xstim).reset_index()

    peaks_melted_abf = pd.melt(temp_df[temp_df['unique_trial_id'].isin(trial_ids)],
                               id_vars=['fly_id', 'unique_trial_id', 'trial_type', 't'],
                               value_vars=['xstim']
                               )

    mean_peaks_abf = peaks_melted_abf.groupby(['trial_type',
                                               't', 'variable'], as_index=False).agg(mean=('value', fc.circmean),
                                                                                     sem=('value', fc.circ_stderror))

    peaks_melted_abf_flipped = peaks_melted_abf.copy()
    peaks_melted_abf_flipped['flip'] = peaks_melted_abf_flipped['trial_type'].map({'down': 1, 'up': -1}).astype(int)
    peaks_melted_abf_flipped['flipped_value'] = peaks_melted_abf_flipped['value'] * peaks_melted_abf_flipped['flip']
    mean_peaks_abf_flipped = peaks_melted_abf_flipped.groupby([
        't', 'variable'], as_index=False).agg(mean=('flipped_value', fc.circmean),
                                              sem=('flipped_value', fc.circ_stderror))

    return peaks_melted_flipped, mean_peaks_flipped, peaks_melted_abf_flipped, mean_peaks_abf_flipped


def get_synthetic_data(VT065306_AD_VT029306_DBD):
    VT065306_AD_VT029306_DBD_synthetic = copy.copy(VT065306_AD_VT029306_DBD)

    # interplate to data to 12 columns
    fb_z = VT065306_AD_VT029306_DBD_synthetic.merged_im_df.filter(regex='fb_c1_.*.*z').values
    x_interp, row_interp = rd.FanShapedBody.interpolate_wedges(self=None, data=fb_z, kind='cubic', dinterp=1.4)

    for i in range(12):
        VT065306_AD_VT029306_DBD_synthetic.merged_im_df['12_roi_' + str(i)] = row_interp[:, i]

    # Max-min normalize so each column so it ranges from -1 to 1
    def max_min_normalize(data):
        rois = data.filter(regex='12_roi_.*').values
        for i in range(np.shape(rois)[1]):
            max_val = np.nanmax(rois[:, i])
            min_val = np.nanmin(rois[:, i])
            data['max_min_12_roi_' + str(i)] = (((rois[:, i] - min_val) / (max_val - min_val)) * 2) - 1
        return data

    VT065306_AD_VT029306_DBD_synthetic.merged_im_df = VT065306_AD_VT029306_DBD_synthetic.merged_im_df.groupby(
        'rec_name').apply(max_min_normalize)

    # comupte phase-to-bar offset for each rec, exludes standing abd bar in back
    def get_offset(data, signal):
        xstim = data['xstim']
        query = '(xstim > -135) & (xstim < 135)& `dforw_boxcar_average_0.5_s`>1'
        idx = data.query(query).index
        offset = fc.circmean(data[signal][idx] - xstim[idx])
        data['phase_offset'] = offset
        return data

    VT065306_AD_VT029306_DBD_synthetic.merged_im_df = VT065306_AD_VT029306_DBD_synthetic.merged_im_df.groupby(
        'rec_name').apply(get_offset, 'fb_c1_phase')

    def shift_signal(data, signal, shift_n_rows):
        data['shifted_' + signal] = data[signal].shift(periods=shift_n_rows)
        return data

    # this will make xstim "later" (1 volume, so ~200 ms), which makes sense, since EPG phase lags
    VT065306_AD_VT029306_DBD_synthetic.merged_im_df = VT065306_AD_VT029306_DBD_synthetic.merged_im_df.groupby(
        'rec_name').apply(shift_signal, signal='xstim', shift_n_rows=1)
    # we also make turning "earlier" because behaviour lags heading-to-goal distance (see Green et al. 2019)
    VT065306_AD_VT029306_DBD_synthetic.merged_im_df = VT065306_AD_VT029306_DBD_synthetic.merged_im_df.groupby(
        'rec_name').apply(shift_signal, signal='dheading_boxcar_average_0.5_s', shift_n_rows=-1)

    # shift xstim by fc2-to-bar offset note that phase_offset is not computed using shifted xstim (this is also the
    # case with EPG data, and in Green et al) but should not have much of a difference when using FC2 phase
    VT065306_AD_VT029306_DBD_synthetic.merged_im_df['EPG_phase'] = VT065306_AD_VT029306_DBD_synthetic.merged_im_df[
                                                                       'shifted_xstim'] + \
                                                                   VT065306_AD_VT029306_DBD_synthetic.merged_im_df[
                                                                       'phase_offset']

    # TODO put model stuff outside?
    bridge_angles = [337.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 22.5, 67.5, 112.5, 157.5, 202.5,
                     247.5, 292.5, 337.5, 22.5]

    # compute synthetic heading signal
    for i in range(18):
        # bridge angle - phase is equivalent to heading - bridge angle
        VT065306_AD_VT029306_DBD_synthetic.merged_im_df['pb_roi_' + str(i)] = np.cos(
            np.deg2rad(bridge_angles[i]) - np.deg2rad(VT065306_AD_VT029306_DBD_synthetic.merged_im_df['EPG_phase']))

    def get_f(pb, fb):
        # these are from Vm so they are prob completely off
        # Parameters from data fit
        a = 29.2282
        b = 2.1736
        c = -0.7011
        d = 0.6299

        V = (d * fb) + pb
        f = a * np.log(1 + np.exp(b * (V + c)))
        return f

    def get_rml(data):
        pb = data.filter(regex="pb_roi_.*").values
        fb = data.filter(regex="max_min_12_roi_.*").values
        r_pfl3_pb = np.array([pb[:, 2], pb[:, 3], pb[:, 4], pb[:, 4],
                              pb[:, 5], pb[:, 6], pb[:, 6], pb[:, 7],
                              pb[:, 8], pb[:, 8], pb[:, 9], pb[:, 10]])

        l_pfl3_pb = np.array([pb[:, 7], pb[:, 8], pb[:, 9], pb[:, 9],
                              pb[:, 10], pb[:, 11], pb[:, 11], pb[:, 12],
                              pb[:, 13], pb[:, 13], pb[:, 14], pb[:, 15]])
        r_pfl3 = get_f(r_pfl3_pb.T, fb)
        l_pfl3 = get_f(l_pfl3_pb.T, fb)
        data['rml'] = np.sum(r_pfl3, axis=1) - np.sum(l_pfl3, axis=1)
        return data

    # predicted PFL3 activity
    VT065306_AD_VT029306_DBD_synthetic.merged_im_df = VT065306_AD_VT029306_DBD_synthetic.merged_im_df.groupby(
        'rec_name').apply(get_rml)

    # we shift R-L since activity lags heading-to-goal (see PFL3 transients plots)
    VT065306_AD_VT029306_DBD_synthetic.merged_im_df = VT065306_AD_VT029306_DBD_synthetic.merged_im_df.groupby(
        'rec_name').apply(shift_signal, signal='rml', shift_n_rows=-1)

    # z-score PFL3 activity
    def get_rml_zscore(data):
        signal = data['shifted_rml']
        norm_signal = (signal - signal.mean()) / signal.std()
        data['shifted_rml_z'] = norm_signal
        return data

    VT065306_AD_VT029306_DBD_synthetic.merged_im_df = VT065306_AD_VT029306_DBD_synthetic.merged_im_df.groupby(
        'rec_name').apply(get_rml_zscore)

    return VT065306_AD_VT029306_DBD_synthetic


def get_predicted_rml_distance_to_goal(merged_im_df, save=False, savepath=None, fname=None):
    binned_df = ap.get_binned_df(merged_im_df,
                                 bin_values={'distance_to_goal': np.linspace(-180, 180, 9)},
                                 labels=['shifted_rml', 'shifted_dheading_boxcar_average_0.5_s'],
                                 id_vars=['fly_id'],
                                 query='is_fixating==True & (`dforw_boxcar_average_0.5_s`>1)')
    print(len(binned_df['fly_id'].unique()),'flies')
    mean_df = binned_df.groupby(['variable',
                                 'distance_to_goal_bin_center']).agg(mean=('mean_value', np.nanmean),
                                                                     sem=('mean_value', lambda x: sc.stats.sem(x,
                                                                                                               nan_policy='omit'))).reset_index()

    if save:
        mean_df.to_csv(savepath + fname)

    return binned_df, mean_df


def get_synthetic_trials(VT065306_AD_VT029306_DBD_synthetic):
    synthetic_abf_trials_df, \
        synthetic_im_trials_df = get_bar_jump_trials(
        genotypes={'VT065306_AD_VT029306_DBD': VT065306_AD_VT029306_DBD_synthetic})

    def get_mean_rml_during(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values <= 2)
        data['mean_rml_during'] = np.mean(data['rml'].values[idx])
        return data

    synthetic_im_trials_df = synthetic_im_trials_df.groupby(['unique_trial_id']).apply(get_mean_rml_during)

    def get_mean_turning_during(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values <= 2)
        data['mean_turning_during'] = np.mean(data['dheading'].values[idx])
        return data

    synthetic_abf_trials_df = synthetic_abf_trials_df.groupby(['unique_trial_id']).apply(get_mean_turning_during)

    return synthetic_abf_trials_df, synthetic_im_trials_df


def get_predicted_rml_dheading_df(synthetic_abf_trials_df, synthetic_im_trials_df,
                                  save=False, savepath=None, fname=None):
    synthetic_summary_abf_trials_df = synthetic_abf_trials_df.drop_duplicates('unique_trial_id')
    synthetic_summary_abf_trials_df['mean_rml_during'] = synthetic_summary_abf_trials_df['unique_trial_id'].map(
        dict(zip(synthetic_im_trials_df['unique_trial_id'], synthetic_im_trials_df['mean_rml_during'])))
    query = '(jump_pos<=135) & (jump_pos >=-135) & (abs_goal_diff<45) & (mean_amplitude_during>0.3) & jump_in_fix==True'
    valid_trials = synthetic_summary_abf_trials_df.query(query)['unique_trial_id'].tolist()
    predicted_rml_dheading_df = synthetic_summary_abf_trials_df.filter(
        ['unique_trial_id', 'mean_rml_during', 'mean_turning_during']).reset_index(drop=True)
    predicted_rml_dheading_df['selected'] = predicted_rml_dheading_df['unique_trial_id'].isin(valid_trials)
    if save:
        predicted_rml_dheading_df.drop('unique_trial_id', axis=1).to_csv(savepath + fname)
    return predicted_rml_dheading_df


def get_menotaxis_bouts_raster_and_goals(genotypes, save=False, savepath=None, fname=None):
    temp_df = genotypes['60D05'].merged_im_df.filter(
        ['genotype', 'fly_id', 'rec_name', 't', 'is_fixating', 'goal', 'fixation_event_id']).copy()
    temp_df2 = genotypes['VT065306-AD-VT029306-DBD'].merged_im_df.filter(
        ['genotype', 'fly_id', 'rec_name', 't', 'is_fixating', 'goal', 'fixation_event_id']).copy()
    menotaxis_bouts_raster_df = pd.concat([temp_df, temp_df2])
    menotaxis_bouts_raster_df['t_bin'] = pd.cut(menotaxis_bouts_raster_df['t'], bins=np.arange(-0.5, 1536.5)).apply(
        lambda x: np.round(x.mid, 2))
    print(menotaxis_bouts_raster_df.groupby(['genotype'])['is_fixating'].mean())
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
        values='is_fixating').reindex(index=['60D05', 'VT065306-AD-VT029306-DBD'], level='genotype').reindex(
        index=np.arange(0, 100), level='fly_id')

    # goal from fixation events is based on bar position, but here we want heading
    # not ideal place to do this...
    summary_menotaxis_df['goal'] = summary_menotaxis_df['goal'] * -1
    if save:
        summary_menotaxis_df.filter(['genotype', 'fly_id', 'goal']).to_csv(savepath + fname)
    return menotaxis_bouts_raster_df, summary_menotaxis_df


# ------------------ Plotting functions ----------------------- #

def plot_phase_corr(corr_df, genotype_corr, save=False, savepath=None, fname=None):
    order = ['60D05', 'VT065306-AD-VT029306-DBD']
    print(corr_df.groupby(['genotype']).apply(lambda x: len(x['fly_id'].unique())))
    fig = plt.figure(1, [0.6, 0.6], dpi=dpi)
    ax = sns.swarmplot(x="genotype", y="corr", data=corr_df, size=2,
                       order=order,
                       palette=genotype_palette,
                       marker='o', alpha=1, color='grey', clip_on=False)
    lw = 0.5
    xvals = pd.Categorical(genotype_corr['genotype'].values, order).argsort()

    e1 = plt.errorbar(x=xvals + 0.5, y=genotype_corr['mean'].values,
                      yerr=genotype_corr['sem'].values,
                      ls='none', elinewidth=lw, ecolor='k', capsize=1.5, capthick=lw)

    e2 = plt.errorbar(x=xvals + 0.5, y=genotype_corr['mean'].values,
                      xerr=0.2, ls='none', elinewidth=lw, ecolor='k', capsize=None)

    for b in e1[1]:
        b.set_clip_on(False)
    for b in e2[1]:
        b.set_clip_on(False)
    for b in e1[2]:
        b.set_clip_on(False)
    for b in e2[2]:
        b.set_clip_on(False)

    # hack to avoid overlappingg dots
    fig.set_size_inches(0.4, 1.25)
    ax.set_ylim([0, 1])
    ax.set_xticklabels([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    ph.despine_axes([ax], style=['left'], adjust_spines_kws={'pad': 2,
                                                             'lw': axis_lw,
                                                             'ticklen': axis_ticklen})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_phase_xcorr(xcorr_df, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(xcorr_df,
                     unit='genotype',
                     fig_kws={'figsize': [1, 1], 'dpi': dpi}
                     )

    def plot(data):
        genotype = data['genotype'].values[0]
        color = genotype_palette[genotype]

        plt.plot(data['t'], data['mean'], color, lw=1)
        ax = plt.gca()
        ax.fill_between(data['t'],
                        data['mean'] - data['sem'],
                        data['mean'] + data['sem'],
                        color=ph.color_to_facecolor(color), lw=0, )
        ax.set_xlim([-1, 1])
        ax.set_ylim([-0.8, 0.4])
        ax.set_yticks([-0.8, -0.4, 0, 0.4])
        ax.axvline(x=0, ls=':', color='grey', lw=0.5)

    g.map_dataframe(plot)

    ax = plt.gca()
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0, alpha=0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5, alpha=0)

    ph.despine_axes(g.axes, style=['left', 'bottom'],
                    adjust_spines_kws={'pad': 2, 'lw': axis_lw, 'ticklen': axis_ticklen})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_trajectory(recs, rec_name, tlim,
                            save=False, savepath=None, fname=None):
    plt.figure(1, [1, 1], dpi=dpi)
    x = recs[rec_name].abf.df['x']
    y = recs[rec_name].abf.df['y']
    idx = (recs[rec_name].abf.df['t'].values >= tlim[0]) & (recs[rec_name].abf.df['t'].values <= tlim[1])

    ap.plot_trajectory(x, y, idx=idx, color='#a8a8a8', color_idx='k')

    ax = plt.gca()
    sb = ph.add_scalebar(ax, sizex=200, sizey=0, barwidth=1, barcolor='k', loc='lower right',
                         pad=0,
                         bbox_transform=ax.transAxes,
                         )
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_fixation_bouts(abf_fixation_df, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(abf_fixation_df.query('is_fixating==True'),
                     unit='unique_fixation_event_id',
                     fig_kws={'figsize': [1.3815, 1.3815], 'dpi': dpi}
                     )

    def plot(data):
        x = data['x'].values
        y = data['y'].values
        x = x - x[0]
        y = y - y[0]
        plt.plot(x, y, lw=0.5, color='k', alpha=0.2, clip_on=False)
        plt.scatter(0, 0, s=2, color='r', zorder=3)

    g.map_dataframe(plot)

    ax = plt.gca()

    xmin = ax.get_xlim()[0]
    xmax = ax.get_xlim()[1]

    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]

    x_range = xmax - xmin
    y_range = ymax - ymin
    max_range = max(x_range, y_range)
    x_center = (xmax + xmin) / 2
    y_center = (ymax + ymin) / 2
    plt.xlim(x_center - max_range / 2, x_center + max_range / 2)
    plt.ylim(y_center - max_range / 2, y_center + max_range / 2)

    ax.set_aspect('equal')
    ax.axis('off')
    sb = ph.add_scalebar(ax, sizex=200, sizey=0, barwidth=axis_lw * 2, barcolor='k',
                         pad=0,
                         bbox_transform=ax.transAxes,
                         bbox_to_anchor=[0.65, 0.1]
                         )

    print(len(abf_fixation_df.query('is_fixating==True')['unique_fixation_event_id'].unique()))
    print(len(abf_fixation_df.query('is_fixating==True & genotype=="60D05"')['fly_id'].unique()) + len(
        abf_fixation_df.query('is_fixating==True & genotype=="VT065306-AD-VT029306-DBD"')['fly_id'].unique()))

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_trials_phase_and_xstim_zeroed(genotype, variable_palette,
                                       melted_df, im_mean_df, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(
        melted_df.query(f'genotype=="{genotype}" & ((variable=="phase_zeroed")|(variable=="xstim_zeroed"))'),
        unit='unique_trial_id',
        row='variable',
        col='trial_type',
        col_order=['+90 jump', '-90 jump'],
        fig_kws={'figsize': [1.5, 1.75], 'dpi': dpi},
        gridspec_kws={'wspace': 0.2,
                      'hspace': 0.5},
    )

    def plot(data):
        variable = data['variable'].values[0]
        color = variable_palette[variable]
        ph.circplot(data['trial_time'], data['value'], color=color, circ='y', lw=0.25, alpha=0.3)

    g.map_dataframe(plot)

    for ax in g.axes.ravel():
        ax.set_ylim([-180, 180])
        ax.set_yticks([-180, 0, 180])
        ax.set_xlim([-5, 10])
        ax.set_xticks([-5, 0, 5, 10])
        ax.axvspan(0, 2, facecolor=ph.color_to_facecolor('#c4c4c4'), edgecolor='none', zorder=-1)

    ax = g.axes[1, 1]
    trial_type = '-90 jump'
    data = im_mean_df.query(f'trial_type=="{trial_type}" & genotype=="{genotype}" & variable=="phase_zeroed"')
    ph.circplot(data['trial_time'], data['mean'], circ='y', color=variable_palette['phase_zeroed'], ax=ax)

    ax = g.axes[1, 0]
    trial_type = '+90 jump'
    data = im_mean_df.query(f'trial_type=="{trial_type}" & genotype=="{genotype}" & variable=="phase_zeroed"')
    ph.circplot(data['trial_time'], data['mean'], circ='y', color=variable_palette['phase_zeroed'], ax=ax)
    ph.despine_axes(g.axes, style=['left', 'bottom'],
                    adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')

    print(len(melted_df.query(f'trial_type=="+90 jump" & genotype=="{genotype}"').drop_duplicates('unique_trial_id')),
          ' +90 trials')
    print(len(melted_df.query(f'trial_type=="-90 jump" & genotype=="{genotype}"').drop_duplicates('unique_trial_id')),
          ' -90 trials')


def plot_change_in_phase(fly_mean_phase_df, genotype_mean_phase_df, save=False, savepath=None, fname=None):
    fly_mean_phase_df_copy = fly_mean_phase_df.copy()
    fly_mean_phase_df_copy['mean'] = fc.wrap(fly_mean_phase_df_copy['mean'] - 90)
    genotype_mean_phase_df_copy = genotype_mean_phase_df.copy()
    genotype_mean_phase_df_copy['mean'] = fc.wrap(genotype_mean_phase_df_copy['mean'] - 90)

    order = ['60D05', 'VT065306-AD-VT029306-DBD']
    fig = plt.figure(1, [0.6, 0.6], dpi=dpi)
    ax = sns.swarmplot(x="genotype", y="mean", data=fly_mean_phase_df_copy, size=2.5,
                       order=order,
                       palette=genotype_palette,
                       marker='o', alpha=1, color='grey', clip_on=False)

    lw = 0.5
    xvals = pd.Categorical(genotype_mean_phase_df_copy['genotype'].values, order).argsort()
    e1 = plt.errorbar(x=xvals + 0.4, y=genotype_mean_phase_df_copy['mean'].values,
                      yerr=genotype_mean_phase_df_copy['sem'].values,
                      ls='none', elinewidth=lw, ecolor='k', capsize=1.5, capthick=lw, zorder=10)

    e2 = plt.errorbar(x=xvals + 0.4, y=genotype_mean_phase_df_copy['mean'].values,
                      xerr=0.2, ls='none', elinewidth=lw, ecolor='k', capsize=None, zorder=10)

    for b in e1[1]:
        b.set_clip_on(False)
    for b in e2[1]:
        b.set_clip_on(False)
    for e in e1[2]:
        b.set_clip_on(False)
    for b in e2[2]:
        b.set_clip_on(False)

    # hack to avoid overlappingg dots
    fig.set_size_inches(0.5, 1.5)
    ax.set_ylim([-180, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_yticklabels([-90, 0, 90, 180, -90])
    ax.set_xticklabels([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.axhline(y=0, ls=":", lw=0.5, color='k')
    ph.despine_axes([ax], style=['left'], adjust_spines_kws={'pad': 2,
                                                             'lw': axis_lw,
                                                             'ticklen': axis_ticklen
                                                             })

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_change_in_amplitude(fly_mean_diff, genotype_mean_diff, ylim=None,
                             save=False, savepath=None, fname=None):
    order = ['60D05', 'VT065306-AD-VT029306-DBD']
    fig = plt.figure(1, [0.6, 0.6], dpi=dpi)
    ax = sns.swarmplot(x="genotype", y="mean", data=fly_mean_diff, size=2,
                       order=order,
                       palette=genotype_palette,
                       marker='o', alpha=1, color='grey', clip_on=False)

    plt.axhline(y=0, ls=':', color='grey', lw=0.5)
    if ylim is not None:
        plt.ylim(ylim)

    lw = 0.5
    xvals = pd.Categorical(genotype_mean_diff['genotype'].values, order).argsort()
    e1 = plt.errorbar(x=xvals + 0.4, y=genotype_mean_diff['mean'].values,
                      yerr=genotype_mean_diff['sem'].values,
                      ls='none', elinewidth=lw, ecolor='k', capsize=1.5, capthick=lw)

    e2 = plt.errorbar(x=xvals + 0.4, y=genotype_mean_diff['mean'].values,
                      xerr=0.2, ls='none', elinewidth=lw, ecolor='k', capsize=None)

    for b in e1[1]:
        b.set_clip_on(False)
    for b in e2[1]:
        b.set_clip_on(False)
    for e in e1[2]:
        b.set_clip_on(False)
    for b in e2[2]:
        b.set_clip_on(False)

    # hack to avoid overlappingg dots
    fig.set_size_inches(0.3, 0.75)

    # ax.set_ylim([-180,180])

    # ax.set_yticks([-180,-90,0,90,180])
    ax.set_xticklabels([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    ph.despine_axes([ax], style=['left'], adjust_spines_kws={'pad': 2,
                                                             'lw': axis_lw,
                                                             'ticklen': axis_ticklen})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_amplitudes_vs_walking(binned_amplitudes_dfs, genotype, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(binned_amplitudes_dfs.query(f'genotype=="{genotype}"'),
                     row='variable',
                     row_order=['mean_dF/F', 'max_min', 'pva_amplitude'],
                     col='bin_var',
                     fig_kws={'figsize': [1, 1.25], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.5,
                                   'hspace': 0.5
                                   },
                     )

    def plot(data):
        genotype = data['genotype'].values[0]
        color = genotype_palette[genotype]

        plt.plot(data['bin_center'], data['mean'],
                 lw=0.5, color=color, clip_on=False)

        ax = plt.gca()
        ax.fill_between(data['bin_center'],
                        data['mean'] - data['sem'],
                        data['mean'] + data['sem'],
                        color=ph.color_to_facecolor(color), lw=0, clip_on=False)

        variable = data['variable'].values[0]
        if variable == 'mean_dF/F':
            plt.ylim([0.4, 0.7])
        elif variable == 'max_min':
            plt.ylim([0.8, 1.6])

        elif variable == 'pva_amplitude':
            plt.ylim([0.3, 0.5])

        bin_var = data['bin_var'].values[0]
        if bin_var == 'dheading':
            plt.xlim([-200, 200])
        elif bin_var == 'dforw':
            plt.xlim([-1, 10])

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['left', 'bottom'],
                    adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_amplitude_vs_r(binned_amplitudes_r_dfs, genotype, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(binned_amplitudes_r_dfs.query(f'genotype=="{genotype}"'),
                     row='variable',
                     row_order=['mean_dF/F', 'max_min', 'pva_amplitude'],
                     col='bin_var',
                     col_order=['r_30', 'r_60', 'r_120'],
                     fig_kws={'figsize': [1.25, 1.25], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.5,
                                   'hspace': 0.5
                                   },
                     )

    def plot(data):
        genotype = data['genotype'].values[0]
        color = genotype_palette[genotype]

        plt.plot(data['bin_center'], data['mean'],
                 lw=0.5, color=color, clip_on=False)

        ax = plt.gca()
        ax.fill_between(data['bin_center'],
                        data['mean'] - data['sem'],
                        data['mean'] + data['sem'],
                        color=ph.color_to_facecolor(color), lw=0, clip_on=False)

        variable = data['variable'].values[0]
        if variable == 'mean_dF/F':
            plt.ylim([0.4, 0.7])
        elif variable == 'max_min':
            plt.ylim([0.8, 1.6])
        elif variable == 'pva_amplitude':
            plt.ylim([0.3, 0.5])

        plt.xlim([0, 1])

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['left', 'bottom'],
                    adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_trajectory_r(merged_im_df, rec_name, label, save=False, savepath=None, fname=None):
    df = merged_im_df.query(f'rec_name=="{rec_name}"').copy()

    r = df[label].values
    idx = ~np.isnan(r)
    r = r[idx]

    x = df['x'].values[idx]
    y = df['y'].values[idx]
    x = x - x[0]
    y = y - y[0]

    plt.figure(1, [2.5, 2.5], dpi=dpi)

    plt.scatter(x, y, c=plt.cm.viridis(r), s=0.01, clip_on=False)
    plt.scatter(0, 0, s=1, color='r', zorder=3, clip_on=False)

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

    sb = ph.add_scalebar(ax, sizex=200, sizey=0, barwidth=1, barcolor='k', loc='lower right',
                         pad=0,
                         bbox_transform=ax.transAxes,
                         )
    if save:
        plt.savefig(savepath + fname + '.pdf',
                    transparent=True, bbox_inches='tight')

    plt.show()
    fig = plt.figure(1, [0.05, 0.25], dpi=dpi)

    cb = mpl.colorbar.ColorbarBase(plt.gca(), orientation='vertical',
                                   cmap=plt.cm.viridis)

    cb.ax.tick_params(size=2.5, width=0.4)
    cb.outline.set_visible(False)
    if save:
        plt.savefig(savepath + fname + '_corlobar.pdf',
                    transparent=True, bbox_inches='tight')


def plot_example_epg_with_forw(recs, rec_name, tlim, dforw_lim=None, return_axes=False, save=False, savepath=None,
                               fname=None):
    sb_kws = {'sizex': 0, 'sizey': 5, 'barwidth': 0.8, 'barcolor': 'k', 'loc': 'center left', 'pad': -0.5}
    if dforw_lim is None:
        dforw_lim = [-1, 15]
    axes = ap.plot(recs, rec_name,
                   im_colormesh_signals=["pb_c1_roi.*.dF/F"],
                   im_overlay_signals=['pb_c1_phase_offset_subtracted'],
                   bh_overlay_signals=['xstim'],
                   bh_signals=['dforw_boxcar_average_0.5_s'],
                   bh_signals_kws={'dforw_boxcar_average_0.5_s': {'c': '#4a4a4a',
                                                                  'lw': 0.4,
                                                                  'xlim': dforw_lim}
                                   },
                   cmaps=[plt.cm.Greys],
                   phase_color=['grey'],
                   cb_kws=dict(orientation='horizontal',
                               fraction=1,
                               aspect=5,
                               shrink=0.5,
                               pad=0,
                               anchor=(0.0, 1.0),
                               use_gridspec=False),
                   plot_kws={'lw': 0.75},
                   gridspec_kws={
                       'height_ratios': [0.9, 0.1],
                       'width_ratios': [0.4, 0.4, 0.2],
                       'wspace': 0.5,
                       'hspace': 0.05},

                   adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2},
                   tlim=tlim,
                   fig_kws={'figsize': [1.5, 2], 'dpi': dpi},
                   sb_kws=sb_kws,
                   xlabel=False

                   )

    ax = axes[0]
    ph.adjust_spines(ax, [])
    sb = ph.add_scalebar(ax, sizex=0, sizey=5, barwidth=axis_lw * 2, barcolor='k', loc='center left',
                         pad=-0.5,
                         bbox_transform=ax.transAxes,
                         )

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')
    if return_axes:
        return axes


def plot_example_fc2(recs, rec_name, tlim, vlim=None, sb_kws=None, return_axes=False, save=False, savepath=None,
                     fname=None):
    if vlim is None:
        vlim = [0, 1]
    if sb_kws is None:
        sb_kws = {'sizex': 0, 'sizey': 5, 'barwidth': 0.8, 'barcolor': 'k', 'loc': 'center left', 'pad': -0.5}

    axes = ap.plot(recs, rec_name,
                   im_colormesh_signals=["fb_c1_roi.*.dF/F"],
                   im_overlay_signals=['fb_c1_phase_offset_subtracted'],
                   bh_overlay_signals=['xstim'],
                   bh_signals=[],
                   cmaps=[plt.cm.Purples],
                   phase_color=['#ac46cf'],
                   cb_kws=dict(orientation='horizontal',
                               fraction=1,
                               aspect=5,
                               shrink=0.5,
                               pad=0,
                               anchor=(0.0, 1.0),
                               use_gridspec=False),
                   plot_kws={'lw': 0.75},
                   gridspec_kws={
                       'height_ratios': [0.9, 0.1],
                       'wspace': 0.5,
                       'hspace': 0.05},

                   adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2},
                   tlim=tlim,
                   vlims=[vlim],
                   fig_kws={'figsize': [1, 2], 'dpi': dpi},
                   sb=True,
                   sb_kws=sb_kws
                   )

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')
    if return_axes:
        return axes


def plot_example_fc2_with_forw(recs, rec_name, tlim, vlim=None, dforw_lim=None, sb_kws=None, return_axes=False,
                               save=False, savepath=None, fname=None):
    if dforw_lim is None:
        dforw_lim = [-1, 15]

    if vlim is None:
        vlim = [0, 1]
    if sb_kws is None:
        sb_kws = {'sizex': 0, 'sizey': 5, 'barwidth': 0.8, 'barcolor': 'k', 'loc': 'center left', 'pad': -0.5}

    axes = ap.plot(recs, rec_name,
                   im_colormesh_signals=["fb_c1_roi.*.dF/F"],
                   im_overlay_signals=['fb_c1_phase_offset_subtracted'],
                   bh_overlay_signals=['xstim'],
                   bh_signals=['dforw_boxcar_average_0.5_s'],
                   bh_signals_kws={'dforw_boxcar_average_0.5_s': {'c': '#4a4a4a',
                                                                  'lw': 0.4,
                                                                  'xlim': dforw_lim}
                                   },
                   cmaps=[plt.cm.Purples],
                   phase_color=['#ac46cf'],
                   cb_kws=dict(orientation='horizontal',
                               fraction=1,
                               aspect=5,
                               shrink=0.5,
                               pad=0,
                               anchor=(0.0, 1.0),
                               use_gridspec=False),
                   plot_kws={'lw': 0.75},
                   gridspec_kws={
                       'height_ratios': [0.9, 0.1],
                       'width_ratios': [0.4, 0.4, 0.2],
                       'wspace': 0.5,
                       'hspace': 0.05},

                   adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2},
                   tlim=tlim,
                   vlims=[vlim],
                   fig_kws={'figsize': [1.5, 2], 'dpi': dpi},
                   sb=True,
                   sb_kws=sb_kws,
                   xlabel=False
                   )

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')
    if return_axes:
        return axes


def plot_example_fc2_drift(recs, rec_name, tlim, vlim, save=False, savepath=None, fname=None):
    sb_kws = {'sizex': 0, 'sizey': 5, 'barwidth': 0.8, 'barcolor': 'k',
              'loc': 'center left', 'pad': -0.5, 'bbox_to_anchor': [-0.1, 0.2]}
    axes = plot_example_fc2(recs, rec_name, tlim, vlim, sb_kws, save=False, savepath=None, return_axes=True)

    ax = axes[1]
    tline = [850, 880]
    ax.vlines(x=200, ymin=tline[0], ymax=tline[1], color='#009444', lw=0.5, clip_on=False)

    tline = [895, 925]
    ax.vlines(x=200, ymin=tline[0], ymax=tline[1], color='#f7941d', lw=0.5, clip_on=False)
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_fc_drift_zoom(recs, rec_name, tlim, vlim, save=False, savepath=None, fname=None):
    sb_kws = {'sizex': 0, 'sizey': 5, 'barwidth': 0.8, 'barcolor': 'k',
              'loc': 'center right', 'pad': -0.5}
    plot_example_fc2(recs, rec_name, tlim, vlim, sb_kws, save=False, savepath=None)
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_phase_transients(peaks_melted_flipped, mean_peaks_flipped, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(peaks_melted_flipped,
                     unit='unique_trial_id',
                     fig_kws={'figsize': [0.75, 1.5], 'dpi': dpi},
                     #                gridspec_kws={'wspace':0.2,
                     #                              'hspace':0.5
                     #                             },
                     )

    def plot(data):
        variable = data['variable'].values[0]
        if variable == 'xstim':
            color = 'k'
        else:
            color = genotype_palette['VT065306-AD-VT029306-DBD']
        ph.circplot(data['flipped_value'], data['t'], lw=0.25, alpha=0.25, circ='x',
                    color=color)
        ax = plt.gca()
        plt.xlim([-180, 180])
        plt.xticks([-180, 0, 180])
        plt.yticks([-2, -1, 0, 1, 2])
        plt.ylim([-2, 2])
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['left', 'top'], adjust_spines_kws={'pad': 4, 'lw': axis_lw, 'ticklen': axis_ticklen})

    variable = mean_peaks_flipped['variable'].values[0]
    if variable == 'xstim':
        color = 'k'
    else:
        color = genotype_palette['VT065306-AD-VT029306-DBD']

    ph.circplot(mean_peaks_flipped['mean'], mean_peaks_flipped['t'], lw=1, alpha=1, circ='x',
                color=color)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_phase_transient(recs, im_peaks_df, unique_trial_id, save=False, savepath=None, fname=None):
    t0 = im_peaks_df.query(f'unique_trial_id=="{unique_trial_id}"')['t_0'].values[0]
    rec_name = im_peaks_df.query(f'unique_trial_id=="{unique_trial_id}"')['rec_name'].values[0]
    tlim = [t0 - 6, t0 + 6, ]

    print(rec_name, tlim)
    axes = ap.plot(recs, rec_name,
                   im_colormesh_signals=["fb_c1_roi.*.dF/F"],
                   im_overlay_signals=['fb_c1_phase_offset_subtracted'],
                   im_signals=[],
                   bh_overlay_signals=['xstim'],
                   bh_signals=[],
                   cmaps=[plt.cm.Purples, plt.cm.Purples],
                   phase_color=['#ac46cf'],
                   cb_kws=dict(orientation='horizontal',
                               fraction=1,
                               aspect=5,
                               shrink=0.5,
                               pad=0,
                               anchor=(0.0, 1.0),
                               use_gridspec=False),
                   plot_kws={'lw': 0.75},
                   gridspec_kws={
                       #                      'height_ratios':[0.98,0.02],
                       'height_ratios': [0.9, 0.1],
                       'wspace': 0.5,
                       'hspace': 0.05},

                   adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2},
                   tlim=tlim,
                   vlims=[[0, 1], None],
                   fig_kws={'figsize': [1.5, 1.5], 'dpi': dpi},
                   )

    ax = axes[0]

    ph.adjust_spines(ax, [])
    sb = ph.add_scalebar(ax, sizex=0, sizey=2, barwidth=axis_lw * 2, barcolor='k', loc='center left',
                         pad=-0.5,
                         bbox_transform=ax.transAxes,
                         )

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_fc2_timeframe(VT065306_AD_VT029306_DBD, save=False, savepath=None, fnames=None):
    temp_vals = VT065306_AD_VT029306_DBD.merged_im_df.filter(regex='fb_c1_.*.*dF/F').values[200, :]

    plt.figure(1, [1, 1], dpi=dpi)
    color = genotype_palette['VT065306-AD-VT029306-DBD']
    plt.scatter(np.arange(16), temp_vals, color=color, s=1)
    plt.plot(np.arange(16), temp_vals, color=color, lw=0.5)

    plt.axhline(y=np.mean(temp_vals), ls=':', color='grey', lw=0.5)

    plt.axhline(y=np.min(temp_vals), ls=':', color='grey', lw=0.5)

    plt.axhline(y=np.max(temp_vals), ls=':', color='grey', lw=0.5)

    ax = plt.gca()
    ph.despine_axes([ax], style=['left'], adjust_spines_kws={'pad': 5, 'lw': axis_lw, 'ticklen': axis_ticklen})
    if save:
        plt.savefig(savepath + fnames[0] + '_activity_profile.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()

    step = (2 * np.pi) / 16
    angles = (np.arange(0, 2 * np.pi, step) + step / 2.) + np.pi / 2

    x, y = fc.polar2cart(temp_vals, angles)

    plt.figure(1, [1, 1], dpi=dpi)
    for x_i, y_i in zip(x, y):
        ax = plt.gca()
        ax.arrow(0, 0, x_i, y_i, head_width=0.1, head_length=0.1, fc=ph.purple, ec=ph.purple, lw=0.5, alpha=0.5)

    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.axis('off')

    ax.arrow(0, 0, x.mean(), y.mean(), head_width=0.2, head_length=0.2, fc=ph.purple, ec=ph.purple, lw=1)

    if save:
        plt.savefig(savepath + fnames[1] + '_PVA_calc.pdf',
                    transparent=True, bbox_inches='tight')

    plt.show()

    plt.figure(1, [1, 1], dpi=dpi)
    # for x_i,y_i in zip(x,y):
    #     ax=plt.gca()
    #     ax.arrow(0, 0, x_i, y_i, head_width=0.1, head_length=0.1, fc=ph.purple, ec=ph.purple,lw=0.5,alpha=0.5)

    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.axis('off')
    ax = plt.gca()

    ax.arrow(0, 0, x.mean(), y.mean(), head_width=0.2, head_length=0.2, fc=ph.purple, ec=ph.purple, lw=1)
    if save:
        plt.savefig(savepath + fnames[1] + '_PVA_vector.pdf',
                    transparent=True, bbox_inches='tight')


def plot_predicted_rml_vs_distance_to_goal(mean_df, save=False, savepath=None, fname=None):
    plt.figure(1, [1, 1], dpi=dpi)

    ax = plt.gca()

    data = mean_df.query('variable=="shifted_rml"')
    x = data['distance_to_goal_bin_center']
    mean = data['mean']
    sem = data['sem']
    ax.plot(x, mean, lw=0.5, color='k')
    ax.scatter(x, mean, s=2, color='k')

    e1 = ax.errorbar(x=x, y=mean,
                     yerr=sem,
                     ls='none', elinewidth=0.5, ecolor='k', capsize=1.5, capthick=0.5, zorder=10)

    ax.set_ylim([-30, 30])
    ax.set_yticks([-30, -15, 0, 15, 30])
    ax.set_xlim([-180, 180])
    ax.set_xticks([-180, -90, 0, 90, 180])

    ax2 = ax.twinx()

    data = mean_df.query('variable=="shifted_dheading_boxcar_average_0.5_s"')
    x = data['distance_to_goal_bin_center']
    mean = data['mean']
    sem = data['sem']
    ax2.plot(x, mean, color='grey', lw=0.5)
    ax2.scatter(x, mean, s=2, color='grey')
    ax2.set_ylim([-100, 100])

    ax2.tick_params(axis='y', colors='grey')
    ax2.spines['right'].set_color('grey')

    e2 = ax2.errorbar(x=x, y=mean,
                      yerr=sem,
                      ls='none', elinewidth=0.5, ecolor='grey', capsize=1.5, capthick=0.5, zorder=10)

    ph.adjust_spines(ax, spines=['bottom', 'left'], pad=2, lw=axis_lw, ticklen=axis_ticklen)
    ph.adjust_spines(ax2, spines=['bottom', 'right'], pad=2, lw=axis_lw, ticklen=axis_ticklen)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_synthetic_example_trace(VT065306_AD_VT029306_DBD_synthetic, save=False, savepath=None, fname=None):
    rec_name = '2022_06_13_0008'
    tlim = [298, 310]
    axes = ap.plot(VT065306_AD_VT029306_DBD_synthetic, rec_name,
                   im_colormesh_signals=["max_min_12_roi_.*", "pb_roi_*."],
                   im_overlay_signals=['fb_c1_phase_offset_subtracted'],
                   im_signals=['rml'],
                   bh_overlay_signals=['xstim'],
                   bh_signals=['dforw_boxcar_average_0.5_s', 'dheading_boxcar_average_0.5_s'],
                   cmaps=[plt.cm.Purples, plt.cm.Greys],
                   phase_color=['#ac46cf'],
                   cb_kws=dict(orientation='horizontal',
                               fraction=1,
                               aspect=5,
                               shrink=0.5,
                               pad=0,
                               anchor=(0.0, 1.0),
                               use_gridspec=False),
                   plot_kws={'lw': 0.75},
                   gridspec_kws={
                       #                      'height_ratios':[0.98,0.02],
                       'height_ratios': [0.9, 0.1],
                       'wspace': 0.5,
                       'hspace': 0.05},

                   adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2},
                   tlim=tlim,
                   vlims=[[-1, 1], [-1, 1]],
                   fig_kws={'figsize': [4, 2], 'dpi': dpi},
                   )

    ax = axes[-1]
    ax.set_xlim([-60, 60])
    ax = axes[0]
    ph.adjust_spines(ax, [])
    sb = ph.add_scalebar(ax, sizex=0, sizey=2, barwidth=axis_lw * 2, barcolor='k', loc='center left',
                         pad=-0.5,
                         bbox_transform=ax.transAxes,
                         )
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_predicted_rml_vs_dheading(predicted_rml_dheading_df, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(predicted_rml_dheading_df,
                     unit='unique_trial_id',
                     fig_kws={'figsize': [1, 1], 'dpi': dpi}
                     )

    def plot(data):
        x = data['mean_rml_during'].values
        y = data['mean_turning_during'].values
        unique_trial_id = data['unique_trial_id'].values[0]
        selected = data['selected'].values[0]
        if unique_trial_id == "2022_06_13_0008_1":
            color = ph.red
            plt.scatter(x, y, s=4, color=color, linewidths=0, clip_on=False, zorder=3)
        else:
            if selected:
                color = 'k'
                plt.scatter(x, y, s=4, color=color, linewidths=0, clip_on=False, zorder=3)
            else:
                color = 'grey'
                plt.scatter(x, y, s=4, color=color, linewidths=0, clip_on=False, alpha=0.75)

        plt.ylim([-150, 150])
        plt.yticks([-150, 0, 150])
        plt.xlim([-70, 70])
        plt.xticks([-70, 0, 70])

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['left', 'bottom'],
                    adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen})

    ax = plt.gca()
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(3))
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(3))
    ax.grid(b=True, which='major', color='w', linewidth=1.0, alpha=0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5, alpha=0)

    trials = predicted_rml_dheading_df.copy()
    x = trials['mean_rml_during'].astype(float)
    y = trials['mean_turning_during']
    print(np.corrcoef(x, y)[0][1])

    trials = predicted_rml_dheading_df.query('selected==True').copy()
    x = trials['mean_rml_during'].astype(float)
    y = trials['mean_turning_during']
    print(np.corrcoef(x, y)[0][1])

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


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
        if genotype == "VT065306-AD-VT029306-DBD":
            columns = [
                't_abf',
                'fb_c1_roi_1_F',
                'fb_c1_roi_2_F',
                'fb_c1_roi_3_F',
                'fb_c1_roi_4_F',
                'fb_c1_roi_5_F',
                'fb_c1_roi_6_F',
                'fb_c1_roi_7_F',
                'fb_c1_roi_8_F',
                'fb_c1_roi_9_F',
                'fb_c1_roi_10_F',
                'fb_c1_roi_11_F',
                'fb_c1_roi_12_F',
                'fb_c1_roi_13_F',
                'fb_c1_roi_14_F',
                'fb_c1_roi_15_F',
                'fb_c1_roi_16_F',
                'fb_c2_roi_1_F',
                'fb_c2_roi_2_F',
                'fb_c2_roi_3_F',
                'fb_c2_roi_4_F',
                'fb_c2_roi_5_F',
                'fb_c2_roi_6_F',
                'fb_c2_roi_7_F',
                'fb_c2_roi_8_F',
                'fb_c2_roi_9_F',
                'fb_c2_roi_10_F',
                'fb_c2_roi_11_F',
                'fb_c2_roi_12_F',
                'fb_c2_roi_13_F',
                'fb_c2_roi_14_F',
                'fb_c2_roi_15_F',
                'fb_c2_roi_16_F',
            ]
        elif genotype == "60D05":
            columns = [
                't_abf',
                'pb_c1_roi_2_F',
                'pb_c1_roi_3_F',
                'pb_c1_roi_4_F',
                'pb_c1_roi_5_F',
                'pb_c1_roi_6_F',
                'pb_c1_roi_7_F',
                'pb_c1_roi_8_F',
                'pb_c1_roi_9_F',
                'pb_c1_roi_10_F',
                'pb_c1_roi_11_F',
                'pb_c1_roi_12_F',
                'pb_c1_roi_13_F',
                'pb_c1_roi_14_F',
                'pb_c1_roi_15_F',
                'pb_c1_roi_16_F',
                'pb_c1_roi_17_F',
                'pb_c2_roi_2_F',
                'pb_c2_roi_3_F',
                'pb_c2_roi_4_F',
                'pb_c2_roi_5_F',
                'pb_c2_roi_6_F',
                'pb_c2_roi_7_F',
                'pb_c2_roi_8_F',
                'pb_c2_roi_9_F',
                'pb_c2_roi_10_F',
                'pb_c2_roi_11_F',
                'pb_c2_roi_12_F',
                'pb_c2_roi_13_F',
                'pb_c2_roi_14_F',
                'pb_c2_roi_15_F',
                'pb_c2_roi_16_F',
                'pb_c2_roi_17_F',

            ]
        df = df.filter(columns).rename({'t_abf': 't'}, axis=1).reset_index(drop=True)
        df.to_hdf(PROCESSED_DATA_PATH + genotype + os.path.sep + rec_name + '_im.h5',
                  key='df', mode='w')

    summary_recs = []
    for genotype, recs in genotypes.items():
        recs.merged_abf_df.groupby(['rec_name']).apply(save_abf_as_hd5f)
        recs.merged_im_df.groupby(['rec_name']).apply(save_im_as_hd5f)
        summary_recs.append(recs.merged_im_df.drop_duplicates('rec_name').copy().sort_values(['fly_id']).filter(
            ['genotype', 'rec_name', 'fly_id']).reset_index(drop=True))

    pd.concat(summary_recs).reset_index(drop=True).to_csv(PROCESSED_DATA_PATH + 'summary.csv')
