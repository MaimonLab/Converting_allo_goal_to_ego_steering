"""fc2_stim.py

Analysis and plotting functions for FC2_stimulation.ipynb

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

import astropy.stats.circstats as astropy_circstats
import cv2

# ------------------ Plotting parameters ----------------------- #

dpi = 300
axis_lw = 0.4
axis_ticklen = 2.5
pad = 2

font = {'family': 'arial',
        'weight': 'normal',
        'size': 5}
mpl.rc('font', **font)

stimulation_palette = {'col1': ph.blue, 'col2': ph.orange}
order = ['VT065306-AD-VT029306-DBD-sytGC7f-Chr', 'VT065306-AD-VT029306-DBD-sytGC7f']
is_in_stim_palette = {True: ph.purple, False: 'k'}

genotype_palette = {'VT065306-AD-VT029306-DBD-sytGC7f-Chr': 'k',
                    'VT065306-AD-VT029306-DBD-sytGC7f': 'grey'}


# ---------------------------- Load data ---------------------------- #
def load_data(DATA_PATH, reprocess=False):
    cschrimson_rec_names = [

        (
            '2022_02_06_0002',
            '2022_02_06_0003',
        ),

        (
            '2022_02_07_0001',
            '2022_02_07_0002',
        ),

        (
            '2022_02_07_0003',
            '2022_02_07_0004',
        ),

        (
            '2022_02_07_0005',
        ),

        (
            '2022_02_07_0008',
            '2022_02_07_0009',
        ),

        (
            '2022_02_07_0010',
            '2022_02_07_0011',
        ),

        (
            '2022_02_08_0001',
            '2022_02_08_0002',
        ),

        (
            '2022_02_08_0003',
            '2022_02_08_0005',
        ),

        (
            '2022_02_08_0006',
            '2022_02_08_0008',
        ),

        (
            '2022_02_08_0010',
            '2022_02_08_0011',
        ),

        (
            '2022_02_08_0012',
            '2022_02_08_0013',
        ),

        (
            '2022_02_14_0006',
            '2022_02_14_0007',
        ),

        (
            '2022_02_14_0008',
            '2022_02_14_0009',
        ),

        (
            '2022_02_14_0010',
            '2022_02_14_0011',
        ),

        (
            '2022_02_15_0008',
        ),

        (
            '2022_02_15_0010',
            '2022_02_15_0011',
        ),

    ]
    parent_folder = DATA_PATH + 'FC2_stimulation' + os.path.sep + 'VT065306-AD-VT029306-DBD-sytGC7f-Chr' + os.path.sep
    chrimson_recs = rd.quickload_experiment(parent_folder + 'nature_VT065306_AD_VT029306_DBD_sytGC7f_Chr.pkl',
                                            rec_names=cschrimson_rec_names,
                                            exp_kws={
                                                'rec_type': rd.ImagingRec,
                                                'parent_folder': parent_folder,
                                                'merge_df': True,
                                                'genotype': 'VT065306-AD-VT029306-DBD-sytGC7f-Chr',
                                                'trim_times': None,
                                            },
                                            reprocess=reprocess,
                                            bh_kws={'angle_offset': 86,
                                                    'boxcar_average': {'dforw': 0.5, 'dheading': 0.5, }},
                                            roi_types=[rd.FanShapedBody],
                                            roi_kws={'celltypes': {'c1': 'FC2', 'c2': 'FC2'}},
                                            photostim=True,
                                            photostim_kws={
                                                'galvo_label': 'mean_x_galvo_pos_slice_1',
                                                'photostim_scanfield_z': [0, 1, 2]
                                            }
                                            )

    no_cschrimson_rec_names = [

        (
            '2022_02_06_0004',
            '2022_02_06_0005',
        ),

        (
            '2022_02_06_0006',
            '2022_02_06_0007',
        ),

        (
            '2022_02_06_0009',
        ),

        (
            '2022_02_07_0012',
            '2022_02_07_0013',
        ),

        (
            '2022_02_08_0014',
            '2022_02_08_0015',
        ),

        (
            '2022_02_14_0002',
            '2022_02_14_0003',
        ),

        (
            '2022_02_14_0004',
            '2022_02_14_0005',
        ),

        (
            '2022_02_15_0002',
        ),

        (
            '2022_02_15_0004',
        ),

        (
            '2022_02_15_0005',
            '2022_02_15_0006',
        ),

    ]

    parent_folder = DATA_PATH + 'FC2_stimulation' + os.path.sep + 'VT065306-AD-VT029306-DBD-sytGC7f' + os.path.sep
    control_recs = rd.quickload_experiment(parent_folder + 'nature_VT065306_AD_VT029306_DBD_sytGC7f.pkl',
                                           rec_names=no_cschrimson_rec_names,
                                           exp_kws={
                                               'rec_type': rd.ImagingRec,
                                               'parent_folder': parent_folder,
                                               'merge_df': True,
                                               'genotype': 'VT065306-AD-VT029306-DBD-sytGC7f',
                                               'trim_times': None,
                                           },
                                           reprocess=reprocess,
                                           bh_kws={'angle_offset': 86,
                                                   'boxcar_average': {'dforw': 0.5, 'dheading': 0.5, }},
                                           roi_types=[rd.FanShapedBody],
                                           roi_kws={'celltypes': {'c1': 'FC2', 'c2': 'FC2'}},
                                           photostim=True,
                                           photostim_kws={
                                               'galvo_label': 'mean_x_galvo_pos_slice_1',
                                               'photostim_scanfield_z': [0, 1, 2]
                                           }
                                           )

    genotypes = {'VT065306-AD-VT029306-DBD-sytGC7f-Chr': chrimson_recs,
                 'VT065306-AD-VT029306-DBD-sytGC7f': control_recs}
    return genotypes


# ------------------ Data processing functions ----------------------- #

def extra_pre_processing(recs):
    # modifies recs.merged_im_df
    def get_phase(data):
        photostim = data.filter(regex="fb.*.photostim")
        theta, r = rd.ROI.get_pva(None, photostim)
        data['photostim_phase'] = theta
        return data

    recs.merged_im_df = recs.merged_im_df.groupby(['rec_name']).apply(get_phase)
    # subtract fc2 phase offset
    recs = ap.subtract_offset(recs, 'fb_c1_phase',
                              idx='(xstim > -135) & (xstim < 135) & (scanfield_config_name=="control")& `dforw_boxcar_average_0.5_s`>1')

    return recs


def get_photostim_trials(genotypes):
    abf_trials_dfs = []
    im_trials_dfs = []

    for genotype, recs in genotypes.items():
        abf_trials_df, im_trials_df = ap.get_photostim_trials_df(recs, pad_s=60)
        abf_trials_dfs.append(abf_trials_df)
        im_trials_dfs.append(im_trials_df)

    abf_trials_df = pd.concat(abf_trials_dfs, ignore_index=True)
    im_trials_df = pd.concat(im_trials_dfs, ignore_index=True)

    # Get xstim zeroed
    # throughout we use xstim (which is inverse of heading) and we swap heading axis to make it more intuitive
    temp_df = abf_trials_df.query(
        '(trial_time>=0) & (trial_time<30) & `dforw_boxcar_average_0.5_s`>1 & scanfield_config_name=="col1"').groupby(
        ['genotype', 'fly_id'], observed=True, as_index=False).agg(mean_heading_during_col1=('xstim', fc.circmean))
    abf_trials_df = abf_trials_df.set_index(['genotype', 'fly_id']).join(
        temp_df.set_index(['genotype', 'fly_id'])).reset_index()
    abf_trials_df['xstim_zeroed'] = fc.wrap(abf_trials_df['xstim'] - abf_trials_df['mean_heading_during_col1'])

    # repeat for inverse!
    temp_df = abf_trials_df.query(
        '(trial_time>=0) & (trial_time<30) & `dforw_boxcar_average_0.5_s`>1 & scanfield_config_name=="col2"').groupby(
        ['genotype', 'fly_id'], observed=True, as_index=False).agg(mean_heading_during_col2=('xstim', fc.circmean))
    abf_trials_df = abf_trials_df.set_index(['genotype', 'fly_id']).join(
        temp_df.set_index(['genotype', 'fly_id'])).reset_index()
    abf_trials_df['xstim_zeroed2'] = fc.wrap(abf_trials_df['xstim'] - abf_trials_df['mean_heading_during_col2'])

    # not useed
    def get_min_time(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values < 30) & (
                data['dforw_boxcar_average_0.5_s'].values > 1)
        if np.sum(idx) < 250:
            data['min_time'] = False
        else:
            data['min_time'] = True
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_min_time)

    def get_mean_phase_during(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values < 30) & (
                data['dforw_boxcar_average_0.5_s'].values > 1)
        data['mean_phase_during'] = fc.circmean(data['fb_c1_phase'].values[idx])
        return data

    im_trials_df = im_trials_df.groupby(['unique_trial_id']).apply(get_mean_phase_during)

    def get_mean_photostim_phase_during(data):
        # the filtering on forward shouldnt do anyting of couse bc this is the same throughhout a recording for each stim
        # position!
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values < 30) & (
                data['dforw_boxcar_average_0.5_s'].values > 1)
        data['mean_photostim_phase_during'] = fc.circmean(data['photostim_phase'].values[idx])
        return data

    im_trials_df = im_trials_df.groupby(['unique_trial_id']).apply(get_mean_photostim_phase_during)

    def get_mean_xstim_during(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values < 30) & (
                data['dforw_boxcar_average_0.5_s'].values > 1)
        data['mean_xstim_during'] = fc.circmean(data['xstim'].values[idx])
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id']).apply(get_mean_xstim_during)

    def get_stimulated_col(data):
        # could  look at a single timepoint instead of taking mean, since does not change over time
        mean = np.mean(data.query(' (trial_time>=0) & (trial_time<30) ').filter(regex='fb.*.photostim'))
        max_col = np.argmax(mean) + 1
        data['max_col'] = max_col
        return data

    im_trials_df = im_trials_df.groupby(['unique_trial_id']).apply(get_stimulated_col)

    return abf_trials_df, im_trials_df


def get_summary_df(abf_trials_df, im_trials_df):
    summary_im_trials_df = im_trials_df.drop_duplicates('unique_trial_id', ignore_index=True).copy()
    summary_abf_trials_df = abf_trials_df.drop_duplicates('unique_trial_id', ignore_index=True).copy()

    temp_dic = dict(zip(abf_trials_df['unique_trial_id'], abf_trials_df['mean_xstim_during']))
    summary_im_trials_df['mean_xstim_during'] = summary_im_trials_df['unique_trial_id'].map(temp_dic).astype('float')

    summary_abf_trials_df['mean_heading_during_zeroed'] = fc.wrap(
        summary_abf_trials_df['mean_xstim_during'] - summary_abf_trials_df['mean_heading_during_col1'])
    temp_dic = dict(zip(summary_abf_trials_df['unique_trial_id'], summary_abf_trials_df['mean_heading_during_zeroed']))
    summary_im_trials_df['mean_heading_during_zeroed'] = summary_im_trials_df['unique_trial_id'].map(temp_dic).astype(
        'float')

    temp_dic = dict(zip(summary_abf_trials_df['unique_trial_id'], summary_abf_trials_df['min_time']))
    summary_im_trials_df['min_time'] = summary_im_trials_df['unique_trial_id'].map(temp_dic).astype('float')

    return summary_abf_trials_df, summary_im_trials_df


def get_diff_df(summary_im_trials_df):
    diff_df = summary_im_trials_df.groupby(['genotype', 'fly_id', 'scanfield_config_name'],
                                           observed=True,
                                           as_index=False).agg(fly_mean_xstim_during=('mean_xstim_during', fc.circmean),
                                                               fly_mean_phase_during=('mean_phase_during', fc.circmean),
                                                               fly_mean_photostim_phase_during=(
                                                                   'mean_photostim_phase_during', fc.circmean))

    pivot_df = pd.pivot_table(diff_df, index=['genotype', 'fly_id'],
                              columns=['scanfield_config_name'],
                              values=['fly_mean_xstim_during',
                                      'fly_mean_phase_during',
                                      'fly_mean_photostim_phase_during'])

    phase_diff = fc.wrap(pivot_df['fly_mean_phase_during', 'col1'] - pivot_df['fly_mean_phase_during', 'col2'])
    head_diff = fc.wrap(pivot_df['fly_mean_xstim_during', 'col1'] - pivot_df['fly_mean_xstim_during', 'col2'])
    stim_diff = fc.wrap(
        pivot_df['fly_mean_photostim_phase_during', 'col1'] - pivot_df['fly_mean_photostim_phase_during', 'col2'])

    head_diff.rename('head_diff', inplace=True)
    phase_diff.rename('phase_diff', inplace=True)
    stim_diff.rename('stim_diff', inplace=True)

    diff_df = head_diff.to_frame().join(phase_diff).join(stim_diff)
    diff_df.reset_index(inplace=True)

    return diff_df


def get_genotype_mean_head_diff(diff_df, save=False, savepath=None, fname=None):
    diff_df_copy = diff_df.copy()
    genotype_mean_head_diff = diff_df_copy.groupby(['genotype'], as_index=False).agg(mean=('head_diff', fc.circmean),
                                                                                     sem=(
                                                                                         'head_diff', fc.circ_stderror))
    if save:
        pd.concat([diff_df_copy.drop(['phase_diff', 'stim_diff'], axis=1).rename({'head_diff': 'value'}, axis=1),
                   genotype_mean_head_diff]).to_csv(savepath + fname)
    return genotype_mean_head_diff


def get_heading_diff_stats(diff_df):
    mu_deg_chr = fc.circmean(diff_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f-Chr"')['stim_diff'].values)
    mu = np.deg2rad(mu_deg_chr)
    x = np.deg2rad(diff_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f-Chr"')['head_diff'].values)
    mean_chr = fc.circmean(diff_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f-Chr"')['head_diff'].values)
    print('Expected diff for CsChrimson:', mu_deg_chr)
    print('Actual diff for CsChrimson:', mean_chr)
    print('p-val:', astropy_circstats.vtest(x, mu=mu))
    mu_deg_ctl = fc.circmean(diff_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f"')['stim_diff'].values)
    mu = np.deg2rad(mu_deg_ctl)
    x = np.deg2rad(diff_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f"')['head_diff'].values)
    mean_ctl = fc.circmean(diff_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f"')['head_diff'].values)
    print('Expected diff for CsChrimson:', mu_deg_ctl)
    print('Actual diff for CsChrimson:', mean_ctl)
    print('p-val:', astropy_circstats.vtest(x, mu=mu))
    return mu_deg_chr, mu_deg_ctl


def get_zeroed_heading_hist_df(abf_trials_df):
    idx = ('trial_time>=0 & trial_time < 30 & `dforw_boxcar_average_0.5_s`>1')
    hist_df = ap.get_hist_df(abf_trials_df.query(idx), value='xstim_zeroed',
                             id_vars=['genotype', 'scanfield_config_name', 'fly_id'],
                             bins=np.linspace(-180, 180, 37))

    zeroed_heading_hist_df = hist_df.groupby(['genotype',
                                              'scanfield_config_name',
                                              'xstim_zeroed_bin_center'])['norm_counts'].mean()
    zeroed_heading_hist_df = pd.DataFrame(zeroed_heading_hist_df).reset_index()

    zeroed_heading_hist_df['norm_counts'] = zeroed_heading_hist_df.groupby(['genotype', 'scanfield_config_name']).apply(
        lambda x: x['norm_counts'] / np.sum(x['norm_counts'])).values

    return zeroed_heading_hist_df


def get_roi_df(im_trials_df):
    im_trials_df_melted = pd.melt(im_trials_df,
                                  id_vars=['genotype', 'fly_id', 'rec_name', 'unique_trial_id', 'scanfield_config_name',
                                           'trial_time', 'max_col'],
                                  value_vars=im_trials_df.filter(regex='fb_c1_roi_.*.dF/F').columns
                                  )

    im_trials_df_melted_2 = pd.melt(im_trials_df,
                                    id_vars=['genotype', 'fly_id', 'rec_name', 'unique_trial_id',
                                             'scanfield_config_name', 'trial_time', 'max_col'],
                                    value_vars=im_trials_df.filter(regex='fb.*.photostim').columns)

    # careful with this!
    im_trials_df_melted['pct'] = im_trials_df_melted_2['value']

    def get_pct(data):
        t = data['trial_time'].values
        idx = (t > 1) & (t < 2)
        pct = data['pct'].values[idx][0]
        data['pct'] = pct
        return data

    im_trials_df_melted = im_trials_df_melted.groupby(['genotype', 'unique_trial_id', 'variable']).apply(get_pct)
    im_trials_df_melted['is_in_stim'] = im_trials_df_melted['pct'] > 0
    im_trials_df_melted['col'] = im_trials_df_melted['variable'].apply(lambda x: int(x.split('_')[3]))
    im_trials_df_melted['col_deg'] = ((im_trials_df_melted['col'] / 16.) * 360.) - 180.
    im_trials_df_melted['max_col_deg'] = ((im_trials_df_melted['max_col'] / 16.) * 360.) - 180.
    im_trials_df_melted['col_diff'] = im_trials_df_melted['max_col'] - im_trials_df_melted['col']
    im_trials_df_melted['abs_col_diff'] = np.abs(
        fc.wrap(im_trials_df_melted['col_deg'] - im_trials_df_melted['max_col_deg']))

    def get_change_in_deltaF(data):
        idx_before = (data['trial_time'].values >= -5) & (data['trial_time'].values < 0)
        idx_during = (data['trial_time'].values >= 5) & (data['trial_time'].values < 30)
        signal = data['value'].values
        data['mean_before'] = np.mean(signal[idx_before])
        data['mean_during'] = np.mean(signal[idx_during])
        data['mean_during_over_before'] = data['mean_during'] / data['mean_before']
        return data

    im_trials_df_melted = im_trials_df_melted.groupby(['unique_trial_id', 'variable']).apply(get_change_in_deltaF)
    return im_trials_df_melted


def get_in_vs_out_stim_df(roi_df):
    mean_df = roi_df.groupby(['genotype', 'fly_id', 'is_in_stim', 'trial_time'], as_index=False).agg(
        mean_value=('value', np.nanmean),
        std_err_value=('value', sc.stats.sem))
    in_vs_out_stim_df = mean_df.groupby(['genotype', 'is_in_stim', 'trial_time'], as_index=False).agg(
        mean_value=('mean_value', np.nanmean),
        std_err_value=('mean_value', sc.stats.sem))

    return in_vs_out_stim_df


def get_dist_to_stim_df(roi_df, save=False, savepath=None, fname=None):
    dist_to_stim_mean_df = roi_df.query('is_in_stim==False').groupby(['genotype', 'fly_id', 'abs_col_diff'],
                                                                     as_index=False, observed=True).agg(
        fly_mean_during_over_before=('mean_during_over_before', np.nanmean))
    dist_to_stim_fly_df = dist_to_stim_mean_df.groupby(['genotype', 'abs_col_diff'],
                                                       as_index=False, observed=True).agg(
        mean_value=('fly_mean_during_over_before', np.nanmean),
        sem=('fly_mean_during_over_before', sc.stats.sem))

    print(len(dist_to_stim_mean_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f-Chr"')['fly_id'].unique()),
          'flies')
    if save:
        dist_to_stim_mean_df_copy = dist_to_stim_mean_df.copy()
        ang_to_col = dict(zip(dist_to_stim_mean_df_copy['abs_col_diff'].unique().tolist(), np.arange(7) + 2))
        dist_to_stim_mean_df_copy['distance'] = dist_to_stim_mean_df_copy['abs_col_diff'].map(ang_to_col)
        dist_to_stim_mean_df_copy.rename({'fly_mean_during_over_before': 'value'}, axis=1, inplace=True)

        dist_to_stim_fly_df_copy = dist_to_stim_fly_df.copy()
        dist_to_stim_fly_df_copy['distance'] = dist_to_stim_fly_df_copy['abs_col_diff'].map(ang_to_col)
        dist_to_stim_fly_df_copy.rename({'mean_value': 'mean'}, axis=1, inplace=True)
        pd.concat([dist_to_stim_mean_df_copy, dist_to_stim_fly_df_copy]).filter(['genotype',
                                                                                 'fly_id',
                                                                                 'distance',
                                                                                 'value',
                                                                                 'mean',
                                                                                 'sem']).to_csv(savepath + fname)

    return dist_to_stim_mean_df, dist_to_stim_fly_df


def get_distance_to_target_heading_vals(abf_trials_df, diff_df):
    def get_distance_to_target_heading(data):
        scanfield_config_name = data['scanfield_config_name'].values[0]
        genotype = data['genotype'].values[0]
        fly_id = data['fly_id'].values[0]

        stim_diff = diff_df.query(f'genotype=="{genotype}" & fly_id=={fly_id}')['stim_diff'].values[0]
        if scanfield_config_name == 'col2':
            # I checked signs, and I think this makes sense:
            # we want y = xstim_goal_col2 - xstim -> this is the same as heading_goal_col2 - heading, which is what we call distance to goal
            # stim_diff = xstim_goal_col1 - xstim_goal_col2
            # xstim_zeroed = xstim - xstim_goal_col1
            y = fc.wrap(-stim_diff - data['xstim_zeroed'])

        elif scanfield_config_name == 'col1':
            # I checked signs, and I think this makes sense:
            # we want y = xstim_goal_col1 - xstim
            # stim_diff = xstim_goal_col1 - xstim_goal_col2
            # xstim_zeroed2 = xstim - xstim_goal_col2
            y = fc.wrap(stim_diff - data['xstim_zeroed2'])
        data['distance_to_target_heading'] = y
        return data

    abf_trials_df = abf_trials_df.groupby(['genotype', 'fly_id', 'scanfield_config_name'], as_index=False).apply(
        get_distance_to_target_heading)

    def get_distance_to_target_heading_before(data):
        idx = (data['trial_time'].values >= -2) & (data['trial_time'].values < 0)
        data['distance_to_target_heading_before'] = fc.circmean(data['distance_to_target_heading'].values[idx])
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id'], as_index=False).apply(
        get_distance_to_target_heading_before)

    def get_is_standing_before(data):
        idx = (data['trial_time'].values >= -5) & (data['trial_time'].values < 0)
        data['is_standing_before'] = (0 == (np.sum(data['dforw_boxcar_average_0.5_s'].values[idx] > 1)))
        return data

    abf_trials_df = abf_trials_df.groupby(['unique_trial_id'], as_index=False).apply(get_is_standing_before)

    abf_trials_df['abs_distance_to_target_heading'] = np.abs(abf_trials_df['distance_to_target_heading'])
    abf_trials_df['distance_to_target_heading_before_bin_center'] = pd.cut(
        abf_trials_df['distance_to_target_heading_before'], bins=np.linspace(-180, 180, 9)).apply(
        lambda x: np.round(x.mid, 2))

    return abf_trials_df


def get_dforw_vs_dist_df(abf_trials_df, save=False, savepath=None, fname=None):
    dforw_fly_df = ap.get_binned_df(abf_trials_df, {'distance_to_target_heading': np.linspace(-180, 180, 25)},
                                    labels=['dforw_boxcar_average_0.5_s'],
                                    query='(trial_time>=0) & (trial_time<30)',
                                    id_vars=['genotype', 'fly_id'], metric=None, observed=True)
    dforw_mean_df = dforw_fly_df.groupby(['genotype', 'distance_to_target_heading_bin_center'], as_index=False).agg(
        mean=('mean_value', np.mean),
        sem=('mean_value', sc.stats.sem))

    if save:
        dforw_mean_df.rename({'distance_to_target_heading_bin_center': 'distance_to_predicted_goal'}, axis=1).to_csv(
            savepath + fname)

    return dforw_mean_df


def get_dist_target_heading_standing_df(abf_trials_df):
    # could make this make this less confusing?
    target_heading_mean_df_standing1 = abf_trials_df.query('trial_time>-10 & trial_time < 40').groupby(['genotype',
                                                                                                        'is_standing_before',
                                                                                                        'trial_time'],
                                                                                                       as_index=False,
                                                                                                       observed=True).agg(
        mean_value=('abs_distance_to_target_heading', fc.circmean),
        std_err_value=('abs_distance_to_target_heading', fc.circ_stderror))

    target_heading_mean_df_standing1['variable'] = 'abs_distance_to_target_heading'

    target_heading_mean_df_standing2 = abf_trials_df.query('trial_time>-10 & trial_time < 40').groupby(['genotype',
                                                                                                        'is_standing_before',
                                                                                                        'trial_time'],
                                                                                                       as_index=False,
                                                                                                       observed=True).agg(
        mean_value=('dforw_boxcar_average_0.5_s', np.mean),
        std_err_value=('dforw_boxcar_average_0.5_s', sc.stats.sem))
    #

    target_heading_mean_df_standing2['variable'] = 'dforw_boxcar_average_0.5_s'

    target_heading_mean_df_standing = pd.concat([target_heading_mean_df_standing1, target_heading_mean_df_standing2])

    return target_heading_mean_df_standing


def get_dist_to_target_heading_df(abf_trials_df):
    target_heading_mean_df_dist = abf_trials_df.query('trial_time>-10 & trial_time < 40').groupby(['genotype',
                                                                                                   'distance_to_target_heading_before_bin_center',
                                                                                                   'trial_time'],
                                                                                                  as_index=False).agg(
        mean_value=('distance_to_target_heading', fc.circmean),
        std_err_value=('distance_to_target_heading', fc.circ_stderror))
    return target_heading_mean_df_dist


# ------------------ Plotting functions ----------------------- #

def plot_stim_diff_hist(diff_df, save=False, savepath=None, fname=None):
    print(fc.circmean(diff_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f-Chr"')['stim_diff'], 0, 360),
          'diff. in stim. positions for CsChrimson flie')
    print(fc.circmean(diff_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f"')['stim_diff'], 0, 360),
          'diff. in stim. positions for no CsChrimson flie')

    plot_df = diff_df.copy()
    plot_df['stim_diff_shifted'] = fc.wrap(plot_df['stim_diff'], 0, 360)

    def plot(data):
        color = genotype_palette[data['genotype'].values[0]]
        hist, bin_edges = np.histogram(data['stim_diff_shifted'].values, np.linspace(0, 360, 19),
                                       density=False)
        hist = hist / hist.sum()
        bin_diff = np.diff(bin_edges)
        widths = bin_diff / 2.
        bin_centers = bin_edges[:-1] + widths
        plt.bar(bin_centers, hist, clip_on=False, color=color, width=widths[0] * 2)
        plt.xlim([0, 360])
        plt.xticks([0, 180, 360])
        plt.ylim([0, 0.75])
        plt.yticks([0, 0.75])

    g = ph.FacetGrid(plot_df, col='genotype', row=None,
                     fig_kws={'figsize': [1.25, 0.5], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.5,
                                   'hspace': 0},
                     col_order=order
                     )
    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['left', 'bottom'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': pad})
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_phase_diff_hist(diff_df, save=False, savepath=None, fname=None):
    print(fc.circmean(diff_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f-Chr"')['phase_diff'], 0, 360),
          'diff. in phase positions during stim. for CsChrimson flie')
    print(fc.circmean(diff_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f"')['phase_diff'], 0, 360),
          'diff. in phase positions during stim. for CsChrimson flie')

    plot_df = diff_df.copy()
    plot_df['phase_diff_shifted'] = fc.wrap(plot_df['phase_diff'], 0, 360)

    def plot(data):
        color = genotype_palette[data['genotype'].values[0]]

        hist, bin_edges = np.histogram(data['phase_diff_shifted'].values, np.linspace(0, 360, 19),
                                       density=False)
        hist = hist / hist.sum()
        bin_diff = np.diff(bin_edges)
        widths = bin_diff / 2.
        bin_centers = bin_edges[:-1] + widths
        plt.bar(bin_centers, hist, clip_on=False, color=color, width=widths[0] * 2)
        plt.xlim([0, 360])
        plt.xticks([0, 180, 360])
        plt.ylim([0, 0.4])
        plt.yticks([0, 0.4])

    g = ph.FacetGrid(plot_df, col='genotype', row=None,
                     fig_kws={'figsize': [1.25, 0.5], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.5,
                                   'hspace': 0},
                     #                subplot_kws={'projection':'polar'},
                     col_order=order
                     )

    g.map_dataframe(plot)

    ph.despine_axes(g.axes, style=['left', 'bottom'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': pad})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_head_diff_scatter(diff_df, genotype_mean_head_diff, mu_deg_chr, mu_deg_ctl,
                           save=False, savepath=None, fname=None):
    plot_df = diff_df.copy()
    plot_df['head_diff_shifted'] = fc.wrap(plot_df['head_diff'], -90, 270)

    print('Total flies:\n', plot_df.groupby(['genotype']).apply(lambda x: len(x['fly_id'])))

    fig = plt.figure(1, [0.5, 0.5], dpi=dpi)
    ax = sns.swarmplot(x='genotype', y='head_diff_shifted', data=plot_df, size=1.5,
                       marker='o', alpha=1, color='grey',
                       order=order, palette=genotype_palette)
    lw = 0.5
    xvals = pd.Categorical(genotype_mean_head_diff['genotype'].values, order).argsort()
    e1 = plt.errorbar(x=xvals + 0.4, y=genotype_mean_head_diff['mean'].values,
                      yerr=genotype_mean_head_diff['sem'].values,
                      ls='none', elinewidth=lw, ecolor='k', capsize=1.5, capthick=lw)

    e2 = plt.errorbar(x=xvals + 0.4, y=genotype_mean_head_diff['mean'].values,
                      xerr=0.1, ls='none', elinewidth=lw, ecolor='k', capsize=None)
    ph.error_bar_clip_on([e1, e2])
    expected_chr = fc.wrap(np.array([mu_deg_chr]), -90, 270)
    plt.axhline(y=expected_chr, ls=":", xmin=.1, xmax=0.4,
                lw=0.75,
                color=ph.red,
                )

    expected_ctl = fc.wrap(np.array([mu_deg_ctl]), -90, 270)
    plt.axhline(y=expected_ctl, ls=":", xmin=0.6, xmax=0.9,
                lw=0.75,
                color=ph.red,
                )

    ax.set_yticks([-90, 0, 90, 180, 270])
    ax.set_ylabel('')

    ph.despine_axes([ax], style=['left'], adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': pad})

    # hack to avoid overlappingg dots
    fig.set_size_inches(0.75, 1)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_phase_vs_xtim_chr(summary_im_trials_df, save=False, savepath=None, fname=None):
    def plot(data):
        color = stimulation_palette[data['scanfield_config_name'].values[0]]
        # by convention 0 is left of FB,
        # but this is confusing here, so we make 0 the middle of FB
        x = fc.wrap(data['mean_phase_during'].values - 180)
        # print(np.isnan(data['mean_phase_during'].values))
        plt.scatter(x, data['mean_xstim_during'].values,
                    s=1, alpha=0.75, color=color, clip_on=False)

    g = ph.FacetGrid(summary_im_trials_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f-Chr"'),
                     col_wrap=4,
                     col='fly_id',
                     unit='unique_trial_id',
                     fig_kws={'figsize': [3, 3], 'dpi': dpi},
                     gridspec_kws={
                         'wspace': 0.5,
                         'hspace': 0.5},
                     col_order=np.arange(16) + 1
                     )

    g.map_dataframe(plot)
    for ax in g.axes.ravel():
        ax.set_xticks([-180, 0, 180])
        ax.set_yticks([-180, 0, 180])
        # flip labels bc we are using bar position not heading
        ax.set_yticklabels([180, 0, -180])
        ax.set_xlim([-180, 180])
        ax.set_ylim([-180, 180])

    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': pad})
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_phase_vs_xstim_all_flies(summary_im_trials_df, save_figure=False,
                                  save_figure_path=None, figure_fname=None,
                                  save_source_data=False,
                                  save_source_data_path=None,
                                  source_data_fname=None):
    def plot(data):
        color = stimulation_palette[data['scanfield_config_name'].values[0]]
        x = data['mean_phase_during'].values
        x = fc.wrap(x - 180)
        y = data['mean_xstim_during'].values
        # y=fc.wrap(y+90) # why?!??!?
        plt.scatter(x, y,
                    s=0.1, alpha=0.75, color=color, clip_on=False)

    g = ph.FacetGrid(summary_im_trials_df, col='genotype', row=None,
                     unit='unique_trial_id',
                     fig_kws={'figsize': [1.5, 0.6], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.5,
                                   'hspace': 0},
                     col_order=order
                     )

    g.map_dataframe(plot)
    for ax in g.axes.ravel():
        ax.set_xticks([-180, 0, 180])
        ax.set_yticks([-180, 0, 180])
        ax.set_xlim([-180, 180])
        ax.set_ylim([-180, 180])
        ax.set_yticklabels([180, 0, -180])
    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': pad})

    if save_figure:
        plt.savefig(save_figure_path + figure_fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)
    if save_source_data:
        summary_im_trials_df_copy = summary_im_trials_df.copy()
        summary_im_trials_df_copy['heading'] = summary_im_trials_df_copy['mean_xstim_during'] * -1
        summary_im_trials_df_copy['phase'] = fc.wrap(summary_im_trials_df_copy['mean_phase_during'] - 180)
        summary_im_trials_df_copy['scanfield_config_name'] = summary_im_trials_df_copy['scanfield_config_name'].map(
            {'col1': 'A', 'col2': 'B'})
        summary_im_trials_df_copy = summary_im_trials_df_copy.filter(['genotype',
                                                                      'fly_id',
                                                                      'rec_name',
                                                                      'photostim_trial_id',
                                                                      'scanfield_config_name',
                                                                      'phase',
                                                                      'heading'])
        summary_im_trials_df_copy = summary_im_trials_df_copy.rename({'scanfield_config_name': 'trial_type'}, axis=1)
        summary_im_trials_df_copy.to_csv(save_source_data_path + source_data_fname)


def plot_zeroed_heading_hist(zeroed_heading_hist_df, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(zeroed_heading_hist_df,
                     col='genotype',
                     unit='scanfield_config_name',
                     fig_kws={'figsize': [1.3, 0.5], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.25,
                                   'hspace': 0},
                     col_order=order
                     )

    def plot(data):
        scanfield_config_name = data['scanfield_config_name'].values[0]
        color = stimulation_palette[scanfield_config_name]
        plt.bar(data['xstim_zeroed_bin_center'], data['norm_counts'], color=color, width=10,
                alpha=0.5, clip_on=False)

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 1})

    for ax in g.axes.ravel():
        ax.set_xlim([-180, 180])
        ax.set_xticks([-180, 0, 180])
        #  flip labels  bc we are using bar position, not heading
        ax.set_xticklabels([180, 0, -180])
        ax.set_ylim([0, 0.08])
        ax.set_yticks([0, 0.08])
        ax.tick_params(axis='x', labelrotation=45)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_zeroed_heading_hist_example_flies(abf_trials_df, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(
        abf_trials_df.query('(trial_time>=0) & (trial_time < 30) & (fly_id<11) & (`dforw_boxcar_average_0.5_s`>1) '),
        col='genotype',
        row='fly_id', unit='scanfield_config_name',
        fig_kws={'figsize': [1, 3], 'dpi': dpi},
        gridspec_kws={'wspace': 0.5,
                      'hspace': 0},
        row_order=np.arange(1, 11),  # hardcoded to max number of flies
        col_order=order
    )

    def plot(data):
        scanfield_config_name = data['scanfield_config_name'].values[0]
        color = stimulation_palette[scanfield_config_name]
        plt.hist(data['xstim_zeroed'].values, color=color, alpha=0.5, bins=np.linspace(-180, 180, 37), density=True)

    g.map_dataframe(plot)
    for ax in g.axes.ravel():
        ax.axis('off')
        ax.set_xlim([-180, 180])
        ax.set_ylim([0, 0.025])
        #  flip labels  bc we are using bar position, not heading
        ax.set_xticklabels([180, 0, -180])

    ax = g.axes[-1, 0]
    sb = ph.add_scalebar(ax, sizex=0, sizey=0.01, barwidth=axis_lw, barcolor='k', loc='lower left',
                         pad=0,
                         bbox_transform=ax.transAxes,
                         bbox_to_anchor=[-0.1, 0]
                         )

    ax.axis('on')
    ax.set_xticks([-180, 0, 180])
    ph.adjust_spines(ax, ['bottom'], lw=axis_lw, ticklen=axis_ticklen, pad=1)
    # ax.tick_params(axis='x', labelrotation=45)

    ax = g.axes[-1, 1]
    ax.axis('on')
    ax.set_xticks([-180, 0, 180])
    ph.adjust_spines(ax, ['bottom'], lw=axis_lw, ticklen=axis_ticklen, pad=1)
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_trace(recs, rec_name, im_trials_df, summary_im_trials_df,
                       save=False, savepath=None, fname=None):
    rec = recs[rec_name]

    # Photostim FB heatmap
    photostim = rec.im.df.query('scanfield_config_name=="col1"').filter(regex="fb.*.photostim")
    plt.figure(1, [0.7, 0.08], dpi=dpi)
    plt.pcolormesh(photostim.values[0:1, :], cmap=plt.cm.Reds, edgecolors='k', linewidth=axis_lw, vmin=0, vmax=1)
    plt.axis('off')

    if save:
        plt.savefig(savepath + fname + '_col1_photostim_array.pdf',
                    transparent=True, bbox_inches='tight')

    plt.show()

    photostim = rec.im.df.query('scanfield_config_name=="col2"').filter(regex="fb.*.photostim")
    plt.figure(1, [0.7, 0.08], dpi=dpi)
    plt.pcolormesh(photostim.values[0:1, :], cmap=plt.cm.Reds, edgecolors='k', linewidth=axis_lw, vmin=0, vmax=1)
    plt.axis('off')

    if save:
        plt.savefig(savepath + fname + '_col2_photostim_array.pdf',
                    transparent=True, bbox_inches='tight')

    plt.show()

    # example trace
    axes, gs = ap.plot(recs, rec_name,
                       im_colormesh_signals=["fb_c1_roi.*.dF/F",
                                             ],
                       im_signals=[],
                       bh_overlay_signals=['xstim'],
                       bh_signals=[],
                       tlim=None,
                       fig_kws={'figsize': [1.5, 3], 'dpi': dpi},
                       plot_kws={'lw': 0.25},
                       vlims=[None],
                       cmaps=[plt.cm.Purples],
                       cb_kws=dict(orientation='horizontal',
                                   fraction=1,
                                   aspect=5,
                                   shrink=0.5,
                                   pad=0,
                                   anchor=(0.0, 1.0),
                                   use_gridspec=False),
                       gridspec_kws={
                           'height_ratios': [0.9, 0.1],
                           'wspace': 0.25,
                           'hspace': 0.05},

                       adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2},
                       return_gs=True
                       )
    ax = axes[0]
    ph.adjust_spines(ax, [])
    sb = ph.add_scalebar(ax, sizex=0, sizey=100, barwidth=axis_lw, barcolor='k', loc='center left',
                         pad=-0.5,
                         bbox_transform=ax.transAxes,
                         )

    ax = axes[1]
    for irow, row in summary_im_trials_df.query(f'rec_name=="{rec_name}"').iterrows():
        color = stimulation_palette[row['scanfield_config_name']]
        # make pad independent!
        ax.axhspan(row['start'] + 60, row['end'] - 60, facecolor=ph.color_to_facecolor(color, alpha=0.3),
                   edgecolor='none', zorder=0)
    ph.adjust_spines(ax, [], pad=0)

    ### Hist

    ax = plt.subplot(gs[1, 1], sharex=axes[1])
    data = im_trials_df.query(
        f'trial_time>=0 & trial_time < 30 & rec_name=="{rec_name}" & scanfield_config_name=="col1" & `dforw_boxcar_average_0.5_s`>1')
    ax.hist(data['xstim'].values, color=ph.blue, alpha=0.5, bins=np.linspace(-180, 180, 37), density=True)

    data = im_trials_df.query(
        f'trial_time>=0 & trial_time < 30 & rec_name=="{rec_name}" & scanfield_config_name=="col2"& `dforw_boxcar_average_0.5_s`>1')
    ax.hist(data['xstim'].values, color=ph.orange, alpha=0.5, bins=np.linspace(-180, 180, 37), density=True)

    ax.set_ylim([0, 0.015])
    ax.set_yticks([0, 0.015])
    ax.set_xlim([-180, 180])
    ax.set_xticks([-180, 0, 180])
    # ax.yaxis.tick_right()

    ph.adjust_spines(ax, ['left', 'bottom'], pad=1, lw=axis_lw, ticklen=axis_ticklen)

    if save:
        plt.savefig(savepath + fname + '.pdf',
                    transparent=True, bbox_inches='tight')

    plt.show()

    # photostim colorbar

    fig = plt.figure(1, [0.05, 0.25], dpi=dpi)

    cb = mpl.colorbar.ColorbarBase(plt.gca(), orientation='vertical',
                                   cmap=plt.cm.Reds)

    cb.ax.tick_params(size=2.5, width=0.4)
    cb.outline.set_visible(False)
    # cb.ax.yaxis.set_ticks_position('left')

    if save:
        plt.savefig(savepath + fname + '_photostim_colorbar.pdf',
                    transparent=True, bbox_inches='tight')


def plot_in_vs_out_stim(in_vs_out_stim_df, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(in_vs_out_stim_df,
                     col='genotype',
                     unit='is_in_stim',
                     fig_kws={'figsize': [2, 1], 'dpi': dpi},
                     col_order=order,
                     gridspec_kws={'wspace': 0.5,
                                   'hspace': 0},

                     )

    def plot(data):
        is_in_stim = data['is_in_stim'].values[0]
        color = is_in_stim_palette[is_in_stim]
        plt.plot(data['trial_time'], data['mean_value'], color=color, lw=0.5)
        plt.fill_between(x=data['trial_time'],
                         y1=data['mean_value'] - data['std_err_value'],
                         y2=data['mean_value'] + data['std_err_value'],
                         color=ph.color_to_facecolor(color), lw=0)

    g.map_dataframe(plot)

    for ax in g.axes.ravel():
        ax.set_xlim([-30, 60])
        ax.set_xticks([-30, 0, 30, 60])
        ax.axvspan(0, 30, facecolor='#ffeded', zorder=-1)
        ph.adjust_spines(ax, ['left'], pad=1, lw=axis_lw, ticklen=axis_ticklen)

    ax = g.axes[0, 0]
    ax.set_ylim([0.1, 0.4])
    ax.set_yticks([0.2, 0.4])
    ax = g.axes[0, 1]
    ax.set_ylim([0.2, 0.5])
    ax.set_yticks([0.2, 0.4])

    sb = ph.add_scalebar(ax, sizex=30, sizey=0, barwidth=axis_lw, barcolor='k', loc='lower center',
                         pad=-0.5,
                         bbox_transform=ax.transAxes,
                         )
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_dist_to_stim_change(dist_to_stim_mean_df, dist_to_stim_fly_df, save=False, savepath=None, fname=None):
    plt.figure(1, [1, 1], dpi=dpi)

    # plots vals for each fly
    temp_df = dist_to_stim_mean_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f-Chr"')
    plt.scatter(temp_df['abs_col_diff'], temp_df['fly_mean_during_over_before'], color='grey', alpha=0.5, s=0.5)

    # plots mean +/- s.e.m. across flies
    temp_df = dist_to_stim_fly_df.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f-Chr"')
    plt.errorbar(x=temp_df['abs_col_diff'].values, y=temp_df['mean_value'].values,
                 yerr=temp_df['sem'].values, color='k', lw=1, capsize=1.5
                 #              ls='none',elinewidth=lw,ecolor='k',capsize=1.5,capthick=lw
                 )

    plt.scatter(temp_df['abs_col_diff'], temp_df['mean_value'], color='k', s=2)
    plt.axhline(y=1, ls=':', color='grey', lw=1, zorder=0)
    ax = plt.gca()
    ax.set_xticks(temp_df['abs_col_diff'].unique().tolist())
    ax.set_xticklabels(np.arange(7) + 2)

    ax.set_xlabel('')
    ax.set_ylabel('')

    ph.adjust_spines(ax, spines=['left', 'bottom'], pad=1, lw=axis_lw, ticklen=axis_ticklen)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_dforw_vs_dist(dforw_mean_df, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(dforw_mean_df,
                     col='genotype',
                     fig_kws={'figsize': [1.25, 0.75], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.5,
                                   'hspace': 0.25},
                     col_order=order,
                     )

    def plot(data):
        genotype = data['genotype'].values[0]
        plt.plot(data['distance_to_target_heading_bin_center'], data['mean'], lw=1,
                 color=genotype_palette[genotype], clip_on=False)
        ax = plt.gca()
        ax.fill_between(data['distance_to_target_heading_bin_center'],
                        data['mean'] - data['sem'],
                        data['mean'] + data['sem'],
                        color=ph.color_to_facecolor(genotype_palette[genotype]), lw=0, clip_on=False)

    g.map_dataframe(plot)

    for ax in g.axes.ravel():
        ax.set_xlim([-180, 180])
        ax.set_xticks([-180, 0, 180])
        ax.set_ylim([1, 5])
        ax.set_yticks([1, 3, 5])

    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'pad': pad, 'lw': axis_lw, 'ticklen': axis_ticklen})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_dist_target_heading_standing(target_heading_mean_df_standing, save=False, savepath=None, fname=None):
    is_standing_before_palette = {True: '#696969', False: 'k'}

    g = ph.FacetGrid(target_heading_mean_df_standing.query('genotype=="VT065306-AD-VT029306-DBD-sytGC7f-Chr" '),
                     unit='is_standing_before',
                     row='variable',
                     fig_kws={'figsize': [1, 2], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.25,
                                   'hspace': 0.25},
                     row_order=['dforw_boxcar_average_0.5_s', 'abs_distance_to_target_heading']
                     )

    def plot(data):
        is_standing_before = data['is_standing_before'].values[0]
        color = is_standing_before_palette[is_standing_before]
        variable = data['variable'].values[0]
        plt.plot(data['trial_time'].values, data['mean_value'].values, alpha=1, lw=0.5, color=color, clip_on=False)
        plt.fill_between(x=data['trial_time'],
                         y1=data['mean_value'] - data['std_err_value'],
                         y2=data['mean_value'] + data['std_err_value'],
                         color=ph.color_to_facecolor(color), lw=0, clip_on=False)
        ax = plt.gca()
        if variable == "abs_distance_to_target_heading":
            ax.set_ylim([0, 120])
            ax.set_yticks([0, 90])
        else:
            ax.set_ylim([-1, 5])
            ax.set_yticks([0, 5])

    g.map_dataframe(plot)

    for ax in g.axes.ravel():
        ax.set_xlim([-10, 40])
        ax.axvspan(0, 30, facecolor=ph.color_to_facecolor(ph.red), edgecolor='none', zorder=-1)
        ax.set_xticks([0, 30])

    ph.despine_axes(g.axes, style=['left', 'bottom'],
                    adjust_spines_kws={'pad': pad, 'lw': axis_lw, 'ticklen': axis_ticklen})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_dist_target_heading(target_heading_mean_df_dist, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(target_heading_mean_df_dist,
                     unit='distance_to_target_heading_before_bin_center',
                     col='genotype',
                     fig_kws={'figsize': [2, 1], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.25,
                                   'hspace': 0.25},
                     col_order=order
                     )

    def plot(data):
        distance_to_target_heading_before_bin_center = data['distance_to_target_heading_before_bin_center'].values[0]

        cmap = plt.cm.twilight_shifted
        #     color='k'
        color = cmap((distance_to_target_heading_before_bin_center + 180.) / 360.)
        ph.circplot(data['trial_time'], data['mean_value'], alpha=1, circ='y', lw=0.5, color=color)
        plt.fill_between(x=data['trial_time'],
                         y1=data['mean_value'] - data['std_err_value'],
                         y2=data['mean_value'] + data['std_err_value'],
                         color=ph.color_to_facecolor(color), lw=0)

    g.map_dataframe(plot)
    for ax in g.axes.ravel():
        ax.set_xlim([-10, 40])
        ax.set_xticks([-10, 0, 10, 20, 30, 40])
        ax.set_ylim([-180, 180])
        ax.axvspan(0, 30, facecolor=ph.color_to_facecolor(ph.red), edgecolor='none', zorder=-1)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.axhline(y=0, ls=':', color='grey', lw=1)

    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'pad': pad, 'lw': axis_lw, 'ticklen': axis_ticklen})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_avg_proj(DATA_PATH,chrimson_recs, t0, t1, scalebar=False, colorbar=False, col=None,
                          save=False, savepath=None, fname=None):
    rec = chrimson_recs['2022_02_08_0003']
    # TODO update path
    img = rd.Image(
        folder=DATA_PATH+'FC2_stimulation/VT065306-AD-VT029306-DBD-sytGC7f-Chr/2022_02_08/2022_02_08_0003')
    plt.figure(1, [1, 0.5], dpi=dpi)
    fb = np.mean(img.tif[t0:t1, :, 0, :64, :], axis=(0, 1))

    im = plt.imshow(fb, plt.cm.Purples, interpolation='gaussian', vmin=17, vmax=47)
    if scalebar:
        # objective_res stored in SI is incorrect
        objective_res = 34.4596
        dim = rec.photostim.scanfields['imScanfield']['pixelResolutionXY']
        pixelToRefTransform = rec.photostim.scanfields['imScanfield']['pixelToRefTransform']
        coords = rec.photostim.get_ref_coords(dim, pixelToRefTransform)
        microns = (coords[-1, 0] - coords[0, 0]) * objective_res
        pixels_per_mircons = dim[0] / microns
        scale_bar_size_microns = 30
        scale_bar_size_pixels = pixels_per_mircons * scale_bar_size_microns

        ax = plt.gca()
        sb = ph.add_scalebar(ax=ax, sizex=scale_bar_size_pixels, sizey=0, barwidth=0.75, barcolor='k',
                             loc='lower right',
                             pad=0.25,
                             )

    if col is not None:
        mask = rec.photostim.photo_stim_masks[col][0]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        plt.fill(contours[0][:, 0, 0], contours[0][:, 0, 1], color=(0, 0, 0, 0), edgecolor=(1, 0, 0, 1), linewidth=0.3)

    plt.axis('off')
    if save:
        plt.savefig(savepath + fname + '.png', transparent=True, bbox_inches='tight', dpi=dpi)

    plt.show()
    if colorbar:
        fig = plt.figure(1, [0.25, 0.05], dpi=dpi)
        cb = plt.colorbar(im, cax=plt.gca(), orientation='horizontal', )
        cb.ax.tick_params(size=2.5, width=0.4)
        cb.outline.set_visible(False)
        if save:
            plt.savefig(savepath + fname + '_colorbar.pdf', transparent=True, bbox_inches='tight', dpi=dpi)


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
            'fb_roi_1_photostim',
            'fb_roi_2_photostim',
            'fb_roi_3_photostim',
            'fb_roi_4_photostim',
            'fb_roi_5_photostim',
            'fb_roi_6_photostim',
            'fb_roi_7_photostim',
            'fb_roi_8_photostim',
            'fb_roi_9_photostim',
            'fb_roi_10_photostim',
            'fb_roi_11_photostim',
            'fb_roi_12_photostim',
            'fb_roi_13_photostim',
            'fb_roi_14_photostim',
            'fb_roi_15_photostim',
            'fb_roi_16_photostim',
            'scanfield_config_name'
        ]).rename({'t_abf': 't', 'scanfield_config_name': 'stim_pos'}, axis=1).reset_index(drop=True)
        df['stim_pos'] = df['stim_pos'].map({'col1': 'A', 'col2': 'B', 'control': 'inter-trial'})
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
