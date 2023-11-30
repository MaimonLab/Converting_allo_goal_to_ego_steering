"""wind_task.py

Analysis and plotting functions for Wind_task.ipynb

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
pad = 2
font = {'family': 'arial',
        'weight': 'normal',
        'size': 5}
mpl.rc('font', **font)
wind_color = '#46b76e'
simulation_color = '#bd8459'

genotype_color_dic = {
    '57C10_AD_VT037220_DBD_TNT_E_rep1': '#c22d2d',
    '57C10_AD_VT037220_DBD_TNT_Q_rep1': 'k',
    '57C10_AD_VT037220_DBD_TNT_E_rep2': '#c22d2d',
    '57C10_AD_VT037220_DBD_TNT_Q_rep2': 'k',
    '57C10_AD_VT037220_DBD_shibire': '#c22d2d',
    '60D05_shibire': '#e3a8a8',
    'empty_AD_empty_DBD_shibire': 'k',
    'empty_AD_empty_DBD_shibire_no_wind': 'grey',
    '27E08_AD_VT037220_DBD_TNT_Q': 'k',
    '27E08_AD_VT037220_DBD_TNT_E': '#c22d2d'
}

genotype_order = [
    '57C10_AD_VT037220_DBD_TNT_Q_rep1',
    '57C10_AD_VT037220_DBD_TNT_E_rep1',
    '57C10_AD_VT037220_DBD_TNT_Q_rep2',
    '57C10_AD_VT037220_DBD_TNT_E_rep2',
    '27E08_AD_VT037220_DBD_TNT_Q',
    '27E08_AD_VT037220_DBD_TNT_E',
    'empty_AD_empty_DBD_shibire',
    '57C10_AD_VT037220_DBD_shibire',
    '60D05_shibire',
    'empty_AD_empty_DBD_shibire_no_wind',
]

genotype_order2 = [
    '57C10_AD_VT037220_DBD_TNT_Q_rep1',
    '57C10_AD_VT037220_DBD_TNT_E_rep1',
    '57C10_AD_VT037220_DBD_TNT_Q_rep2',
    '57C10_AD_VT037220_DBD_TNT_E_rep2',
    'empty_AD_empty_DBD_shibire',
    '60D05_shibire',
    'empty_AD_empty_DBD_shibire_no_wind',
]


# ------------------ Load data ----------------------- #

def load_data(DATA_PATH, reprocess=False):
    parent_folder = DATA_PATH + 'Wind_task' + os.path.sep
    rec_names_dict = {
        '57C10_AD_VT037220_DBD_TNT_E_rep1': [
            ('2023_06_05_0001',),
            ('2023_06_05_0003',),
            ('2023_06_05_0005',),
            ('2023_06_06_0000',),
            ('2023_06_06_0002',),
            ('2023_06_06_0004',),
            ('2023_06_07_0001',),
            ('2023_06_07_0003',),
            ('2023_06_07_0005',),
            ('2023_06_07_0007',),
            ('2023_06_15_0001',),
            ('2023_06_15_0003',),
            ('2023_06_15_0005',),
            ('2023_06_19_0003',),
            ('2023_06_19_0005',),
            ('2023_06_19_0007',),
            ('2023_06_20_0000',),
            ('2023_06_20_0002',),
            ('2023_06_21_0000',),
            ('2023_06_21_0002',),
            ('2023_06_21_0004',),
            ('2023_06_22_0000',),
            ('2023_06_22_0002',),
            ('2023_06_22_0004',),
            ('2023_06_22_0006',),
        ],
        '57C10_AD_VT037220_DBD_TNT_Q_rep1': [
            ('2023_06_05_0000',),
            ('2023_06_05_0002',),
            ('2023_06_05_0004',),
            ('2023_06_06_0001',),
            ('2023_06_06_0003',),
            ('2023_06_06_0005',),
            ('2023_06_07_0002',),
            ('2023_06_07_0004',),
            ('2023_06_07_0006',),
            ('2023_06_15_0002',),
            ('2023_06_15_0004',),
            ('2023_06_19_0000',),
            ('2023_06_19_0002',),
            ('2023_06_19_0004',),
            ('2023_06_20_0001',),
            ('2023_06_20_0003',),
            ('2023_06_21_0001',),
            ('2023_06_21_0003',),
            ('2023_06_21_0005',),
            ('2023_06_22_0001',),
            ('2023_06_22_0003',),
            ('2023_06_22_0005',),
        ],
        '57C10_AD_VT037220_DBD_TNT_E_rep2': [
            ('2023_09_14_0001',),
            ('2023_09_14_0003',),
            ('2023_09_14_9001',),
            ('2023_09_14_9003',),
            ('2023_09_14_9004',),
            ('2023_09_15_0001',),
            ('2023_09_15_0003',),
            ('2023_09_15_0004',),
            ('2023_09_15_9001',),
            ('2023_09_15_9003',),
            ('2023_09_15_9004',),
            ('2023_09_18_0001',),
            ('2023_09_18_0003',),
            ('2023_09_18_0005',),
            ('2023_09_18_9001',),
            ('2023_09_18_9003',),
            ('2023_09_18_9005',),
            ('2023_09_20_0000',),
            ('2023_09_20_0002',),
            ('2023_09_20_0004',),
            ('2023_09_20_9000',),
            ('2023_09_20_9002',),
            ('2023_09_20_9003',),
            ('2023_09_20_9004',),
            ('2023_09_21_0001',),
            ('2023_09_21_0003',),
            ('2023_09_21_0005',),
            ('2023_09_21_9001',),
            ('2023_09_21_9003',),
            ('2023_09_21_9005',),
            ('2023_09_23_0000',),
            ('2023_09_23_0002',),
            ('2023_09_23_0004',),
            ('2023_09_23_9000',),
            ('2023_09_23_9002',),
            ('2023_09_23_9004',),
            ('2023_09_25_0001',),
            ('2023_09_25_0003',),
            ('2023_09_25_0005',),
            ('2023_09_25_9001',),
            ('2023_09_25_9003',),
            ('2023_09_25_9005',),
            ('2023_09_26_0000',),
            ('2023_09_26_0002',),
            ('2023_09_26_0004',),
            ('2023_09_26_9000',),
            ('2023_09_26_9002',),
            ('2023_09_26_9004',),
            ('2023_09_27_0001',),
            ('2023_09_27_0003',),
            ('2023_09_27_0004',),
            ('2023_09_27_9001',),
            ('2023_09_27_9004',),
            ('2023_09_28_0000',),
            ('2023_09_28_0002',),
            ('2023_09_28_0004',),
            ('2023_09_28_9000',),
            ('2023_09_28_9002',),
            ('2023_09_28_9004',),
        ],
        '57C10_AD_VT037220_DBD_TNT_Q_rep2': [
            ('2023_09_14_0000',),
            ('2023_09_14_0002',),
            ('2023_09_14_0004',),
            ('2023_09_14_9000',),
            ('2023_09_14_9002',),
            ('2023_09_15_0000',),
            ('2023_09_15_0002',),
            ('2023_09_15_9000',),
            ('2023_09_15_9002',),
            ('2023_09_18_0000',),
            ('2023_09_18_0002',),
            ('2023_09_18_0004',),
            ('2023_09_18_9000',),
            ('2023_09_18_9002',),
            ('2023_09_18_9004',),
            ('2023_09_20_0001',),
            ('2023_09_20_0003',),
            ('2023_09_20_9001',),
            ('2023_09_21_0000',),
            ('2023_09_21_0002',),
            ('2023_09_21_0004',),
            ('2023_09_21_9000',),
            ('2023_09_21_9002',),
            ('2023_09_21_9004',),
            ('2023_09_23_0001',),
            ('2023_09_23_0003',),
            ('2023_09_23_0005',),
            ('2023_09_23_9001',),
            ('2023_09_23_9003',),
            ('2023_09_23_9005',),
            ('2023_09_25_0000',),
            ('2023_09_25_0002',),
            ('2023_09_25_0004',),
            ('2023_09_25_9000',),
            ('2023_09_25_9002',),
            ('2023_09_25_9004',),
            ('2023_09_26_0001',),
            ('2023_09_26_0003',),
            ('2023_09_26_0005',),
            ('2023_09_26_9001',),
            ('2023_09_26_9003',),
            ('2023_09_26_9005',),
            ('2023_09_27_0000',),
            ('2023_09_27_0002',),
            ('2023_09_27_9000',),
            ('2023_09_27_9002',),
            ('2023_09_28_0001',),
            ('2023_09_28_0003',),
            ('2023_09_28_0005',),
            ('2023_09_28_9001',),
            ('2023_09_28_9003',),
        ],
        '27E08_AD_VT037220_DBD_TNT_E': [
            ('2023_10_13_0003',),
            ('2023_10_13_9003',),
            ('2023_10_13_9004',),
            ('2023_10_15_0001',),
            ('2023_10_15_0003',),
            ('2023_10_15_9001',),
            ('2023_10_15_9003',),
            ('2023_10_15_9004',),
            ('2023_10_16_0000',),
            ('2023_10_16_0002',),
            ('2023_10_16_0004',),
            ('2023_10_16_0006',),
            ('2023_10_16_9000',),
            ('2023_10_16_9002',),
            ('2023_10_16_9004',),
            ('2023_10_16_9006',),
            ('2023_10_17_0001',),
            ('2023_10_17_0003',),
            ('2023_10_17_0005',),
            ('2023_10_17_9001',),
            ('2023_10_17_9003',),
            ('2023_10_17_9005',),
            ('2023_10_17_9006',),
            ('2023_10_18_0000',),
            ('2023_10_18_0002',),
            ('2023_10_18_0004',),
            ('2023_10_18_9000',),
            ('2023_10_18_9002',),
            ('2023_10_18_9004',),
            ('2023_10_19_0001',),
            ('2023_10_19_0003',),
            ('2023_10_19_0005',),
            ('2023_10_19_9001',),
            ('2023_10_19_9003',),
            ('2023_10_19_9005',),
            ('2023_10_21_0000',),
            ('2023_10_21_0002',),
            ('2023_10_21_0004',),
            ('2023_10_21_0006',),
            ('2023_10_21_0008',),
            ('2023_10_21_9000',),
            ('2023_10_21_9002',),
            ('2023_10_21_9004',),
            ('2023_10_21_9006',),
            ('2023_10_21_9008',),
            ('2023_10_23_0002',),
            ('2023_10_23_0004',),
            ('2023_10_23_0006',),
            ('2023_10_23_9002',),
            ('2023_10_23_9004',),
            ('2023_10_23_9006',),
        ],
        '27E08_AD_VT037220_DBD_TNT_Q': [
            ('2023_10_13_0004',),
            ('2023_10_13_0005',),
            ('2023_10_13_9005',),
            ('2023_10_15_0000',),
            ('2023_10_15_0002',),
            ('2023_10_15_0004',),
            ('2023_10_15_9000',),
            ('2023_10_15_9002',),
            ('2023_10_16_0001',),
            ('2023_10_16_0003',),
            ('2023_10_16_0005',),
            ('2023_10_16_9001',),
            ('2023_10_16_9003',),
            ('2023_10_16_9005',),
            ('2023_10_17_0000',),
            ('2023_10_17_0002',),
            ('2023_10_17_0004',),
            ('2023_10_17_9000',),
            ('2023_10_17_9002',),
            ('2023_10_17_9004',),
            ('2023_10_18_0001',),
            ('2023_10_18_0003',),
            ('2023_10_18_0005',),
            ('2023_10_18_9001',),
            ('2023_10_18_9003',),
            ('2023_10_18_9005',),
            ('2023_10_19_0000',),
            ('2023_10_19_0002',),
            ('2023_10_19_0004',),
            ('2023_10_19_9000',),
            ('2023_10_19_9002',),
            ('2023_10_19_9004',),
            ('2023_10_21_0001',),
            ('2023_10_21_0003',),
            ('2023_10_21_0005',),
            ('2023_10_21_0007',),
            ('2023_10_21_9001',),
            ('2023_10_21_9003',),
            ('2023_10_21_9005',),
            ('2023_10_21_9007',),
            ('2023_10_21_9009',),
            ('2023_10_23_0001',),
            ('2023_10_23_0003',),
            ('2023_10_23_0005',),
            ('2023_10_23_0007',),
            ('2023_10_23_9001',),
            ('2023_10_23_9003',),
            ('2023_10_23_9005',),
            ('2023_10_23_9007',),
        ],
        'empty_AD_empty_DBD_shibire': [
            (
                '2023_07_05_0007',
            ), (
                '2023_07_05_0010',
            ), (
                '2023_07_06_0000',
            ), (
                '2023_07_06_0003',
            ), (
                '2023_07_06_0007',
            ), (
                '2023_07_07_0002',
            ), (
                '2023_07_07_0006',
            ), (
                '2023_07_10_0001',
            ), (
                '2023_07_10_0005',
            ), (
                '2023_07_11_0000',
            ), (
                '2023_07_11_0003',
            ), (
                '2023_07_11_0005',
            ),

            (
                '2023_07_12_0000',
            ),

            (
                '2023_07_12_0005',
            ),

            (
                '2023_07_13_0000',
            ),

            (
                '2023_07_13_0004',
            ),

            (
                '2023_07_13_0007',
            ),

            (
                '2023_07_13_0009',
            ),

            (
                '2023_07_14_0000',
            ),

            (
                '2023_07_14_0003',
            ),

            (
                '2023_07_17_0000',
            ),

            (
                '2023_07_18_0000',
            ),

        ],
        '57C10_AD_VT037220_DBD_shibire': [
            (
                '2023_07_05_0006',
            ), (
                '2023_07_05_0009',
            ), (
                '2023_07_06_0001',
            ), (
                '2023_07_06_0004',
            ), (
                '2023_07_06_0008',
            ), (
                '2023_07_07_0003',
            ), (
                '2023_07_07_0007',
            ), (
                '2023_07_10_0002',
            ), (
                '2023_07_10_0006',
            ), (
                '2023_07_11_0001',
            ), (
                '2023_07_11_0004',
            ), (
                '2023_07_11_0006',
            ),

            (
                '2023_07_12_0001',
            ),

            (
                '2023_07_13_0001',
            ),

            (
                '2023_07_13_0005',
            ),

            (
                '2023_07_13_0008',
            ),

            (
                '2023_07_14_0001',
            ),

            (
                '2023_07_14_0004',
            ),

            (
                '2023_07_17_0001',
            ),

            (
                '2023_07_17_0004',
            ),

            (
                '2023_07_18_0002',
            ),

        ],
        '60D05_shibire': [
            (
                '2023_07_05_0004',
            ), (
                '2023_07_05_0008',
            ), (
                '2023_07_06_0002',
            ), (
                '2023_07_06_0006',
            ), (
                '2023_07_07_0005',
            ), (
                '2023_07_10_0003',
            ), (
                '2023_07_10_0007',
            ),

            (
                '2023_07_12_0002',
            ),

            (
                '2023_07_12_0008',
            ),

            (
                '2023_07_17_0002',
            ),

        ],
        'empty_AD_empty_DBD_shibire_no_wind': [
            (
                '2023_07_29_0000',
            ),
            (
                '2023_07_29_0001',
            ),

            (
                '2023_07_29_0002',
            ),

            (
                '2023_07_29_0003',
            ),

            (
                '2023_07_29_0004',
            ),

            (
                '2023_07_29_0005',
            ),

            (
                '2023_07_29_0006',
            ),

            (
                '2023_07_31_0000',
            ),

            (
                '2023_07_31_0001',
            ),

            (
                '2023_07_31_0002',
            ),

            (
                '2023_07_31_0003',
            ),

            (
                '2023_07_31_0004',
            ),

            (
                '2023_07_31_0005',
            ),

            (
                '2023_08_02_0000',
            ),

            (
                '2023_08_02_0001',
            ),

            (
                '2023_08_02_0002',
            ),

            (
                '2023_08_02_0003',
            ),

            (
                '2023_08_02_0004',
            ),

            (
                '2023_08_02_0005',
            ),

            (
                '2023_08_02_0006',
            ),

            (
                '2023_08_02_0007',
            ),

        ]
    }

    genotypes = {}
    for genotype, rec_names in rec_names_dict.items():
        print(genotype)
        genotypes[genotype] = rd.quickload_experiment(parent_folder + 'nature_' + genotype + '.pkl',
                                                      rec_names=rec_names,
                                                      exp_kws={
                                                          'rec_type': rd.Rec,
                                                          'parent_folder': parent_folder,
                                                          'merge_df': True,
                                                          'genotype': genotype,
                                                      },
                                                      light=True,
                                                      reprocess=reprocess,
                                                      bh_kws={'angle_offset': 0,
                                                              'camtrig_label': None,
                                                              'subsampling_rate': 50,  # no camera trigs
                                                              'delay_ms': 30,
                                                              'ball_diam_mm': 8,
                                                              'boxcar_average': {'dforw': 0.5, 'dheading': 0.5, }
                                                              }
                                                      )
    return genotypes


# ------------------ Process & analyze data ----------------------- #

def get_trials(genotypes):
    trials_dfs = []
    for genotype, recs in genotypes.items():
        if (genotype not in ['empty_AD_empty_DBD_shibire_no_wind']):

            trials = ap.get_trials_df(recs, pad_s=60, im=False,
                                      stimid_map={'wind_on': [0.2, 7], 'wind_off': [-1, 0.1]}, stimid_label='flow')
        else:
            #  since flow is usually used to determine trials, and in no wind experiments flow is zero
            # we use the bar jump to determine trials instead (which is stored in "meta" channel)
            trials = ap.get_trials_df(recs, pad_s=90, im=False,
                                      stimid_map={'wind_on': [2.5, 3.5], 'wind_off': [-1, 1.5]}, stimid_label='meta')
            trials['trial_time'] = trials['trial_time'] + 29.5
            trials = trials.query('trial_time<90').copy().reset_index(drop=True)
            # is_OL here is a misnomer, and instead refers to whether the wind is on or not (binary of flow channel)
            trials.loc[(trials['trial_time'] >= 0) & (trials['trial_time'] < 30), 'is_OL'] = 1
            trials.loc[(trials['trial_time'] < 0), 'is_OL'] = 0
            trials.loc[(trials['trial_time'] >= 30), 'is_OL'] = 0
        trials_dfs.append(trials)

    trials = pd.concat(trials_dfs, ignore_index=True)
    trials['allowind'] = fc.wrap(trials['xstim'] - trials['servopos'])

    def get_allowind_during(data):
        idx = (data['trial_time'].values >= 0) & (data['trial_time'].values < 30) & (data['is_OL'] == True)
        allowind_during = fc.circmean(data['allowind'].values[idx])
        data['allowind_during_true'] = allowind_during
        allowind_during = int(np.round(allowind_during))
        # forced
        allowinds = np.array([-135, -90, -45, 45, 90, 135])
        data['allowind_during'] = allowinds[np.argmin(np.abs(fc.wrap(allowind_during - allowinds)))]
        return data

    trials = trials.groupby(['unique_trial_id']).apply(get_allowind_during)

    def get_n_trial(x):
        x = x % 3
        if x == 0:
            x = 3
        return x

    trials['n_trial'] = trials['trial_id'].apply(lambda x: (get_n_trial(x)))

    def get_mean_xstim_during(data):
        t = data['trial_time'].values
        xstim = data['xstim'].values
        dforw = data['dforw_boxcar_average_0.5_s'].values
        idx = (t > 0) & (t < 30) & (dforw > 1) & (data['is_OL'] == True)
        data['mean_xstim_during'] = fc.circmean(xstim[idx])
        return data

    trials = trials.groupby(['unique_trial_id']).apply(get_mean_xstim_during)

    def get_mean_xstim_after(data):
        t = data['trial_time'].values
        xstim = data['xstim'].values
        dforw = data['dforw_boxcar_average_0.5_s'].values
        idx = (t > 35) & (t < 65) & (dforw > 1)
        data['mean_xstim_after'] = fc.circmean(xstim[idx])
        return data

    trials = trials.groupby(['unique_trial_id']).apply(get_mean_xstim_after)

    def get_std_xstim_after(data):
        t = data['trial_time'].values
        xstim = data['xstim'].values
        dforw = data['dforw_boxcar_average_0.5_s'].values
        idx = (t > 35) & (t < 65) & (dforw > 1)
        data['std_xstim_after'] = fc.circstd(xstim[idx])
        return data

    trials = trials.groupby(['unique_trial_id']).apply(get_std_xstim_after)

    trials['dist_to_wind'] = fc.wrap(trials['xstim'] - trials['allowind_during_true'])
    trials['abs_dist_to_wind'] = np.abs(trials['dist_to_wind'])

    def get_abs_dist_to_wind_after(data):
        t = data['trial_time'].values
        dforw = data['dforw_boxcar_average_0.5_s'].values
        idx = (t > 35) & (t < 65) & (dforw > 1)
        data['abs_dist_to_wind_after'] = fc.circmean(data['abs_dist_to_wind'].values[idx])
        return data

    trials = trials.groupby(['unique_trial_id']).apply(get_abs_dist_to_wind_after)
    trials['is_upwind'] = trials['abs_dist_to_wind'] < 90
    return trials


def get_summary_trials(trials):
    summary_trials = trials.drop_duplicates('unique_trial_id').copy()

    # data were collected on two different rigs
    def get_rig(data):
        n = data['rec_name'].apply(lambda x: x.split('_')[-1][0]).values[0]
        if n == '0':
            data['rig'] = 1
        else:
            data['rig'] = 2
        return data

    summary_trials = summary_trials.groupby(['rec_name']).apply(get_rig)
    return summary_trials


def get_heading_vs_wind_df(summary_trials):
    df = summary_trials.copy()
    heading_vs_wind_df = df.query('n_trial!=1').groupby(['genotype',
                                                         'rec_name',
                                                         'fly_id',
                                                         'allowind_during'],
                                                        as_index=False).agg(
        allowind_during_true=('allowind_during_true', fc.circmean),
        mean_xstim_after=('mean_xstim_after', fc.circmean),
        mean_xstim_during=('mean_xstim_during', fc.circmean),
        std_xstim_after=('std_xstim_after', np.mean),
    )

    heading_vs_wind_df['signed_error_after'] = fc.wrap(
        heading_vs_wind_df['mean_xstim_after'] - heading_vs_wind_df['allowind_during_true'])
    heading_vs_wind_df['error_after'] = np.abs(heading_vs_wind_df['signed_error_after'])
    heading_vs_wind_df['signed_error_during'] = fc.wrap(
        heading_vs_wind_df['mean_xstim_during'] - heading_vs_wind_df['allowind_during_true'])
    heading_vs_wind_df['error_during'] = np.abs(heading_vs_wind_df['signed_error_during'])

    return heading_vs_wind_df


def exclude_flies(trials, summary_trials, heading_vs_wind_df):
    """
    These flies have no timepoints where the filtered forward walking velocity was above 1 mm/s for both the second and
    third trial for at least one of the wind direction blocks. We exclude these flies because their mean heading for one of the wind
    directions would be undefined.
     """

    exclude_rec_names = heading_vs_wind_df[heading_vs_wind_df['mean_xstim_after'].isna()]['rec_name'].unique().tolist()
    all_rec_names = heading_vs_wind_df['rec_name'].unique().tolist()
    print(len(exclude_rec_names), 'flies excluded from', len(all_rec_names), 'total flies')
    print(exclude_rec_names)
    trials = trials[~trials['rec_name'].isin(exclude_rec_names)].reset_index(drop=True)
    summary_trials = summary_trials[~summary_trials['rec_name'].isin(exclude_rec_names)].reset_index(drop=True)
    heading_vs_wind_df = heading_vs_wind_df[~heading_vs_wind_df['rec_name'].isin(exclude_rec_names)].reset_index(
        drop=True)

    print('Total flies in included dataset:')
    print(heading_vs_wind_df.groupby(['genotype']).apply(lambda x: len(np.unique(x['fly_id']))))

    return trials, summary_trials, heading_vs_wind_df


def write_heading_vs_wind_df(heading_vs_wind_df, save=False, savepath=None, fname=None):
    df = heading_vs_wind_df.filter(
        ['genotype', 'fly_id', 'allowind_during_true', 'mean_xstim_after', 'std_xstim_after']).copy()
    df['mean_xstim_after'] = df['mean_xstim_after'] * -1  # bc we want heading not bar position
    df['allowind_during_true'] = df['allowind_during_true'] * -1
    df.rename({'allowind_during_true': 'wind_direction',
               'mean_xstim_after': 'heading',
               'std_xstim_after': 'stdev'}, axis=1, inplace=True)

    def is_example(data):
        example_flies = {'57C10_AD_VT037220_DBD_TNT_Q_rep1': [4, 5, 8, 3],
                         '57C10_AD_VT037220_DBD_TNT_E_rep1': [2, 8, 16, 21],
                         'empty_AD_empty_DBD_shibire': [1, 5, 6, 8],
                         '60D05_shibire': [1, 2, 3, 4],
                         'empty_AD_empty_DBD_shibire_no_wind': [2, 3, 4, 9],
                         }

        genotype = data['genotype'].values[0]
        fly_id = data['fly_id'].values[0]
        if genotype in list(example_flies.keys()):
            if fly_id in example_flies[genotype]:
                data['is_example_fly'] = True
        return data

    df['is_example_fly'] = False
    df = df.groupby(['genotype', 'fly_id']).apply(is_example)
    if save:
        df.to_csv(savepath + fname)


def get_direction_hist(trials, query, label, bins=None):
    if bins is None:
        bins = np.linspace(-180, 180, 37)
    dx = np.diff(bins)[0]

    trial_hist_df = ap.get_hist_df(trials,
                                   value=label,
                                   bins=bins,
                                   query=query,
                                   id_vars=['genotype', 'fly_id', 'unique_trial_id'],
                                   density=True)

    fly_hist_df = trial_hist_df.groupby(['genotype',
                                         'fly_id',
                                         label + '_bin_center']).agg(norm_counts=('norm_counts', np.mean)).reset_index()

    fly_hist_df['norm_counts'] = fly_hist_df.groupby(['genotype', 'fly_id']).apply(
        lambda x: x['norm_counts'] / np.sum(x['norm_counts'])).values
    fly_hist_df['norm_counts'] = fly_hist_df['norm_counts'] / dx

    genotype_hist_df = fly_hist_df.groupby(['genotype',
                                            label + '_bin_center']).agg(norm_counts=('norm_counts', np.mean),
                                                                        sem=('norm_counts',
                                                                             lambda x: sc.stats.sem(x))).reset_index()

    genotype_hist_df['norm_counts'] = genotype_hist_df.groupby(['genotype']).apply(
        lambda x: x['norm_counts'] / np.sum(x['norm_counts'])).values
    genotype_hist_df['norm_counts'] = genotype_hist_df['norm_counts'] / dx

    return fly_hist_df, genotype_hist_df


def get_dist_to_wind_over_time(trials):
    binned_df = ap.get_binned_df(trials.query('trial_time>0 & (trial_id!=0) '),
                                 bin_values={'trial_time': np.linspace(0, 90, 91)},
                                 labels=['abs_dist_to_wind'], query=None, id_vars=['genotype', 'fly_id'],
                                 metric=fc.circmean)

    abs_dist_to_wind_vs_time = binned_df.groupby(['genotype', 'trial_time_bin_center'], as_index=False).agg(
        mean_value=('mean_value', fc.circmean),
        sem=('mean_value', fc.circ_stderror))
    return abs_dist_to_wind_vs_time


def get_error(heading_vs_wind_df, when='during', save=False, savepath=None, fname=None):
    df = heading_vs_wind_df.copy()
    fly_error_df = df.groupby(['genotype', 'fly_id'], as_index=False).agg(mean=('error_' + when, fc.circmean),
                                                                          )
    genotype_error_df = fly_error_df.groupby(['genotype'], as_index=False).agg(mean=('mean', fc.circmean),
                                                                               sem=('mean', fc.circ_stderror)
                                                                               )
    genotype_error_df['genotype'] = pd.Categorical(genotype_error_df['genotype'], categories=genotype_order,
                                                   ordered=True)
    genotype_error_df = genotype_error_df.sort_values(by='genotype')
    if save:
        pd.concat([fly_error_df.rename({'mean': 'value'}, axis=1), genotype_error_df]).to_csv(savepath + fname)
    return fly_error_df, genotype_error_df


def get_below_30_counts(heading_vs_wind_df, cutoff=30, save=False, savepath=None, fname=None):
    mean_error = heading_vs_wind_df.copy()
    mean_error['below_30'] = mean_error['error_after'] < cutoff
    below_30_counts = pd.DataFrame(
        mean_error.groupby(['genotype', 'fly_id']).apply(lambda x: np.sum(x['below_30']))).reset_index()
    below_30_counts = below_30_counts.rename({0: 'counts'}, axis=1)
    mean_counts = below_30_counts.groupby(['genotype']).agg(mean=('counts', np.mean),
                                                            sem=('counts', sc.stats.sem)).reset_index()
    mean_counts['genotype'] = pd.Categorical(mean_counts['genotype'], categories=genotype_order, ordered=True)
    mean_counts = mean_counts.sort_values(by='genotype')
    if save:
        pd.concat([below_30_counts, mean_counts]).to_csv(savepath + fname)
    return below_30_counts, mean_counts


def get_PI(trials, query, save=False, savepath=None, fname=None):
    PI_fly_df = trials.query(query).groupby(['genotype', 'fly_id'], as_index=False).agg(
        mean_value=('is_upwind', np.mean))

    PI_fly_df['PI'] = PI_fly_df['mean_value'] - (1 - PI_fly_df['mean_value'])

    PI_mean_df = PI_fly_df.groupby(['genotype'], as_index=False).agg(mean=('PI', np.mean),
                                                                     sem=('PI', sc.stats.sem))

    PI_mean_df['genotype'] = pd.Categorical(PI_mean_df['genotype'], categories=genotype_order, ordered=True)
    PI_mean_df = PI_mean_df.sort_values(by='genotype')
    if save:
        pd.concat([PI_fly_df.rename({'PI': 'value'}, axis=1).drop('mean_value', axis=1), PI_mean_df]).to_csv(
            savepath + fname)
    return PI_fly_df, PI_mean_df


def get_abs_dist_to_wind_by_ntrial(summary_trials, save=False, savepath=None, fname=None):
    fly_df = summary_trials.groupby(['genotype', 'fly_id', 'n_trial']).agg(
        abs_dist_to_wind_after=('abs_dist_to_wind_after', fc.circmean)).reset_index()
    genotype_df = fly_df.groupby(['genotype', 'n_trial']).agg(mean=('abs_dist_to_wind_after', fc.circmean),
                                                              sem=(
                                                                  'abs_dist_to_wind_after',
                                                                  fc.circ_stderror)).reset_index()
    if save:
        pd.concat([fly_df.rename({'abs_dist_to_wind_after': 'value'}, axis=1),
                   genotype_df]).to_csv(savepath + fname)
    return fly_df, genotype_df


# ------------------ Plotting functions ----------------------- #

def plot_example_2nd_trial(trials, genotype, fly_id, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(trials.query(
        f'(genotype=="{genotype}")&(fly_id=={fly_id}) & ((trial_id==2)|(trial_id==5)|(trial_id==8)|(trial_id==11)|(trial_id==14)|(trial_id==17))'),
        col='allowind_during',
        fig_kws={'figsize': [5, 1], 'dpi': dpi},
        gridspec_kws={
            'wspace': 0.2,
            'hspace': 0})

    def plot(data):
        ax = plt.gca()
        ph.circplot(data['trial_time'], -1 * data['xstim'], alpha=1, lw=0.5, circ='y')
        idx = (data['flow'].values > 0.2)
        t0 = data['trial_time'].values[idx][0]
        t1 = data['trial_time'].values[idx][-1]
        ax.axvspan(t0, t1, facecolor=ph.color_to_facecolor(ph.blue), edgecolor='none', zorder=-1)
        n_trial = data['n_trial'].values[0]
        if n_trial == 1:
            idx = (data['trial_time'].values > t1)
            ax.plot(data['trial_time'][idx], data['allowind_during_true'][idx] * -1, lw=1, color=wind_color, ls=':')
        else:
            ax.plot(data['trial_time'], data['allowind_during_true'] * -1, lw=1, color=wind_color, ls=':')

        ax.set_ylim([-180, 180])
        ax.set_yticks([-180, 0, 180])
        ax.set_xlim([-5, 65])
        ax.set_xticks([0, 0, 30, 60])

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['left'], adjust_spines_kws={'pad': 2, 'lw': axis_lw, 'ticklen': axis_ticklen})
    # scale bar
    ax = g.axes[0, 1]
    ax.plot([0, 30], [-200, -200], lw=0.75, color='k', clip_on=False)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_first_three(trials, genotype, fly_id, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(
        trials.query(f'(genotype=="{genotype}")&(fly_id=={fly_id}) & ((trial_id==1)|(trial_id==2)|(trial_id==3))'),
        #                  col='trial_id',
        row='trial_id',
        #                  unit='',
        fig_kws={'figsize': [2.25, 2], 'dpi': dpi},
        gridspec_kws={
            'wspace': 0,
            'hspace': 0.3})

    def plot(data):
        ax = plt.gca()
        ph.circplot(data['trial_time'], -1 * data['xstim'], alpha=1, lw=0.5, circ='y')
        idx = (data['flow'].values > 0.2)
        t0 = data['trial_time'].values[idx][0]
        t1 = data['trial_time'].values[idx][-1]
        ax.axvspan(t0, t1, facecolor=ph.color_to_facecolor(ph.blue), edgecolor='none', zorder=-1)
        ax.plot(data['trial_time'][idx], data['allowind_during_true'][idx] * -1, lw=1, color=wind_color)
        n_trial = data['n_trial'].values[0]

        if n_trial == 1:
            idx = (data['trial_time'].values > t1)
            ax.plot(data['trial_time'][idx], data['allowind_during_true'][idx] * -1, lw=1, color=wind_color, ls=':')
        else:
            ax.plot(data['trial_time'], data['allowind_during_true'] * -1, lw=1, color=wind_color, ls=':')

        ax.set_ylim([-180, 180])
        ax.set_yticks([-180, 0, 180])
        ax.set_xlim([-20, 65])
        ax.set_xticks([0, 0, 30, 60])

    g.map_dataframe(plot)

    ph.despine_axes(g.axes, style=['left'], adjust_spines_kws={'pad': 2, 'lw': axis_lw, 'ticklen': axis_ticklen})

    ax = g.axes[-1, 0]

    ax.plot([35, 65], [-200, -200], lw=0.5, color='k', clip_on=False)

    sb = ph.add_scalebar(ax, sizex=10, sizey=0, barwidth=0.75, barcolor='k',
                         #                     pad=-0.5,
                         bbox_transform=ax.transAxes,
                         bbox_to_anchor=[0.1, 0.5],

                         loc='lower center',
                         )

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_mean_xstim_after_vs_allowind_during_example(heading_vs_wind_df, genotype, fly_ids, save=False, savepath=None,
                                                     fname=None):
    plot_df = heading_vs_wind_df.query(f'genotype=="{genotype}"').copy()
    plot_df = plot_df[plot_df['fly_id'].isin(fly_ids)].reset_index(drop=True)

    g = ph.FacetGrid(plot_df,
                     col='fly_id',
                     col_wrap=4,
                     col_order=fly_ids,
                     fig_kws={'figsize': [2.25, 0.6], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.5}
                     )

    def plot(data):
        ax = plt.gca()

        genotype = data['genotype'].values[0]
        color = genotype_color_dic[genotype]

        # flip because we want heading not xstim
        x = data['allowind_during_true'].values * -1
        y = data['mean_xstim_after'].values * -1
        ax.scatter(x, y, clip_on=False, color=color, s=1)
        e1 = ax.errorbar(x=x, y=y,
                         yerr=data['std_xstim_after'].values,
                         ls='none', elinewidth=0.5, ecolor=color, capsize=1, capthick=0.5)
        ph.error_bar_clip_on([e1])
        fly_id = data['fly_id'].values[0]
        error = np.round(fc.circmean(np.abs(fc.wrap(x - y))), 1)
        ax.set_title(f"{error}", fontsize=5, pad=-0.25)

        ax.set_ylim([-300, 300])
        ax.set_xlim([-180, 180])

        ax.set_xticks([-180, 0, 180])
        ax.set_yticks([-180, 0, 180])

        l = np.linspace(-360, 360, 5)
        ph.circplot(l, l, color='k', circ='y', lw=0.5, ls=':')
        ph.circplot(l, l + 360, color='k', circ='y', lw=0.5, ls=':')
        ph.circplot(l, l - +360, color='k', circ='y', lw=0.5, ls=':')

    g.map_dataframe(plot)

    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen})
    for ax in g.axes.ravel():
        for loc, spine in ax.spines.items():
            if loc in 'bottom':
                spine.set_position(('outward', 4))

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_dist_to_wind_hist(dist_to_wind_fly_hist_df, dist_to_wind_genotype_hist_df, genotype, during=False,
                           save=False, savepath=None, fname=None):
    ylim = [0, 0.017]
    g = ph.FacetGrid(dist_to_wind_fly_hist_df.query(f'genotype=="{genotype}"'),
                     unit='fly_id',
                     fig_kws={'figsize': [0.6, 0.6], 'dpi': dpi},
                     )

    def plot(data):
        genotype = data['genotype'].values[0]
        color = genotype_color_dic[genotype]
        x = data['dist_to_wind_bin_center'].values
        y = data['norm_counts'].values
        ax = plt.gca()
        ax.plot(x, y, color=color, lw=0.25, clip_on=False, alpha=0.5)
        ax.set_xlim([-180, 180])
        ax.set_xticks([-180, 0, 180])
        ax.set_ylim(ylim)
        ax.set_yticks(ylim)

    data = dist_to_wind_genotype_hist_df.query(f'genotype=="{genotype}"')
    color = genotype_color_dic[genotype]
    ax = plt.gca()
    x = data['dist_to_wind_bin_center'].values
    y = data['norm_counts'].values
    ax.plot(x, y, color=color, lw=1, clip_on=False, alpha=1)

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen})

    if during:
        ax.fill_between([-180, 180], [ylim[0], ylim[0]], [ylim[1], ylim[1]],
                        color=ph.color_to_facecolor(ph.blue), lw=0, clip_on=False)
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_abs_dist_to_wind_over_time(abs_dist_to_wind_vs_time, genotypes=None,
                                    save=False, savepath=None, fname=None):
    abs_dist_to_wind_vs_time_copy = abs_dist_to_wind_vs_time[
        abs_dist_to_wind_vs_time['genotype'].isin(genotypes)].copy().reset_index()
    g = ph.FacetGrid(abs_dist_to_wind_vs_time_copy,
                     unit='genotype',
                     fig_kws={'figsize': [0.5, 1.5], 'dpi': dpi},
                     )

    def plot(data):
        genotype = data['genotype'].values[0]
        color = genotype_color_dic[genotype]
        x = data['trial_time_bin_center'].values
        y = data['mean_value'].values
        sem = data['sem'].values
        ls = '-'
        ax = plt.gca()
        ax.plot(x, y, color=color, lw=0.5, ls=ls)
        ax.set_ylim([0, 180])
        ax.set_yticks([0, 90, 180])
        ax.set_xlim([0, 90])
        ax.set_xticks([0, 30, 60, 90])
        ax.axhline(y=90, ls=':', color='grey', lw=0.5)
        ax.fill_between(x, y - sem, y + sem,
                        color=ph.color_to_facecolor(color), lw=0)

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen})
    ax = plt.gca()
    ax.axvspan(0, 30, facecolor=ph.color_to_facecolor(ph.blue), edgecolor='none', zorder=-2)
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_error(fly_error_df, genotype_error_df, order=1, hlines=None,
               pre_size=None,
               figsize=None, during=False, save=False, savepath=None, fname=None):
    if pre_size is None:
        pre_size = [2, 0.75]
    if figsize is None:
        figsize = [2.15, 0.8]
    if order == 1:
        order = genotype_order
    elif order == 2:
        order = genotype_order2

    fly_error_df = fly_error_df[fly_error_df['genotype'].isin(order)].copy()
    genotype_error_df = genotype_error_df[genotype_error_df['genotype'].isin(order)].copy()

    fig = plt.figure(1, pre_size, dpi=dpi)
    ax = sns.swarmplot(x="genotype", y="mean", data=fly_error_df, size=1.25,
                       order=order,
                       palette=genotype_color_dic,
                       marker='o', alpha=1, color='k', clip_on=False)
    if len(order) == 10:
        offsets = [0, 0, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.9, 0.9]
    elif len(order) == 7:
        offsets = [0, -0.2, 0, -0.2, 0, -0.2, 0]
    else:
        offsets = np.zeros(len(order)).tolist()
    for i, offset in enumerate(offsets):
        add_offset(i, offset)

    lw = 0.5
    genotype_error_df['genotype'] = pd.Categorical(genotype_error_df['genotype'], categories=order, ordered=True)
    genotype_error_df = genotype_error_df.sort_values(by='genotype')
    xvals = np.arange(len(genotype_error_df)) + offsets

    e1 = plt.errorbar(x=xvals + 0.4, y=genotype_error_df['mean'].values,
                      yerr=genotype_error_df['sem'].values,
                      ls='none', elinewidth=lw, ecolor='k', capsize=1.5, capthick=lw, zorder=10)

    e2 = plt.errorbar(x=xvals + 0.4, y=genotype_error_df['mean'].values,
                      xerr=0.2, ls='none', elinewidth=lw, ecolor='k', capsize=None, zorder=10)

    ph.error_bar_clip_on([e1, e2])

    plt.ylim([0, 140])
    plt.yticks([0, 45, 90])
    plt.ylabel('')
    fig.set_size_inches(figsize[0], figsize[1])
    ax.set_xlim(right=xvals.max() + 0.6)
    if during:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.fill_between([xlim[0], xlim[1]], [ylim[0], ylim[0]], [ylim[1], ylim[1]],
                        color=ph.color_to_facecolor(ph.blue), lw=0, clip_on=False)

    ph.adjust_spines(ax, spines=['left'], lw=axis_lw, ticklen=axis_ticklen, pad=1)
    ax.axhline(y=90, ls=':', color='grey', lw=0.5)
    if hlines is not None:
        for hline in hlines:
            ax.axhline(xmin=1, xmax=1.1, y=hline, color='k', lw=0.5)
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_below_30(below_30_counts, mean_counts, figsize=None,
                  all_geno=True, save=False, savepath=None, fname=None):
    if figsize is None:
        figsize = [2.075, 0.75]
    if all_geno == True:
        order = genotype_order
    else:
        order = genotype_order2
        below_30_counts = below_30_counts[below_30_counts['genotype'].isin(genotype_order2)].copy()
        mean_counts = mean_counts[mean_counts['genotype'].isin(genotype_order2)].copy()
    fig = plt.figure(1, [4, 0.5], dpi=dpi)
    ax = sns.swarmplot(x="genotype", y="counts", data=below_30_counts, size=1.25,
                       order=order,
                       palette=genotype_color_dic,
                       marker='o', alpha=0.5, color='k', clip_on=False)

    if all_geno == True:
        offsets = [0, 0, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.9, 0.9]
    else:
        offsets = [0, -0.2, 0, -0.2, 0, -0.2, 0]
    for i, offset in enumerate(offsets):
        add_offset(i, offset)

    lw = 0.5

    xvals = np.arange(len(mean_counts)) + offsets
    e1 = plt.errorbar(x=xvals + 0.3, y=mean_counts['mean'].values,
                      yerr=mean_counts['sem'].values,
                      ls='none', elinewidth=lw, ecolor='k', capsize=1.5, capthick=lw, zorder=10)
    e2 = plt.errorbar(x=xvals + 0.3, y=mean_counts['mean'].values,
                      xerr=0.2, ls='none', elinewidth=lw, ecolor='k', capsize=None, zorder=10)

    ph.error_bar_clip_on([e1, e2])
    fig.set_size_inches(figsize[0], figsize[1])
    plt.ylim([0, 6])
    ax.set_xlim(right=xvals.max() + 0.3)
    plt.yticks([0, 3, 6])
    plt.ylabel('')

    ph.adjust_spines(ax, spines=['left'], lw=axis_lw, ticklen=axis_ticklen, pad=2)

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_PI(PI_fly_df, PI_mean_df, during=False, save=False, savepath=None, fname=None):
    # TODO  THINK ABOUT ERROR BARS!!!!!!!
    fig = plt.figure(1, [4, 0.75], dpi=dpi)

    ax = sns.swarmplot(x="genotype", y="PI", data=PI_fly_df, size=1,
                       order=genotype_order,
                       palette=genotype_color_dic,
                       marker='o', alpha=1, color='k', clip_on=False)
    offsets = [0, 0, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.9, 0.9]
    for i, offset in enumerate(offsets):
        add_offset(i, offset)

    lw = 0.5

    xvals = np.arange(len(PI_mean_df)) + offsets
    e1 = plt.errorbar(x=xvals + 0.3, y=PI_mean_df['mean'].values,
                      yerr=PI_mean_df['sem'].values,
                      ls='none', elinewidth=lw, ecolor='k', capsize=1.5, capthick=lw, zorder=10)

    e2 = plt.errorbar(x=xvals + 0.3, y=PI_mean_df['mean'].values,
                      xerr=0.2, ls='none', elinewidth=lw, ecolor='k', capsize=None, zorder=10)

    ph.error_bar_clip_on([e1, e2])
    fig.set_size_inches(2.5, 0.8)
    plt.ylim([-0.6, 1])
    ax.set_xlim(right=xvals.max() + 0.6)
    plt.ylabel('')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if during:
        ax.fill_between([xlim[0], xlim[1]], [ylim[0], ylim[0]], [ylim[1], ylim[1]],
                        color=ph.color_to_facecolor(ph.blue), lw=0, clip_on=False)

    ph.adjust_spines(ax, spines=['left'], lw=axis_lw, ticklen=axis_ticklen, pad=1)
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_abs_dist_to_wind_by_ntrial(mean_fly_df, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(mean_fly_df,
                     col='genotype',
                     #                  unit='genotype',
                     col_order=genotype_order,
                     fig_kws={'figsize': [2.4, 0.5], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.5,
                                   'hspace': 0}, )

    def plot(data):
        genotype = data['genotype'].values[0]
        color = genotype_color_dic[genotype]
        x = data['n_trial'].values
        y = data['mean'].values
        sem = data['sem'].values
        ax = plt.gca()

        ax.plot(x, y, color=color, lw=0.5, clip_on=False)
        ax.scatter(x, y, color=color, s=0.5, clip_on=False)
        ax.set_xlim([1, 3])
        lw = 0.5
        e1 = plt.errorbar(x=x, y=y,
                          yerr=sem,
                          ls='none', elinewidth=lw, ecolor=color, capsize=1.5, capthick=lw, zorder=10)

        e2 = plt.errorbar(x=x, y=y,
                          xerr=0.2, ls='none', elinewidth=lw, ecolor=color, capsize=None, zorder=10)
        ph.error_bar_clip_on([e1, e2])
        ax.set_ylim([30, 100])
        ax.set_yticks([45, 90])
        ax.axhline(y=90, ls=':', color='grey', lw=0.5)

    g.map_dataframe(plot)

    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'pad': 3, 'lw': axis_lw, 'ticklen': axis_ticklen})
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_abs_dist_to_wind_by_ntrial_one_geno(abs_dist_to_wind_by_ntrial_fly, abs_dist_to_wind_by_ntrial_genotype
                                             , genotype, save=False, savepath=None, fname=None):
    temp_df = abs_dist_to_wind_by_ntrial_genotype.copy()
    temp_df['fly_id'] = 'mean'
    plot_df = pd.concat([abs_dist_to_wind_by_ntrial_fly, temp_df])

    g = ph.FacetGrid(plot_df.query(f'genotype=="{genotype}"'),
                     unit='fly_id',
                     fig_kws={'figsize': [0.75, 0.75], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.25,
                                   'hspace': 0}, )

    def plot(data):
        genotype = data['genotype'].values[0]
        color = genotype_color_dic[genotype]
        x = data['n_trial'].values
        ax = plt.gca()

        if data['fly_id'].values[0] != 'mean':
            y = data['abs_dist_to_wind_after']
            ax.plot(x, y, color=color, lw=0.25, alpha=0.25, clip_on=False)
            # ax.scatter(x,y,color=color,s=0.025,alpha=0.25,clip_on=False)

        else:

            y = data['mean'].values
            sem = data['sem'].values

            ax.plot(x, y, color=color, lw=0.5, clip_on=False)
            ax.scatter(x, y, color=color, s=0.5, clip_on=False)
            lw = 0.5

            e1 = plt.errorbar(x=x, y=y,
                              yerr=sem,
                              ls='none', elinewidth=lw, ecolor=color, capsize=1.5, capthick=lw, zorder=10)

            e2 = plt.errorbar(x=x, y=y,
                              xerr=0.2, ls='none', elinewidth=lw, ecolor=color, capsize=None, zorder=10)

            ph.error_bar_clip_on([e1, e2])
        ax.set_ylim([30, 100])
        ax.set_yticks([45, 90])
        ax.axhline(y=90, ls=':', color='grey', lw=0.5)

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'pad': 3, 'lw': axis_lw, 'ticklen': axis_ticklen})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_predicted_error(genotype_error_after_df, save=False, savepath=None, fname=None):
    # from simulation (done in MATLAB)
    predicted_error = [
        40.1546,
        41.4513,
        42.7368,
        44.0005,
        45.5586,
        46.8444,
        48.5119,
        49.8157,
        51.4622,
        53.5447,
        54.9953,
        56.8349,
        59.2264,
        60.7980,
        62.7363,
        64.9493,
        66.8765,
        69.3570,
        71.9968,
        75.9175,
        80.4671,
        85.5457,
        91.0517,
        90.5266,
        90.2877,
    ]
    plt.figure(1, [1, 0.75], dpi=dpi)
    ax = plt.gca()
    n_cells = np.arange(len(predicted_error))
    ax.plot(n_cells, predicted_error, color=simulation_color, lw=0.5)
    ax.scatter(n_cells, predicted_error, facecolors='none', lw=0.5, edgecolors=simulation_color, s=2.5, clip_on=False,
               zorder=10)

    # simulation used tnt inactive to determine baseline noise
    ctl_error = genotype_error_after_df.query('genotype=="57C10_AD_VT037220_DBD_TNT_Q_rep1"')['mean']
    ctl_mean = np.zeros(len(predicted_error))
    ctl_mean[:] = ctl_error
    ctl_sem = genotype_error_after_df.query('genotype=="57C10_AD_VT037220_DBD_TNT_Q_rep1"')['sem'].values[0]
    e1 = plt.errorbar(x=[0], y=ctl_mean[0],
                      yerr=ctl_sem,
                      ls='none', elinewidth=0.5, ecolor='k', capsize=1.5, capthick=0.5)
    ph.error_bar_clip_on([e1])

    # simulation used tnt inactive to determine baseline noise
    ctl_error = genotype_error_after_df.query('genotype=="57C10_AD_VT037220_DBD_TNT_Q_rep2"')['mean']
    ctl_mean = np.zeros(len(predicted_error))
    ctl_mean[:] = ctl_error
    ctl_sem = genotype_error_after_df.query('genotype=="57C10_AD_VT037220_DBD_TNT_Q_rep2"')['sem'].values[0]
    e1 = plt.errorbar(x=[0], y=ctl_mean[0],
                      yerr=ctl_sem,
                      ls='none', elinewidth=0.5, ecolor='grey', capsize=1.5, capthick=0.5)
    ph.error_bar_clip_on([e1])

    tnt_error = genotype_error_after_df.query('genotype=="57C10_AD_VT037220_DBD_TNT_E_rep1"')['mean']
    tnt_mean = np.zeros(len(predicted_error))
    tnt_mean[:] = tnt_error
    tnt_sem = genotype_error_after_df.query('genotype=="57C10_AD_VT037220_DBD_TNT_E_rep1"')['sem'].values[0]
    color = genotype_color_dic['57C10_AD_VT037220_DBD_TNT_E_rep1']
    ax.plot(n_cells, tnt_mean, color=color, lw=0.5)
    ax.fill_between(n_cells,
                    tnt_mean - tnt_sem,
                    tnt_mean + tnt_sem,
                    color=ph.color_to_facecolor(color), lw=0, clip_on=False)

    tnt_error = genotype_error_after_df.query('genotype=="57C10_AD_VT037220_DBD_TNT_E_rep2"')['mean']
    tnt_mean = np.zeros(len(predicted_error))
    tnt_mean[:] = tnt_error
    tnt_sem = genotype_error_after_df.query('genotype=="57C10_AD_VT037220_DBD_TNT_E_rep2"')['sem'].values[0]
    color = genotype_color_dic['57C10_AD_VT037220_DBD_TNT_E_rep2']
    ax.plot(n_cells, tnt_mean, color=color, lw=0.5, ls=':')
    ax.fill_between(n_cells,
                    tnt_mean - tnt_sem,
                    tnt_mean + tnt_sem,
                    color=ph.color_to_facecolor(color), lw=0, clip_on=False)

    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_xlim([0, 24])
    ax.set_ylim([22.5, 100])
    ax.set_yticks([45, 90])

    ph.despine_axes([ax], style=['bottom', 'left'],
                    adjust_spines_kws={'pad': 3, 'lw': axis_lw, 'ticklen': axis_ticklen})
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_predicted_correct(mean_counts, save=False, savepath=None, fname=None):
    # from simulation (done in MATLAB)
    predicted_correct = [2.8260,
                         2.7208,
                         2.6806,
                         2.6092,
                         2.5090,
                         2.4222,
                         2.3646,
                         2.2914,
                         2.2400,
                         2.1402,
                         2.0632,
                         2.0164,
                         1.9010,
                         1.8776,
                         1.8004,
                         1.7302,
                         1.6316,
                         1.5720,
                         1.4712,
                         1.3580,
                         1.2688,
                         1.1078,
                         0.9708,
                         0.9600,
                         1.0132,
                         ]
    plt.figure(1, [1, 0.75], dpi=dpi)
    ax = plt.gca()
    n_cells = np.arange(len(predicted_correct))
    ax.plot(n_cells, predicted_correct, color=simulation_color, lw=0.5)
    ax.scatter(n_cells, predicted_correct, facecolors='none', lw=0.5, edgecolors=simulation_color, s=2.5, clip_on=False,
               zorder=10)

    # simulation used tnt inactive to determine baseline noise
    ctl_error = mean_counts.query('genotype=="57C10_AD_VT037220_DBD_TNT_Q_rep1"')['mean']
    ctl_mean = np.zeros(len(predicted_correct))
    ctl_mean[:] = ctl_error
    ctl_sem = mean_counts.query('genotype=="57C10_AD_VT037220_DBD_TNT_Q_rep1"')['sem'].values[0]
    e1 = plt.errorbar(x=[0], y=ctl_mean[0],
                      yerr=ctl_sem,
                      ls='none', elinewidth=0.5, ecolor='k', capsize=1.5, capthick=0.5)
    ph.error_bar_clip_on([e1])

    # simulation used tnt inactive to determine baseline noise
    ctl_error = mean_counts.query('genotype=="57C10_AD_VT037220_DBD_TNT_Q_rep2"')['mean']
    ctl_mean = np.zeros(len(predicted_correct))
    ctl_mean[:] = ctl_error
    ctl_sem = mean_counts.query('genotype=="57C10_AD_VT037220_DBD_TNT_Q_rep2"')['sem'].values[0]
    e1 = plt.errorbar(x=[0], y=ctl_mean[0],
                      yerr=ctl_sem,
                      ls='none', elinewidth=0.5, ecolor='grey', capsize=1.5, capthick=0.5)
    ph.error_bar_clip_on([e1])

    tnt_error = mean_counts.query('genotype=="57C10_AD_VT037220_DBD_TNT_E_rep1"')['mean']
    tnt_mean = np.zeros(len(predicted_correct))
    tnt_mean[:] = tnt_error
    tnt_sem = mean_counts.query('genotype=="57C10_AD_VT037220_DBD_TNT_E_rep1"')['sem'].values[0]
    color = genotype_color_dic['57C10_AD_VT037220_DBD_TNT_E_rep1']
    ax.plot(n_cells, tnt_mean, color=color, lw=0.5)
    ax.fill_between(n_cells,
                    tnt_mean - tnt_sem,
                    tnt_mean + tnt_sem,
                    color=ph.color_to_facecolor(color), lw=0, clip_on=False)

    tnt_error = mean_counts.query('genotype=="57C10_AD_VT037220_DBD_TNT_E_rep2"')['mean']
    tnt_mean = np.zeros(len(predicted_correct))
    tnt_mean[:] = tnt_error
    tnt_sem = mean_counts.query('genotype=="57C10_AD_VT037220_DBD_TNT_E_rep2"')['sem'].values[0]
    color = genotype_color_dic['57C10_AD_VT037220_DBD_TNT_E_rep1']
    ax.plot(n_cells, tnt_mean, color=color, lw=0.5, ls=':')
    ax.fill_between(n_cells,
                    tnt_mean - tnt_sem,
                    tnt_mean + tnt_sem,
                    color=ph.color_to_facecolor(color), lw=0, clip_on=False)

    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_xlim([0, 24])
    ax.set_ylim([0, 3.5])
    ax.set_yticks([0, 1, 2, 3])
    ph.despine_axes([ax], style=['bottom', 'left'],
                    adjust_spines_kws={'pad': 3, 'lw': axis_lw, 'ticklen': axis_ticklen})
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_predicted_error_v2(genotype_error_after_df, save=False, savepath=None, fname=None):
    # from simulation (done in MATLAB)
    predicted_error = [
        40.1546,
        41.4513,
        42.7368,
        44.0005,
        45.5586,
        46.8444,
        48.5119,
        49.8157,
        51.4622,
        53.5447,
        54.9953,
        56.8349,
        59.2264,
        60.7980,
        62.7363,
        64.9493,
        66.8765,
        69.3570,
        71.9968,
        75.9175,
        80.4671,
        85.5457,
        91.0517,
        90.5266,
        90.2877,
    ]
    plt.figure(1, [1, 0.75], dpi=dpi)
    ax = plt.gca()
    n_cells = np.arange(len(predicted_error))
    ax.plot(n_cells, predicted_error, color=simulation_color, lw=0.5)
    ax.scatter(n_cells, predicted_error, facecolors='none', lw=0.5, edgecolors=simulation_color, s=2.5, clip_on=False,
               zorder=10)

    # simulation used tnt inactive to determine baseline noise
    ctl = genotype_error_after_df.query(
        '(genotype=="57C10_AD_VT037220_DBD_TNT_Q_rep1")|(genotype=="57C10_AD_VT037220_DBD_TNT_Q_rep2")')
    ctl_error = np.mean(ctl['mean'])
    ctl_mean = np.zeros(len(predicted_error))
    ctl_mean[:] = ctl_error
    ctl_sem = np.mean(ctl['sem'])
    e1 = plt.errorbar(x=[0], y=ctl_mean[0],
                      yerr=ctl_sem,
                      ls='none', elinewidth=0.5, ecolor='k', capsize=1.5, capthick=0.5)
    ph.error_bar_clip_on([e1])

    tnt = genotype_error_after_df.query(
        '(genotype=="57C10_AD_VT037220_DBD_TNT_E_rep1")|(genotype=="57C10_AD_VT037220_DBD_TNT_E_rep2")')
    tnt_error = np.mean(tnt['mean'])
    tnt_mean = np.zeros(len(predicted_error))
    tnt_mean[:] = tnt_error
    tnt_sem = np.mean(tnt['sem'])
    color = genotype_color_dic['57C10_AD_VT037220_DBD_TNT_E_rep1']
    ax.plot(n_cells, tnt_mean, color=color, lw=0.5)
    ax.fill_between(n_cells,
                    tnt_mean - tnt_sem,
                    tnt_mean + tnt_sem,
                    color=ph.color_to_facecolor(color), lw=0, clip_on=False)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_xlim([0, 24])
    ax.set_ylim([22.5, 100])
    ax.set_yticks([45, 90])

    ph.despine_axes([ax], style=['bottom', 'left'],
                    adjust_spines_kws={'pad': 3, 'lw': axis_lw, 'ticklen': axis_ticklen})
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def plot_predicted_correct_v2(mean_counts, save=False, savepath=None, fname=None):
    # from simulation (done in MATLAB)
    predicted_correct = [2.8260,
                         2.7208,
                         2.6806,
                         2.6092,
                         2.5090,
                         2.4222,
                         2.3646,
                         2.2914,
                         2.2400,
                         2.1402,
                         2.0632,
                         2.0164,
                         1.9010,
                         1.8776,
                         1.8004,
                         1.7302,
                         1.6316,
                         1.5720,
                         1.4712,
                         1.3580,
                         1.2688,
                         1.1078,
                         0.9708,
                         0.9600,
                         1.0132,
                         ]
    plt.figure(1, [1, 0.75], dpi=dpi)
    ax = plt.gca()
    n_cells = np.arange(len(predicted_correct))
    ax.plot(n_cells, predicted_correct, color=simulation_color, lw=0.5)
    ax.scatter(n_cells, predicted_correct, facecolors='none', lw=0.5, edgecolors=simulation_color, s=2.5, clip_on=False,
               zorder=10)

    # simulation used tnt inactive to determine baseline noise
    ctl = mean_counts.query(
        '(genotype=="57C10_AD_VT037220_DBD_TNT_Q_rep1")|(genotype=="57C10_AD_VT037220_DBD_TNT_Q_rep2")')
    ctl_error = np.mean(ctl['mean'])
    ctl_mean = np.zeros(len(predicted_correct))
    ctl_mean[:] = ctl_error
    ctl_sem = np.mean(ctl['sem'])
    e1 = plt.errorbar(x=[0], y=ctl_mean[0],
                      yerr=ctl_sem,
                      ls='none', elinewidth=0.5, ecolor='k', capsize=1.5, capthick=0.5)
    ph.error_bar_clip_on([e1])
    tnt = mean_counts.query(
        '(genotype=="57C10_AD_VT037220_DBD_TNT_E_rep1")|(genotype=="57C10_AD_VT037220_DBD_TNT_E_rep2")')
    tnt_error = np.mean(tnt['mean'])
    tnt_mean = np.zeros(len(predicted_correct))
    tnt_mean[:] = tnt_error
    tnt_sem = np.mean(tnt['sem'])
    color = genotype_color_dic['57C10_AD_VT037220_DBD_TNT_E_rep1']
    ax.plot(n_cells, tnt_mean, color=color, lw=0.5)
    ax.fill_between(n_cells,
                    tnt_mean - tnt_sem,
                    tnt_mean + tnt_sem,
                    color=ph.color_to_facecolor(color), lw=0, clip_on=False)

    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_xlim([0, 24])
    ax.set_ylim([0, 3.5])
    ax.set_yticks([0, 1, 2, 3])
    ph.despine_axes([ax], style=['bottom', 'left'],
                    adjust_spines_kws={'pad': 3, 'lw': axis_lw, 'ticklen': axis_ticklen})
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight', dpi=dpi)


def add_offset(i, offset):
    # get first patchcollection
    ax = plt.gca()
    c = ax.get_children()[i]
    x, y = np.array(c.get_offsets()).T
    # Add offset to x values
    xnew = x + offset
    offsets = list(zip(xnew, y))
    # set newoffsets
    c.set_offsets(offsets)


# ------------------ Save processed data ----------------------- #
def save_processed_data(PROCESSED_DATA_PATH, genotypes):
    def save_as_hd5f(data):
        df = data.copy()
        rec_name = df['rec_name'].values[0]
        genotype = df['genotype'].values[0]
        df = df.filter([
            't',
            'heading',
            'side',
            'forw',
            'meta',
            'flow',
            'xstim',
            'wstim',
            'servopos']).reset_index(drop=True)
        df.to_hdf(PROCESSED_DATA_PATH + genotype + os.path.sep + rec_name + '.h5',
                  key='df', mode='w')

    for genotype, recs in genotypes.items():
        recs.merged_abf_df.groupby(['rec_name']).apply(save_as_hd5f)
