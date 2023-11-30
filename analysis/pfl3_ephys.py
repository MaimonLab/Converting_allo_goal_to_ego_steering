import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sc
import matplotlib as mpl
import matplotlib.gridspec as gridspec

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

# ------------------ Model ----------------------- #
# Model from 2023/08/16
a = 29.2282
b = 2.1736
c = -0.7011
d = 0.6299
phiGoal = np.deg2rad(-48)
H = np.linspace(-np.pi, np.pi, 100)


def get_vMa(H, G, d, phiGoal):
    vR = d * np.cos(G - phiGoal) + np.cos(H)
    return vR


def get_f(vMa, a, b, c):
    f = a * np.log(1 + np.exp(b * (vMa + c)))
    return f


# ------------------ Load data ----------------------- #

def load_data(DATA_PATH, reprocess=False):
    rec_names = [

        (
            '2021_07_05_0001',
        ),

        (
            '2021_07_05_0002',
        ),
        (
            '2021_07_08_0003',
        ),

        ('2021_07_09_0003',
         ),

        ('2021_07_12_0002',
         ),
        ('2021_07_12_0008',
         ),
        ('2021_07_13_0001',
         ),
        ('2021_07_13_0005',
         ),
        ('2021_07_15_0002',
         ),
        ('2021_07_22_0002',
         ),
        ('2021_07_25_0001',
         ),
        ('2021_07_25_0003',
         ),
        ('2021_07_25_0004',
         ),

        ('2021_09_06_0001',
         ),

        ('2021_09_10_0001',
         ),

        ('2021_09_11_0004',
         ),

        ('2021_09_17_0002',
         ),

        ('2021_09_18_0003',
         ),

        ('2021_09_20_0001',
         ),

        ('2021_09_24_0002',
         ),

        ('2021_09_27_0002',
         ),

        ('2021_10_01_0002',
         ),

        ('2021_10_05_0002',
         ),

        ('2021_10_05_0004',
         ),

        (
            '2021_10_06_0004',
        ),

        (
            '2021_10_07_0002',
        ),

        (
            '2021_10_14_0006',
        ),

        (
            '2021_10_18_0002',
        ),

        (
            '2021_10_29_0001',
        ),

        (
            '2021_11_01_0002',
        ),

        (
            '2021_11_01_0006',
        ),

    ]

    # Start and end trim times determined by manually inspecting recordings
    trim_times = {
        '2021_07_05_0001': [0, 1600],
        '2021_07_05_0002': [0, 2600],
        '2021_07_08_0003': [0, 380],
        '2021_07_09_0003': [0, 1000],
        '2021_07_12_0002': [0, np.inf],
        '2021_07_12_0008': [0, 700],
        '2021_07_13_0001': [0, np.inf],
        '2021_07_13_0005': [0, np.inf],
        '2021_07_15_0002': [0, 2550],
        '2021_07_22_0002': [0, 1000],
        '2021_07_25_0001': [0, np.inf],
        '2021_07_25_0003': [0, 1200],
        '2021_07_25_0004': [0, 1900],
        '2021_09_06_0001': [0, np.inf],
        '2021_09_10_0001': [5600, np.inf],
        '2021_09_11_0004': [4000, np.inf],
        '2021_09_17_0002': [0, 3000],
        '2021_09_18_0003': [0, np.inf],
        '2021_09_20_0001': [0, 3000],
        '2021_09_24_0002': [0, 3600],
        '2021_09_27_0002': [0, np.inf],
        '2021_10_01_0002': [0, np.inf],
        '2021_10_05_0002': [0, np.inf],
        '2021_10_05_0004': [0, 1150],
        '2021_10_06_0004': [0, 2000],
        '2021_10_07_0002': [0, np.inf],
        '2021_10_14_0006': [0, 3700],
        '2021_10_18_0002': [0, np.inf],
        '2021_10_29_0001': [0, np.inf],
        '2021_11_01_0002': [0, np.inf],
        '2021_11_01_0006': [0, 4300],
    }

    # Spike detection parameters determined by manually inspecting recordings
    rec_params = {
        # default is 100-3000 Hz butterworth bandpass
        '2021_07_05_0001': {'threshold': 0.4, 'freq_range': [125, 3000]},
        '2021_07_05_0002': {'threshold': 0.4},
        '2021_07_08_0003': {'threshold': 0.5},
        '2021_07_09_0003': {'threshold': 0.3, 'freq_range': [110, 3000]},
        '2021_07_12_0002': {'threshold': 0.5},
        '2021_07_12_0008': {'threshold': 0.4, 'freq_range': [115, 3000]},
        '2021_07_13_0001': {'threshold': 0.2, 'freq_range': [120, 3000]},
        '2021_07_13_0005': {'threshold': 0.2, 'freq_range': [110, 3000]},
        '2021_07_15_0002': {'threshold': 0.3},
        '2021_07_22_0002': {'threshold': 0.25, 'freq_range': [110, 3000]},
        '2021_07_25_0001': {'threshold': 0.3, 'freq_range': [110, 3000]},
        '2021_07_25_0003': {'threshold': 0.3},
        '2021_07_25_0004': {'threshold': 0.25},
        '2021_09_06_0001': {'threshold': 0.4},
        '2021_09_10_0001': {'threshold': 1},
        '2021_09_11_0004': {'threshold': 0.4},
        '2021_09_17_0002': {'threshold': 0.2, 'freq_range': [110, 3000]},
        '2021_09_18_0003': {'threshold': 0.3, 'freq_range': [125, 3000]},
        '2021_09_20_0001': {'threshold': 0.3},
        '2021_09_24_0002': {'threshold': 0.3},
        '2021_09_27_0002': {'threshold': 0.34},
        '2021_10_01_0002': {'threshold': 0.2},
        '2021_10_05_0002': {'threshold': 0.2},
        '2021_10_05_0004': {'threshold': 0.4},
        '2021_10_06_0004': {'threshold': 0.25},
        '2021_10_07_0002': {'threshold': 0.2, 'freq_range': [120, 3000]},
        '2021_10_14_0006': {'threshold': 0.4},
        '2021_10_18_0002': {'threshold': 0.4},
        '2021_10_29_0001': {'threshold': 0.4},
        '2021_11_01_0002': {'threshold': 0.4},
        '2021_11_01_0006': {'threshold': 0.3, 'freq_range': [110, 3000]},
    }

    parent_folder = DATA_PATH + 'PFL3_ephys' + os.path.sep + 'VT000355-AD-VT037220-DBD' + os.path.sep
    recs = rd.quickload_experiment(parent_folder + 'nature_VT000355_AD_VT037220_DBD_2xeGFP.pkl',
                                   rec_names=rec_names,
                                   exp_kws={
                                       'rec_type': rd.Rec,
                                       'parent_folder': parent_folder,
                                       'merge_df': True,
                                       'genotype': 'VT000355-AD-VT037220-DBD',
                                   },
                                   light=['t', 'camtrig', 'patch_1', 'sp', 'vm_sp_subtracted'],
                                   hdf5=True,
                                   reprocess=reprocess,
                                   patch=True,
                                   camera=True,
                                   trim_times=trim_times,
                                   patch_kws={
                                       'detection_method': 'butterworth',
                                       'detection_params': {
                                           'freq_range': [100, 3000],
                                           'threshold': 10,
                                           'min_distance_s': 0.005
                                       },
                                       'rec_params': rec_params},
                                   bh_kws={'angle_offset': 80.78, 'delay_ms': 30,
                                           'boxcar_average': {'dforw': 0.5, 'dheading': 0.5, 'sp_rate': 1}}
                                   )

    # Side of LAL (or Gall in case of PEG neurons) that cell projects to.
    LAL_side = {
        '2021_07_05_0001': 'Left',
        '2021_07_05_0002': 'Left',
        '2021_07_08_0003': 'Left',
        '2021_07_09_0003': 'Right',
        '2021_07_12_0002': 'Left',
        '2021_07_12_0008': 'Right',
        '2021_07_13_0001': 'Left',
        '2021_07_13_0005': 'Left',
        '2021_07_15_0002': 'Left',
        '2021_07_22_0002': 'Left',
        '2021_07_25_0001': 'Right',
        '2021_07_25_0003': 'Right',
        '2021_07_25_0004': 'Left',
        '2021_09_06_0001': 'Left',
        '2021_09_10_0001': 'Left',
        '2021_09_11_0004': 'Left',
        '2021_09_17_0002': 'Left',
        '2021_09_18_0003': 'Left',
        '2021_09_20_0001': 'Left',
        '2021_09_24_0002': 'Right',
        '2021_09_27_0002': 'Left',
        '2021_10_01_0002': 'Left',
        '2021_10_05_0002': 'Left',
        '2021_10_05_0004': 'Left',
        '2021_10_06_0004': 'Left',
        '2021_10_07_0002': 'Left',
        '2021_10_14_0006': 'Left',
        '2021_10_18_0002': 'Right',
        '2021_10_29_0001': 'Right',
        '2021_11_01_0002': 'Right',
        '2021_11_01_0006': 'Right',
    }
    recs.merged_abf_df['LAL_side'] = recs.merged_abf_df['rec_name'].map(LAL_side)

    celltype = {
        '2021_07_05_0001': 'PFL3',
        '2021_07_05_0002': 'PEG',
        '2021_07_08_0003': 'PEG',
        '2021_07_09_0003': 'PFL3',
        '2021_07_12_0002': 'PEG',
        '2021_07_12_0008': 'PFL3',
        '2021_07_13_0001': 'PFL3',
        '2021_07_13_0005': 'PFL3',
        '2021_07_15_0002': 'PFL3',
        '2021_07_22_0002': 'PFL3',
        '2021_07_25_0001': 'PFL3',
        '2021_07_25_0003': 'PFL3',
        '2021_07_25_0004': 'PFL3',
        '2021_09_06_0001': 'PFL3',
        '2021_09_10_0001': 'PEG',
        '2021_09_11_0004': 'PEG',
        '2021_09_17_0002': 'PFL3',
        '2021_09_18_0003': 'PFL3',
        '2021_09_20_0001': 'PFL3',
        '2021_09_24_0002': 'PFL3',
        '2021_09_27_0002': 'PFL3',
        '2021_10_01_0002': 'PFL3',
        '2021_10_05_0002': 'PFL3',
        '2021_10_05_0004': 'PEG',
        '2021_10_06_0004': 'PFL3',
        '2021_10_07_0002': 'PFL3',
        '2021_10_14_0006': 'PEG',
        '2021_10_18_0002': 'PEG',
        '2021_10_29_0001': 'PEG',
        '2021_11_01_0002': 'PEG',
        '2021_11_01_0006': 'PFL3',
    }
    recs.merged_abf_df['celltype'] = recs.merged_abf_df['rec_name'].map(celltype)

    return recs


# ------------------ Process & analyze data ----------------------- #


def get_vh(recs):
    recs.merged_abf_df['vh'] = recs.merged_abf_df['xstim'] * -1
    for rec in recs:
        rec.rec.abf.df['vh'] = rec.rec.abf.df['xstim'] * -1


def correct_ljp(recs):
    # correct for liquid junction potential
    recs.merged_abf_df['vm'] = recs.merged_abf_df['vm'] - 13
    recs.merged_abf_df['vm_sp_subtracted'] = recs.merged_abf_df['vm_sp_subtracted'] - 13

    for rec in recs:
        rec.rec.abf.df['vm'] = rec.rec.abf.df['vm'] - 13
        rec.rec.abf.df['vm_sp_subtracted'] = rec.rec.abf.df['vm_sp_subtracted'] - 13

        rec.rec.abf.df_orig['patch_1'] = rec.rec.abf.df_orig['patch_1'] - 13
        rec.rec.abf.df_orig['vm_sp_subtracted'] = rec.rec.abf.df_orig['vm_sp_subtracted'] - 13


def get_pd_df(recs):
    pd_df = ap.get_binned_df(recs.merged_abf_df,
                             bin_values={'vh': np.linspace(-180, 180, 25)},
                             labels=[
                                 'sp_rate',
                                 'vm_sp_subtracted'
                             ],
                             id_vars=['celltype', 'fly_id', 'rec_name', 'LAL_side'],
                             query='`dforw_boxcar_average_0.5_s`>1'
                             )

    # From Larry's fits
    rec_name_to_pd = {
        '2021_07_05_0001': 49.5307,
        '2021_07_09_0003': -167.0907,
        '2021_07_12_0008': 27.6938,
        '2021_07_13_0001': 31.0801,
        '2021_07_13_0005': -176.8249,
        '2021_07_15_0002': 113.7386,
        '2021_07_22_0002': 123.2512,
        '2021_07_25_0001': 53.6430,
        '2021_07_25_0003': -85.9983,
        '2021_07_25_0004': -28.3176,
        '2021_09_06_0001': 94.6029,
        '2021_09_17_0002': 100.0541,
        '2021_09_18_0003': -82.1752,
        '2021_09_20_0001': -127.5489,
        '2021_09_24_0002': -42.8779,
        '2021_09_27_0002': -34.3190,
        '2021_10_01_0002': 46.8383,
        '2021_10_05_0002': 41.2362,
        '2021_10_06_0004': -167.6846,
        '2021_10_07_0002': -168.4983,
        '2021_11_01_0006': 176.6928
    }

    pd_df['pd'] = pd_df['rec_name'].map(rec_name_to_pd)

    # adds  pd to merged_abf_df
    recs.merged_abf_df['pd'] = recs.merged_abf_df['rec_name'].map(rec_name_to_pd)

    pd_df['vh_pd_subtracted'] = fc.wrap(pd_df['vh_bin_center'].astype(float) - pd_df['pd'])
    pd_df.sort_values(by=['rec_name', 'vh_pd_subtracted'], inplace=True)

    return pd_df


def get_fits_df(pd_df, path):
    temp_df = pd_df.query('celltype=="PFL3"')
    temp_df = temp_df.filter(['rec_name', 'vh_bin_center', 'variable', 'mean_value'], axis=1)
    temp_df = temp_df.pivot_table(index=['rec_name', 'vh_bin_center'], columns=['variable'],
                                  values=['mean_value']).reset_index()
    rec_names = temp_df['rec_name'].unique()
    rec_to_cell_nb = dict(zip(rec_names, np.arange(len(rec_names)) + 1))

    cell_nb_to_rec = {}
    for key, item in rec_to_cell_nb.items():
        cell_nb_to_rec[item] = key
    fits_df = pd.read_csv(path + 'VMFits.csv', header=None)
    fits_df.reset_index(inplace=True)
    fits_df['index'] = fits_df['index'] + 1
    fits_df['rec_name'] = fits_df['index'].map(cell_nb_to_rec)

    # Amp, offSet and phase
    fits_df = fits_df.rename({0: 'amp', 1: 'offset', 2: 'phase'}, axis=1)
    return fits_df


def get_fixation_events(recs):
    abf_fixation_df = ap.get_fixation_events_df(recs, detection_parameters={'RDP_epsilon': 25,
                                                                            'min_length': 200}, inplace=True, im=False)
    # a bit confusing, but default is to use xstim to calculate goal and distance to goal
    # but in this dataset we want virtual heading since we don't compare goal with phase
    # so goal and distance_to_goal need to be flipped
    # should probably standardize variable names across datasets...
    abf_fixation_df['goal'] = abf_fixation_df['goal'] * -1
    abf_fixation_df['distance_to_goal'] = abf_fixation_df['distance_to_goal'] * -1

    abf_fixation_df['vh_pd_subtracted'] = fc.wrap(abf_fixation_df['vh'] - abf_fixation_df['pd'])
    abf_fixation_df['goal_to_pd_distance'] = fc.wrap(abf_fixation_df['goal'] - abf_fixation_df['pd'])

    abf_fixation_df['vm_sp_subtracted_zeroed'] = abf_fixation_df.groupby(['rec_name'])['vm_sp_subtracted'].apply(
        lambda x: x - x.min())
    return abf_fixation_df


def get_modulation_df(abf_fixation_df):
    vh_to_pd_bins = np.linspace(-180, 180, 9)
    goal_to_pd_bins = np.linspace(-180, 180, 9)

    event_df = ap.get_binned_df(abf_fixation_df,
                                bin_values={'vh_pd_subtracted': vh_to_pd_bins,
                                            'goal_to_pd_distance': goal_to_pd_bins},
                                labels=['sp_rate', 'vm_sp_subtracted', 'vm', 'vm_sp_subtracted_zeroed'],
                                id_vars=['LAL_side', 'celltype', 'rec_name'],
                                query='is_fixating==True & `dforw_boxcar_average_0.5_s`>1',
                                )

    # gets mean across flies
    fly_df = event_df.groupby(['LAL_side',
                               'celltype',
                               'vh_pd_subtracted_bin_center',
                               'goal_to_pd_distance_bin_center',
                               'variable']).agg(mean=('mean_value', np.nanmean),
                                                count=('mean_value', len), sd=('mean_value', np.nanstd))
    fly_df.reset_index(inplace=True)
    fly_df['rec_name'] = 'mean'
    modulation_df = pd.concat([fly_df, event_df])
    return event_df, fly_df, modulation_df


def get_event_df_sem(abf_fixation_df, event_df):
    # adds sem
    vh_to_pd_bins = np.linspace(-180, 180, 9)
    goal_to_pd_bins = np.linspace(-180, 180, 9)

    event_df_sem = ap.get_binned_df(abf_fixation_df,
                                    bin_values={'vh_pd_subtracted': vh_to_pd_bins,
                                                'goal_to_pd_distance': goal_to_pd_bins},
                                    labels=['sp_rate', 'vm_sp_subtracted', 'vm', 'vm_sp_subtracted_zeroed'],
                                    id_vars=['LAL_side', 'celltype', 'rec_name'],
                                    query='is_fixating==True & `dforw_boxcar_average_0.5_s`>1',
                                    metric=lambda x: sc.stats.sem(x, nan_policy='omit'),
                                    )

    event_df_sem = event_df_sem.rename({'mean_value': 'sem'}, axis=1)
    event_df_sem = event_df.join(event_df_sem, rsuffix='drop')
    return event_df_sem

    # new_df.sort_values(by=['pd','vh_bin_center'],inplace=True)


def get_modulation_df_ctl_walking(abf_fixation_df):
    vh_to_pd_bins = np.linspace(-180, 180, 9)
    goal_to_pd_bins = np.linspace(-180, 180, 9)

    event_df = ap.get_binned_df(abf_fixation_df,
                                bin_values={'vh_pd_subtracted': vh_to_pd_bins,
                                            'goal_to_pd_distance': goal_to_pd_bins},
                                labels=['sp_rate',
                                        'dheading_boxcar_average_0.5_s', 'dforw_boxcar_average_0.5_s'],
                                id_vars=['LAL_side', 'celltype', 'rec_name'],
                                query='is_fixating==True & `dheading_boxcar_average_0.5_s`<5 & `dheading_boxcar_average_0.5_s`>-5  & `dforw_boxcar_average_0.5_s`<0.5 & `dforw_boxcar_average_0.5_s`>-0.5',
                                )

    event_df['flip'] = event_df['LAL_side'].map({'Right': -1, 'Left': 1})
    event_df['goal_to_pd_distance_bin_center_flipped'] = event_df['goal_to_pd_distance_bin_center'].astype(float) * \
                                                         event_df['flip']
    event_df['vh_pd_subtracted_bin_center_flipped'] = event_df['vh_pd_subtracted_bin_center'].astype(float) * event_df[
        'flip']

    # gets mean across flies (i.e. cells)
    fly_df = event_df.groupby([
        'celltype',
        'vh_pd_subtracted_bin_center_flipped',
        'goal_to_pd_distance_bin_center_flipped',
        'variable']).agg(mean=('mean_value', np.mean))
    fly_df.reset_index(inplace=True)
    fly_df['rec_name'] = 'mean'
    modulation_df_ctl_walk = pd.concat([fly_df, event_df])
    return modulation_df_ctl_walk


def get_sp_by_dforw(recs, save=False, savepath=None, fname=None):
    binned_df = ap.get_binned_df(recs.merged_abf_df,
                                 bin_values={'dforw_boxcar_average_0.5_s': np.linspace(-1, 11, 11),
                                             'vh_pd_subtracted': np.linspace(-180, 180, 37)},
                                 labels=['sp_rate', 'vm_sp_subtracted_zeroed'],
                                 id_vars=['LAL_side', 'fly_id'],
                                 query='celltype=="PFL3"',
                                 )

    print(len(binned_df['fly_id'].unique()), 'cells')

    binned_df['flip'] = binned_df['LAL_side'].map({'Right': -1, 'Left': 1})
    binned_df['vh_pd_subtracted_bin_center_flipped'] = binned_df['vh_pd_subtracted_bin_center'].astype(float) * \
                                                       binned_df['flip']

    mean_df = binned_df.groupby(
        ['variable', 'dforw_boxcar_average_0.5_s_bin_center', 'vh_pd_subtracted_bin_center_flipped'],
        as_index=False).agg(mean=('mean_value', np.mean))

    mean_df['dforw_boxcar_average_0.5_s_bin_center'] = mean_df['dforw_boxcar_average_0.5_s_bin_center'].astype(float)

    sp_by_dforw_df = pd.pivot_table(mean_df.query('variable=="sp_rate"'), values='mean',
                                    columns='vh_pd_subtracted_bin_center_flipped',
                                    index='dforw_boxcar_average_0.5_s_bin_center')

    if save:
        sp_by_dforw_df.to_csv(savepath + fname)

    return sp_by_dforw_df


def get_sp_by_dheading(recs, save=False, savepath=None, fname=None):
    binned_df = ap.get_binned_df(recs.merged_abf_df,
                                 bin_values={'dheading_boxcar_average_0.5_s': np.linspace(-200, 200, 11),
                                             'vh_pd_subtracted': np.linspace(-180, 180, 37)},
                                 labels=['sp_rate', 'vm_sp_subtracted_zeroed'],
                                 id_vars=['LAL_side', 'fly_id'],
                                 query='celltype=="PFL3"',
                                 )

    binned_df['flip'] = binned_df['LAL_side'].map({'Right': -1, 'Left': 1})
    binned_df['vh_pd_subtracted_bin_center_flipped'] = binned_df['vh_pd_subtracted_bin_center'].astype(float) * \
                                                       binned_df['flip']
    binned_df['dheading_boxcar_average_0.5_s_bin_center_flipped'] = binned_df[
                                                                        'dheading_boxcar_average_0.5_s_bin_center'].astype(
        float) * binned_df['flip']

    mean_df = binned_df.groupby(
        ['variable', 'dheading_boxcar_average_0.5_s_bin_center_flipped', 'vh_pd_subtracted_bin_center_flipped'],
        as_index=False).agg(mean=('mean_value', np.mean))

    sp_by_dheading_df = pd.pivot_table(mean_df.query('variable=="sp_rate"'), values='mean',
                                       columns='vh_pd_subtracted_bin_center_flipped',
                                       index='dheading_boxcar_average_0.5_s_bin_center_flipped')

    if save:
        sp_by_dheading_df.to_csv(savepath + fname)

    return sp_by_dheading_df


def get_spike_vs_vm_df(abf_fixation_df, save=False, savepath=None, fname=None):
    filter_df = abf_fixation_df.query(
        '(celltype=="PFL3") & (is_fixating==True) & (`dforw_boxcar_average_0.5_s`>1)').reset_index(drop=True)

    filter_df = filter_df.filter(['LAL_side', 'rec_name', 'goal_to_pd_distance', 'vm_sp_subtracted', 'sp_rate'])

    rec_names = filter_df['rec_name'].unique()
    rec_to_cell_nb = dict(zip(rec_names, np.arange(len(rec_names)) + 1))

    filter_df['cell'] = filter_df['rec_name'].map(rec_to_cell_nb)

    filter_df['flip'] = filter_df['LAL_side'].map({'Right': -1, 'Left': 1})
    filter_df['goal_to_pd_distance_flipped'] = filter_df['goal_to_pd_distance'].astype(float) * filter_df['flip']

    filter_df = filter_df.drop(['LAL_side', 'flip', 'goal_to_pd_distance'], axis=1)

    spike_vs_vm_df = ap.get_binned_df(filter_df,
                                      bin_values={'vm_sp_subtracted': np.linspace(-86, -46, 11),
                                                  'goal_to_pd_distance_flipped': np.linspace(-180, 180, 9)},
                                      labels='sp_rate',
                                      id_vars=['rec_name'])

    mean_spike_vs_vm_df = spike_vs_vm_df.groupby([
        'goal_to_pd_distance_flipped_bin_center',
        'vm_sp_subtracted_bin_center']).agg(mean=('mean_value', np.nanmean)).reset_index()

    if save:
        mean_spike_vs_vm_df.rename({'mean': 'sp_rate'}, axis=1).to_csv(savepath + fname)

    return mean_spike_vs_vm_df


def get_menotaxis_bouts_raster_and_goals(recs, save=False, savepath=None, fname=None):
    menotaxis_bouts_raster_df = recs.merged_abf_df.query('celltype=="PFL3"').filter(
        ['genotype', 'fly_id', 'rec_name', 't', 'is_fixating', 'goal', 'fixation_event_id']).copy()
    menotaxis_bouts_raster_df['new_fly_id'] = menotaxis_bouts_raster_df['fly_id'].map(dict(
        zip(np.sort(menotaxis_bouts_raster_df['fly_id'].unique()),
            np.arange(len(menotaxis_bouts_raster_df['fly_id'].unique())) + 1)))
    menotaxis_bouts_raster_df['fly_id'] = menotaxis_bouts_raster_df['new_fly_id']
    menotaxis_bouts_raster_df.drop('new_fly_id', axis=1, inplace=True)
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
    if save:
        summary_menotaxis_df.filter(['fly_id', 'goal']).to_csv(savepath + fname)

    return menotaxis_bouts_raster_df, summary_menotaxis_df


# ------------------ Plotting functions ----------------------- #

def plot_all_vm_aligned(pd_df, rec_name_to_cell_no, save_figure=False, save_figure_path=None, figure_fname=None,
                        save_source_data=False, save_source_data_path=None, source_data_fname=None):
    plot_df = pd_df.copy()
    plot_df['cell_no'] = plot_df['rec_name'].map(rec_name_to_cell_no)
    plot_df = plot_df.query('celltype=="PFL3" & variable=="vm_sp_subtracted"').filter(
        ['cell_no', 'vh_pd_subtracted', 'mean_value']).copy().reset_index(drop=True)
    g = ph.FacetGrid(plot_df,
                     unit='cell_no',
                     fig_kws={'figsize': [0.75, 0.75], 'dpi': dpi},
                     gridspec_kws={'wspace': 0,
                                   'hspace': 0},
                     )

    def plot(data):
        plt.plot(data['vh_pd_subtracted'].values, data['mean_value'].values,
                 color='k', lw=0.25, alpha=0.5, clip_on=False)

    g.map_dataframe(plot)
    ax = g.axes[0, 0]
    ax.set_xticks([-180, 0, 180])
    ax.set_xlim([-180, 180])
    ax.set_ylim([-80, -30])
    ax.set_yticks([-80, -30])
    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2})

    if save_figure:
        plt.savefig(save_figure_path + figure_fname,
                    transparent=True, bbox_inches='tight')

    if save_source_data:
        plot_df.to_csv(save_source_data_path + source_data_fname)


def plot_all_sp_rate(pd_df, rec_name_to_cell_no, PFL3_row_order, save_figure=False, save_figure_path=None,
                     figure_fname=None,
                     save_source_data=False, save_source_data_path=None, source_data_fname=None):
    plot_df = pd_df.copy()
    plot_df['cell_no'] = plot_df['rec_name'].map(rec_name_to_cell_no)
    plot_df = plot_df.query('celltype=="PFL3" & variable=="sp_rate"').sort_values(
        by=['pd', 'vh_bin_center']).copy().reset_index(drop=True)
    g = ph.FacetGrid(plot_df,
                     row='rec_name',
                     fig_kws={'figsize': [0.5, 6], 'dpi': dpi},
                     gridspec_kws={'wspace': 0,
                                   'hspace': 0},
                     row_order=PFL3_row_order
                     )

    def plot(data):
        plt.plot(data['vh_bin_center'], data['mean_value'], color='k', lw=0.5, clip_on=False)

    g.map_dataframe(plot)
    for ax in g.axes.ravel():
        ax.axvspan(-180, -135, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0, ymax=0.9)
        ax.axvspan(135, 180, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0, ymax=0.9)
        ax.set_ylim([-5, 70])
        ax.set_xticks([-180, 0, 180])
        ax.set_xlim([-180, 180])
        rect = ax.patch
        rect.set_alpha(0)

    g.gs.update(hspace=-0.1)

    ph.despine_axes(g.axes, style=['bottom'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2})

    sb = ph.add_scalebar(g.axes[-1, 0], sizex=0, sizey=50, barwidth=axis_lw * 2, barcolor='k', loc=3,
                         pad=0,
                         bbox_transform=ax.transAxes,
                         bbox_to_anchor=[-0.1, -0.1]
                         #                                         bbox_to_anchor=[1.1,-0.1]

                         )
    if save_figure:
        plt.savefig(save_figure_path + figure_fname,
                    transparent=True, bbox_inches='tight')

    if save_source_data:
        plot_df.filter(['cell_no', 'vh_bin_center', 'mean_value']).to_csv(save_source_data_path + source_data_fname)


def plot_all_vm(pd_df, fits_df, rec_name_to_cell_no, PFL3_row_order, save_figure=False, save_figure_path=None,
                figure_fname=None,
                save_source_data=False, save_source_data_path=None, source_data_fname=None):
    pd_df_vm = pd_df.copy()
    pd_df_vm['cell_no'] = pd_df_vm['rec_name'].map(rec_name_to_cell_no)
    pd_df_vm = pd_df_vm.query('variable=="vm_sp_subtracted"').copy().reset_index(drop=True)

    # a bit funky
    def zero_vm(data):
        data['min'] = np.nanmin(data['mean_value'].values)
        data['zero_vm'] = data['mean_value'].values - np.nanmin(data['mean_value'].values)
        return data

    pd_df_vm['zero_vm'] = np.nan
    pd_df_vm = pd_df_vm.groupby(['rec_name']).apply(zero_vm)
    pd_df_vm = pd_df_vm.sort_values(by=['vh_bin_center']).query('celltype=="PFL3"')
    g = ph.FacetGrid(pd_df_vm,
                     row='rec_name',
                     fig_kws={'figsize': [0.5, 6], 'dpi': dpi},
                     gridspec_kws={'wspace': 0,
                                   'hspace': 0},
                     row_order=PFL3_row_order
                     )

    def plot(data):
        rec_name = data['rec_name'].values[0]
        fit = fits_df.query(f'rec_name=="{rec_name}"')
        amp = fit['amp'].values[0]
        phase = fit['phase'].values[0]
        min_val = data['min'].values[0]
        offset = fit['offset'].values[0] - min_val
        y = amp * np.cos(np.deg2rad(data['vh_bin_center'].astype(float)) - np.deg2rad(phase)) + offset
        plt.plot(data['vh_bin_center'], y, clip_on=False, ls=':', color='#36454F', lw=1.5, zorder=1)
        plt.plot(data['vh_bin_center'], data['zero_vm'], color='k', lw=0.5,
                 clip_on=False, zorder=1)

    #     plt.plot(data['vh_bin_center'],y,ls=':',color='#d99898',lw=0.5,)

    g.map_dataframe(plot)

    for ax in g.axes.ravel():
        ax.axvspan(-180, -135, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0, ymax=0.9)
        ax.axvspan(135, 180, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0, ymax=0.9)
        ax.set_ylim([0, 40])
        ax.set_xticks([-180, 0, 180])
        ax.set_xlim([-180, 180])
        rect = ax.patch
        rect.set_alpha(0)

    g.gs.update(hspace=-0.1)

    ph.despine_axes(g.axes, style=['bottom'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2})

    sb = ph.add_scalebar(g.axes[-1, 0], sizex=0, sizey=20, barwidth=axis_lw * 2, barcolor='k', loc=3,
                         pad=0,
                         bbox_transform=ax.transAxes,
                         bbox_to_anchor=[-0.1, -0.1]
                         )

    if save_figure:
        plt.savefig(save_figure_path + figure_fname,
                    transparent=True, bbox_inches='tight')
    if save_source_data:
        pd_df_vm.filter(['cell_no', 'vh_bin_center', 'zero_vm']).sort_values(['cell_no', 'vh_bin_center']).reset_index(
            drop=True).to_csv(save_source_data_path + source_data_fname)


def plot_example_tuning_curves_vm(rec_names, pd_df, rec_name_to_cell_no,
                                  save=False, savepath=None, fname=None):
    pd_df.sort_values(by=['rec_name', 'vh_bin_center'], inplace=True)

    row_order = rec_names
    print([rec_name_to_cell_no[rec] for rec in row_order])

    g = ph.FacetGrid(pd_df.query('variable=="vm_sp_subtracted"')[pd_df['rec_name'].isin(row_order)],
                     row='rec_name',
                     row_order=row_order,
                     unit='rec_name',
                     fig_kws={'figsize': [0.5, 1.75], 'dpi': dpi},
                     gridspec_kws={'wspace': 0,
                                   'hspace': 0.5},
                     )

    def plot(data):
        plt.plot(data['vh_bin_center'].values, data['mean_value'].values, color='k', lw=1, alpha=1, clip_on=False)

    g.map_dataframe(plot)

    for ax in g.axes.ravel():
        ax.set_xticks([-180, 0, 180])
        ax.set_xlim([-180, 180])
        ax.set_ylim([-80, -40])
        ax.set_yticks([-80, -40])
        ax.axvspan(-180, -135, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0)
        ax.axvspan(135, 180, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0)
    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_tuning_curves_sp_rate(rec_names, pd_df, rec_name_to_cell_no, save=False, savepath=None, fname=None):
    pd_df.sort_values(by=['rec_name', 'vh_bin_center'], inplace=True)
    row_order = rec_names
    print([rec_name_to_cell_no[rec] for rec in row_order])
    g = ph.FacetGrid(pd_df.query('variable=="sp_rate"')[pd_df['rec_name'].isin(row_order)],
                     row='rec_name',
                     row_order=row_order,
                     unit='rec_name',
                     fig_kws={'figsize': [0.5, 1.75], 'dpi': dpi},
                     gridspec_kws={'wspace': 0,
                                   'hspace': 0.5},
                     )

    def plot(data):
        plt.plot(data['vh_bin_center'].values, data['mean_value'].values, color='k', lw=1, alpha=1, clip_on=False)

    g.map_dataframe(plot)

    for ax in g.axes.ravel():
        ax.set_xticks([-180, 0, 180])
        ax.set_xlim([-180, 180])
        ax.set_ylim([0, 70])
        ax.set_yticks([0, 70])
        ax.axvspan(-180, -135, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0)
        ax.axvspan(135, 180, facecolor=(0.9, 0.9, 0.9), edgecolor='none', zorder=0)
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_label_position("right")
        ph.adjust_spines(ax, spines=['right'], lw=axis_lw, ticklen=axis_ticklen, pad=2)

    ax = g.axes[-1, 0]
    ph.adjust_spines(ax, spines=['right', 'bottom'], lw=axis_lw, ticklen=axis_ticklen, pad=2)
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_example_trace(recs, rec_name_to_cell_no, pd_df, save=False, savepath=None, fname=None):
    rec_name = '2021_09_24_0002'

    print(rec_name_to_cell_no[rec_name])

    sp_rate_ylim = [0, 70]

    tlim1 = [2460, 2620]
    tlim2 = [2537.3, 2538.3]

    vm_ylim = [-80, -30]
    vm_ylim2 = [-60, -40]

    ap.plot_patch(recs, rec_name, tlim1, pd_df,
                  fig_kws={'figsize': [4.5, 1.5],
                           'dpi': dpi},
                  adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen},
                  vm_ylim=vm_ylim, sp_rate_ylim=sp_rate_ylim,
                  tline=tlim2, density=None,
                  height_ratios=[0, 0.4, 0.1, 0.5], hspace=0.5)

    ax = plt.gca()
    sb = ph.add_scalebar(plt.gca(), sizex=10, sizey=0, barwidth=axis_lw, barcolor='k',
                         loc='lower center',
                         pad=0,
                         bbox_transform=ax.transAxes,
                         #                     bbox_to_anchor=[2.5,0]
                         #                                         bbox_to_anchor=[5,-1.7]

                         )
    if save:
        plt.savefig(savepath + fname + '.pdf',
                    transparent=True, bbox_inches='tight')

    plt.show()

    ap.plot_patch3(recs, rec_name, tlim=tlim2,
                   fig_kws={'figsize': [2, 0.5],
                            'dpi': dpi},
                   adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen},
                   vm_ylim=vm_ylim2, tline_s=0.1)
    if save:
        plt.savefig(savepath + fname + '_zoom.pdf',
                    transparent=True, bbox_inches='tight')


def plot_pfl3_vs_peg(recs, rec_name_to_cell_no, save=False, savepath=None, fnames=None):
    # PFL3

    rec_name = '2021_10_07_0002'
    print(rec_name_to_cell_no[rec_name])

    tlim = [298.8, 299.2]
    vm_ylim = [-55, -40]
    ap.plot_patch3(recs, rec_name, tlim, fig_kws={'figsize': [2, 1],
                                                  'dpi': dpi},
                   adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen},
                   vm_ylim=vm_ylim, tline_s=0.1
                   )

    if save:
        plt.savefig(savepath + fnames[0] + '_left.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()

    tlim = [1838, 1838 + 4]
    vm_ylim = [-80, -60]
    ap.plot_patch3(recs, rec_name, tlim, fig_kws={'figsize': [2, 1],
                                                  'dpi': dpi},
                   adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen},
                   vm_ylim=vm_ylim, tline_s=0.5
                   )
    if save:
        plt.savefig(savepath + fnames[1] + '_left.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()

    # PEG
    rec_name = '2021_10_05_0004'
    tlim = [681.8, 682.2]
    vm_ylim = [-55, -40]
    ap.plot_patch3(recs, rec_name, tlim, fig_kws={'figsize': [2, 1],
                                                  'dpi': dpi},
                   adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen},
                   vm_ylim=vm_ylim, tline_s=0.1
                   )

    if save:
        plt.savefig(savepath + fnames[0] + '_right.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()

    rec_name = '2021_10_05_0004'
    tlim = [482, 482 + 4]
    vm_ylim = [-80, -60]
    ap.plot_patch3(recs, rec_name, tlim, fig_kws={'figsize': [2, 1],
                                                  'dpi': dpi},
                   adjust_spines_kws={'pad': 1, 'lw': axis_lw, 'ticklen': axis_ticklen},
                   vm_ylim=vm_ylim, tline_s=0.5
                   )
    if save:
        plt.savefig(savepath + fnames[1] + '_right.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()


def plot_example_tuning_curve_modulation(abf_fixation_df, rec_name_to_cell_no,
                                         pd_df, save=False, savepath=None, fname=None):
    vh_bins = np.linspace(-180, 180, 9)
    goal_to_pd_bins = np.linspace(-180, 180, 9)

    color_A = 'k'
    color_B = 'grey'

    rec_name = '2021_07_13_0001'
    print(rec_name_to_cell_no[rec_name])
    example_goal_to_pd = [-112.5, 22.5]
    plot_paired_fixation_tuning_aligned(rec_name, abf_fixation_df, pd_df, example_goal_to_pd,
                                        color_A, color_B,
                                        dpi, axis_ticklen, axis_lw, vh_bins, goal_to_pd_bins)

    if save:
        plt.savefig(savepath + fname + '_fist_col.pdf',
                    transparent=True, bbox_inches='tight')

    plt.show()

    rec_name = '2021_07_13_0005'
    print(rec_name_to_cell_no[rec_name])
    example_goal_to_pd = [-22.5, 67.5]
    plot_paired_fixation_tuning_aligned(rec_name, abf_fixation_df, pd_df, example_goal_to_pd,
                                        color_A, color_B,
                                        dpi=dpi, axis_ticklen=axis_ticklen, axis_lw=axis_lw,
                                        vh_bins=vh_bins, goal_to_pd_bins=goal_to_pd_bins
                                        )
    if save:
        plt.savefig(savepath + fname + '_second_col.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()

    rec_name = '2021_09_06_0001'
    example_goal_to_pd = [-22.5, 157.5]
    print(rec_name_to_cell_no[rec_name])
    plot_paired_fixation_tuning_aligned(rec_name, abf_fixation_df, pd_df, example_goal_to_pd,
                                        color_A, color_B,
                                        dpi=dpi, axis_ticklen=axis_ticklen, axis_lw=axis_lw,
                                        vh_bins=vh_bins, goal_to_pd_bins=goal_to_pd_bins)

    if save:
        plt.savefig(savepath + fname + '_third_col.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()


def plot_left_sp_rate_modulation(modulation_df, rec_name_to_cell_no,
                                 save_figure=False,
                                 save_figure_path=None,
                                 figure_fname=None,
                                 save_source_data=False,
                                 save_source_data_path=None,
                                 source_data_fname=None):
    vh_to_pd_bins = np.linspace(-180, 180, 9)  # make sure that these are bins use in get_modulation_df()
    circplot_period = np.diff(vh_to_pd_bins)[0] * 2  # used to prevent drawing lines between non adjacent bins
    # could alternatively add NaNs to empty bins...

    print(len(modulation_df.query('celltype=="PFL3" & LAL_side=="Left" & (rec_name!="mean")')['rec_name'].unique()),
          'cells')
    plot_df = modulation_df.query('celltype=="PFL3" & LAL_side=="Left" & (variable=="sp_rate") ').copy()

    g = ph.FacetGrid(plot_df,
                     col='goal_to_pd_distance_bin_center',
                     row='variable',
                     unit='rec_name',
                     fig_kws={'figsize': [4.5, 1.25], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.2,
                                   'hspace': 0.4},
                     col_order=modulation_df['goal_to_pd_distance_bin_center'].unique().sort_values(ascending=True)
                     )

    def plot(data):
        goal_to_pd_distance_bin_center = data['goal_to_pd_distance_bin_center'].values[0]
        cmap = plt.cm.twilight_shifted
        #     color='k'
        color = cmap((goal_to_pd_distance_bin_center + 180.) / 360.)

        if data['rec_name'].iloc[0] != 'mean':
            ph.circplot(data['vh_pd_subtracted_bin_center'].values, data['mean_value'].values, circ='x', color=color,
                        period=circplot_period, lw=0.2, args={'clip_on': False})

            plt.scatter(data['vh_pd_subtracted_bin_center'].values, data['mean_value'].values, color=color, s=0.1,
                        clip_on=False)


        else:
            plt.scatter(data['vh_pd_subtracted_bin_center'].values, data['mean'].values,
                        facecolors='none', edgecolors=color, s=10, clip_on=False)

            if data['variable'].values[0] == "sp_rate":
                vMa = get_vMa(H=H, G=np.deg2rad(goal_to_pd_distance_bin_center),
                              d=d, phiGoal=phiGoal)
                f = get_f(vMa=vMa, a=a, b=b, c=c)
                plt.plot(np.rad2deg(H), f, color=color, lw=1, clip_on=False)

    g.map_dataframe(plot)

    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2})

    for i, row in enumerate(g.axes):
        for ax in row:
            if i == 0:
                ax.set_ylim([0, 80])
                ax.set_yticks([0, 80])
            else:
                ax.set_ylim([-70, -20])
                ax.set_yticks([-70, -20])
            ax.set_xticks([-180, 0, 180])
            ax.set_xlim([-180, 180])
            ax.set_xticklabels([])

    ax = g.axes[0, 0]
    ax.set_xticklabels([-180, 0, 180])

    if save_figure:
        plt.savefig(save_figure_path + figure_fname,
                    transparent=True, bbox_inches='tight')
    if save_source_data:
        plot_df['cell_no'] = plot_df['rec_name'].map(rec_name_to_cell_no)
        plot_df = plot_df.filter(
            ['cell_no', 'vh_pd_subtracted_bin_center', 'goal_to_pd_distance_bin_center', 'mean_value', 'mean'])
        plot_df.rename({'mean_value': 'sp_rate'}, axis=1, inplace=True)
        plot_df.to_csv(save_source_data_path + source_data_fname)


def plot_left_right_sp_rate_modulation(modulation_df, rec_name_to_cell_no,
                                       save_figure=False,
                                       save_figure_path=None,
                                       figure_fname=None,
                                       save_source_data=False,
                                       save_source_data_path=None,
                                       source_data_fname=None):
    vh_to_pd_bins = np.linspace(-180, 180, 9)  # make sure that these are bins use in get_modulation_df()
    circplot_period = np.diff(vh_to_pd_bins)[0] * 2  # used to prevent drawing lines between non adjacent bins
    # could alternatively add NaNs to empty bins...

    print(len(modulation_df.query('celltype=="PFL3" & LAL_side=="Left" & (rec_name!="mean")')['rec_name'].unique()),
          'left cells')
    print(len(modulation_df.query('celltype=="PFL3" & LAL_side=="Right" & (rec_name!="mean")')['rec_name'].unique()),
          'right cells')

    plot_df = modulation_df.query('celltype=="PFL3" & variable=="sp_rate"').copy()
    g = ph.FacetGrid(plot_df,
                     col='goal_to_pd_distance_bin_center',
                     row='LAL_side',
                     unit='rec_name',
                     fig_kws={'figsize': [3, 1], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.2,
                                   'hspace': 0.4},
                     col_order=modulation_df['goal_to_pd_distance_bin_center'].unique().sort_values(ascending=True)
                     )

    def plot(data):
        goal_to_pd_distance_bin_center = data['goal_to_pd_distance_bin_center'].values[0]
        cmap = plt.cm.twilight_shifted
        color = cmap((goal_to_pd_distance_bin_center + 180.) / 360.)

        if data['rec_name'].iloc[0] != 'mean':
            ph.circplot(data['vh_pd_subtracted_bin_center'].values, data['mean_value'].values, circ='x', color=color,
                        period=circplot_period, lw=0.2, args={'clip_on': False})

            plt.scatter(data['vh_pd_subtracted_bin_center'].values, data['mean_value'].values, color=color, s=0.1,
                        clip_on=False)

        else:
            # plot mean
            ph.circplot(data['vh_pd_subtracted_bin_center'].values, data['mean'].values, circ='x',
                        color=color, period=circplot_period, lw=1, args={'clip_on': False})

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2})

    for ax in g.axes.ravel():
        ax.set_ylim([0, 80])
        ax.set_yticks([0, 80])
        ax.set_xticks([-180, 0, 180])
        ax.set_xlim([-180, 180])
        ax.set_xticklabels([])
    #     ax.axvline(x=0,ls=':',lw=0.5,color='k')

    ax = g.axes[1, 0]
    ax.set_xticklabels([-180, 0, 180])

    if save_figure:
        plt.savefig(save_figure_path + figure_fname,
                    transparent=True, bbox_inches='tight')
    if save_source_data:
        plot_df['cell_no'] = plot_df['rec_name'].map(rec_name_to_cell_no)
        plot_df = plot_df.filter(
            ['LAL_side', 'cell_no', 'vh_pd_subtracted_bin_center', 'goal_to_pd_distance_bin_center', 'mean_value',
             'mean'])
        plot_df.rename({'mean_value': 'sp_rate'}, axis=1, inplace=True)
        plot_df.to_csv(save_source_data_path + source_data_fname)


def plot_left_right_vm_modulation(modulation_df, rec_name_to_cell_no,
                                  save_figure=False,
                                  save_figure_path=None,
                                  figure_fname=None,
                                  save_source_data=False,
                                  save_source_data_path=None,
                                  source_data_fname=None):
    vh_to_pd_bins = np.linspace(-180, 180, 9)  # make sure that these are bins use in get_modulation_df()

    circplot_period = np.diff(vh_to_pd_bins)[0] * 2  # used to prevent drawing lines between non adjacent bins
    # could alternatively add NaNs to empty bins...
    plot_df = modulation_df.query('celltype=="PFL3" & variable=="vm_sp_subtracted"').copy()
    g = ph.FacetGrid(plot_df,
                     col='goal_to_pd_distance_bin_center',
                     row='LAL_side',
                     unit='rec_name',
                     fig_kws={'figsize': [3, 1], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.2,
                                   'hspace': 0.4},
                     )

    def plot(data):
        goal_to_pd_distance_bin_center = data['goal_to_pd_distance_bin_center'].values[0]
        cmap = plt.cm.twilight_shifted
        color = cmap((goal_to_pd_distance_bin_center + 180.) / 360.)

        if data['rec_name'].iloc[0] != 'mean':
            ph.circplot(data['vh_pd_subtracted_bin_center'].values, data['mean_value'].values, circ='x', color=color,
                        period=circplot_period, lw=0.2, args={'clip_on': False})

            plt.scatter(data['vh_pd_subtracted_bin_center'].values, data['mean_value'].values, color=color, s=0.1,
                        clip_on=False)

        else:
            # plot mean
            ph.circplot(data['vh_pd_subtracted_bin_center'].values, data['mean'].values, circ='x',
                        color=color, period=circplot_period, lw=1, args={'clip_on': False})

    g.map_dataframe(plot)
    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2})

    for ax in g.axes.ravel():
        ax.set_ylim([-80, -30])
        ax.set_yticks([-80, -30])
        ax.set_xticks([-180, 0, 180])
        ax.set_xlim([-180, 180])
        ax.set_xticklabels([])

    ax = g.axes[-1, 0]
    ax.set_xticklabels([-180, 0, 180])

    if save_figure:
        plt.savefig(save_figure_path + figure_fname,
                    transparent=True, bbox_inches='tight')
    if save_source_data:
        plot_df['cell_no'] = plot_df['rec_name'].map(rec_name_to_cell_no)
        plot_df = plot_df.filter(
            ['LAL_side', 'cell_no', 'vh_pd_subtracted_bin_center', 'goal_to_pd_distance_bin_center', 'mean_value',
             'mean'])
        plot_df.rename({'mean_value': 'vm'}, axis=1, inplace=True)
        plot_df.to_csv(save_source_data_path + source_data_fname)


def plot_all_cells_sp_rate_modulation(event_df_sem, rec_name_to_cell_no, save=False, savepath=None, fname=None):
    # Cell nb order

    # LEFT AND RIGHT IN SAME PLOT
    event_df_sem['cell_no'] = event_df_sem['rec_name'].map(rec_name_to_cell_no)
    row_order = \
        event_df_sem.query('celltype=="PFL3"').drop_duplicates(['rec_name']).sort_values(['LAL_side', 'cell_no'])[
            'rec_name']
    print([rec_name_to_cell_no[rec] for rec in row_order])

    vh_to_pd_bins = np.linspace(-180, 180, 9)

    circplot_period = np.diff(vh_to_pd_bins)[0] * 2  # used to prevent drawing lines between non adjacent bins
    # could alternatively add NaNs to empty bins...

    g = ph.FacetGrid(event_df_sem.query('celltype=="PFL3"& variable=="sp_rate"'),
                     col='goal_to_pd_distance_bin_center',
                     row='rec_name',
                     #                unit='rec_name',
                     fig_kws={'figsize': [3, 4], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.2,
                                   'hspace': 0.2},
                     col_order=event_df_sem['goal_to_pd_distance_bin_center'].unique().sort_values(ascending=True),
                     row_order=row_order
                     )

    def plot(data):
        goal_to_pd_distance_bin_center = data['goal_to_pd_distance_bin_center'].values[0]
        cmap = plt.cm.twilight_shifted
        color = cmap((goal_to_pd_distance_bin_center + 180.) / 360.)
        ph.circplot(data['vh_pd_subtracted_bin_center'].values, data['mean_value'].values, circ='x', color=color,
                    period=circplot_period, lw=0.75, args={'clip_on': False}, zorder=1)
        plt.scatter(data['vh_pd_subtracted_bin_center'].values, data['mean_value'].values, s=0.25, color=color,
                    zorder=2,
                    clip_on=False,
                    )

        ax = plt.gca()
        e2 = ax.errorbar(x=data['vh_pd_subtracted_bin_center'].values, y=data['mean_value'].values,
                         yerr=data['sem'].values,
                         ls='none', elinewidth=0.5, ecolor=color, capsize=1.5, capthick=0.5, zorder=10)

    g.map_dataframe(plot)

    ph.despine_axes(g.axes, style=['bottom'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 0.5})

    for ax in g.axes.ravel():
        ax.set_ylim([-5, 85])
        ax.set_xticks([-180, 0, 180])
        ax.set_xlim([-180, 180])
        ax.set_xticklabels([])
        ax.axhline(y=0, ls=':', color='grey', lw=0.25, zorder=0, alpha=0.5)

    ax = g.axes[-1, 0]
    ax.set_xticklabels([-180, 0, 180])

    sb = ph.add_scalebar(g.axes[-1, 0], sizex=0, sizey=50, barwidth=axis_lw * 2, barcolor='k', loc=3,
                         pad=0,
                         bbox_transform=g.axes[-1, 0].transAxes,
                         bbox_to_anchor=[-0.1, 0.1]
                         )
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_all_cells_vm_modulation(event_df_sem, rec_name_to_cell_no,
                                 save=False,
                                 savepath=None,
                                 fname=None):
    # Cell nb order

    # LEFT AND RIGHT IN SAME PLOT
    event_df_sem['cell_no'] = event_df_sem['rec_name'].map(rec_name_to_cell_no)
    row_order = \
        event_df_sem.query('celltype=="PFL3"').drop_duplicates(['rec_name']).sort_values(['LAL_side', 'cell_no'])[
            'rec_name']
    print([rec_name_to_cell_no[rec] for rec in row_order])

    vh_to_pd_bins = np.linspace(-180, 180, 9)
    circplot_period = np.diff(vh_to_pd_bins)[0] * 2  # used to prevent drawing lines between non adjacent bins

    # could alternatively add NaNs to empty bins...

    # a bit funky
    def zero_vm(data):
        if data['variable'].values[0] == 'vm_sp_subtracted':
            data['zero_vm'] = data['mean_value'].values - np.nanmin(data['mean_value'].values)
        return data

    event_df_sem['zero_vm'] = np.nan
    event_df_sem = event_df_sem.groupby(['rec_name', 'variable']).apply(zero_vm)

    g = ph.FacetGrid(event_df_sem.query('celltype=="PFL3"& variable=="vm_sp_subtracted"'),
                     col='goal_to_pd_distance_bin_center',
                     row='rec_name',
                     #                unit='rec_name',
                     fig_kws={'figsize': [3, 4], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.2,
                                   'hspace': 0.2},
                     col_order=event_df_sem['goal_to_pd_distance_bin_center'].unique().sort_values(ascending=True),
                     row_order=row_order
                     )

    def plot(data):
        goal_to_pd_distance_bin_center = data['goal_to_pd_distance_bin_center'].values[0]
        cmap = plt.cm.twilight_shifted
        color = cmap((goal_to_pd_distance_bin_center + 180.) / 360.)

        ph.circplot(data['vh_pd_subtracted_bin_center'].values, data['zero_vm'].values, circ='x', color=color,
                    period=circplot_period, lw=0.75, args={'clip_on': False}, zorder=1)
        plt.scatter(data['vh_pd_subtracted_bin_center'].values, data['zero_vm'].values, s=0.25, color=color, zorder=2,
                    clip_on=False,
                    )
        ax = plt.gca()
        e2 = ax.errorbar(x=data['vh_pd_subtracted_bin_center'].values, y=data['zero_vm'].values,
                         yerr=data['sem'].values,
                         ls='none', elinewidth=0.5, ecolor=color, capsize=1.5, capthick=0.5, zorder=10)

    g.map_dataframe(plot)

    ph.despine_axes(g.axes, style=['bottom'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 0.5})

    for ax in g.axes.ravel():
        ax.set_ylim([0, 30])
        ax.set_xticks([-180, 0, 180])
        ax.set_xlim([-180, 180])
        ax.set_xticklabels([])
        #     ax.set_yticklabels([])
        ax.axhline(y=0, ls=':', color='grey', lw=0.25, zorder=0, alpha=0.5)

    # ph.adjust_spines(g.axes[-1,0],['left','bottom'],lw=axis_lw,ticklen=axis_ticklen,pad=0.5)
    ax = g.axes[-1, 0]
    ax.set_xticklabels([-180, 0, 180])

    sb = ph.add_scalebar(g.axes[-1, 0], sizex=0, sizey=10, barwidth=axis_lw * 2, barcolor='k', loc=3,
                         pad=0,
                         bbox_transform=g.axes[-1, 0].transAxes,
                         bbox_to_anchor=[-0.1, 0.1]
                         )

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_ctl_for_walking(modulation_df_ctl_walk, rec_name_to_cell_no,
                         save_figure=False,
                         save_figure_path=False,
                         figure_fname=None,
                         save_source_data=False,
                         save_source_data_path=False,
                         source_data_fname=None):
    vh_to_pd_bins = np.linspace(-180, 180, 9)

    print(len(modulation_df_ctl_walk.query('celltype=="PFL3" &(rec_name!="mean")')['rec_name'].unique()), 'flies')

    circplot_period = np.diff(vh_to_pd_bins)[0] * 2  # used to prevent drawing lines between non adjacent bins
    # could alternatively add NaNs to empty bins...
    plot_df = modulation_df_ctl_walk.query('celltype=="PFL3"').copy()

    g = ph.FacetGrid(plot_df,
                     col='goal_to_pd_distance_bin_center_flipped',
                     row='variable',
                     unit='rec_name',
                     fig_kws={'figsize': [3, 2], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.2,
                                   'hspace': 0.4},
                     col_order=np.sort(plot_df['goal_to_pd_distance_bin_center_flipped'].unique()),
                     row_order=[
                         'sp_rate', 'dforw_boxcar_average_0.5_s', 'dheading_boxcar_average_0.5_s']
                     )

    def plot(data):
        goal_to_pd_distance_bin_center = data['goal_to_pd_distance_bin_center_flipped'].values[0]
        cmap = plt.cm.twilight_shifted
        color = cmap((goal_to_pd_distance_bin_center + 180.) / 360.)

        if data['rec_name'].iloc[0] != 'mean':
            ph.circplot(data['vh_pd_subtracted_bin_center_flipped'].values, data['mean_value'].values, circ='x',
                        color=color,
                        period=circplot_period, lw=0.2, args={'clip_on': False})
            plt.scatter(data['vh_pd_subtracted_bin_center_flipped'].values, data['mean_value'].values, s=0.1,
                        color=color, zorder=2,
                        clip_on=False,
                        )

        else:
            # plot mean
            ph.circplot(data['vh_pd_subtracted_bin_center_flipped'].values, data['mean'].values, circ='x',
                        color=color, period=circplot_period, lw=1, args={'clip_on': False})

    g.map_dataframe(plot)

    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 2})

    for irow, row in enumerate(g.axes):
        for icol, ax in enumerate(row):
            if irow == 1:
                ax.set_ylim([-1, 2])
                ax.set_yticks([-1, 0, 1, 2])
            elif irow == 2:
                ax.set_ylim([-20, 20])
                ax.set_yticks([-20, 0, 20])
            elif irow == 0:
                ax.set_ylim([0, 70])
                ax.set_yticks([0, 70])
            ax.set_xlim([-180, 180])
            ax.set_xticks([-180, 0, 180])
            ax.set_xticklabels([])

    ax = g.axes[-1, 0]
    ax.set_xticklabels([-180, 0, 180])

    if save_figure:
        plt.savefig(save_figure_path + figure_fname,
                    transparent=True, bbox_inches='tight')
    if save_source_data:
        plot_df['cell_no'] = plot_df['rec_name'].map(rec_name_to_cell_no)
        plot_df.rename({'mean_value': 'value'}, axis=1, inplace=True)
        plot_df = plot_df.filter(['LAL_side', 'cell_no', 'variable', 'vh_pd_subtracted_bin_center_flipped',
                                  'goal_to_pd_distance_bin_center_flipped', 'value', 'mean'])
        plot_df.to_csv(save_source_data_path + source_data_fname)


def plot_sp_by_dforw(sp_by_dforw_df, save=False, savepath=None, fname=None):
    plt.figure(1, [1, 1], dpi=dpi)

    ax = plt.gca()
    x = np.linspace(-180, 180, 37)
    y = np.linspace(-1, 11, 11)
    vmin = sp_by_dforw_df.values.min()
    vmax = sp_by_dforw_df.values.max()
    cmap = sns.color_palette("rocket", as_cmap=True)
    # cmap=plt.cm.hot

    pc = ax.pcolormesh(
        x, y, sp_by_dforw_df.values,
        cmap=cmap,
        #                 edgecolors='grey',linewidth=0.1,
        clip_on=False,
        vmin=vmin, vmax=vmax)

    ax.set_xticks([-180, 0, 180])
    # ax.set_xlim([0,300])

    ph.adjust_spines(ax, ['left', 'bottom'], pad=1, lw=axis_lw, ticklen=axis_ticklen)

    if save:
        plt.savefig(savepath + fname + '.pdf',
                    transparent=True, bbox_inches='tight')

    plt.show()

    fig = plt.figure(1, [0.05, 0.25], dpi=dpi)

    cb = mpl.colorbar.ColorbarBase(plt.gca(), orientation='vertical',
                                   cmap=cmap
                                   )

    cb.ax.tick_params(size=2.5, width=0.4)
    cb.outline.set_visible(False)
    cb.ax.set_yticklabels([int(np.round(vmin)), int(np.round(vmax))])

    if save:
        plt.savefig(savepath + fname + '_colorbar.pdf',
                    transparent=True, bbox_inches='tight')


def plot_sp_by_dheading(sp_by_dheading_df, save=False, savepath=None, fname=None):
    plt.figure(1, [1, 1], dpi=dpi)

    ax = plt.gca()
    x = np.linspace(-180, 180, 37)
    y = np.linspace(-200, 200, 11)
    vmin = sp_by_dheading_df.values.min()
    vmax = sp_by_dheading_df.values.max()
    cmap = sns.color_palette("rocket", as_cmap=True)
    # cmap=plt.cm.hot

    pc = ax.pcolormesh(
        x, y, sp_by_dheading_df.values,
        cmap=cmap,
        #                 edgecolors='grey',linewidth=0.1,
        clip_on=False,
        vmin=vmin, vmax=vmax)

    ax.set_xticks([-180, 0, 180])
    # ax.set_xlim([0,300])

    ph.adjust_spines(ax, ['left', 'bottom'], pad=1, lw=axis_lw, ticklen=axis_ticklen)

    if save:
        plt.savefig(savepath + fname + '.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()

    fig = plt.figure(1, [0.05, 0.25], dpi=dpi)

    cb = mpl.colorbar.ColorbarBase(plt.gca(), orientation='vertical',
                                   cmap=cmap
                                   )

    cb.ax.tick_params(size=2.5, width=0.4)
    cb.outline.set_visible(False)
    cb.ax.set_yticklabels([int(np.round(vmin)), int(np.round(vmax))])

    if save:
        plt.savefig(savepath + fname + '_colorbar.pdf',
                    transparent=True, bbox_inches='tight')


def plot_sp_vs_vm(mean_spike_vs_vm_df, save=False, savepath=None, fname=None):
    g = ph.FacetGrid(mean_spike_vs_vm_df,
                     unit='goal_to_pd_distance_flipped_bin_center',
                     fig_kws={'figsize': [1, 1], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.4,
                                   'hspace': 0},
                     )

    def plot(data):
        goal_to_pd_distance_bin_center = data['goal_to_pd_distance_flipped_bin_center'].values[0]
        #     goal_bin_to_xy[goal_to_pd_distance_bin_center] = [data['vm_sp_subtracted_bin_center'].values.astype(float),
        #                                                      data['mean'].values]
        cmap = plt.cm.twilight_shifted
        color = cmap((goal_to_pd_distance_bin_center + 180.) / 360.)
        plt.plot(data['vm_sp_subtracted_bin_center'], data['mean'], color=color, lw=1, clip_on=False)
        plt.scatter(data['vm_sp_subtracted_bin_center'], data['mean'], color=color, s=5, clip_on=False)

    g.map_dataframe(plot)

    ax = g.axes[0, 0]
    ax.set_xlim([-90, -40])
    ax.set_xticks([-90, -80, -70, -60, -50, -40])

    ax.set_ylim([0, 40])
    ax.set_yticks([0, 20, 40])

    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 5})

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_vm_shifts(mean_spike_vs_vm_df, save=False, savepath=None, fnames=None):
    # From Larry #2023/08/16
    goals = np.array([
        -157.5000,
        -112.5000,
        -67.5000,
        -22.5000,
        22.5000,
        67.5000,
        112.5000,
        157.5000
    ])
    shifts = np.array([58.6751,
                       65.2687,
                       65.5067,
                       64.0321,
                       61.7986,
                       58.5442,
                       51.4630,
                       54.9146])

    goal_to_shfift = dict(zip(goals, shifts))

    G = np.linspace(-np.pi, np.pi, 100)
    amp = 6.4252
    offset = 60.0254
    phase = np.deg2rad(-56.7007)

    plt.figure(1, [1, 1], dpi=dpi)
    plt.plot(np.rad2deg(G), amp * np.cos(G - phase) + offset, clip_on=False, lw=1, color='grey')
    plt.scatter(goals, shifts, s=2, clip_on=False, color='k', zorder=3)
    plt.xlim([-180, 180])
    plt.xticks([-180, 0, 180])
    plt.ylim([50, 70])
    plt.yticks([50, 60, 70])
    ax = plt.gca()
    ph.adjust_spines(ax, ['left', 'bottom'], pad=2, lw=axis_lw, ticklen=axis_ticklen)

    if save:
        plt.savefig(savepath + fnames[0] + '.pdf',
                    transparent=True, bbox_inches='tight')

    plt.show()

    g = ph.FacetGrid(mean_spike_vs_vm_df,
                     #                col='goal_to_pd_distance_bin_center',
                     unit='goal_to_pd_distance_flipped_bin_center',
                     fig_kws={'figsize': [1, 1], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.4,
                                   'hspace': 0},
                     )

    def plot(data):
        goal_to_pd_distance_bin_center = data['goal_to_pd_distance_flipped_bin_center'].values[0]
        #     goal_bin_to_xy[goal_to_pd_distance_bin_center] = [data['vm_sp_subtracted_bin_center'].values.astype(float),
        #                                                      data['mean'].values]

        shift = goal_to_shfift[goal_to_pd_distance_bin_center]
        cmap = plt.cm.twilight_shifted
        color = cmap((goal_to_pd_distance_bin_center + 180.) / 360.)
        #     plt.plot(data['vm_sp_subtracted_bin_center']+shift,data['mean'],color=color,lw=1,clip_on=False)
        plt.scatter(data['vm_sp_subtracted_bin_center'].values.astype(float) + shift, data['mean'], color=color, s=1,
                    clip_on=False)

    g.map_dataframe(plot)
    ax = g.axes[0, 0]
    ax.set_xlim([-30, -20])
    ax.set_xticks([-20, 0, 20])
    ax.set_ylim([0, 40])
    ax.set_yticks([0, 20, 40])

    # #The black curve is a*log(1 + exp(b*VShifted)) with a = 6.3197 and b = 0.2878.
    Vm = np.linspace(-30, 20, 100)
    a = 6.3197
    b = 0.2878
    plt.plot(Vm, a * np.log(1 + np.exp(b * (Vm))), color='k', zorder=3, clip_on=False, lw=1)
    ph.despine_axes(g.axes, style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 5})
    if save:
        plt.savefig(savepath + fnames[1] + '.pdf',
                    transparent=True, bbox_inches='tight')

    plt.show()


def plot_paired_fixation_tuning_aligned(rec_name, abf_fixation_df, pd_df, example_goal_to_pd,
                                        color_A, color_B,
                                        dpi, axis_ticklen, axis_lw, vh_bins, goal_to_pd_bins):
    circplot_period = np.diff(vh_bins)[0] * 2  # used to prevent drawing lines between non adjacent bins

    fixation_data = abf_fixation_df.query(f'rec_name=="{rec_name}"').copy()

    fixation_data['goal_to_pd_distance_bin_center'] = pd.cut(fixation_data['goal_to_pd_distance'],
                                                             bins=goal_to_pd_bins).apply(lambda x: np.round(x.mid, 2))

    binned_df = ap.get_binned_df(fixation_data,
                                 bin_values={'vh_pd_subtracted': vh_bins,
                                             'goal_to_pd_distance': goal_to_pd_bins},
                                 labels=['sp_rate'],
                                 id_vars=['LAL_side', 'celltype', 'rec_name'],
                                 query='is_fixating==True & `dforw_boxcar_average_0.5_s`>1',
                                 )

    fig = plt.figure(1, [1.25, 0.75], dpi=dpi)
    gs = gridspec.GridSpec(figure=fig, nrows=2, ncols=2, wspace=0.5, hspace=0.4,
                           height_ratios=[0.3, 0.7], )

    # Hist
    ax = plt.subplot(gs[0, 0])
    data = fixation_data.query(
        f'goal_to_pd_distance_bin_center=={example_goal_to_pd[0]} & `dforw_boxcar_average_0.5_s`>1')
    plt.hist(data['vh_pd_subtracted'].values, bins=np.linspace(-180, 180, 37), density=True, color=color_A, alpha=1)
    ph.adjust_spines(ax, spines=['left'], lw=axis_lw, ticklen=axis_ticklen, pad=1)

    ax1 = plt.subplot(gs[0, 1], sharey=ax)
    data = fixation_data.query(
        f'goal_to_pd_distance_bin_center=={example_goal_to_pd[1]} & `dforw_boxcar_average_0.5_s`>1')
    plt.hist(data['vh_pd_subtracted'].values, bins=np.linspace(-180, 180, 37), density=True, color=color_B, alpha=1)
    ph.adjust_spines(ax1, spines=[], lw=axis_lw, ticklen=axis_ticklen, pad=1)
    plt.ylim([-0.001, 0.025])

    # tuning curves
    ax2 = plt.subplot(gs[1, 0])
    data = binned_df.query(f'goal_to_pd_distance_bin_center=={example_goal_to_pd[0]}')
    ph.circplot(data['vh_pd_subtracted_bin_center'].values, data['mean_value'].values, color=color_A, lw=0.5,
                circ='x', period=circplot_period)
    plt.scatter(data['vh_pd_subtracted_bin_center'].values, data['mean_value'].values, color=color_A, s=1,
                clip_on=False)

    pd_df_data = pd_df.query(f'rec_name=="{rec_name}" & variable=="sp_rate"').copy()
    pd_df_data.sort_values(by=['vh_pd_subtracted'], inplace=True)

    plt.plot(pd_df_data['vh_pd_subtracted'],
             pd_df_data['mean_value'], ls=':', lw=0.5, color='grey')

    ph.adjust_spines(ax2, spines=['left', 'bottom'], lw=axis_lw, ticklen=axis_ticklen, pad=2)

    ax3 = plt.subplot(gs[1, 1], sharex=ax2, sharey=ax2)
    data = binned_df.query(f'goal_to_pd_distance_bin_center=={example_goal_to_pd[1]}')
    ph.circplot(data['vh_pd_subtracted_bin_center'].values, data['mean_value'].values, color=color_B, lw=0.5,
                circ='x', period=circplot_period)
    plt.scatter(data['vh_pd_subtracted_bin_center'].values, data['mean_value'].values, color=color_B, s=1,
                clip_on=False)
    plt.plot(pd_df_data['vh_pd_subtracted'],
             pd_df_data['mean_value'], ls=':', lw=0.5, color='grey')

    ph.adjust_spines(ax3, spines=['bottom'], lw=axis_lw, ticklen=axis_ticklen, pad=2)

    plt.ylim([0, 80])
    plt.yticks([0, 80])
    plt.xlim([-180, 180])
    plt.xticks([-180, 0, 180])


# ------------------ Save processed data ----------------------- #
def save_processed_data(PROCESSED_DATA_PATH, recs):
    def save_as_hd5f(data):
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
            'vm_sp_subtracted',
            'sp_rate',
        ]).reset_index(drop=True)
        df.to_hdf(PROCESSED_DATA_PATH + genotype + os.path.sep + rec_name + '.h5',
                  key='df', mode='w')

        df_orig = recs[rec_name].abf.df_orig.copy()
        df_orig.filter(['t', 'patch_1', 'sp']).rename({'patch_1': 'vm'}, axis=1).to_hdf(
            PROCESSED_DATA_PATH + genotype + os.path.sep + rec_name + '_orig.h5',
            key='df_orig', mode='w')

    recs.merged_abf_df.groupby(['rec_name']).apply(save_as_hd5f)
    summary_recs = recs.merged_abf_df.drop_duplicates('rec_name').copy()
    summary_recs.filter(['rec_name', 'celltype', 'LAL_side']).rename({'LAL_side': 'side'}, axis=1).reset_index(
        drop=True).to_csv(PROCESSED_DATA_PATH + 'summary.csv')
