import neuprint
from neuprint import fetch_neurons, fetch_adjacencies, NeuronCriteria
from neuprint import fetch_synapse_connections, SynapseCriteria
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import plotting_help as ph

# ------------------ Plotting parameters ----------------------- #
dpi = 300
axis_lw = 0.4
axis_ticklen = 2.5
font = {'family': 'arial',
        'weight': 'normal',
        'size': 5}
mpl.rc('font', **font)

glom_sorter_delta7 = ['L8R1R9', 'L7R2', 'L7R3', 'L6R3', 'L6R4', 'L5R4', 'L4R5', 'L4R6', 'L3R6', 'L2R7', 'L1L9R8']
glom_sorter_PFL3 = ['L7', 'L6', 'L5', 'L4', 'L3/L4', 'L3', 'L2', 'L1', 'R1', 'R2', 'R3', 'R3/R4', 'R4', 'R5', 'R6',
                    'R7']

PFL3_bodyId_to_glom = {
    1004700437: 'L7',
    941939879: 'L6',
    911569552: 'L5',
    880875927: 'L5',
    789130596: 'L4',
    850855138: 'L3/L4',
    912925080: 'L3',
    757694775: 'L2',
    1258686925: 'L2',
    1258073453: 'L1',
    851493896: 'L1',
    944262351: 'L1',
    787374226: 'R1',
    1134253374: 'R1',
    666308023: 'R1',
    1097718659: 'R2',
    666994301: 'R2',
    912488890: 'R3',
    1008028537: 'R3/R4',
    942172835: 'R4',
    910447181: 'R5',
    911134017: 'R5',
    941132430: 'R6',
    1200032115: 'R7',
}

PFL3_bodyId_to_col = {
    1004700437: 'C1',
    941939879: 'C2',
    911569552: 'C3',
    880875927: 'C4',
    789130596: 'C5',
    850855138: 'C6',
    912925080: 'C7',
    757694775: 'C1',
    1258686925: 'C8',
    1258073453: 'C2',
    851493896: 'C9',
    944262351: 'C10',
    787374226: 'C3',
    1134253374: 'C4',
    666308023: 'C11',
    1097718659: 'C5',
    666994301: 'C12',
    912488890: 'C6',
    1008028537: 'C7',
    942172835: 'C8',
    910447181: 'C9',
    911134017: 'C10',
    941132430: 'C11',
    1200032115: 'C12'

}

PFL3_LAL_dict = {
    1004700437: 'R',
    941939879: 'R',
    911569552: 'R',
    880875927: 'R',
    789130596: 'R',
    850855138: 'R',
    912925080: 'R',
    757694775: 'L',
    1258686925: 'R',
    1258073453: 'L',
    851493896: 'R',
    944262351: 'R',
    787374226: 'L',
    1134253374: 'L',
    666308023: 'R',
    1097718659: 'L',
    666994301: 'R',
    912488890: 'L',
    1008028537: 'L',
    942172835: 'L',
    910447181: 'L',
    911134017: 'L',
    941132430: 'L',
    1200032115: 'L'

}
col_sorter = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
side_sorter = ['L', 'R']


# ------------------ Get data ----------------------- #

def get_PFL3_PB_input(c):
    criteria = NeuronCriteria(type='PFL3', regex=True)
    neurons_df, roi_counts_df = fetch_neurons(criteria, client=c)
    post_body_ids = neurons_df['bodyId'].unique()
    neurons_df, roi_conn_df = fetch_adjacencies(targets=post_body_ids, client=c, rois='PB')
    PFL3_PB_input_df = neuprint.utils.merge_neuron_properties(neurons_df, roi_conn_df, properties=['type', 'instance'])
    PFL3_PB_input_sum_df = pd.DataFrame(PFL3_PB_input_df.groupby(['type_pre'])['weight'].sum()).reset_index()
    print('Delta7 ' + str(
        PFL3_PB_input_sum_df.query('type_pre=="Delta7"')['weight'].values[0] / PFL3_PB_input_sum_df['weight'].sum()))
    print('EPG ' + str(
        PFL3_PB_input_sum_df.query('type_pre=="EPG"')['weight'].values[0] / PFL3_PB_input_sum_df['weight'].sum()))


def get_D7_to_PFL3(c):
    criteria = NeuronCriteria(type='Delta7', regex=True)
    neurons_df, roi_counts_df = fetch_neurons(criteria, client=c)
    pre_body_ids = neurons_df['bodyId'].unique()
    criteria = NeuronCriteria(type='PFL3', regex=True)
    neurons_df, roi_counts_df = fetch_neurons(criteria, client=c)
    post_body_ids = neurons_df['bodyId'].unique()
    neurons_df, roi_conn_df = fetch_adjacencies(sources=pre_body_ids, targets=post_body_ids, client=c, rois='PB')
    D7_to_PFL3_df = neuprint.utils.merge_neuron_properties(neurons_df, roi_conn_df, properties=['type', 'instance'])

    def get_glom_Delta7(s):
        l = s.split('_')
        glom = l[1]
        return glom

    D7_to_PFL3_df['D7_glom'] = D7_to_PFL3_df['instance_pre'].apply(lambda x: get_glom_Delta7(x))
    D7_to_PFL3_df['PFL3_col'] = D7_to_PFL3_df['bodyId_post'].map(PFL3_bodyId_to_col)
    D7_to_PFL3_df['PFL3_glom'] = D7_to_PFL3_df['bodyId_post'].map(PFL3_bodyId_to_glom)
    D7_to_PFL3_df['PFL3_id'] = D7_to_PFL3_df['PFL3_glom'] + '_' + D7_to_PFL3_df['PFL3_col']

    # temp_df =pd.DataFrame(D7_to_PFL3_df.groupby(['D7_glom','PFL3_id'])['weight'].sum()).reset_index()
    # pd.pivot_table(temp_df,values='weight',index='D7_glom',columns='PFL3_id').idxmax(axis=0)

    D7_to_PFL3_df_matrix_df = D7_to_PFL3_df.pivot_table(
        index=[
            'D7_glom',
            'bodyId_pre',

        ],
        columns=[
            'PFL3_glom',
            'bodyId_post',

        ],
        values='weight')
    D7_to_PFL3_df_matrix_df = D7_to_PFL3_df_matrix_df.reindex(index=glom_sorter_delta7, level='D7_glom').reindex(
        columns=glom_sorter_PFL3, level='PFL3_glom').fillna(0)

    return D7_to_PFL3_df_matrix_df


def get_PFL3_FB_input(c):
    criteria = NeuronCriteria(type='PFL3', regex=True)
    neurons_df, roi_counts_df = fetch_neurons(criteria, client=c)
    post_body_ids = neurons_df['bodyId'].unique()

    neurons_df, roi_conn_df = fetch_adjacencies(targets=post_body_ids, client=c, rois='FB')
    df = neuprint.utils.merge_neuron_properties(neurons_df, roi_conn_df, properties=['type', 'instance'])
    bar_df = df.groupby(['type_pre', 'type_post', 'roi'], as_index=False).weight.sum()
    bar_df = bar_df.sort_values('weight', ascending=False).reset_index(drop=True)

    def get_type(x):
        if x[0:2] == "FB":
            return 0
        elif x == "ExR5":
            return 0
        else:
            return 1

    bar_df['type'] = bar_df['type_pre'].apply(get_type)

    total_columnar = bar_df.query('type==1')['weight'].sum()

    total_FC = bar_df.query('type_pre=="FC2A"|type_pre=="FC2B"|type_pre=="FC2C"')['weight'].sum()
    print('Fraction FC2 ' + str(total_FC / total_columnar))

    total_synapses = bar_df['weight'].sum()

    bar_df = bar_df.head(50)
    show_synapses = bar_df['weight'].sum()
    print('Fraction plotted ' + str(show_synapses / total_synapses))

    bar_df = bar_df.sort_values(['type', 'weight'], ascending=False).reset_index(drop=True)

    return bar_df


def get_FC2_to_PFL3(c):
    criteria = NeuronCriteria(type='FC2.*', regex=True)
    neurons_df, roi_counts_df = fetch_neurons(criteria, client=c)
    pre_body_ids = neurons_df['bodyId'].unique()
    criteria = NeuronCriteria(type='PFL3', regex=True)
    neurons_df, roi_counts_df = fetch_neurons(criteria, client=c)
    post_body_ids = neurons_df['bodyId'].unique()
    neurons_df, roi_conn_df = fetch_adjacencies(sources=pre_body_ids, targets=post_body_ids, client=c, rois='FB')
    df = neuprint.utils.merge_neuron_properties(neurons_df, roi_conn_df, properties=['type', 'instance'])
    df['post_col'] = df['bodyId_post'].map(PFL3_bodyId_to_col)
    df['post_side'] = df['bodyId_post'].map(PFL3_LAL_dict)

    synapses = fetch_synapse_connections(source_criteria='FC2.*', target_criteria='PFL3',
                                         synapse_criteria=SynapseCriteria(rois=['FB'], primary_only=True))
    synpase_pos = synapses.groupby(['bodyId_pre'])['x_pre'].mean()
    synpase_pos = pd.DataFrame(synpase_pos).reset_index()
    FC2_bodyId_order = synpase_pos.sort_values('x_pre')['bodyId_pre']

    def get_col(s):
        l = s.split('_')
        col = [i for i in l if i[0] == 'C']
        if col:
            col = col[0]
        else:
            col = np.nan
        return col

    df['pre_col'] = df['instance_pre'].apply(lambda x: get_col(x))

    matrix_df = df.pivot_table(
        index=[
            'type_pre',
            'pre_col',
            'bodyId_pre',
        ],
        columns=[
            'post_col',
            'post_side',
            'bodyId_post',
            #                    'instance_post'
        ],
        values='weight').reindex(columns=col_sorter, level='post_col').reindex(columns=side_sorter,
                                                                               level='post_side').reindex(
        index=['FC2C', 'FC2B', 'FC2A'], level='type_pre').fillna(0).reindex(index=FC2_bodyId_order,
                                                                            level='bodyId_pre').fillna(0)

    return matrix_df


def get_FC2_to_PFL3_pw_corr(c):
    criteria = NeuronCriteria(type='FC2.*', regex=True)
    neurons_df, roi_counts_df = fetch_neurons(criteria, client=c)
    pre_body_ids = neurons_df['bodyId'].unique()
    criteria = NeuronCriteria(type='PFL3', regex=True)
    neurons_df, roi_counts_df = fetch_neurons(criteria, client=c)
    post_body_ids = neurons_df['bodyId'].unique()
    neurons_df, roi_conn_df = fetch_adjacencies(sources=pre_body_ids, targets=post_body_ids, client=c, rois='FB')
    df = neuprint.utils.merge_neuron_properties(neurons_df, roi_conn_df, properties=['type', 'instance'])
    df['post_col'] = df['bodyId_post'].map(PFL3_bodyId_to_col)
    df['post_side'] = df['bodyId_post'].map(PFL3_LAL_dict)

    def get_col(s):
        l = s.split('_')
        col = [i for i in l if i[0] == 'C']
        if col:
            col = col[0]
        else:
            col = np.nan
        return col

    df['pre_col'] = df['instance_pre'].apply(lambda x: get_col(x))

    matrix_df = df.pivot_table(
        index=[
            'pre_col',
            'bodyId_pre',
        ],
        columns=[
            'post_col',
            'post_side',
            'bodyId_post',
            #                    'instance_post'
        ],
        values='weight').reindex(columns=col_sorter, level='post_col').reindex(columns=side_sorter,
                                                                               level='post_side').fillna(0)

    corr_df = matrix_df.corr()
    return corr_df


# ------------------ Plotting functions ----------------------- #

def plot_D7_to_PFL3(D7_to_PFL3_df_matrix_df, save=False, savepath=None, fname=None):
    plt.figure(1, [3, 3], dpi=dpi)

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 3}
    mpl.rc('font', **font)

    ax = plt.gca()
    ax.set_aspect('equal')
    vmax = D7_to_PFL3_df_matrix_df.values.max()
    vmin = D7_to_PFL3_df_matrix_df.values.min()
    pc = ax.pcolormesh(D7_to_PFL3_df_matrix_df.values,
                       cmap=plt.cm.Greys, edgecolors='grey', linewidth=0.1, vmin=vmin, vmax=vmax)
    columns = D7_to_PFL3_df_matrix_df.columns.get_level_values(0)
    ax.set_xticks(np.arange(len(columns)) + 0.5)
    ax.set_xticklabels(columns)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ph.adjust_spines(ax, ['left', 'top', 'right', 'bottom'], pad=0, lw=0.2, ticklen=0)
    # ax.tick_params('both', length=axis_ticklen, width=axis_lw, which='major')
    plt.xticks(rotation=90)
    rows = D7_to_PFL3_df_matrix_df.index.get_level_values(0)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows)

    if save:
        plt.savefig(savepath + fname + '.pdf', transparent=True, bbox_inches='tight')
    plt.show()

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 5}
    mpl.rc('font', **font)

    fig = plt.figure(1, [0.05, 0.25], dpi=dpi)
    cb = plt.colorbar(pc, cax=plt.gca(), orientation='vertical', )
    cb.ax.tick_params(size=2.5, width=0.4)
    cb.outline.set_visible(False)
    if save:
        plt.savefig(savepath + fname + '_colorbar.pdf', transparent=True, bbox_inches='tight')
    plt.show()


def plot_top_FB_inputs(bar_df, save=False, savepath=None, fname=None):
    colors = np.zeros(len(bar_df['type_pre'])).astype('str')
    colors[:] = '#c2c2c2'
    color_dict = dict(zip(bar_df['type_pre'], colors))
    color_dict['FC2A'] = '#803399'
    color_dict['FC2B'] = '#803399'
    color_dict['FC2C'] = '#803399'
    plt.figure(1, [0.5, 6], dpi=dpi)
    ax = sns.barplot(x='weight', y='type_pre', data=bar_df.head(50),
                     ci=None, orient='h', palette=color_dict)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ph.despine_axes([ax], style=['left', 'top'], adjust_spines_kws={'pad': 2,
                                                                    'lw': axis_lw,
                                                                    'ticklen': axis_ticklen}
                    )

    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_FC2_to_PFL3(matrix_df, save=False, savepath=None, fname=None):
    fig = plt.figure(1, [1.5, 5.5], dpi=dpi)

    ax = plt.gca()
    ax.set_aspect('equal')
    vmax = matrix_df.values.max()
    vmin = matrix_df.values.min()

    pc = ax.pcolormesh(matrix_df.values,
                       cmap=plt.cm.Purples, edgecolors='grey', linewidth=0.1, vmin=vmin, vmax=vmax)

    columns = matrix_df.columns.get_level_values(1)
    ax.set_xticks(np.arange(len(columns)) + 0.5)
    ax.set_xticklabels(columns)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ph.adjust_spines(ax, ['left', 'top', 'right', 'bottom'], pad=0, lw=axis_lw, ticklen=2)

    plt.xticks(rotation=0)
    ax.set_yticks([])
    ax.set_yticklabels([])

    if save:
        plt.savefig(savepath + fname + '.pdf',
                    transparent=True, bbox_inches='tight')

    plt.show()

    fig = plt.figure(1, [0.05, 0.25], dpi=dpi)

    cb = plt.colorbar(pc, cax=plt.gca(), orientation='vertical', )
    cb.ax.tick_params(size=2.5, width=0.4)
    cb.outline.set_visible(False)
    if save:
        plt.savefig(savepath + fname + '_colobar.pdf',
                    transparent=True, bbox_inches='tight')


def plot_corr_matrix(corr_df, save=False, savepath=None, fname=None):
    plt.figure(1, [2, 2], dpi=dpi)
    ax = plt.gca()
    ax.set_aspect('equal')
    vmax = corr_df.values.max()
    vmin = corr_df.values.min()
    pc = ax.pcolormesh(corr_df.values,
                       cmap="rocket",
                       #                    edgecolors='grey',linewidth=0.1,
                       vmin=vmin, vmax=vmax)
    columns = corr_df.columns.get_level_values(1)
    ax.set_xticks(np.arange(len(columns)) + 0.5)
    ax.set_xticklabels(columns)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.invert_yaxis()
    ph.adjust_spines(ax, ['left', 'top', 'right', 'bottom'], pad=0, lw=axis_lw, ticklen=axis_ticklen)
    rows = corr_df.index.get_level_values(1)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows)

    if save:
        plt.savefig(savepath + fname + '.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()

    fig = plt.figure(1, [0.05, 0.25], dpi=dpi)
    cb = plt.colorbar(pc, cax=plt.gca(), orientation='vertical', )
    cb.ax.tick_params(size=2.5, width=0.4)
    cb.outline.set_visible(False)
    if save:
        plt.savefig(savepath + fname + '_colorbar.pdf',
                    transparent=True, bbox_inches='tight')
