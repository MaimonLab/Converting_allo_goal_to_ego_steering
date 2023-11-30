"""model.py

Functions for Model.ipynb

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import pandas as pd

import functions as fc
import plotting_help as ph

dpi = 300
axis_lw = 0.4
axis_ticklen = 2.5

font = {'family': 'arial',
        'weight': 'normal',
        'size': 5}
mpl.rc('font', **font)

l_color = '#1c75bc'
r_color = '#b51700'
heading_color = '#6d6e71'

# TODO check with Larry model params
# Define PB and FB angles

thetaGH = np.arange(-180, 180 + 0.5, 0.5)

# same numbers as in Methods, but shifted by a 180º offset (each fly has an idiosyncratic bump
# to heading offset). We do this because these plots we want 0º to signify that the goal bump
# is in the middle of the FB
thetaPB = np.array([337.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5,
                    22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 22.5])
thetaPB = -1*fc.wrap(thetaPB + 180)

thetaFB = np.arange(0, 330 + 30, 30) + 15
thetaFB = -1*fc.wrap(thetaFB + 180)

nFB = len(thetaFB)
thetaRight = np.array([thetaPB[2], thetaPB[3], thetaPB[4], thetaPB[4],
                       thetaPB[5], thetaPB[6], thetaPB[6], thetaPB[7],
                       thetaPB[8], thetaPB[8], thetaPB[9], thetaPB[10]])
thetaLeft = np.array([thetaPB[7], thetaPB[8], thetaPB[9], thetaPB[9],
                      thetaPB[10], thetaPB[11], thetaPB[11], thetaPB[12],
                      thetaPB[13], thetaPB[13], thetaPB[14], thetaPB[15]])

# Parameters from data fit
a = 29.2282
b = 2.1736
c = -0.7011
d = 0.6299

# thetaGH is a range of heading and goal angles
# iGZero is the index of thetaGH = 0
iGZero = np.where(thetaGH == 0)[0][0]


def get_activity():
    # Left and Right PFL3 rates
    #   indexed by (heading,goal,column)
    rML = np.zeros([len(thetaGH), len(thetaGH), nFB])
    rMR = np.zeros([len(thetaGH), len(thetaGH), nFB])

    for i in range(nFB):
        for j in range(len(thetaGH)):
            gSig = d * np.cos(np.pi * (thetaGH[j] - thetaFB[i]) / 180)
            hSig = np.cos(np.pi * (thetaGH - thetaLeft[i]) / 180)
            rML[:, j, i] = a * np.log(1 + np.exp(b * (hSig + gSig + c)))
            hSig = np.cos(np.pi * (thetaGH - thetaRight[i]) / 180)
            rMR[:, j, i] = a * np.log(1 + np.exp(b * (hSig + gSig + c)))

    # Left and right LAL outputs and turning signal
    rLLAL = np.sum(rML, 2)
    rRLAL = np.sum(rMR, 2)
    rTurn = rRLAL - rLLAL

    return rML, rMR, rLLAL, rRLAL, rTurn


def plot_pb(H_deg, save=False, savepath=None, fname=None):
    plt.figure(1, [1.8471, 0.1], dpi=dpi)
    plt.bar(np.arange(0, 14), np.cos(np.pi * (H_deg-thetaPB[2:-2]) / 180) + 1, color=heading_color)
    plt.axis('off')
    if save:
        plt.savefig(savepath + fname + f'_pb_{H_deg}.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()


def plot_goal(G_deg, save=False, savepath=None, fname=None):
    plt.figure(1, [1.5119, 0.1], dpi=dpi)
    plt.bar(np.arange(0, 12), np.cos(np.pi * (G_deg - thetaFB) / 180) + 1, color='#c467ef')

    plt.axis('off')
    if save:
        plt.savefig(savepath + fname + f'_goal_{G_deg}.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()

def plot_PFL3_activity(H_deg, G_deg, rMR, rML, save=False, savepath=None, fname=None):
    iH = np.where(thetaGH == H_deg)[0][0]
    iGZero = np.where(thetaGH == G_deg)[0][0]
    R_PFL3 = rMR[iH, iGZero, :]
    R_PFL3 = pd.DataFrame({'fb_col': np.arange(0, 12), 'f': R_PFL3})
    R_PFL3['side'] = 'R'
    L_PFL3 = rML[iH, iGZero, :]
    L_PFL3 = pd.DataFrame({'fb_col': np.arange(0, 12), 'f': L_PFL3})
    L_PFL3['side'] = 'L'

    PFL3_df = pd.concat([R_PFL3, L_PFL3])

    g = ph.FacetGrid(PFL3_df,
                     col='fb_col',
                     unit='side',
                     fig_kws={'figsize': [1.7, 0.2], 'dpi': dpi},
                     gridspec_kws={'wspace': 0.5}

                     )

    def plot(data):
        side = data['side'].values[0]
        shift = 1
        if side == "L":
            color = l_color
        else:
            color = r_color
            shift = 0

        plt.bar(data['fb_col'] + shift, data['f'], clip_on=False, color=color)

    g.map_dataframe(plot)

    for ax in g.axes.ravel():
        ax.set_ylim([0, 80])
        ax.axis('off')

    if save:
        plt.savefig(savepath + fname + f'_col_activity_goal_H_{H_deg}_G_{G_deg}.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()
    sum_df = PFL3_df.groupby(['side'])['f'].sum()
    plt.figure(1, [0.0994, 0.1452], dpi=dpi)
    plt.bar([0, 1], sum_df.values, color=[l_color, r_color])
    plt.ylim([0, 300])
    plt.axis('off')
    if save:
        plt.savefig(savepath + fname + f'_lal_sum_activity_H_{H_deg}_G_{G_deg}.pdf',
                    transparent=True, bbox_inches='tight')
    plt.show()

def plot_lal_turning_signal(csv_path, save=False, savepath=None, fname=None):
    right_LAL = pd.read_csv(csv_path + 'RightLAL.csv', names=['right_LAL']).values.flatten()
    left_LAL = pd.read_csv(csv_path + 'LeftLAL.csv', names=['left_LAL']).values.flatten()
    rml_LAL = pd.read_csv(csv_path + 'Turning.csv', names=['Turning']).values.flatten()
    x = np.linspace(-180, 180, 360)

    fig = plt.figure(1, [3.5, 0.75], dpi=dpi)

    gs = gridspec.GridSpec(figure=fig, nrows=1, ncols=3,
                           wspace=0.5,
                           hspace=0)

    ax = plt.subplot(gs[0, 0])
    ax.plot(x, left_LAL, clip_on=False, color=l_color, lw=1)
    ax.set_ylim([-75, 75])
    ax.set_yticks([])

    ax.set_xlim([-180, 180])
    ax.set_xticks([-180, 0, 180])
    ax.set_xticklabels([])
    ax.axvline(x=0, ls=':', color='k', lw=0.5)

    ph.adjust_spines(ax, ['left', 'bottom'], pad=2, lw=axis_lw, ticklen=axis_ticklen)

    ax1 = plt.subplot(gs[0, 2])
    ax1.plot(x, right_LAL, clip_on=False, color=r_color, lw=1)
    ax1.set_ylim([-75, 75])
    ax1.set_yticks([])
    ax1.set_xlim([-180, 180])
    ax1.set_xticks([-180, 0, 180])
    ax1.set_xticklabels([])
    ax1.axvline(x=0, ls=':', color='k', lw=0.5)
    ph.adjust_spines(ax1, ['left', 'bottom'], pad=2, lw=axis_lw, ticklen=axis_ticklen)

    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(x, rml_LAL, clip_on=False, color='k', lw=1)
    ax2.axhline(y=0, ls=':', color='k', lw=0.5)
    ax2.axvline(x=0, ls=':', color='k', lw=0.5)

    # ax2.set_ylim([-150,150])
    ax2.set_yticks([0])
    ax2.set_xlim([-180, 180])
    ax2.set_xticks([-180, 0, 180])
    ax2.set_xticklabels([])
    ph.adjust_spines(ax2, ['left', 'bottom'], pad=2, lw=axis_lw, ticklen=axis_ticklen)
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')


def plot_model_error(save=False, savepath=None, fname=None):
    plt.figure(1, [0.75, 0.75], dpi=dpi)
    ax = plt.gca()
    x = np.linspace(-180, 180, 360)
    y = -0.085 * np.sin(4 * np.pi * x / 180)
    ax.plot(x, y, color='k', lw=0.5)
    ax.set_xlim([-180, 180])
    ax.set_xticks([-180, 0, 180])
    ax.set_ylim([-0.1, 0.1])
    ax.set_yticks([-0.1, 0, 0.1])
    ph.despine_axes([ax], style=['bottom', 'left'],
                    adjust_spines_kws={'lw': axis_lw, 'ticklen': axis_ticklen, 'pad': 5})
    if save:
        plt.savefig(savepath + fname,
                    transparent=True, bbox_inches='tight')
