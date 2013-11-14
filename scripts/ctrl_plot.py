from glob import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np

from . import base, files


## Figure 9

def signals(pklf, tr):
    d = base.load_pickle(pklf)
    if not d.has_key(tr + '_sig'):
        return
    sd = d[tr + '_sig']

    plt.figure(1)
    for ix in range(2):
        plt.plot(sd['signals'][ix][1],
                 sd['signals'][ix][2] - ix * 0.5,
                 label=sd['signals'][ix][0])

    plt.yticks(())
    plt.xticks(sd['s_ticks'], sd['s_labels'])

    if 'simple' in pklf and tr == 'c':
        leg = plt.legend(loc='upper right', prop={'size':10}, frameon=True)
        leg.get_frame().set_edgecolor('none')
        plt.title("Cue-responding correct trial", fontsize='medium')
    elif 'simple' in pklf:
        plt.title("Cue-responding late trial", fontsize='medium')
    elif 'adaptive' in pklf and tr == 'c':
        plt.title("Adaptive control correct trial", fontsize='medium')
    elif 'adaptive' in pklf:
        plt.title("Adaptive control premature trial", fontsize='medium')

    if 'adaptive' in pklf:
        plt.gca().xaxis.get_major_ticks()[-2].set_pad(16)

    plt.grid()
    plt.gca().xaxis.set_ticks_position('none')

    for side in ['left', 'top', 'right', 'bottom']:
        plt.gca().spines[side].set_visible(False)

    tight = {'rect': [0.05, -0.02, 0.95, 1.02]}
    base.save_or_show(plt.gcf(), pklf, tr+'_signals', fs=(3, 1.5), tight=tight)


def traj(pklf, tr):
    d = base.load_pickle(pklf)
    if not d.has_key(tr + '_sig'):
        return
    sd = d[tr + '_sig']

    plt.figure(1)
    plt.grid()
    plt.axhline(0, color='0.2')
    plt.axvline(0, color='0.2')
    x, y = sd['signals'][2][2], sd['signals'][3][2]
    plt.plot(y, x, lw=2)
    plt.plot(y[0], x[0], 'ow', ms=6, label='$t_s$')
    plt.plot(y[-1], x[-1], 'ok', ms=6, label='$t_{ITI}$')
    plt.xlabel("$x_2$")
    plt.ylabel("$x_1$")
    leg = plt.legend(handlelength=0.8, frameon=True, loc='best')
    leg.get_frame().set_edgecolor('none')
    edge = 1.6
    plt.xlim(-edge, edge)
    plt.ylim(-edge, edge)
    plt.gca().xaxis.set_ticklabels(())
    plt.gca().yaxis.set_ticklabels(())
    for side in ['top', 'right', 'bottom', 'left']:
        plt.gca().spines[side].set_visible(False)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')

    base.save_or_show(plt.gcf(), pklf, tr + '_traj', fs=(3, 3))


def fig9():
    sig_w, sig_h = 3 * 72, 1.5 * 72
    traj_w = traj_h = 3 * 72

    simple = 'plots/ctrl-simple-93_c_signals.svg'
    late = 'plots/ctrl-simple-857_l_signals.svg'
    adapt = 'plots/ctrl-adaptive-35_c_signals.svg'
    pre = 'plots/ctrl-adaptive-35_p_signals.svg'
    cc = 'plots/ctrl-adaptive-35_c_traj.svg'
    ec = 'plots/ctrl-adaptive-39_c_traj.svg'

    fig = base.svgfig(sig_w + traj_w, traj_h * 2)
    fig.append(base.el('A', simple, 0, 0, offset=(0, 30)))
    fig.append(base.el(None, late, 0, sig_h))
    fig.append(base.el(None, adapt, 0, sig_h * 2))
    fig.append(base.el(None, pre, 0, sig_h * 3))
    fig.append(base.el('B', cc, sig_w, 0, offset=(-2, 30)))
    fig.append(base.el('C', ec, sig_w, traj_h, offset=(-2, 30)))
    base.savefig(fig, files.pdf('fig9'))


## Figure 10

def performance(pklf):
    d = base.load_pickle(pklf)
    cor, prem, late = d['correct'], d['premature'], d['late']

    plt.figure(1)
    plt.subplot(1, 1, 1)
    left = np.arange(0.25, len(cor) - 0.25)
    plt.plot([0.5, 0.5], [1, 1], lw=2, color='k', label='Correct')
    if not 'simple' in pklf:
        plt.plot([0.5, 0.5], [1, 1], lw=2, color='0.4', label='Premature')
    if not 'adaptive' in pklf:
        plt.plot([0.5, 0.5], [1, 1], lw=2, color='0.8', label='Late')
    plt.legend(loc='upper left', prop={'size':10}, bbox_to_anchor=(0,0,1,1.05))

    plt.bar(left, late + prem + cor, width=0.5, color='k')
    plt.bar(left, late + prem, width=0.5, color='0.4')
    plt.bar(left, late, width=0.5, color='0.8')

    plt.axis((0, len(cor), 0, 250))
    plt.xticks(left + 0.25, np.arange(len(cor)) + 1)
    if 'expt' in pklf:
        plt.xlabel("Experimental subject")
    elif 'simple' in pklf:
        plt.xlabel("Cue-responding model")
    elif 'adaptive' in pklf:
        plt.xlabel("Adaptive control model")

    plt.gca().spines['left'].set_visible('expt' in pklf)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()

    if 'expt' in pklf:
        plt.ylabel("Trials")
    else:
        plt.yticks(())

    fd = 3.5 + 0.2 * (1 if 'expt' in pklf else -1)
    base.save_or_show(plt.gcf(), pklf, 'perf', fs=(fd, 3.5))


def reactiontimes(pklf):
    d = base.load_pickle(pklf)
    rts, rtstats = d['rts'], d['rtstats']

    plt.figure(1)
    plt.subplot(1, 1, 1)

    plt.axhline(rtstats[0], color='k', lw=1.5, ls=':')  # mean
    lines = plt.boxplot(rts, sym='')
    for line in lines.values():
        plt.setp(line, color='black')
    plt.ylim((0, 0.6))

    plt.gca().spines['left'].set_visible('expt' in pklf)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()

    if 'expt' in pklf:
        plt.ylabel("Reaction time (s)")
    else:
        plt.yticks(())

    if 'expt' in pklf:
        plt.xlabel("Experimental subject")
    elif 'simple' in pklf:
        plt.xlabel("Cue-responding model")
    elif 'adaptive' in pklf:
        plt.xlabel("Adaptive control model")

    fd = 3.5 + 0.2 * (1 if 'expt' in pklf else -1)
    base.save_or_show(plt.gcf(), pklf, 'rts', fs=(fd, 3.5))


def fig10():
    l_w, o_w = (3.5 + 0.2) * 72, (3.5 - 0.2) * 72
    h = 3.5 * 72

    fig = base.svgfig(l_w + o_w * 2, h * 2)
    fig.append(base.el(None, 'plots/expt_perf.svg', 0, 0))
    fig.append(base.el(None, 'plots/ctrl-simple_perf.svg', l_w, 0))
    fig.append(base.el(None, 'plots/ctrl-adaptive_perf.svg', l_w + o_w, 0))
    fig.append(base.el(None, 'plots/expt_rts.svg', 0, h))
    fig.append(base.el(None, 'plots/ctrl-simple_rts.svg', l_w, h))
    fig.append(base.el(None, 'plots/ctrl-adaptive_rts.svg', l_w + o_w, h))
    base.savefig(fig, files.pdf('fig10'))
