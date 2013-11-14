from glob import glob
import os.path
import random

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import numpy as np
from scipy import stats

from . import base, files

## Figures 7 and 8

def pcs(simf, expf, stname):
    exp = base.load_pickle(expf)
    sim = base.load_pickle(simf)
    ids = sim['id']
    name = '-'.join([ids['exp'], ids['seed']])

    try:
        comps = sim[stname]['evecs'].shape[1]
    except:
        if stname == 'sint':
            return
        raise
    norm = plt.Normalize(vmin=-3, vmax=3)
    w = 4.0
    wr = [10] * comps
    if 'lc' in simf:
        wr.append(0.5)
        w += 0.07 * w
    if 'cc' in simf:
        w += 0.125 * w

    fig = plt.figure(1)
    if stname == 'dint':
        gs = GridSpec(3, len(wr), height_ratios=(1, 2, 2), width_ratios=wr)
    elif stname == 'sint':
        gs = GridSpec(2, len(wr), height_ratios=(1, 2), width_ratios=wr)

    for comp in xrange(comps):
        # Determine normal and flipped PR, use the right one
        exp_ev = exp['evecs_%s' % sim['seq']].T[comp]
        sim_ev = sim[stname]['evecs'].T[comp]
        sim_sdf = sim[stname]['sdfs_%d' % comp]
        pr, _ = stats.pearsonr(exp_ev, sim_ev)
        if stats.pearsonr(exp_ev, sim_ev * -1)[0] > pr:
            sim_ev *= -1
            pr, _ = stats.pearsonr(exp_ev, sim_ev)
            sim_sdf = sim_sdf[::-1]  # reverse SDFs
        rsquared = pr ** 2
        if stname == 'dint' and comp == 0:
            print files.pkl(simf, True), 'R$^2$=%.2f ' % rsquared

        # Top: PC
        fig.add_subplot(gs[0, comp])
        plt.plot(exp['t'], exp_ev, lw=2, c='r')
        plt.plot(sim['t'], sim_ev, lw=2, c='k')
        plt.axvline(0.0, ls=':', color='k')
        plt.axvline(1.0, ls=':', color='k')
        plt.ylim(-2.5, 2.5)
        plt.xlim(-4, 4)

        plt.text(1.0, 1.0, 'R$^2$=%.2f' % rsquared, fontsize=10,
                 va='top', ha='right', transform=plt.gca().transAxes)

        plt.gca().spines['left'].set_visible(comp == 0 and 'cc' in simf)
        for side in ['top', 'right', 'bottom']:
            plt.gca().spines[side].set_visible(False)
        plt.gca().xaxis.tick_bottom()
        plt.gca().yaxis.tick_left()
        plt.xticks(())
        plt.yticks(())

        if comp == 0 and 'cc' in simf:
            plt.yticks((-1.5, 0, 1.5))
            plt.ylabel("Z-score")
        if comp == 0 and (('cc' in simf and stname == 'dint')
                          or ('pc' in simf and stname == 'sint')):
            loc = 'upper left' if sim_ev[20] < 0 else 'lower left'
            plt.legend(("Expt", "Sim"), loc=loc, prop={'size': 9})

        plt.title("PC%d" % (comp + 1))

        # Middle: Simulation SDF
        fig.add_subplot(gs[1, comp])
        cm = plt.imshow(sim_sdf,
                        aspect='auto',
                        norm=norm,
                        interpolation='nearest',
                        extent=(sim['t'][0], sim['t'][-1], sim['n'], 0))
        plt.axvline(0.0, ls=':', color='k')
        plt.axvline(1.0, ls=':', color='k')

        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.gca().spines['left'].set_visible(comp == 0 and 'cc' in simf)
        for side in ['top', 'right', 'bottom']:
            plt.gca().spines[side].set_visible(False)
        plt.xticks(())
        plt.ylim((sim['n'] - 1, 0))

        if comp == 0 and 'cc' in simf:
            plt.ylabel("Neuron (sim)")
            plt.yticks(np.arange(20, 180, 20))
        else:
            plt.yticks(())

        if stname == 'sint':
            plt.xlabel("Time from press (s)")
        if comp == comps - 1 and 'lc' in simf:
            ax = fig.add_subplot(gs[1, comps])
            plt.colorbar(cm, format='%d', ticks=np.arange(-2, 3), cax=ax)

        # Bottom: Experimental SDF
        if stname == 'dint':
            fig.add_subplot(gs[2, comp])
            cm = plt.imshow(exp['sdfs_%s_%d' % (sim['seq'], comp)],
                            norm=norm,
                            aspect='auto',
                            interpolation='nearest',
                            extent=(exp['t'][0], exp['t'][-1], exp['n'], 0))
            plt.axvline(0.0, ls=':', color='k')
            plt.axvline(1.0, ls=':', color='k')

            plt.gca().xaxis.tick_bottom()
            plt.gca().yaxis.set_ticks_position('none')
            plt.gca().spines['left'].set_visible(comp == 0 and 'cc' in simf)
            for side in ['top', 'right']:
                plt.gca().spines[side].set_visible(False)
            plt.xticks((-2, 0, 2))
            plt.ylim((exp['n'] - 1, 0))

            if comp == 0 and 'cc' in simf:
                plt.ylabel("Neuron (expt)")
                plt.yticks(np.arange(20, 180, 20))
            else:
                plt.yticks(())

            plt.xlabel("Time from press (s)")
            if comp == comps - 1 and 'lc' in simf:
                ax = fig.add_subplot(gs[2, comps])
                plt.colorbar(cm, format='%d', ticks=np.arange(-2, 3), cax=ax)

    tight = {'h_pad': 0.4, 'w_pad': 0.4}
    h = 5 if stname == 'dint' else 3
    base.save_or_show(plt.gcf(), simf, stname + '_pc', fs=(w, h), tight=tight)


def fig7_8(typ='dint'):
    cc_w = (4.0 + 0.125 * 4) * 72
    pc_w = 4.0 * 72
    lc_w = (4.0 + 0.07 * 4) * 72
    h = (5.0 if typ=='dint' else 3.0) * 72

    cc = 'plots/dint-cc-917_' + typ + '_pc.svg'
    pc = 'plots/dint-pc-931_' + typ + '_pc.svg'
    lc = 'plots/dint-lc-931_' + typ + '_pc.svg'

    fig = base.svgfig(cc_w + pc_w + lc_w, h)
    off = (0, 18)
    fig.append(base.el('A', cc, 0, 0, offset=off))
    fig.append(base.el('B', pc, cc_w, 0, offset=off))
    fig.append(base.el('C', lc, cc_w + pc_w, 0, offset=off))
    base.savefig(fig, files.pdf('fig7' if typ == 'dint' else 'fig8'))
