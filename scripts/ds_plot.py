import matplotlib.pyplot as plt
import numpy as np

from . import base, files


## Figure 1
def env_signals(pklf, trial):
    d = base.load_pickle(pklf)

    fig = plt.figure(1)
    lu, = plt.plot(d['t'], d['u']+0.6, lw=2)
    lc, = plt.plot(d['t'], d['u_c']+0.4, lw=2)
    lr, = plt.plot(d['t'], d['u_r']+0.2, lw=2)
    ll, = plt.plot(d['t'], d['u_l'], lw=2)
    plt.xlim(right=5)
    plt.ylim(-0.15, 1.75)

    plt.yticks(())
    if pklf[-5] == 'c':
        plt.xticks((0, 1, 2, 2.6, 4.6),
                   ('$t_s$', '$t_p$', '$t_c$', '$t_{Rw}$', '$t_{ITI}$'))
    elif pklf[-5] == 'p':
        plt.xticks((0, 1, 1.5, 3.5),
                   ('$t_s$', '$t_p$', '$t_{TO}$', '$t_{ITI}$'))
    elif pklf[-5] == 'l':
        plt.xticks((0, 1, 2, 2.6, 4.6),
                   ('$t_s$', '$t_p$', '$t_c$', '$t_{TO}$', '$t_{ITI}$'))

    for side in ['left', 'top', 'right', 'bottom']:
        plt.gca().spines[side].set_visible(False)
    plt.gca().xaxis.set_ticks_position('none')
    plt.grid()

    figlegend = plt.figure(2)
    figlegend.legend([lu, lc, lr, ll],
                     ['Press', 'Cue', 'Reward', 'Timeout'],
                     loc='center')

    base.save_or_show(fig, pklf, trial + '_sig', fs=(2.75, 1))
    base.save_or_show(figlegend, pklf, 'legend', fs=(1.25, 3))


def fig1():
    pad = 15
    tr_w, tr_h = 209.74, 216.43
    sig_w, sig_h = 2.75 * 72, 1 * 72
    leg_w, leg_h = 1.25 * 72, 3.0 * 72
    fsm_w, fsm_h = 497.74, 250.0
    fig = base.svgfig(tr_w + sig_w + leg_w + pad, tr_h + fsm_h + pad*2)
    fig.append(base.el('A', files.diag('trials'), 0, 10, offset=(0, 10)))
    fig.append(base.el('B',
                       files.svg('ds_c', 'simple_sig'),
                       tr_w + pad,
                       0,
                       offset=(-pad, 20)))
    fig.append(base.el(None, files.svg('ds_p', 'simple_sig'), tr_w + pad, sig_h))
    fig.append(base.el(None, files.svg('ds_l', 'simple_sig'), tr_w + pad, sig_h * 2))
    fig.append(base.el(None, files.svg('ds_c', 'legend'), tr_w + pad + sig_w, 0))
    fig.append(base.el('C', files.diag('fsm'), 0, tr_h + pad*2, offset=(10, 15)))
    base.savefig(fig, files.pdf('fig1'))

## Figure 2
def fig2():
    fig = base.svgfig(820.14, 557.58)
    fig.append(base.el(None, files.diag('fig2'), 0, 0))
    base.savefig(fig, files.pdf('fig2'))


## Figure 3
def fig3():
    fig = base.svgfig(262.97, 180.73)
    fig.append(base.el(None, files.diag('neural'), 0, 0))
    base.savefig(fig, files.pdf('fig3'))


## Figure 4
def param(pklf, par):
    plt.rc('axes', labelsize='large')
    plt.rc('xtick', labelsize='large')
    d = base.load_pickle(pklf)
    x1, x2 = d[par + 'trajs']['x1'], d[par + 'trajs']['x2']
    right = 2.0 if par in ('beta', 'ics') else 4.6
    rightix = d['t'] <= right
    t = d['t'][rightix]

    plt.figure(1)
    plt.subplot(2, 1, 1)
    for x, y, pval in zip(x1, x2, d[par]):
        x, y = x[rightix], y[rightix]
        plt.plot(t, x, lw=2, ls='--', c='0.4')
        plt.plot(t, y, lw=2, c='k')
        if par != 'ics':
            ybump = 0.05 if par == 'beta' and pval == 0.5 else 0
            plt.text(right * 1.02,
                     (x[-1] if par in 'RE' else y[-1]) + ybump,
                     ('$\\' if par == 'beta' else '$')+par+'='+str(pval)+'$',
                     ha='left',
                     va='center',
                     bbox={'fc': 'white', 'ec': 'none'})

    if par == 'beta':
        plt.legend(['$x_1$', '$x_2$'], loc='upper left')

    if par == 'ics':
        plt.yticks((d[par][:, 0]))
        plt.gca().yaxis.set_ticks_position('left')
    else:
        plt.yticks(())
        plt.xlim(0, right * 1.24)
        plt.gca().spines['left'].set_visible(False)

    plt.gca().xaxis.grid(True)
    plt.axhline(0, color='k')
    plt.ylim(-1.1, 1.1)

    if par in ('beta', 'ics'):
        plt.xticks((0, 1, right), ('$t_s$', '$t_p$', '$t_c$'))
    elif par in 'RE':
        xlabel = '$t_{Rw}$' if par == 'R' else '$t_{TO}$'
        plt.xticks((0, 1, 2, 2.6, right),
                   ('$t_s$', '$t_p$', '$t_c$', xlabel, '$t_{ITI}$'))

    for side in ['top', 'right', 'bottom']:
        plt.gca().spines[side].set_visible(False)
    plt.gca().xaxis.set_ticks_position('none')

    plt.subplot(2, 1, 2)

    # plt.grid()
    plt.axhline(0, color='0.2')
    plt.axvline(0, color='0.2')
    left = 0.0 if par == 'ics' else 1.0
    arrow = {'overhang': 0.2, 'head_width': 0.12, 'color': 'k'}

    for x, y, pval in zip(x1, x2, d[par]):
        x, y = x[rightix], y[rightix]
        plt.plot(y, x, lw=2, c='k')
        st, = plt.plot(y[t == 0], x[t == 0], 'ow', ms=6)
        md, = plt.plot(y[t == 1], x[t == 1], 'o', c='0.5', ms=6)
        ed, = plt.plot(y[-1], x[-1], 'ok', ms=6)
        if par in ('beta', 'ics'):
            arrowix = np.where(t == 0.84)[0][0]
        elif par in 'RE':
            arrowix = np.where(t == 1.2)[0][0]
        if not (par == 'ics' and y[0] > 0.4):
            plt.arrow(y[arrowix],
                      x[arrowix],
                      y[arrowix+1] - y[arrowix],
                      x[arrowix+1] - x[arrowix], **arrow)

    if par == 'R':
        arrowix = np.where(t == 2.79)[0][0]
    elif par == 'E':
        arrowix = np.where(t == 4.32)[0][0]
    if par in 'RE':
        plt.arrow(x2[-1][arrowix],
                  x1[-1][arrowix],
                  x2[-1][arrowix+1] - y[arrowix],
                  x1[-1][arrowix+1] - x[arrowix], **arrow)

    if par == 'R':
        leg = plt.legend([md, ed],
                         ['$t_p$', '$t_c$'],
                         handlelength=0.8,
                         frameon=True,
                         loc='best')
    else:
        leg = plt.legend([st, md, ed],
                         ['$t_s$','$t_p$','$t_{ITI}$' if par=='E' else '$t_c$'],
                         handlelength=0.8,
                         frameon=True,
                         loc='best')
    leg.get_frame().set_edgecolor('none')
    plt.xlabel("$x_2$")
    plt.ylabel("$x_1$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().xaxis.set_ticklabels(())
    plt.gca().yaxis.set_ticklabels(())
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    for side in ['top', 'right', 'bottom', 'left']:
        plt.gca().spines[side].set_visible(False)

    tight = {'h_pad': 1.5, 'rect': [0, 0, 0.97, 1]}
    base.save_or_show(plt.gcf(), pklf, 'param_' + par, fs=(3, 6), tight=tight)
    plt.rc('xtick', labelsize='medium')
    plt.rc('axes', labelsize='medium')


def fig4():
    w, h = 216, 432
    fig = base.svgfig(w * 4, h)
    off = (-8, 18)
    fig.append(base.el('A', files.svg('ds_c', 'param_ics'), 8, 0, offset=off))
    off = (8, 18)
    fig.append(base.el('B', files.svg('ds_c', 'param_beta'), w, 0, offset=off))
    fig.append(base.el('C', files.svg('ds_c', 'param_R'), w * 2, 0, offset=off))
    fig.append(base.el('D', files.svg('ds_l', 'param_E'), w * 3, 0, offset=off))
    base.savefig(fig, files.pdf('fig4'))


## Figure 5
def ctrl_signals(pklf, trial):
    plt.rc('axes', labelsize='large')
    plt.rc('xtick', labelsize='large')
    d = base.load_pickle(pklf)

    fig = plt.figure(1)
    plt.plot(d['t'], d[trial]['x2'], lw=2, c='k', label="$x_2$")
    plt.plot(d['t'], d[trial]['lever'], lw=2, ls='--', c='k', label="$L$")
    plt.plot(d['t'], d['u'], lw=2, c='0.5', label="$u$")
    plt.plot(d['t'], d[trial]['release'], lw=2, c='0.75', label="$u_r$")

    for side in ['top', 'right', 'bottom']:
        plt.gca().spines[side].set_visible(False)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().xaxis.grid(True)

    if trial == 'simple':
        plt.xlim(left=0.5)
        leg = plt.legend(frameon=True, loc='best')
        leg.get_frame().set_edgecolor('none')
        plt.gca().yaxis.set_ticks_position('left')
    else:
        plt.gca().spines['left'].set_visible(False)
        plt.yticks(())
    plt.ylim(-1.1, 1.1)
    plt.xlim(right=3.5)

    if trial == 'adaptive_p':
        plt.xticks((0, 1, 1.5), ('$t_s$', '$t_p$', '$t_{TO}$'))
    else:
        plt.xticks((0, 1, 2, 2.6), ('$t_s$', '$t_p$', '$t_c$', '$t_{Rw}$'))

    tight = {'rect': [0.05, 0, 0.98, 1]}
    base.save_or_show(fig, pklf, trial, fs=(3, 3), tight=tight)
    plt.rc('xtick', labelsize='medium')
    plt.rc('axes', labelsize='medium')


def fig5():
    pad = 8
    w = h = 216
    fig = base.svgfig(w*4 + pad*3, h)
    off = (0, 18)
    fig.append(base.el('A', files.svg('ds_c', 'simple'), 0, 0, offset=(4, 18)))
    fig.append(base.el(
        'B', files.svg('ds_c', 'adaptive_c'), w + pad, 0, offset=off))
    fig.append(base.el(
        'C', files.svg('ds_p', 'adaptive_p'), w*2 + pad*2, 0, offset=off))
    fig.append(base.el(
        'D', files.svg('ds_c', 'adaptive_pc'), w*3 + pad*3, 0, offset=off))
    base.savefig(fig, files.pdf('fig5'))


## Figure 6
def learning(pklf):
    d = base.load_pickle(pklf)
    x1, x2 = d['simple']['x1'], d['simple']['x2']

    def sp(ix, text):
        plt.subplot(5, 1, ix)
        plt.xlim(0.4, 6)
        plt.ylim(-0.2, 1.2)
        plt.xticks(())
        plt.yticks(())
        for side in ['top', 'right', 'bottom', 'left']:
            plt.gca().spines[side].set_visible(False)
        plt.text(4.8, 1.0, text, ha='left', va='top', fontsize=19)

    plt.figure(1)
    sp(1, '$S = x_1$')
    plt.plot(d['t'], x1, lw=3.5)

    sp(2, '$V = x_2$')
    plt.plot(d['t'], x2, lw=3.5)

    sp(3, '$r = u_{Rw}$')
    plt.plot(d['t'], d['u_r'], lw=3.5)

    sp(4, '$\omega^N = \max$ \\vspace{0.2em}\\\\ $(0, x_2 - u_{Rw})$')
    plt.plot(d['t'], np.maximum(np.zeros_like(x2), x2 - d['u_r']), lw=3.5)

    sp(5, '$\omega^P = \max$ \\vspace{0.2em}\\\\ $(0, u_{Rw} - x_2)$')
    plt.plot(d['t'], np.maximum(np.zeros_like(x2), d['u_r'] - x2), lw=3.5)

    fs = (1.372 * 3, 1.80 * 3)
    tight = {'rect': [0, 0, 0.82, 1]}
    base.save_or_show(plt.gcf(), pklf, 'learn', fs=fs, tight=tight)


def fig6():
    ab_w, ab_h = 98.8, 129.61
    fig = base.svgfig(ab_w * 2 + 3, ab_h)
    fig.append(base.el(None, files.diag('alexander2011'), 0, 3))
    fig.append(base.el(None, files.svg('ds_c', 'learn'), ab_w, 0, scale=1./3.))
    base.savefig(fig, files.pdf('fig6'))
