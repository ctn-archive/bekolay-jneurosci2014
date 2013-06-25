import matplotlib
matplotlib.use('Agg')
font = {'family': 'sans-serif',
        'sans-serif': 'Arial'}
matplotlib.rc('font', **font)
matplotlib.rc('figure', dpi=300)

import numpy as np
import quantities as pq
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap


def save_or_show(path, ext='.svg', fig=1, save=True):
    if save:
        plt.savefig(path + ext, dpi=300)
        print "Saved " + path + ext
        plt.close(fig)
    else:
        plt.show()


def plot_pc_summary(exp, sim, name='pc_summary', pos='left'):
    print "Plotting PCs: %s" % name
    comps = sim['evecs'].shape[1]  # Comps is always 2, but whatever
    norm = plt.Normalize(vmin=-3, vmax=3)
    w = 2 * comps
    wr = [10] * comps
    left = 0.025
    right = 0.975
    if pos == 'right':
        wr += [1.0 / comps]
        w += (wr[-1] / np.sum(wr)) * w
        right -= (wr[-1] / np.sum(wr))
    if pos == 'left':
        w += 0.125 * w
        left += 0.125

    fig = plt.figure(figsize=(w, 6))
    gs = gridspec.GridSpec(3, len(wr), height_ratios=(1, 2, 2), width_ratios=wr)

    for comp in xrange(comps):
        # Top: PC
        ax = fig.add_subplot(gs[0, comp])

        ax.plot(exp['t'], exp['evecs_%s' % sim['seq']].T[comp], lw=2, color='r')
        ax.plot(sim['t'], sim['evecs'].T[comp], lw=2, color='k')
        # ax.axhline(0, color='k')
        ax.axvline(0.0, ls=':', color='k')
        ax.axvline(1.0, ls=':', color='k')
        ax.set_ylim((-2.5, 2.5))
        ax.set_xlim((-4, 4))

        # Legend with amount of variance explained
        # ax.legend(("Expt, %.1f%% var" % (
        #     exp['evals_%s' % sim['seq']][comp] * 100),
        #            "Sim, %.1f%% var" % (sim['evals'][comp] * 100)),
        #           loc=legendloc, prop={'size': 8}, frameon=False)

        pr, _ = pearsonr(exp['evecs_%s' % sim['seq']].T[comp],
                         sim['evecs'].T[comp])
        rsquared = pr ** 2
        ax.text(1.0, 1.0, 'R$^2$=%.2f' % rsquared, fontsize=10,
                va='top', ha='right', transform=ax.transAxes)

        ax.spines['left'].set_visible(comp == 0 and pos=='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xticks(())
        ax.set_yticks(())

        if comp == 0 and pos == 'left':
            if sim['evecs'].T[comp][20] < 0:
                legendloc = 2
            else:
                legendloc = 3
            ax.legend(("Expt", "Sim"),
                      loc=legendloc, prop={'size': 9}, frameon=False)
            ax.set_yticks((-1.5, 0, 1.5))
            ax.set_ylabel("Z-score")
        ax.set_title("PC%d" % (comp + 1))

        # Middle: Simulation SDF
        ax = fig.add_subplot(gs[1, comp])
        cm = ax.imshow(sim['sdfs_%d' % comp], aspect='auto', norm=norm,
                       interpolation='nearest',
                       extent=(sim['t'][0], sim['t'][-1], sim['n'], 0))
        ax.axvline(0.0, ls=':', color='k')
        ax.axvline(1.0, ls=':', color='k')

        ax.get_xaxis().set_ticks_position('none')
        ax.get_yaxis().set_ticks_position('none')
        ax.spines['left'].set_visible(comp == 0 and pos=='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks(())
        ax.set_ylim((sim['n'] - 1, 0))

        if comp == 0 and pos == 'left':
            ax.set_ylabel("Neuron (sim)")
        else:
            ax.set_yticks(())

        if comp == comps - 1 and pos == 'right':
            ax = fig.add_subplot(gs[1, comps])
            plt.colorbar(cm, format='%d', ticks=np.arange(-2, 3), cax=ax)

        # Bottom: Experimental SDF
        ax = fig.add_subplot(gs[2, comp])
        cm = ax.imshow(exp['sdfs_%s_%d' % (sim['seq'], comp)], norm=norm,
                       aspect='auto', interpolation='nearest',
                       extent=(exp['t'][0], exp['t'][-1], exp['n'], 0))
        ax.axvline(0.0, ls=':', color='k')
        ax.axvline(1.0, ls=':', color='k')

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().set_ticks_position('none')
        ax.spines['left'].set_visible(comp == 0 and pos=='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks((-2, 0, 2))
        ax.set_ylim((exp['n'] - 1, 0))

        if comp == 0 and pos == 'left':
            ax.set_ylabel("Neuron (expt)")
        else:
            ax.set_yticks(())

        ax.set_xlabel("Time from press (s)")
        if comp == comps - 1 and pos == 'right':
            ax = fig.add_subplot(gs[2, comps])
            plt.colorbar(cm, format='%d', ticks=np.arange(-2, 3), cax=ax)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05, left=left, right=right)
    save_or_show('plots/%s_pc' % name)


def plot_dynamics(trajectories, vf, name='trajectory', pos='left'):
    print "Plotting trajectory: %s" % name
    plt.figure(1, figsize=(4, 4))

    ax = plt.subplot(111)
    if 'left' in pos:
        plt.ylabel("Task state (arbitrary units)")
    if 'bottom' in pos:
        plt.xlabel("Relative time in task state (arbitrary units)")

    ax.streamplot(vf['x'], vf['y'], vf['dx'], vf['dy'], minlength=0.2,
                  arrowstyle='->', color='0.75', density=0.5,
                  linewidth=2 * vf['speed'] / vf['speed'].max())
    plt.xticks(())
    plt.yticks(())

    trajx = np.concatenate([t[0] for t in trajectories])
    trajy = np.concatenate([t[1] for t in trajectories])
    annote = set()
    for t in trajectories:
        for a in t[2]:
            annote.add(a)

    t = np.arange(trajx.shape[0]) * 0.001
    ax.plot(trajy, trajx, color='k', lw=4)

    nback = 60
    plt.arrow(trajy[-nback], trajx[-nback], trajy[-1] - trajy[-nback],
              trajx[-1] - trajx[-nback],
              width=0.035, overhang=0.25, color='k', zorder=4)

    for (a_t, l) in annote:
        ix = int(a_t / 0.001) - 1
        xy = (trajy[ix], trajx[ix])
        xytext = (-30 if xy[0] > 0 else 30, -30 if xy[1] > 0 else 30)
        if l == 'Cue':
            xytext = (-45, xytext[1])
        if l == 'Start of next trial' and xy[1] > 0:
            l = 'Start of\nnext trial'
            xytext = (-60, 5)
        if l == 'Release\n(premature)':
            xytext = (-15, -35)
        plt.annotate(l, xy, xycoords='data', xytext=xytext, ha='center',
                     va='center', textcoords='offset points',
                     arrowprops={'arrowstyle': '->',
                                 'connectionstyle': 'arc3, rad=0.2'})

    plt.axhline(0.0, color='k')
    plt.axvline(0.0, color='k')
    plt.axis((-1.3, 1.3, -1.3, 1.3))
    plt.subplots_adjust(bottom=0.1, top=1.0, left=0.1, right=1.0)

    save_or_show('plots/' + name + '_traj')


def plot_trajectory(trajectories, name='trajectory', pos='left'):
    print "Plotting trajectory: %s" % name
    plt.figure(1, figsize=(4, 4))

    ax = plt.subplot(111)
    trajx = np.concatenate([t[0] for t in trajectories])
    trajy = np.concatenate([t[1] for t in trajectories])
    annote = set()
    for t in trajectories:
        for a in t[2]:
            annote.add(a)

    cmap = plt.get_cmap('Greys')

    t = np.arange(trajx.shape[0]) * 0.001
    points = np.array([trajy, trajx]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap,
                        norm=plt.Normalize(-t[-1] * 0.5, t[-1] * 0.75))
    lc.set_array(t)
    lc.set_linewidth(3)
    lc.set_rasterized(True)
    ax.add_collection(lc)

    for (a_t, l) in annote:
        ix = int(a_t / 0.001) - 1
        xy = (trajy[ix], trajx[ix])
        xytext = (-30 if xy[0] > 0 else 30, -30 if xy[1] > 0 else 30)
        if l == 'Start of next trial' and xy[1] > 0:
            l = 'Start of\nnext trial'
            xytext = (-50, 5)
        if l == 'Release\n(premature)':
            xytext = (-10, -35)
        plt.annotate(l, xy, xycoords='data', xytext=xytext, ha='center',
                     va='center', textcoords='offset points',
                     arrowprops={'arrowstyle': '->',
                                 'connectionstyle': 'arc3, rad=0.2'})

    plt.axhline(0.0, color='k', ls=":")
    plt.axvline(0.0, color='k', ls=":")
    if 'left' in pos:
        plt.ylabel("Task state (arbitrary units)")
        ax.get_yaxis().tick_left()
    else:
        plt.yticks(())
    ax.get_xaxis().tick_bottom()
    plt.xlabel("Relative time in task state (arbitrary units)")


    plt.axis((-1.5, 1.5, -2.0, 1.5))
    plt.subplots_adjust(bottom=0.12, top=0.97, left=0.17, right=0.97)

    save_or_show('plots/' + name + '_traj')


def plot_performance(cor, prem, late, name='performance', pos='left'):
    print "Plotting performance: %s" % name
    plt.figure(figsize=(4, 4))
    ax = plt.subplot(111)
    left = np.arange(0.25, len(cor) - 0.25)
    plt.plot([0.5, 0.5], [1, 1], lw=2, color='k', label='Correct')
    plt.plot([0.5, 0.5], [1, 1], lw=2, color='0.6', label='Premature')
    plt.plot([0.5, 0.5], [1, 1], lw=2, color='w', label='Late')
    plt.legend(loc=2, prop={'size': 10})

    plt.bar(left, late + prem + cor, width=0.5, color='k')
    plt.bar(left, late + prem, width=0.5, color='0.6')
    plt.bar(left, late, width=0.5, color='w')

    plt.axis((0, len(cor), 0, 250))
    plt.xticks(left + 0.25, np.arange(len(cor)) + 1)
    if name == 'exp':
        plt.xlabel("Experimental subject")
    elif name == 'sr':
        plt.xlabel("Cue-responding model")
    elif name == 'td':
        plt.xlabel("Adaptive control model")

    ax.spines['left'].set_visible(pos=='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if pos == 'left':
        plt.ylabel("Trials")
    else:
        ax.set_yticks(())

    plt.tight_layout()
    save_or_show('plots/' + name + '_perf')


def plot_reactiontimes(rts, rtstats, name='reaction_times', pos='left'):
    print "Plotting reaction times: %s" % name
    plt.figure(figsize=(4, 4))
    ax = plt.subplot(111)

    plt.axhline(rtstats[0], color='k', lw=1.5, ls=':')  # mean
    lines = plt.boxplot(rts, sym='')
    for line in lines.values():
        plt.setp(line, color='black')
    plt.ylim((0, 0.6))

    ax.spines['left'].set_visible(pos=='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if pos == 'left':
        plt.ylabel("Reaction time (s)")
    else:
        ax.set_yticks(())

    if name == 'exp':
        plt.xlabel("Experimental subject")
    elif name == 'sr':
        plt.xlabel("Cue-responding model")
    elif name == 'td':
        plt.xlabel("Adaptive control model")

    plt.tight_layout()
    save_or_show('plots/' + name + "_rts")


def plot_signals(sd, name='signals', pos='left'):
    print "Plotting signals: %s" % name
    w = 2
    left = 0.05
    if pos == 'left':
        w += 0.8
        left += 0.8 / w

    plt.figure(1, figsize=(w, 5))

    for ix, (sname, color, signal) in enumerate(sd['signals']):
        ax = plt.subplot(len(sd['signals']), 1, ix + 1)
        if pos == 'left':
            plt.ylabel(sname)
        plt.plot(signal.times, signal, color='k')
        for (_, t) in sd['signal_events']:
            plt.axvline(t, color='k', ls=":")
        plt.ylim((-.8, 2))

        sigevts = sorted(sd['signal_events'], key=lambda t: t[1])

        if ix == 0:
            offsets = [0 * pq.s for _ in xrange(len(sigevts))]
            min_dist = 0.4 * pq.s
            for jx, t1, t2 in zip(
                    xrange(len(sigevts)),
                    [lt[1] for lt in sigevts[:-1]],
                    [lt[1] for lt in sigevts[1:]]):
                if t2 - t1 < min_dist:
                    offsets[jx] = (-min_dist + (t2 - t1)) * 0.25
                    offsets[jx + 1] = (min_dist - (t1 - t2)) * 0.25

            for jx, (l, t) in enumerate(sigevts):
                plt.text(t + offsets[jx], plt.ylim()[1] + 0.3, l,
                         ha='center', va='center')

        plt.xticks(())
        if ix == len(sd['signals']) - 1:
            plt.xticks((0, 1, 2, 3, 4))

        if pos == 'left':
            plt.yticks((-0.5, 0, 0.5, 1, 1.5))
        else:
            plt.yticks(())

        ax.spines['left'].set_visible(pos=='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.025, top=0.94, left=left, right=0.9)
    save_or_show('plots/' + name + '_signals')


def plot_raster(d, name='raster'):
    print "Plotting raster: %s" % name
    plt.figure(1, figsize=(10, 5))

    for ix, st in enumerate(d['sts']):
        plt.scatter(st, ix * np.ones(st.shape), marker=',', s=2, color='k')

    plt.ylim(len(d['sts']), -1)
    # plt.title(title + ", Component " + str(i + 1))
    plt.axvline(0.0, ls='--', color='k')
    plt.axvline(1.0, ls='--', color='k')

    plt.tight_layout()
    save_or_show('plots/' + name + '_raster', ext='.png')
