from __future__ import absolute_import

import os.path
import random

import numpy as np
import quantities as pq
from scipy import linalg
from scipy import signal, stats
from scipy.io import loadmat

from .sim_io import get_seg
from . import neo

def correct_seq(e):
    return (e['press'], e['toneon'], e['release'], e['pumpon'])


def premature_seq(e):
    return (e['press'], e['release'], e['lightsoff'])


def late_seq(e):
    return (e['press'], e['toneon'], e['lightsoff'])


def reaction_seq(e):
    return (e['toneon'], e['release'])


def delay_seq(e):
    return (e['press'], e['toneon'])


def pca(x, comps=2, flip=True):
    evecs, evals, _ = linalg.svd(x.T, full_matrices=False)
    evecs = stats.zscore(evecs[:, :comps], ddof=1)
    evals = (evals ** 2) / np.sum(evals ** 2)
    loadings = np.dot(x, evecs)

    if not flip:
        return {'evecs': evecs, 'evals': evals, 'loadings': loadings}

    for i in xrange(comps):
        if evecs.T[i][int(evecs.T[i].shape[0] * 0.485)] < 0:
            evecs.T[i] *= -1
            loadings.T[i] *= -1

    return {'evecs': evecs, 'evals': evals, 'loadings': loadings}


def integrator_vf(x1, x2, press=0, reward=0, error=0, degrade=0):
    alpha = 0.2
    x1, x2 = x1.T, x2.T
    dx1 = 0.45 * press + 0.25 * error - reward * x1 - degrade * x1
    dx2 = alpha * x1 - 0.75 * reward * x2 - degrade * x2
    return dx2, dx1


def gaussian_sdf(hist, binsize, sigma=0.015 * pq.s, denoise=True):
    # Put gaussian in terms of bins
    sigma.units = binsize.units
    sigma = float(sigma / binsize)
    gauss = signal.gaussian(int(sigma * 7), sigma)  # +/- 3 stddevs
    gauss /= np.sum(gauss)  # normalize to area 1 (important!)
    gauss /= binsize  # then scale to size of bins
    sdf = np.convolve(hist, gauss, mode='same')

    if denoise:
        sdf = (np.roll(sdf, -1) + sdf + np.roll(sdf, 1)) / 3.0

    return sdf


def merge_all_eas(seg, channels):
    merged = []
    for channel in channels:
        ix = map(lambda ea: ea.annotations['channel_name'],
                 seg.eventarrays).index(channel)
        merged.extend(zip(seg.eventarrays[ix].times,
                          [channel] * len(seg.eventarrays[ix].times)))
    merged.sort()
    return neo.EventArray(np.array(map(lambda x: x[0], merged)),
                          np.array(map(lambda x: x[1], merged)))


def behaviour(seg, e):
    merged = merge_all_eas(seg, e.values())

    # How many correct, premature, late?
    cor = merged.epochs_from(correct_seq(e), "C")
    pre = merged.epochs_from(premature_seq(e), "P")
    lat = merged.epochs_from(late_seq(e), "L")

    # Filter for only long-delay correct trials
    merged = merged.filter_for(correct_seq(e))
    dly = merged.epochs_from(delay_seq(e))
    react = merged.epochs_from(reaction_seq(e))
    if e['toneon'] == 'eventTONEOFF':
        react.times -= 0.1 * pq.s
        react.durations += 0.1 * pq.s
    filt = dly.durations > 0.88  # really 1.0, but leave a grace period

    return {'trials': len(cor) + len(pre) + len(lat),
            'correct': len(cor), 'premature': len(pre),
            'late': len(lat), 'rts': react.durations[filt]}


def valid_presses(seg, e, seq):
    merged = merge_all_eas(seg, e.values())
    # print merged
    merged_ep = (merged.epochs_from(correct_seq(e), name='C') +
                 merged.epochs_from(premature_seq(e), name='P') +
                 merged.epochs_from(late_seq(e), name='L'))
    # print merged_ep
    ep = merged_ep.epochs_from(seq)
    # print ep

    # Everything in seq will have a press,
    # but we only care about the last one.
    return merged.during_epochs(ep).filter_for(
        (e['press'],))[len(seq) - 1::len(seq)]


def get_analogsignal(seg, name):
    # print [a.name for a in seg.analogsignals]
    # print seg.file_origin
    ix = map(lambda ansig: ansig.name,
             seg.analogsignals).index(name)
    return seg.analogsignals[ix]


def perievent_signal(analog, evt, window):
    t = analog.times
    ix = np.logical_and(t > (evt + window[0]),
                        t <= (evt + window[1]))
    return analog[ix]


def decoded_signals(seg, e, trial='C', window=(-1.0, 3.0) * pq.s):
    evts = merge_all_eas(seg, e.values())
    press_evts = valid_presses(seg, e, (trial,))

    # Always choose press event 5. Dunno why!
    if len(press_evts.times) >= 5:
        press = press_evts.times[4]
    else:
        press = press_evts.times[0]

    if trial == 'C':
        seq = correct_seq(e)
        # (e['press'], e['toneon'], e['release'], e['pumpon'])
    elif trial == 'P':
        seq = premature_seq(e)
        # (e['press'], e['release'], e['lightsoff'])
    elif trial == 'L':
        seq = late_seq(e)
        # (e['press'], e['toneon'], e['lightsoff'])
    s_evts = [evts[evts.times >= press].filter_for((evt,)).times[0]
              for evt in seq]
    s_evts = [-window[0] + (evt - press) for evt in s_evts]

    evts = [('P', -window[0])]

    if trial == 'C' or trial == 'L':
        evts.append(('C', s_evts[1]))
    elif trial == 'P':
        evts.append(('R', s_evts[1]))

    if trial == 'L' or trial == 'P':
        evts.append(('E', s_evts[2]))
    elif trial == 'C':
        evts.append(('R', s_evts[2]))
        evts.append(('Rw', s_evts[3]))

    signals = []

    # Get all signals
    signals.append(("Press", 'b', perievent_signal(
        get_analogsignal(seg, 'Press/Release_0'), press, window)))
    signals.append(("Release", 'b', perievent_signal(
        get_analogsignal(seg, 'Press/Release_1'), press, window)))
    signals.append(("Holding", 'g', perievent_signal(
        get_analogsignal(seg, 'Holding_0'), press, window)))
    signals.append(("Trigger", 'k', perievent_signal(
        get_analogsignal(seg, 'Trigger_0'), press, window)))
    try:
        signals.append(("Cue\nprediction", 'r', perievent_signal(
            get_analogsignal(seg, 'Force release_0'),
            press, window)))
    except:
        pass

    return {'signal_events': evts, 'signals': signals}


def perievent_neural(seg, evts, window, binsize):
    extend = 0.1 * pq.s
    window_ext = (window[0] - extend, window[1] + extend)
    bins = int((window_ext[1] - window_ext[0]) / binsize)
    trim = int(extend / binsize)

    d = {k: [] for k in ['sts', 'rates', 'sdfs']}

    for st in seg.spiketrains:
        d['sts'].extend(st.perievent_slices(evts.times, window))

    todelete = []
    for ix, st in enumerate(d['sts']):
        sdf = st.gaussian_sdf(bins=bins)[trim:-trim]
        if np.mean(sdf) < 1:  # filter out < 1 Hz spiketrains
            todelete.append(ix)
            continue
        d['rates'].append(np.mean(sdf))
        d['sdfs'].append(sdf)

    if len(d['sdfs']) == 0:
        raise Exception('No SDFs')

    d['rates'] = np.array(d['rates'])
    d['sdfs'] = stats.zscore(np.vstack(d['sdfs']), axis=1)

    for ix in reversed(todelete):
        del d['sts'][ix]

    return d


def perievent_neural_mat(mat, window, binsize):
    d = {k: [] for k in ['sdfs_cc', 'rates_cc', 'sdfs_ec', 'rates_ec']}

    for k in mat.keys():
        if k.endswith('pC'):
            seq = 'cc'
        elif k.endswith('pE'):
            seq = 'ec'
        else:
            continue

        peh = mat[k].T
        if peh.shape[1] > 80:
            newpeh = np.zeros((peh.shape[0], 80))
            scale = peh.shape[1] / 80
            rnge = np.arange(0, peh.shape[1] + 1, scale)
            rnge[-1] = peh.shape[1]
            for i, s, e in zip(range(len(rnge)), rnge[:-1], rnge[1:]):
                newpeh[:, i] = peh[:, s:e].sum(axis=1)
            peh = newpeh

        for hist in peh:
            sdf = gaussian_sdf(hist, binsize)
            if np.mean(sdf) < 1:  # filter out < 1 Hz spiketrains
                continue
            d['rates_%s' % seq].append(np.mean(sdf))
            d['sdfs_%s' % seq].append(sdf)

    for seq in ('cc', 'ec'):
        sdf = 'sdfs_%s' % seq
        d[sdf] = stats.zscore(np.vstack(d[sdf]),
                              axis=1)[:, window[0]:window[1]]

    return d


def trajectory(seg, e, trial='C'):
    evts = merge_all_eas(seg, e.values())
    press = valid_presses(seg, e, (trial,)).times[0]

    if trial == 'C':
        seq = correct_seq(e)
        press = valid_presses(seg, e, (trial,)).times[1]
        # (e['press'], e['toneon'], e['release'], e['pumpon'])
    elif trial == 'P':
        seq = premature_seq(e)
        # (e['press'], e['release'], e['lightsoff'])
    elif trial == 'L':
        seq = late_seq(e)
        # (e['press'], e['toneon'], e['lightsoff'])
    t_evts = [evts[evts.times > press].filter_for((evt,)).times[0]
              for evt in seq]

    trajannote = []
    trajevts = [
        press - 1.0 * pq.s,  # 1 s before "press"
        press + 0.5 * pq.s,  # 200 ms after "press"
    ]
    trajvf = [{'press': 1}, {}, None, {}]

    if trial == 'C':
        trajevts.append(t_evts[2])  # release time
        trajevts.append(t_evts[3] + 1.0 * pq.s)  # 1 s after reward delivery
        trajvf[2] = {'reward': 1}

        trajannote.append((1, t_evts[1], "Cue"))
        trajannote.append((1, t_evts[2], "Release"))
    elif trial == 'P':
        trajevts.append(t_evts[1])  # release time
        trajevts.append(t_evts[2] + 2.0 * pq.s)  # 2.0 s after lights out
        trajvf[2] = {'error': -1}

        trajannote.append((1, t_evts[1], "Release\n(premature)"))
    elif trial == 'L':
        trajevts.append(t_evts[1] + 0.6 * pq.s)  # 0.6 s after cue
        trajevts.append(t_evts[2] + 2.0 * pq.s)  # 2.0 s after lights out
        trajvf[2] = {'error': -1}

        trajannote.append((1, t_evts[1], "Cue"))

    trajevts.append(t_evts[0] - 2.0 * pq.s)  # 2 s before next press
    trajannote.append((3, t_evts[0] - 2.0 * pq.s, "Start of next trial"))

    x = get_analogsignal(seg, 'Delay state_0')
    y = get_analogsignal(seg, 'Delay state_1')
    t = x.times
    trajs = []
    for ix, t1, t2 in zip(range(len(trajevts) - 1),
                          trajevts[:-1], trajevts[1:]):
        annote = []
        for i, at, l in trajannote:
            if i <= ix:
                annote.append((at - trajevts[0], l))
        tr_ix = np.logical_and(t >= t1, t <= t2)
        trajs.append((x[tr_ix], y[tr_ix], annote))

    return {'traj_evts': trajevts, 'traj_vf': trajvf, 'trajectory': trajs}


def process_sim_dint(path, comps=2, binsize=0.1 * pq.s,
                     window=(-4.0 * pq.s, 4.1 * pq.s)):
    print path
    seg, e = get_seg(path)
    d = {}
    # fs contents:
    #  dint, (deg), direct/spikes, exptype, seed
    fname = os.path.splitext(os.path.basename(path))[0]
    fs = fname.split('-')
    d['id'] = {'model': fs[0]}

    if len(fs) > 4:
        d['id'].update({'degrade': fs[1][-1], 'mode': fs[2],
                        'exp': fs[3], 'seed': fs[4]})
    else:
        d['id'].update({'mode': fs[1], 'exp': fs[2], 'seed': fs[3]})

    if d['id']['exp'] == 'cc':
        seq = ('C', 'C')
        d['seq'] = 'cc'
    elif d['id']['exp'] == 'pc':
        seq = ('P', 'C')
        d['seq'] = 'ec'
    elif d['id']['exp'] == 'lc':
        seq = ('L', 'C')
        d['seq'] = 'ec'
    else:
        raise Exception("%s should start with cc or pc"
                        % os.path.basename(path))

    try:
        evts = valid_presses(seg, e, seq)
        evts[0]
    except:
        print "No valid presses."
        return None
    evts.times += 0.3 * pq.s  # For delay to muscles

    # Only keep 174 neurons (as in experiment)

    # Neural data
    if 'spikes' in path:
        try:
            d.update(perievent_neural(seg, evts, window, binsize))
            keep = np.random.choice(d['rates'].shape[0], 174, replace=False)
            d['rates'] = d['rates'][keep]
            d['sdfs'] = d['sdfs'][keep]

            # PCA
            d.update(pca(d['sdfs'], comps=comps))

            # Sort sdfs by loadings
            for c in xrange(comps):
                d['sdfs_%d' % c] = d['sdfs'][np.argsort(d['loadings'].T[c])]

            # Convenience!
            d['n'] = len(d['sdfs'])
            d['t'] = np.linspace(window[0], window[1], d['sdfs'][0].shape[0])
        except:
            pass

    # Trajectories
    d.update(trajectory(seg, e, seq[0]))

    # Calculate vector fields
    npoints = 201
    xdomain = (-1.3, 1.3)
    ydomain = (-1.3, 1.3)
    x, y = np.meshgrid(np.linspace(xdomain[0], xdomain[1], npoints),
                       np.linspace(ydomain[0], ydomain[1], npoints))
    d['vf'] = [dict.fromkeys(['x', 'y', 'dx', 'dy', 'speed'])
               for _ in xrange(len(d['trajectory']))]
    for vfd, vf_args in zip(d['vf'], d['traj_vf']):
        vfd['x'] = x
        vfd['y'] = y
        vfd['dx'], vfd['dy'] = integrator_vf(x, y, **vf_args)
        vfd['speed'] = np.sqrt(vfd['dx'] * vfd['dx']
                               + vfd['dy'] * vfd['dy'])

    return d


def process_sim_ctrl(path):
    print path
    seg, e = get_seg(path)
    d = {}
    fname = os.path.splitext(os.path.basename(path))[0]
    fs = fname.split('-')
    d['id'] = {'model': fs[0]}

    if len(fs) > 3:
        d['id'].update({'degrade': fs[1][-1], 'ctrl_type': fs[2],
                        'seed': fs[3]})
    else:
        d['id'].update({'ctrl_type': fs[1], 'seed': fs[2]})

    try:
        valid_presses(seg, e, ('C'))[0]
    except:
        print "No valid presses."
        return None

    # Behavioural data
    d.update(behaviour(seg, e))

    # Signals
    d['c_sig'] = decoded_signals(seg, e)

    try:
        d['p_sig'] = decoded_signals(seg, e, 'P')
    except:
        print "No premature trial in " + path

    try:
        d['l_sig'] = decoded_signals(seg, e, 'L')
    except:
        print "No late trial in " + path

    return d


def process_exp(matpath, nexpaths, comps=2, binsize=0.1 * pq.s,
                     window=(-4.0 * pq.s, 4.0 * pq.s)):
    mwindow = (int((window[0] + 4.0 * pq.s) / 0.1 * pq.s),
               int((window[1] + 4.1 * pq.s) / 0.1 * pq.s))
    mat = loadmat(matpath)
    d = {}

    # Neural data
    d.update(perievent_neural_mat(mat, mwindow, binsize))

    # PCA
    for seq in ('cc', 'ec'):
        sdf = 'sdfs_%s' % seq
        loads = 'loadings_%s' % seq
        evecs = 'evecs_%s' % seq
        pca_d = pca(d[sdf], comps=comps)
        for k in pca_d.keys():
            d['%s_%s' % (k, seq)] = pca_d[k]

        # Sort sdfs by loadings
        for c in xrange(comps):
            d['%s_%d' % (sdf, c)] = d[sdf][np.argsort(d[loads].T[c])]

    d['n'] = len(d['sdfs_cc'])
    d['t'] = np.linspace(window[0], window[1], d['sdfs_cc'][0].shape[0])

    # NEX files
    d['behaviour'] = []
    for path in nexpaths:
        seg, e = get_seg(path)
        d['behaviour'].append(behaviour(seg, e))
    d['behaviour'].sort(cmp=lambda x,y: cmp(x['trials'], y['trials']))

    # Aggregate behaviour
    d.update({k: [] for k in d['behaviour'][0].keys()})
    for bdict in d['behaviour']:
        for k in bdict.keys():
            d[k].append(bdict[k])
    for k in d['behaviour'][0].keys():
        if k != 'rts':
            d[k] = np.array(d[k], dtype=float)
    medians = [np.median(rt) for rt in d['rts']]
    d['rtstats'] = (np.mean(medians), np.std(medians))
    return d


def aggregate_dint(dints):
    print 'Merging all degraded trajectories'
    d = {'trajectory': []}
    for dint in dints:
        d['trajectory'].append([
            AnalogSignal(
            np.concatenate([t[0] for t in dint['trajectory']]).magnitude,
            sampling_period=1 * pq.ms, units=pq.mV),
         AnalogSignal(
             np.concatenate([t[1] for t in dint['trajectory']]).magnitude,
             sampling_period=1 * pq.ms, units=pq.mV),
         [(dint['trajectory'][-1][-1][-1][0], '')]])
    return d


def aggregate_ctrl(sims):
    d = {k: [] for k in ('correct', 'premature', 'late',
                         'trials', 'rts', 'seeds')}

    orderedsims = sorted(sims,
                         cmp=lambda x,y: cmp(x['trials'], y['trials']))

    for sim in orderedsims:
        for k in d.keys():
            if k == 'seeds':
                d[k].append(sim['id']['seed'])
            else:
                d[k].append(sim[k])

    for k in d.keys():
        if k != 'rts' and k != 'seeds':
            d[k] = np.array(d[k], dtype=float)
    medians = [np.median(rt) for rt in d['rts']]
    d['rtstats'] = (np.mean(medians), np.std(medians))
    return d
