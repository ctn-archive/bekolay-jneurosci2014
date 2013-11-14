from __future__ import absolute_import, division

import numpy as np
import quantities as pq

from .sim_io import get_seg
from . import base, files


def perievent_signal(analog, evt, window):
    t = analog.times
    ix = np.logical_and(t > (evt + window[0]), t <= (evt + window[1]))
    return [analog[ix].times, analog[ix]]


def decoded_signals(seg, e, trial='C', with_x=True, window=(-1.0, 3.0) * pq.s):
    evts = base.merge_all_eas(seg, e.values())
    press_evts = base.valid_presses(seg, e, (trial,))

    # Choose the 5th press event, if more than 5
    ix = 4 if len(press_evts.times) >= 5 else -1
    press = press_evts.times[ix]

    if trial == 'C':
        seq = base.correct_seq(e)
    elif trial == 'P':
        seq = base.premature_seq(e)
    elif trial == 'L':
        seq = base.late_seq(e)
    s_evts = [evts[evts.times >= press].filter_for((evt,)).times[0]
              for evt in seq]
    s_evts = [-window[0] + (evt - press) for evt in s_evts]

    evt_t = [-window[0]]
    evt_l = ['$t_p$']

    if trial == 'C' or trial == 'L':
        evt_t.append(s_evts[1])
        evt_l.append('$t_c$')
    elif trial == 'P':
        evt_t.append(s_evts[1])
        evt_l.append('$t_r$')

    if trial == 'L' or trial == 'P':
        evt_t.append(s_evts[2])
        evt_l.append('$t_{TO}$')
    elif trial == 'C':
        evt_t.extend(s_evts[2:4])
        evt_l.extend(['$t_r$', '$t_{Rw}$'])

    signals = []

    # Get all signals
    signals.append(["$u$"] + perievent_signal(
        base.get_analogsignal(seg, 'Press/Release_0'), press, window))
    signals.append(["$u_r$"] + perievent_signal(
        base.get_analogsignal(seg, 'Press/Release_1'), press, window))
    if with_x:
        signals.append(["$x_1$"] + perievent_signal(
            base.get_analogsignal(seg, 'Delay state_0'), press, window))
        signals.append(["$x_2$"] + perievent_signal(
            base.get_analogsignal(seg, 'Delay state_1'), press, window))
    return {'s_ticks': evt_t, 's_labels': evt_l, 'signals': signals}


def analyze(path):
    seg, e = get_seg(path)
    d = {}

    try:
        base.valid_presses(seg, e, ('C'))[0]
    except IndexError:
        raise ValueError("No valid presses in " + path)

    # Behavioural data
    d.update(base.behaviour(seg, e))
    print files.pkl(path, True), "median RT: %.3f ms" % np.median(d['rts'])
    print files.pkl(path, True), "perf: %.3f" % (d['correct'] / d['trials'])
    print files.pkl(path, True), "trials: %d" % d['trials']

    # Signals
    with_x = 'adaptive' in path
    d['c_sig'] = decoded_signals(seg, e, with_x=with_x)

    try:
        d['p_sig'] = decoded_signals(seg, e, 'P', with_x=with_x)
    except:
        pass

    try:
        d['l_sig'] = decoded_signals(seg, e, 'L', with_x=with_x)
    except:
        pass

    base.save_pickle(d, files.pkl(path))


def aggregate(paths):
    d = {k: [] for k in ('correct', 'premature', 'late', 'trials', 'rts')}

    orderedsims = sorted([base.load_pickle(path) for path in paths],
                         cmp=lambda x,y: cmp(x['trials'], y['trials']))

    for sim in orderedsims:
        for k in d.keys():
            d[k].append(sim[k])
    for k in d.keys():
        if k != 'rts' and k != 'seeds':
            d[k] = np.array(d[k], dtype=np.float64)
    medians = [np.median(rt) for rt in d['rts']]
    d['rtstats'] = (np.mean(medians), np.std(medians))

    if 'simple' in paths[0]:
        pklfp = files.pkl('ctrl-simple')
    elif 'adaptive' in paths[0]:
        pklfp = files.pkl('ctrl-adaptive')
    print pklfp, "RTs: %.3f +/- %.3f ms" % d['rtstats']
    print pklfp, "perf: %.3f" % (np.sum(d['correct']) / np.sum(d['trials']))
    base.save_pickle(d, pklfp)
