from __future__ import absolute_import

import os.path

import numpy as np
import quantities as pq
from scipy import stats

from .sim_io import get_seg
from . import base


def perievent_neural(seg, evts, window, binsize, stnames):
    extend = 0.1 * pq.s
    window_ext = (window[0] - extend, window[1] + extend)
    bins = int((window_ext[1] - window_ext[0]) / binsize)
    trim = int(extend / binsize)
    sts = []

    d = {k: [] for k in ['rates', 'sdfs']}

    for st in seg.spiketrains:
        for stname in stnames:
            if st.name.startswith(stname):
                sts.extend(st.perievent_slices(evts.times, window))
                break

    todelete = []
    for ix, st in enumerate(sts):
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
    return d

def analyze(path, comps=2, binsize=0.1*pq.s, window=(-4.0*pq.s, 4.1*pq.s)):
    seg, e = get_seg(path)
    d = {}
    # fs contents:
    #  dint, exptype, seed
    fname = os.path.splitext(os.path.basename(path))[0]
    fs = fname.split('-')
    d['id'] = {'exp': fs[1], 'seed': fs[2]}
    seq = fs[1].upper()
    d['seq'] = 'cc' if seq == 'CC' else 'ec'

    try:
        evts = base.valid_presses(seg, e, seq)
        evts[0]
    except IndexError:
        print base.merge_all_eas(seg, e.values())
        raise ValueError("No valid presses in " + path)
    evts.times += 0.3 * pq.s  # For delay to muscles

    rand = np.random.RandomState(120)  # To keep analysis deterministic

    for stname in ('dint', 'sint'):
        # Only keep 174 neurons (as in experiment)
        stnames = ['Delay state'] if stname == 'dint' else ['Delaying']
        try:
            d[stname] = perievent_neural(seg, evts, window, binsize, stnames)
        except:
            if stname == 'sint':
                continue
            raise
        keep = rand.choice(d[stname]['rates'].shape[0], 174, replace=False)
        d[stname]['rates'] = d[stname]['rates'][keep]
        d[stname]['sdfs'] = d[stname]['sdfs'][keep]

        # PCA
        d[stname].update(base.pca(d[stname]['sdfs'], comps=comps))

        # Sort sdfs by loadings
        for c in xrange(comps):
            indices = np.argsort(d[stname]['loadings'].T[c])
            d[stname]['sdfs_%d'%c] = d[stname]['sdfs'][indices]

    # Convenience!
    d['n'] = 174
    d['t'] = np.linspace(window[0], window[1], d['dint']['sdfs'][0].shape[0])

    # Trajectories
    # d.update(trajectory(seg, e, seq[0]))
    base.save_pickle(d, 'analyzed/' + fname + '.pkl')
