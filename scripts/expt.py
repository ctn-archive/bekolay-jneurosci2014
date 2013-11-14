import numpy as np
import quantities as pq
from scipy import signal, stats
from scipy.io import loadmat

from .sim_io import get_seg
from . import base


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
        d[sdf] = stats.zscore(np.vstack(d[sdf]), axis=1)[:, window[0]:window[1]]
    return d


def analyze(matpath, nexpaths, comps=2, binsize=0.1 * pq.s, window=(-4.0 * pq.s, 4.0 * pq.s)):
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
        pca_d = base.pca(d[sdf], comps=comps)
        pca_d['evecs'].T[0] *= -1
        pca_d['loadings'].T[0] *= -1
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
        d['behaviour'].append(base.behaviour(seg, e))
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
    print "expt RTs: %.3f +/- %.3f ms" % d['rtstats']
    print "expt perf: %.3f" % (np.sum(d['correct']) / np.sum(d['trials']))
    base.save_pickle(d, 'analyzed/expt.pkl')
