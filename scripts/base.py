try:
    import cPickle as pickle
except ImportError:
    import pickle
import subprocess

import numpy as np
import quantities as pq
from scipy import linalg, stats
import svgutils.transform as sg

from . import files, neo


### Pickle

def load_pickle(pklf):
    with open(pklf, 'rb') as pklpkl:
        ret = pickle.load(pklpkl)
    return ret


def strip_dict(d):
    for k in d:
        d[k] = stripped(d[k])


def strip_iterable(it):
    for ix in xrange(len(it)):
        it[ix] = stripped(it[ix])


def stripped(el):
    if type(el) == dict:
        strip_dict(el)
    elif type(el) == list:
        strip_iterable(el)
    elif hasattr(el, 'units'):
        # print 'stripping',type(el)
        return el.magnitude
    # elif type(el) == tuple:
    #     print 'tuple'
    # else:
    #     print 'not stripped',type(el)
    return el


def save_pickle(d, pklfp):
    strip_dict(d)
    with open(pklfp, 'wb') as pklf:
        pickle.dump(d, pklf, protocol=pickle.HIGHEST_PROTOCOL)


### Analysis

def pca(x, comps=2):
    evecs, evals, _ = linalg.svd(x.T, full_matrices=False)
    evecs = stats.zscore(evecs[:, :comps], ddof=1)
    evals = (evals ** 2) / np.sum(evals ** 2)
    loadings = np.dot(x, evecs)
    return {'evecs': evecs, 'evals': evals, 'loadings': loadings}


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


### Neo

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

def merge_all_eas(seg, channels):
    times = np.array([])
    labels = np.array([], dtype='S')
    for channel in channels:
        ix = map(lambda ea: ea.annotations['channel_name'],
                 seg.eventarrays).index(channel)
        if seg.eventarrays[ix].times.shape == (0, 1):
            continue
        times = np.hstack([times, seg.eventarrays[ix].times])
        labels = np.hstack(
            [labels, np.array([channel] * len(seg.eventarrays[ix].times))])
    sortix = np.argsort(times)
    times = times[sortix] * pq.s
    labels = labels[sortix]
    return neo.EventArray(times, labels)


def valid_presses(seg, e, seq):
    merged = merge_all_eas(seg, e.values())
    merged_ep = (merged.epochs_from(correct_seq(e), name='C') +
                 merged.epochs_from(premature_seq(e), name='P') +
                 merged.epochs_from(late_seq(e), name='L'))
    ep = merged_ep.epochs_from(seq)

    # Everything in seq will have a press,
    # but we only care about the last one.
    return merged.during_epochs(ep).filter_for(
        (e['press'],))[len(seq) - 1::len(seq)]


def get_analogsignal(seg, name):
    ix = map(lambda ansig: ansig.name,
             seg.analogsignals).index(name)
    return seg.analogsignals[ix]


### Plotting

def save_or_show(fig, pklf, suffix, fs=(8, 6), tight=None):
    if tight is None:
        tight = {'pad': 0.8}
    if tight is not None and not tight.has_key('pad'):
        tight['pad'] = 0.8
    pltf = files.svg(pklf, suffix)
    fig.set_size_inches(*fs)
    fig.set_tight_layout(tight)
    fig.savefig(pltf, dpi=1200)
    fig.clear()


### Combining

class RectElement(sg.FigureElement):
    def __init__(self, x, y):
        s = 18
        rect = sg.etree.Element(sg.SVG+"rect",
                                {"x": str(x), "y": str(y - s),
                                 "width": str(s), "height": str(s),
                                 "style":"fill:white;"})
        sg.FigureElement.__init__(self, rect)

def el(char, path, x, y, scale=1, offset=(10, 30)):
    toret = []
    if char is not None:
        toret.append(RectElement(x + offset[0], y + offset[1]))
        toret.append(sg.TextElement(x + offset[0], y + offset[1],
                        char, size=24, weight='bold', font='Arial'))
    if path.endswith(".svg"):
        svg = sg.fromfile(path)
        svg = svg.getroot()
        svg.moveto(str(x), str(y), scale)
        return [svg] + toret


def svgfig(w, h):
    w = str(w)
    h = str(h)
    return sg.SVGFigure(w, h)


def savefig(fig, out):
    fig.save('%s.svg' % out)
    print "Saved %s.svg" % out
    subprocess.call(['inkscape',
                     '--export-pdf=%s.pdf' % out, '%s.svg' % out])
    subprocess.call(['inkscape', '--export-text-to-path',
                     '--export-eps=%s.eps' % out, '%s.svg' % out])
