from glob import glob
import os.path
import random
import subprocess

import svgutils.transform as sg

outdir = 'figures'
sc = 1.25
dims = {
    'pc': {'w': 405 / sc, 'h': 540 / sc},
    'traj': {'w': 360 / sc, 'h': 360 / sc},
    'perf': {'w': 360 / sc, 'h': 360 / sc},
    'rts': {'w': 360 / sc, 'h': 360 / sc},
    'signals': {'w': 180 / sc, 'h': 450 / sc},
}


def el(char, path, x, y, w=None, h=None, scale=1, offset=(10, 30)):
    toret = []
    if char is not None:
        toret.append(sg.TextElement(x + offset[0], y + offset[1],
                        char, size=24, weight='bold', font='Arial'))
    if path.endswith(".svg"):
        svg = sg.fromfile(path)
        if w is not None and h is not None:
            svg.set_size((w, h))
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


def fig4():
    print 'fig4'
    out = '%s/fig4' % outdir

    cc = sorted(glob('plots/cc*-?_traj.svg'))
    pc = sorted(glob('plots/pc*-?_traj.svg'))
    lc = sorted(glob('plots/lc*-?_traj.svg'))

    fig = svgfig(dims['traj']['w'] * 4, dims['traj']['h'] * 3)

    l = 'A'
    for i, c_f in enumerate(cc):
        fig.append(el(l, c_f, dims['traj']['w'] * i, 0))
        l = None

    l = 'B'
    for i, p_f in enumerate(pc):
        fig.append(el(l, p_f, dims['traj']['w'] * i, dims['traj']['h']))
        l = None

    l = 'C'
    for i, l_f in enumerate(lc):
        fig.append(el(l, l_f, dims['traj']['w'] * i, dims['traj']['h'] * 2))
        l = None
    savefig(fig, out)


def fig5(cc=None, pc=None, lc=None):
    print 'fig5'
    out = '%s/fig5' % outdir

    if cc is None:
        cc = random.choice(glob('plots/cc-???_traj.svg'))
    if pc is None:
        pc = random.choice(glob('plots/pc-???_traj.svg'))
    if lc is None:
        lc = random.choice(glob('plots/lc-???_traj.svg'))

    fig = svgfig(dims['traj']['w'] * 3, dims['traj']['h'])

    fig.append(el(None, cc, 0, 0))
    fig.append(el(None, pc, dims['traj']['w'], 0))
    fig.append(el(None, lc, dims['traj']['w'] * 2, 0))
    savefig(fig, out)


def fig6(cc=None, pc=None, lc=None):
    print 'fig6'
    out = '%s/fig6' % outdir

    if cc is None:
        cc = random.choice(glob('plots/cc*pc.svg'))
    if pc is None:
        pc = random.choice(glob('plots/pc*pc.svg'))
    if lc is None:
        lc = random.choice(glob('plots/lc*pc.svg'))

    fig = svgfig(dims['pc']['w'] * 3 - 64, dims['pc']['h'])
    fig.append(el('A', cc, 0, 0))
    fig.append(el('B', pc, dims['pc']['w'], 0))
    fig.append(el('C', lc, dims['pc']['w'] * 2 - 36, 0))
    savefig(fig, out)


def fig7():
    print 'fig7'
    out = '%s/fig7' % outdir

    fig = svgfig(dims['perf']['w'] * 3, dims['perf']['h'] * 2)
    fig.append(el(None, 'plots/exp_perf.svg', 0, 0))
    fig.append(el(None, 'plots/sr_perf.svg', dims['perf']['w'], 0))
    fig.append(el(None, 'plots/td_perf.svg', dims['perf']['w'] * 2, 0))
    fig.append(el(None, 'plots/exp_rts.svg', 0, dims['perf']['h']))
    fig.append(el(None, 'plots/sr_rts.svg',
                  dims['perf']['w'], dims['perf']['h']))
    fig.append(el(None, 'plots/td_rts.svg',
                  dims['perf']['w'] * 2, dims['perf']['h']))
    savefig(fig, out)


def fig8(simple=None, late=None, adapt=None, pre=None):
    print 'fig8'
    out = '%s/fig8' % outdir

    if simple is None:
        simple = random.choice(glob('plots/simple-*_c_signals.svg'))
    if late is None:
        late = random.choice(glob('plots/simple-*_l_signals.svg'))
    if adapt is None:
        adapt = random.choice(glob('plots/adaptive-*_c_signals.svg'))
    if pre is None:
        pre = random.choice(glob('plots/adaptive-*_p_signals.svg'))

    w = dims['signals']['w']
    w2 = w * 1.375

    fig = svgfig(w * 2 + w2 * 2, dims['signals']['h'])
    fig.append(el('A', simple, 0, 0))
    fig.append(el('B', late, w2, 0))
    fig.append(el('C', adapt, w2 + w, 0))
    fig.append(el('D', pre, w2 * 2 + w, 0))
    savefig(fig, out)
