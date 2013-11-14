from glob import glob
import os.path


## Scripts
home = os.path.expanduser("~")
scripts = os.path.dirname(os.path.realpath(__file__))
jneuro = os.path.realpath(os.path.join(scripts, '..'))
data_sim = os.path.join(jneuro, 'data_sim')

## Nengo
nengo = os.path.join(home, 'nengo-latest')
nengo_cl = os.path.join(nengo, 'nengo-cl')
nengoscripts = os.path.join(nengo, 'trevor')
nef_dint = os.path.join(nengoscripts, 'scripts', 'NEFDoubleIntegrator.py')
nef_ctrl = os.path.join(nengoscripts, 'scripts', 'NEFControl.py')

## Convert
csv = glob(data_sim + '/*.csv')
def h5(csv, basename=False):
    if basename:
        return os.path.basename(os.path.splitext(csv)[0] + '.h5')
    return os.path.splitext(csv)[0] + '.h5'

## Analysis
exp_mat = os.path.join(jneuro, 'data_exp', 'PEHs_MFC_080312.mat')
exp_nex = glob(os.path.join(jneuro, 'data_exp', '*.nex'))
dintf = [os.path.splitext(f)[0] for f in glob(data_sim + '/dint-*')]
ctrlf = [os.path.splitext(f)[0] for f in glob(data_sim + '/ctrl-*')]
def pkl(h5, basename=False):
    bname = os.path.splitext(os.path.basename(h5))[0]
    if basename:
        return bname + '.pkl'
    return os.path.join(jneuro, 'analyzed', bname + '.pkl')


## Plot
def svg(pkl, suffix, basename=False):
    bname = os.path.splitext(os.path.basename(pkl))[0]
    if basename:
        return bname + '_' + suffix + '.svg'
    return os.path.join(jneuro, 'plots', bname + '_' + suffix + '.svg')


## Combine
def pdf(fname, basename=False):
    toret = os.path.join(jneuro, 'figures', fname)
    if basename:
        return os.path.basename(toret)
    return toret


## Diagrams
def diag(fname):
    return os.path.join(jneuro, 'diagrams', fname + '.svg')
