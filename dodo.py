import inspect
import os.path

from doit.action import CmdAction

from scripts import base, expt, files, sim_io
from scripts import ctrl, ctrl_plot, dint, dint_plot, ds, ds_plot


DOIT_CONFIG = {'default_tasks': ['convert'], 'verbosity': 2}


def task_setup_nengo():
    """Make sure scripts will run from Nengo."""
    actions = []
    if not os.path.exists(files.nengo):
        actions.append('mv ' + files.home + '/nengo-* ' + files.nengo)
    if not os.path.exists(files.nengoscripts):
        actions.append('ln -s ' + files.jneuro + ' ' + files.nengoscripts)
    return {'actions': actions, 'targets': [files.nengo, files.nengoscripts]}


def task_sim_dint():
    """Run a double-integrator simulation."""
    return {'actions': [files.nengo_cl + ' ' + files.nef_dint
                        + ' %(nperd)s %(seed)s'],
            'params': [{'name': 'nperd', 'short': 'n', 'default': '600'},
                       {'name': 'seed', 'short': 's', 'default': '933'}]}


def task_sim_ctrl():
    """Run a control circuit simulation."""
    return {'actions': [files.nengo_cl + ' ' + files.nef_ctrl
                        + ' %(control)s %(length)s %(nperd)s %(seed)s'],
            'params': [{'name': 'control', 'short': 'c', 'default': 'adaptive'},
                       {'name': 'length', 'short': 'l', 'default': '200'},
                       {'name': 'nperd', 'short': 'n', 'default': '600'},
                       {'name': 'seed', 'short': 's', 'default': '933'}]}


def task_download_data():
    """Downloads analyzed data from Figshare."""
    def download_data(article_id='715887'):
        import json
        import requests
        r = requests.get("http://api.figshare.com/v1/articles/" + article_id)
        detail = json.loads(r.content)
        for file_info in detail['items'][0]['files']:
            outpath = os.path.join(files.simdata, file_info['name'])
            if os.path.exists(outpath):
                print outpath + " exists. Skipping."
                continue
            with open(outpath, 'wb') as outf:
                print "Downloading " + outpath + "..."
                dl = requests.get(file_info['download_url'])
                outf.write(dl.content)
    return {'actions': [(download_data, ['715887'])]}


def task_convert():
    """Convert CSV files to H5."""
    for csv in files.csv:
        yield {'name': files.h5(csv, True),
               'actions': [(sim_io.convert_to_h5, [csv])],
               'targets': [files.h5(csv)]}


def task_analyze():
    """Analyze H5 files and save results in PKL."""
    def analyze(fname, anl_f, pass_file, *args):
        actionargs = [files.h5(fname)] + list(args) if pass_file else args
        filedeps = [inspect.getsourcefile(anl_f)]
        if pass_file:
            filedeps.append(files.h5(fname))
        return {'name': files.pkl(fname, True),
                'actions': [(anl_f, actionargs)],
                'file_dep': filedeps,
                'targets': [files.pkl(fname)],
                'clean': True}
    yield analyze('expt', expt.analyze, False, files.exp_mat, files.exp_nex)
    for dsf in ('ds_c', 'ds_p', 'ds_l'):
        yield analyze(dsf, ds.analyze, False, dsf[-1])
    for df in files.dintf:
        yield analyze(df, dint.analyze, True)
    for cf in files.ctrlf:
        yield analyze(cf, ctrl.analyze, True)
    for ctrl_type in ('ctrl-simple', 'ctrl-adaptive'):
        ctrl_f = [files.pkl(p) for p in files.ctrlf if ctrl_type + '-' in p]
        yield analyze(ctrl_type, ctrl.aggregate, False, ctrl_f)


def task_plot():
    """Make plots out of the analyzed PKL files."""
    def plot(fname, plt, plt_f, *args):
        return {'name': files.svg(fname, plt, True),
                'actions': [(plt_f, [files.pkl(fname)] + list(args))],
                'file_dep': [files.pkl(fname), inspect.getsourcefile(plt_f)],
                'targets': [files.svg(fname, plt)],
                'clean': True}
    for tr, par in [('c', 'R'), ('l', 'E'), ('c', 'beta'), ('c', 'ics')]:
        yield plot('ds_' + tr, 'param_' + par, ds_plot.param, par)
    for tr in ['adaptive_c', 'adaptive_pc', 'simple']:
        yield plot('ds_c', tr, ds_plot.ctrl_signals, tr)
    yield plot('ds_p', 'adaptive_p', ds_plot.ctrl_signals, 'adaptive_p')
    for dsf in ['ds_c', 'ds_p', 'ds_l']:
        yield plot(dsf, 'simple_sig', ds_plot.env_signals, 'simple')
        yield plot(dsf, 'learn', ds_plot.learning)
    for df in files.dintf:
        yield plot(df, 'dint_pc', dint_plot.pcs, files.pkl('expt'), 'dint')
        yield plot(df, 'sint_pc', dint_plot.pcs, files.pkl('expt'), 'sint')
    for cf in files.ctrlf:
        for st in 'cpl':
            yield plot(cf, st + '_signals', ctrl_plot.signals, st)
            if 'adaptive' in cf:
                yield plot(cf, st + '_traj', ctrl_plot.traj, st)
    for aggf in ('expt', 'ctrl-simple', 'ctrl-adaptive'):
        yield plot(aggf, 'perf', ctrl_plot.performance)
        yield plot(aggf, 'rts', ctrl_plot.reactiontimes)


def task_combine():
    """Combine the SVG plots together into full PDF figures."""
    def combine(fig, cmb_f, *args):
        return {'name': files.pdf(fig, True),
                'actions': [(cmb_f, args)],
                'file_dep': [inspect.getsourcefile(cmb_f)],
                'targets': [files.pdf(fig)]}
    yield combine('fig1', ds_plot.fig1)
    yield combine('fig2', ds_plot.fig2)
    yield combine('fig3', ds_plot.fig3)
    yield combine('fig4', ds_plot.fig4)
    yield combine('fig5', ds_plot.fig5)
    yield combine('fig6', ds_plot.fig6)
    yield combine('fig8', dint_plot.fig7_8, 'dint')
    yield combine('fig9', dint_plot.fig7_8, 'sint')
    yield combine('fig7', ctrl_plot.fig9)
    yield combine('fig10', ctrl_plot.fig10)


def task_paper():
    """Generate the paper with pdflatex."""
    d = os.path.join(files.jneuro, 'paper')
    pdf = CmdAction('pdflatex -interaction=batchmode jneurosci2013.tex', cwd=d)
    bib = CmdAction('bibtex jneurosci2013', cwd=d)
    return {'actions': [pdf, bib, pdf, pdf],
            'targets': ['paper/jneurosci2013.pdf']}


def task_rebuttal():
    """Generate the rebuttal letter with pdflatex."""
    d = os.path.join(files.jneuro, 'paper')
    pdf = CmdAction('pdflatex -interaction=batchmode rebuttal.tex', cwd=d)
    bib = CmdAction('bibtex rebuttal', cwd=d)
    return {'actions': [pdf, bib, pdf, pdf],
            'targets': ['paper/rebuttal.pdf']}
