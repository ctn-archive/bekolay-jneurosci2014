from glob import glob
import os
import os.path
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
font = {'family': 'sans-serif',
        'sans-serif': 'Arial'}
matplotlib.rc('font', **font)
matplotlib.rc('figure', dpi=100)
from IPython import parallel

from scripts.analyze import aggregate_ctrl, aggregate_dint
from scripts.analyze import process_exp, process_sim_ctrl, process_sim_dint
import scripts.combine as c
import scripts.plot as p
from scripts.sim_io import convert_to_h5

unbuffered = os.fdopen(sys.stdout.fileno(), 'w', 0)
#unbuffered = sys.stdout


def download_data():
    import json
    try:
        import requests
    except:
        print ("Requires requests. Please apt-get install python-requests,"
               " or pip install requests.")
        return

    outdir = 'data_sim/'
    article_id = '715887'

    r = requests.get("http://api.figshare.com/v1/articles/" + article_id)
    detail = json.loads(r.content)
    for file_info in detail['items'][0]['files']:
        outpath = os.path.join(outdir, file_info['name'])
        if os.path.exists(outpath):
            print outpath + " exists. Skipping."
            continue
        with open(outpath, 'wb') as outf:
            print "Downloading " + outpath + "..."
            dl = requests.get(file_info['download_url'])
            outf.write(dl.content)

def schedule(lview, func, path, *args, **kwargs):
    if isinstance(path, list):
        for pth in path:
            if not os.path.exists(pth):
                break
        else:
            return
    elif isinstance(path, str):
        if os.path.exists(path):
            return

    if isinstance(path, list):
        res = lview.map(func, *args, **kwargs)
    else:
        res = lview.apply(func, *args, **kwargs)
    return res


def wait_for(jobs, name="Jobs"):
    if len(jobs) == 0:
        return
    unbuffered.write("%s scheduled. Waiting for results..." % name)

    for job in jobs:
        job.wait()
    unbuffered.write("done!\n")

    # Any failures?
    for job in jobs:
        if not job.successful():
            unbuffered.write("Failed. Printing first failure:\n")
            unbuffered.write("--------------------------------------" +
                             "-------------------------------------\n")
            unbuffered.write('%s\n' % job.display_outputs())
            unbuffered.write('%s\n' % job.result())


def filter_failures(jobdict):
    """If the result is None, it failed, so let's just remove it."""
    for k in jobdict.keys():
        if jobdict[k].r is None:
            print "%s failed. Skipping..." % k
            del jobdict[k]


def print_outputs(results):
    for res in results:
        print ("--------------------------------------"
               "--------------------------------------")
        if not res.successful():
            print res.result()
        print res.display_outputs()


if __name__ == '__main__':
    rootdir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(rootdir)

    print ("Options: download_data keep convert plot combine "
           "(dint|ctrl|all) paper")

    # If we're rerunning, we'll delete existing figures
    if 'keep' not in sys.argv:
        if os.path.exists('plots'):
            for f in os.listdir('plots'):
                if f.endswith('svg'):
                    os.unlink(os.path.join('plots', f))

    # Use IPython parallel stuff
    rc = parallel.Client()
    lview = rc.load_balanced_view()
    lview.block = False
    r = []

    if 'download_data' in sys.argv:
        download_data()

    # Convert all CSVs to HDF5
    if 'convert' in sys.argv:
        simfiles = glob('data_sim/*.csv')
        if len(simfiles) > 0:
            r.append(schedule(lview, convert_to_h5, ['__'], simfiles))
        wait_for(r, "CSV conversion")

    # Initialize dictionaries with filename keys
    dint = dict.fromkeys(glob('data_sim/dint-spikes*.h5'))
    dint_direct = dict.fromkeys(glob('data_sim/dint-direct*.h5'))
    sr = dict.fromkeys(glob('data_sim/ctrl-simple*.h5'))
    td = dict.fromkeys(glob('data_sim/ctrl-adaptive*.h5'))
    dint_jobs = (dint, dint_direct)
    ctrl_jobs = (sr, td)
    all_jobs = dint_jobs + ctrl_jobs

    # Save some effort if desired
    if not 'dint' in sys.argv and not 'all' in sys.argv:
        dint, dint_direct = {}, {}
    if not 'ctrl' in sys.argv and not 'all' in sys.argv:
        sr, td = {}, {}

    # Plot all the things (also in parallel)
    if 'plot' in sys.argv:
        exp = schedule(lview, process_exp, None,
                       'data_exp/PEHs_MFC_080312.mat',
                       glob('data_exp/*.nex'))

        for sims in dint_jobs:
            for k in sims.keys():
                sims[k] = schedule(lview, process_sim_dint, None, k)

        for control in ctrl_jobs:
            for k in control.keys():
                control[k] = schedule(lview, process_sim_ctrl, None, k)

        wait_for([exp], "Experimental analysis")
        wait_for([val for d in dint_jobs for val in d.values()],
                 "Double integrator analysis")
        wait_for([val for d in ctrl_jobs for val in d.values()],
                 "Control analysis")

        for jobdict in all_jobs:
            filter_failures(jobdict)

        simple = schedule(lview, aggregate_ctrl, None,
                          [v.r for v in sr.values()])
        adaptive = schedule(lview, aggregate_ctrl, None,
                            [v.r for v in td.values()])
        wait_for([simple, adaptive], "Aggregate analysis")

        r = []

        # Performance / RT plots
        r.append(schedule(lview, p.plot_performance,
                          "exp_perf.svg", exp.r['correct'],
                          exp.r['premature'], exp.r['late'], "exp", "left"))
        r.append(schedule(lview, p.plot_reactiontimes, "exp_rts.svg",
                          exp.r['rts'], exp.r['rtstats'], "exp", "left"))
        print "Experimental RTs: %.3f +/- %.3f ms" % exp.r['rtstats']
        print ("Experimental performance: %.3f"
               % (np.sum(exp.r['correct']) / np.sum(exp.r['trials'])))

        if len(sr) > 0:
            r.append(schedule(lview, p.plot_performance,
                "sr_perf.svg", simple.r['correct'],
                simple.r['premature'], simple.r['late'], "sr", "center"))
            r.append(schedule(lview, p.plot_reactiontimes, "sr_rts.svg",
                simple.r['rts'], simple.r['rtstats'], "sr", "center"))
            print "Cue-responding RTs: %.3f +/- %.3f ms" % simple.r['rtstats']
            print ("Cue-responding performance: %.3f" % (np.sum(
                simple.r['correct']) / np.sum(simple.r['trials'])))
            print simple.r['seeds']

        if len(td) > 0:
            r.append(schedule(lview, p.plot_performance,
                "td_perf.svg", adaptive.r['correct'],
                adaptive.r['premature'], adaptive.r['late'], "td", "right"))
            r.append(schedule(lview, p.plot_reactiontimes, "td_rts.svg",
                adaptive.r['rts'], adaptive.r['rtstats'], "td", "right"))
            print "Adaptive RTs: %.3f +/- %.3f ms" % adaptive.r['rtstats']
            print ("Adaptive performance: %.3f" % (np.sum(
                adaptive.r['correct']) / np.sum(adaptive.r['trials'])))
            print adaptive.r['seeds']

        for k in dint.keys():
            ids = dint[k].r['id']
            name = '-'.join([ids['exp'], ids['seed']])
            if ids['exp'] == 'cc':
                pos = 'left'
            elif ids['exp'] == 'pc':
                pos = 'center'
            elif ids['exp'] == 'lc':
                pos = 'right'

            # Spiking trajectories
            r.append(schedule(lview, p.plot_trajectory, name + "_traj.svg",
                              dint[k].r['trajectory'], name, pos))

            # PC summaries
            r.append(schedule(lview, p.plot_pc_summary, name + "_pc.svg",
                              exp.r, dint[k].r, name, pos))

        # Dynamics breakdown
        for k in dint_direct.keys():
            ids = dint_direct[k].r['id']
            name = '-'.join([ids['exp'], ids['seed']])
            if ids['exp'] == 'cc':
                pos = 'top'
            elif ids['exp'] == 'pc':
                pos = 'middle'
            elif ids['exp'] == 'lc':
                pos = 'bottom'

            for i in xrange(1, len(dint_direct[k].r['trajectory']) + 1):
                pos2 = ' left' if i == 1 else ''
                r.append(schedule(lview, p.plot_dynamics,
                    name + "-%d_traj.svg" % i,
                    dint_direct[k].r['trajectory'][:i],
                    dint_direct[k].r['vf'][i - 1],
                    name + "-%d" % i, pos=pos + pos2))

        # Signals
        for sims in (sr, td):
            for k in sims.keys():
                ids = sims[k].r['id']
                name = '-'.join([ids['ctrl_type'], ids['seed']])

                r.append(schedule(lview, p.plot_signals,
                                  name + "_c_signals.svg",
                                  sims[k].r['c_sig'], name + "_c", pos='left'))
                try:
                    r.append(schedule(lview, p.plot_signals,
                                      name + "_p_signals.svg",
                                      sims[k].r['p_sig'], name + "_p",
                                      pos='right'))
                except:
                    pass
                try:
                    r.append(schedule(lview, p.plot_signals,
                                      name + "_l_signals.svg",
                                      sims[k].r['l_sig'], name + "_l",
                                      pos='right'))
                except:
                    pass

        wait_for(r, "Plotting figures")

    if 'combine' in sys.argv:
        r = []

        r.append(schedule(lview, c.fig4, None))
        r.append(schedule(lview, c.fig5, None, 'plots/cc-933_traj.svg'))
        r.append(schedule(lview, c.fig6, None, 'plots/cc-454_pc.svg'))
        r.append(schedule(lview, c.fig7, None))
        r.append(schedule(lview, c.fig8, None,
                          'plots/simple-420_c_signals.svg',
                          'plots/simple-420_l_signals.svg',
                          'plots/adaptive-35_c_signals.svg',
                          'plots/adaptive-35_p_signals.svg'))

        wait_for(r, "Combining figures")

    if 'paper' in sys.argv:
        os.chdir('paper')
        batch = '-interaction=batchmode'
        tex = 'jneurosci2013.tex'
        subprocess.call(['pdflatex', batch, tex])
        subprocess.call(['bibtex', tex[:-4]])
        subprocess.call(['pdflatex', batch, tex])
        subprocess.call(['pdflatex', batch, tex])
