import math
import sys
sys.path.append('trevor/scripts')

import nef
from NEFBuilder import BaseReactionTask, MPFCModel

from ca.nengo.model import SimulationMode


class DIntReactionTask(BaseReactionTask):
    def __init__(self, name, results=None, **kwargs):
        """Possible results: C (correct), P (premature), L (late)"""
        if results is None:
            self.results = ['C', 'C', 'P', 'C', 'L'] * 10
        self.inc = 0.0018

        BaseReactionTask.__init__(self, name, **kwargs)
        self.reset()

    def reset(self, randomize=False):
        self.release_time = None
        self.press = [0.0]
        self.dpress = 0
        self.release = [0.0]
        self.drelease = 0
        BaseReactionTask.reset(self, randomize)

    ###############################
    # State stuff
    def on_lever_press(self):
        """Figure out when to release."""
        result = 'C'
        if self.trial_num < len(self.results):
            result = self.results[self.trial_num]

        self.release_time = self.t

        if result == 'C':
            self.release_time += self.states['foreperiod'].timer_t + 0.1
        elif result == 'P':
            self.release_time += self.states['foreperiod'].timer_t * 0.5
        elif result == 'L':
            self.release_time += self.states['foreperiod'].timer_t + 0.1
            self.release_time += self.states['trigger on'].timer_t
            self.release_time += self.states['response window'].timer_t
        self.start[0] = 0.0

    def tick(self):
        if self.start[0] == 1.0:
            self.dpress = 0.0009
        if self.release_time is not None and self.t >= self.release_time:
            self.drelease = 0.003
            self.release_time = None

        self.lever[0] -= self.press[0] * 0.01
        self.lever[0] += self.release[0] * 0.01
        self.lever[0] = min(1.0, self.lever[0])
        self.lever[0] = max(-1.0, self.lever[0])

        BaseReactionTask.tick(self)

    ###############################
    # Origins
    def origin_press(self):
        toret = self.press
        self.press[0] += self.dpress
        if self.press[0] < 0.0:
            self.press[0] = 0.0
            self.dpress = 0
        elif self.press[0] > 1.0:
            self.press[0] = 1.0
            self.dpress = -0.01
        return toret

    def origin_release(self):
        toret = self.release
        self.release[0] += self.drelease
        if self.release[0] < 0.0:
            self.release[0] = 0.0
            self.drelease = 0
        elif self.release[0] > 1.0:
            self.release[0] = 1.0
            self.drelease = -0.01
        return toret

    ###############################
    # Other
    def experiment_length(self, trials):
        # Go through each of the trials and determine approximate time
        length = 0.0
        for res in self.results[:trials]:
            # Always start with an ITI
            length += self.states['intertrial interval'].timer_t

            # Then start; we have to approximate the time
            # it takes to actually press the lever.
            length += 1.0

            # Then the foreperiod.
            # This is not quite the full length in the premature case,
            # but we'll include the full length just in case.
            length += self.states['foreperiod'].timer_t

            # In the premature case, we have an error,
            # and then we're done.
            if res == 'P':
                length += self.states['error'].timer_t
                continue

            # C and L have a trigger and response window
            length += self.states['trigger on'].timer_t
            length += self.states['response window'].timer_t

            # C has a reward time
            if res == 'C':
                length += self.states['drinking'].timer_t
            # L has an error timeout
            elif res == 'L':
                length += self.states['error'].timer_t

        # We'll add on an ITI at the end for good measure
        length += self.states['intertrial interval'].timer_t

        # Phew! That was an ordeal.
        return length


class DoubleIntegrator(MPFCModel):
    def __init__(self, nperd=200, seed=None):
        self.nperd = nperd
        self.radius = 1.1
        self.intscale = 2

        super(DoubleIntegrator, self).__init__(seed)

    @staticmethod
    def oscillation(t, amp=1.0, freq=0.185, phase=-2.28):
        return amp * math.sin(2 * math.pi * freq * t + phase)

    def oscillator_make(self, net):
        net.make_input('Oscillation', DoubleIntegrator.oscillation)

    def dint_make(self, net):
        common = {'dimensions': 2,
                  'node_factory': self.alif_factory,
                  'radius': self.radius}

        n = self.nperd * 2

        net.make('Delay state', n, noise=1.5, **common)
        net.make('Delaying', n * self.intscale, noise=0.15, **common)
        net.make('Timer', n * self.intscale, **common)

    def _dint_connect(self, net):
        net.connect('RTTask', 'Delaying', origin_name='press',
                    transform=[0.16, 0], pstc=0.01)
        net.connect('RTTask', 'Timer', origin_name='press',
                    transform=[0, -0.01], pstc=0.01)

    def dint_connect(self, net):
        net.connect('RTTask', 'Delaying', origin_name='reward',
                    transform=[-0.01, -0.2], pstc=0.1)
        net.connect('RTTask', 'Timer', origin_name='reward',
                    transform=[0, -0.15], pstc=0.1)
        net.connect('RTTask', 'Delaying', origin_name='lights',
                    transform=[0.06, 0], pstc=0.01)
        net.connect('Delaying', 'Timer',
                    transform=[[0.04, 0], [0, 0]], pstc=0.01)
        net.connect('Oscillation', 'Delaying',
                    transform=[0.02, 0], pstc=0.01)

        net.connect('Delaying', 'Delaying', transform=[1, 0], pstc=0.05,
                    func=self.dintfunc)
        net.connect('Timer', 'Timer', transform=[1, 0], pstc=0.05,
                    func=self.dintfunc)

        net.connect('Delaying', 'Delay state',
                    index_pre=0, index_post=0, pstc=0.01)
        net.connect('Timer', 'Delay state',
                    index_pre=0, index_post=1, pstc=0.01)

    @staticmethod
    def dintfunc(x):
        return [x[0] * (x[1] + 1)]

    def make(self):
        net = nef.Network('Double integrator', seed=self.seed)
        net.add(DIntReactionTask('RTTask'))
        self.oscillator_make(net)
        self.dint_make(net)
        self._dint_connect(net)
        self.dint_connect(net)
        self.net = net
        return self.net

    def log_nodes(self, log):
        log.add("Delay state", tau=0.1)
        log.add_spikes("Delay state")
        log.add_spikes("Delaying")
        log.add_spikes("Timer")
        log.add("RTTask", origin="lever", name="lever_event", tau=0.0)
        log.add("RTTask", origin="reward", name="reward_event", tau=0.0)
        log.add("RTTask", origin="lights", name="lights_event", tau=0.0)
        log.add("RTTask", origin="trigger", name="tone_event", tau=0.0)

    def filename(self, experiment):
        return 'dint-%s-%d' % (experiment, self.seed)

    def run(self):
        if self.net is None:
            self.make()

        rttask = self.net.get('RTTask')

        experiments = {
            'cc': ['C', 'C', 'C'],
            'pc': ['C', 'P', 'C'],
            'lc': ['C', 'L', 'C'],
        }

        for experiment, results in experiments.iteritems():
            rttask.results = results
            fname = self.filename(experiment)
            exp_length = rttask.experiment_length(len(results))
            super(DoubleIntegrator, self).run(fname, exp_length)

if '__nengo_ui__' in globals():
    dint = DoubleIntegrator(nperd=1000, seed=100)
    dint.make()
    dint.view(True)

if '__nengo_cl__' in globals():
    params = {'nperd': 200, 'seed': None}

    print sys.argv

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print "Usage: nengo-cl NEFDoubleIntegrator.py [nperd seed]"
        sys.exit()

    if len(sys.argv) >= 2:
        params['nperd'] = int(sys.argv[1])
    if len(sys.argv) == 3:
        params['seed'] = int(sys.argv[2])

    dint = DoubleIntegrator(**params)
    dint.run()
