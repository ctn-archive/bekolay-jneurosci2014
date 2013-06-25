import math
import sys
sys.path.append('trevor')

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
    def __init__(self, degrade=None, nperd=200,
                 mode=SimulationMode.DEFAULT, seed=None):
        if degrade is None:
            self.degrade = None
        else:
            self._degrade = degrade
            if mode == SimulationMode.DIRECT:
                self.degrade = degrade * 0.025
            elif mode == SimulationMode.DEFAULT:
                self.degrade = degrade * 0.01
        self.nperd = nperd
        self.mode = mode

        self.radius = 1.1
        self.intscale = 2

        super(DoubleIntegrator, self).__init__(seed)

    @staticmethod
    def oscillation(t, amp=1.0, freq=0.185, phase=-2.3):
        return amp * math.sin(2 * math.pi * freq * t + phase)

    def oscillator_make(self, net):
        net.make_input('Oscillation', DoubleIntegrator.oscillation)

    def dint_make(self, net):
        net.make('Delay state', self.nperd * 2, 2, radius=self.radius,
                 node_factory=self.alif_factory)
        net.make('Delaying', self.nperd * 2 * self.intscale, 2, noise=0.1,
                 radius=self.radius, node_factory=self.alif_factory)
        net.make('Timer', self.nperd * 2 * self.intscale, 2, noise=0.1,
                 node_factory=self.alif_factory, radius=self.radius)

    def _dint_connect(self, net):
        net.connect('RTTask', 'Delaying', origin_name='press',
                    transform=[0.14, 0], pstc=0.01)
        net.connect('RTTask', 'Timer', origin_name='press',
                    transform=[0, -0.01], pstc=0.01)

    def dint_connect(self, net):
        net.connect('RTTask', 'Delaying', origin_name='reward',
                    transform=[-0.005, -0.2], pstc=0.01)
        net.connect('RTTask', 'Timer', origin_name='reward',
                    transform=[0, -0.15], pstc=0.01)
        net.connect('RTTask', 'Delaying', origin_name='lights',
                    transform=[0.06, 0], pstc=0.01)
        net.connect('Delaying', 'Timer',
                    transform=[[0.04, 0], [0, 0]], pstc=0.01)
        if self.mode == SimulationMode.DEFAULT:
            net.connect('Oscillation', 'Delaying',
                        transform=[0.02, 0], pstc=0.01)

        if self.mode == SimulationMode.DEFAULT and self.degrade is not None:
            weight_func = DoubleIntegrator.degradeweights(self.degrade)
        else:
            weight_func = None

        net.connect('Delaying', 'Delaying', transform=[1, 0], pstc=0.05,
                    func=DoubleIntegrator.get_control_2d(
                        self.mode, self.degrade, self.radius),
                    weight_func=weight_func)
        net.connect('Timer', 'Timer', transform=[1, 0], pstc=0.05,
                    func=DoubleIntegrator.get_control_2d(
                        self.mode, self.degrade, self.radius),
                    weight_func=weight_func)

        net.connect('Delaying', 'Delay state',
                    index_pre=0, index_post=0, pstc=0.01)
        net.connect('Timer', 'Delay state',
                    index_pre=0, index_post=1, pstc=0.01)

    def make(self):
        net = nef.Network('Double integrator', seed=self.seed)
        net.add(DIntReactionTask('RTTask'))

        self.oscillator_make(net)
        self.dint_make(net)
        self._dint_connect(net)
        self.dint_connect(net)
        net.network.setMode(self.mode)
        self.net = net
        return self.net

    @staticmethod
    def control_4d(x):
        return [x[0] * (x[2] + 1), x[1] * (x[3] + 1)]

    @staticmethod
    def get_control_2d(mode, degrade, rad):
        def direct_control_2d(x):
            d = 0.0 if degrade is None else degrade
            val = x[0] * (x[1] + 1) * (1 - d)
            if val > 0:
                return min(val, rad)
            else:
                return max(val, -rad)

        def spike_control_2d(x):
            return [x[0] * (x[1] + 1)]
        if mode == SimulationMode.DIRECT:
            return direct_control_2d
        elif mode == SimulationMode.DEFAULT:
            return spike_control_2d

    @staticmethod
    def degradeweights(degrade):
        def degweights(w):
            for i in xrange(len(w)):
                for j in xrange(len(w[0])):
                    w[i][j] *= (1 - degrade)
            return w
        return degweights

    def log_nodes(self, log):
        log.add("Delay state", tau=0.1)
        if self.mode == SimulationMode.DEFAULT:
            log.add_spikes("Delay state")
        log.add("RTTask", origin="lever", name="lever_event", tau=0.0)
        log.add("RTTask", origin="reward", name="reward_event", tau=0.0)
        log.add("RTTask", origin="lights", name="lights_event", tau=0.0)
        log.add("RTTask", origin="trigger", name="tone_event", tau=0.0)

    def filename(self, experiment):
        if self.mode == SimulationMode.DEFAULT:
            mode = 'spikes'
        elif self.mode == SimulationMode.DIRECT:
            mode = 'direct'

        if self.degrade is None:
            return  'dint-%s-%s-%d' % (
                mode, experiment, self.seed)
        else:
            return 'dint-deg%d-%s-%s-%d' % (
                self._degrade, mode, experiment, self.seed)

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
    dint = DoubleIntegrator(degrade=None, mode=SimulationMode.DEFAULT)
    dint.make()
    dint.view(True)

if '__nengo_cl__' in globals():
    params = {
        'mode': SimulationMode.DEFAULT,
        'nperd': 200,
        'degrade': None,
        'seed': None,
    }

    if len(sys.argv) < 2:
        print ("Usage: nengo-cl NEFDoubleIntegrator.py "
               + "(direct|spikes) [degrade nperd seed]")
        sys.exit()

    if sys.argv[1] == 'direct':
        params['mode'] = SimulationMode.DIRECT
    if len(sys.argv) >= 3:
        try:
            params['degrade'] = int(sys.argv[2])
        except:
            pass
    if len(sys.argv) >= 4:
        params['nperd'] = int(sys.argv[3])
    if len(sys.argv) == 5:
        params['seed'] = int(sys.argv[4])
    if len(sys.argv) > 5:
        print "Too many arguments. Ignoring extras..."

    dint = DoubleIntegrator(**params)
    dint.run()
