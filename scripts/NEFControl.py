import math
import sys
sys.path.append('trevor/scripts')

import nef
from NEFBuilder import BaseReactionTask
from NEFDoubleIntegrator import DoubleIntegrator


class TopDownReactionTask(BaseReactionTask):
    def termination_lever(self, x, pstc=0.01, dimensions=2):
        if x[0] > 0.85:
            self.lever[0] -= x[0] * 0.19  # dim0 = press
        if x[1] > 0.75:
            self.lever[0] += x[1] * 0.19  # dim1 = lift
        self.lever[0] = min(1.0, self.lever[0])
        self.lever[0] = max(-1.0, self.lever[0])


class AdaptiveControl(DoubleIntegrator):
    def __init__(self, control='adaptive', **dintparams):
        self.control = control
        self.delaypops = 9
        super(AdaptiveControl, self).__init__(**dintparams)
        self.radius = 1.2

    def simple_make(self, net):
        net.make('Press/Release', self.nperd * 2, 2,
                 radius=self.radius, node_factory=self.alif_factory)
        [net.make('P/R Delay %d' % i, self.nperd * 2, 2,
                  node_factory=self.alif_factory, radius=self.radius)
         for i in xrange(self.delaypops)]

    def simple_connect(self, net):
        net.connect('RTTask', 'Press/Release', origin_name='start',
                    pstc=0.1, transform=[1, 0])
        net.connect('P/R Delay %d' % (self.delaypops - 1),
                    net.get('RTTask').getTermination('lever'))
        net.connect('RTTask', 'Press/Release', origin_name='trigger',
                    pstc=0.1, transform=[0, 3])

        net.connect('Press/Release', 'P/R Delay 0', pstc=0.01)

        for i in xrange(self.delaypops - 1):
            net.connect('P/R Delay %d' % i, 'P/R Delay %d' % (i + 1),
                        pstc=0.05)

    def adaptive_connect(self, net):
        net.connect('Press/Release', 'Delaying',
                    pstc=0.01, transform=[[0.15, 0], [0, 0]])
        net.connect('Press/Release', 'Timer',
                    pstc=0.01, transform=[[0, 0], [-0.3, 0]])
        net.connect('Delay state', 'Press/Release', pstc=0.01,
                    func=self.releasezone, transform=[0, 1.5])

    def make(self):
        net = nef.Network('Simple reaction time', seed=self.seed)
        net.add(TopDownReactionTask('RTTask'))

        self.oscillator_make(net)
        self.simple_make(net)
        self.simple_connect(net)

        if self.control == 'adaptive':
            self.dint_make(net)
            self.dint_connect(net)
            self.adaptive_connect(net)
        self.net = net
        return self.net

    @staticmethod
    def releasezone(x):
        return (1. / (1 + math.exp(-20 * (x[1] - 0.9))))

    def log_nodes(self, log):
        if self.control == 'adaptive':
            log.add("Delay state", tau=0.1)
        log.add("Press/Release", tau=0.1)
        log.add("RTTask", origin="lever", name="lever_event", tau=0.0)
        log.add("RTTask", origin="reward", name="reward_event", tau=0.0)
        log.add("RTTask", origin="lights", name="lights_event", tau=0.0)
        log.add("RTTask", origin="trigger", name="tone_event", tau=0.0)

    def filename(self):
        return 'ctrl-%s-%d' % (self.control, self.seed)

    def run(self, length):
        if self.net is None:
            self.make()

        fname = self.filename()
        super(DoubleIntegrator, self).run(fname, length)

if '__nengo_ui__' in globals():
    ctrl = AdaptiveControl('adaptive', nperd=200, seed=None)
    ctrl.view(True)

if '__nengo_cl__' in globals():
    params = {'control': 'adaptive', 'nperd': 200, 'seed': None}
    length = 200.

    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print ("Usage: nengo-cl NEFControl.py "
               + "(simple|adaptive) [length nperd seed]")
        sys.exit()

    params['control'] = sys.argv[1]
    if len(sys.argv) >= 3:
        length = float(sys.argv[2])
    if len(sys.argv) >= 4:
        params['nperd'] = int(sys.argv[3])
    if len(sys.argv) == 5:
        params['seed'] = int(sys.argv[4])

    ctrl = AdaptiveControl(**params)
    ctrl.run(length)
