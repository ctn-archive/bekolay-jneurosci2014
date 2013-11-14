import os.path
import random

import nef


class BaseReactionTask(nef.FSMNode):
    '''Runs a series of reaction time tasks.

    A trial looks like the following:

                t_foreperiod  t_tone  t_response    t_drink
        |-----|--------------|------|------------|---------|
      Start Press          Tone     Release    Reward

    The subject starts by pressing the lever for `t_foreperiod` seconds.
    At the tone, the subject releases the lever.
    If it was released within `t_reward`, the subject receives reward.
    The subject drinks for `t_drink` seconds.
    '''
    def __init__(self, name, t_foreperiod=1.0, t_timeout=2.0, t_response=0.6,
                 t_tone=0.1, t_reward=2.0, t_iti=5.0):
        self.reset()
        nef.FSMNode.__init__(self, name, 1)

        self.add_state('trial start',
                       enter_f=self.start_trial,
                       exit_f=self.on_lever_press)
        self.add_state('foreperiod',
                       timer_t=t_foreperiod,
                       timer_f=self.after_foreperiod)
        self.add_state('error',
                       timer_t=t_timeout,
                       timer_f=self.finished_trial)
        self.add_state('trigger on',
                       timer_t=t_tone,
                       timer_f=self.turn_off_trigger)
        self.add_state('response window',
                       timer_t=t_response - t_tone,
                       timer_f=self.reward_or_dont)
        self.add_state('drinking',
                       timer_t=t_reward,
                       timer_f=self.finished_trial)
        self.add_state('intertrial interval',
                       timer_t=t_iti,
                       timer_f=lambda: self.transition("trial start"))
        self.reset()

    def reset(self, randomize=False):
        self.trial_num = 0
        self.start = [0.0]
        self.lever = [1.0]
        self.trigger = [0.0]

        self.lights = [0.0]
        self.dlights = 0.0
        self.reward = [0.0]
        self.dreward = 0.0

        if hasattr(self, 'states'):
            self.transition('intertrial interval')
            nef.FSMNode.reset(self, randomize)

    ###############################
    # State functions
    def start_trial(self):
        self.lights[0] = 0.0
        self.dlights = 0.0
        self.start[0] = 1.0

    def on_lever_press(self):
        self.start[0] = 0.0

    def after_foreperiod(self):
        self.trigger[0] = 1.0
        self.transition('trigger on')

    def turn_off_trigger(self):
        self.trigger[0] = 0.0
        self.transition('response window')

    def reward_or_dont(self):
        if self.lever[0] == 1.0:
            self.dreward = 0.01
            self.transition('drinking')
        else:
            self.dlights = -0.01
            self.transition('error')

    def finished_trial(self):
        self.reward[0] = 0.0
        self.lights[0] = 0.0
        self.trial_num += 1
        self.transition('intertrial interval')

    def tick(self):
        if self.state.name == 'trial start' and self.lever[0] == -1.0:
            self.transition('foreperiod')
        elif self.state.name == 'foreperiod' and self.lever[0] == 1.0:
            self.dlights = -0.01
            self.transition('error')

    ###############################
    # Origins
    def origin_start(self):
        return self.start

    def origin_trigger(self):
        return self.trigger

    def origin_lever(self):
        return self.lever

    def origin_lights(self):
        toret = self.lights
        self.lights[0] += self.dlights
        if self.lights[0] < -1.0:
            self.lights[0] = -1.0
            self.dlights = 0
        elif self.lights[0] > 1.0:
            self.lights[0] = 1.0
            self.dlights = 0
        return toret

    def origin_reward(self):
        toret = self.reward
        self.reward[0] += self.dreward
        if self.reward[0] < -1.0:
            self.reward[0] = -1.0
            self.dreward = 0
        elif self.reward[0] > 1.0:
            self.reward[0] = 1.0
            self.dreward = 0
        return toret


class MPFCModel(object):
    def __init__(self, seed=None):
        from ca.nengo.math.impl import IndicatorPDF
        from ca.nengo.model.neuron.impl import ALIFNeuronFactory
        from ca.nengo.model.neuron.impl import LIFNeuronFactory
        maxrate = IndicatorPDF(10, 50)
        intercept = IndicatorPDF(-1, 1)
        tau_rc = 0.02
        tau_ref = 0.001
        self.lif_factory = LIFNeuronFactory(
            tauRC=tau_rc, tauRef=tau_ref,
            maxRate=maxrate, intercept=intercept)
        tau_adapt = 0.01
        inc_adapt = IndicatorPDF(0.001, 0.02)
        self.alif_factory = ALIFNeuronFactory(
            maxRate=maxrate, intercept=intercept, tauRC=tau_rc,
            tauRef=tau_ref, incN=inc_adapt, tauN=tau_adapt)

        # If no seed passed in, we'll generate one
        if seed is None:
            seed = random.randint(0, 1000)  # That's plenty
        self.seed = seed

        self.net = None

    def view(self, iplot=True):
        if self.net is None:
            self.make()
        self.net.add_to_nengo()
        if iplot:
            self.net.view()

    def run(self, name, length):
        log_dir = os.path.expanduser("~/nengo-latest/trevor/data_sim")
        if (os.path.exists('%s/%s.csv' % (log_dir, name)) or
                os.path.exists('%s/%s.h5' % (log_dir, name))):
            print 'This parameter set already run. Stopping.'
            return

        log_node = nef.Log(self.net, "log", dir=log_dir, filename=name)
        self.log_nodes(log_node)
        self.net.network.simulator.run(0, length, 0.001, False)
        self.net.network.removeStepListener(log_node)
