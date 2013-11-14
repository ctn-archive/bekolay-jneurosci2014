import random

import numpy as np

from . import base


class SimpleRTSim(object):
    def __init__(self, trial, E, R, beta, adaptive=True, dt=0.01):
        self.trial = trial.lower()
        self.dt = dt
        self.E = E
        self.R = R
        self.beta = beta
        self.adaptive = adaptive

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt
        self.dt1 = dt * 3 * 0.5
        self.dt2 = dt * 0.5

    ## Constants

    def cue(self, t):
        if self.trial in 'cl':
            return 1 if t > 2.0 and t <= 2.1 else 0
        return 0

    def lights(self, t):
        if self.trial == 'p':
            return 1 if t > 1.5 and t <= 3.5 else 0
        elif self.trial == 'l':
            return 1 if t > 2.6 and t <= 4.6 else 0
        return 0

    def press(self, t):
        return 1 if t > 0.8 and t <= 1.0 else 0

    def reward(self, t):
        if self.trial == 'c':
            return 1 if t > 2.6 and t <= 4.6 else 0
        return 0

    ## Dynamic

    def d_lever(self, t, X, lever):
        return 10 * (self.release(t, X, lever) - self.press(t))

    def d_X(self, t, X):
        dx1 = (2.5 * self.press(t)
               - self.E * self.lights(t)
               - self.R * self.reward(t) * X[0])
        dx2 = self.beta * X[0] - self.R * self.reward(t) * X[1]
        toret = np.array([dx1, dx2])
        toret[toret * self.dt + X > 1] = 0
        toret[toret * self.dt + X < -1] = 0
        return toret

    def release(self, t, X, lever):
        if lever >= 1.0:
            return 0

        if self.trial == 'c':
            ret = 1 if t > 2.0 and t <= 2.2 else 0
        elif self.trial == 'p':
            ret = 0 if t > 1.3 and t <= 1.5 else 0
        elif self.trial == 'l':
            ret = 1 if t > 2.7 and t <= 2.9 else 0

        if self.adaptive:
            ret += 1. / (1 + np.exp(-20 * (X[1] - 0.9)))

        return ret

    def sim(self, until, X0):
        time = np.arange(0, until+self.dt, self.dt)
        X = np.zeros((time.shape[0], 2))
        X[0] = X[1] = X0

        lever = np.ones_like(time)
        release = np.zeros_like(time)

        for ix in xrange(time.shape[0] - 2):
            # Use two-step Adams-Bashforth; pretty cool
            X[ix+2] = (X[ix+1]
                       + self.dt1 * self.d_X(time[ix+1], X[ix+1])
                       + self.dt2 * self.d_X(time[ix], X[ix]))
            release[ix+2] = self.release(time[ix+2], X[ix+2], lever[ix+1])
            lever[ix+2] = min(1, max(-1,
                lever[ix+1]
                + self.dt1 * self.d_lever(time[ix+1], X[ix+1], lever[ix+1])
                + self.dt2 * self.d_lever(time[ix], X[ix], lever[ix])))
        return X.T, time, lever, release


def analyze(trial, tlen=10, beta0=0.44, R0=2, E0=0.5):
    dt = 0.01
    ics0 = np.array([0., 0.])

    # Cue responding
    s = SimpleRTSim(trial, E0, R0, beta0, adaptive=False, dt=dt)
    X, time, lever, release = s.sim(tlen, ics0)
    d = {'t': time,
         'u': np.asarray([s.press(t) for t in time]),
         'u_c': np.asarray([s.cue(t) for t in time]),
         'u_l': np.asarray([s.lights(t) for t in time]),
         'u_r': np.asarray([s.reward(t) for t in time]),
         'beta': np.linspace(0, 0.5, 6),
         'ics': np.vstack([np.linspace(-1, 1, 9), np.linspace(-1, 1, 9)]).T,
         'R': np.array([0, 0.1, 0.2, 0.3, 0.5, 2]),
         'E': np.linspace(0.0, 0.5, 6),
         'simple': {'x1':X[0], 'x2':X[1], 'lever':lever, 'release':release}}

    # Try many initial conditions
    d['icstrajs'] = {'x1': [], 'x2': []}
    for X0 in d['ics']:
        X, time, _, _ = s.sim(tlen, X0)
        d['icstrajs']['x1'].append(X[0])
        d['icstrajs']['x2'].append(X[1])

    # Adaptive (perfect)
    s = SimpleRTSim(trial, E0, R0, beta0, adaptive=True, dt=dt)
    X, _, lever, release = s.sim(tlen, ics0)
    d['adaptive_c'] = {'x1':X[0], 'x2':X[1], 'lever':lever, 'release':release}

    # Adaptive (premature)
    s = SimpleRTSim(trial, E0, R0, 0.94, adaptive=True, dt=dt)
    X, _, lever, release = s.sim(tlen, ics0)
    d['adaptive_p'] = {'x1':X[0], 'x2':X[1], 'lever':lever, 'release':release}

    # Adaptive (post-error)
    s = SimpleRTSim(trial, E0, R0, beta0, adaptive=True, dt=dt)
    X,  _, lever, release = s.sim(tlen, np.array([-1, -1]))
    d['adaptive_pc'] = {'x1':X[0], 'x2':X[1], 'lever':lever, 'release':release}

    for par in 'R', 'beta', 'E':
        s = SimpleRTSim(trial, E0, R0, 0.8, adaptive=False, dt=dt)
        d[par + 'trajs'] = {'x1': [], 'x2': []}
        for val in d[par]:
            setattr(s, par, val)
            X, _, _, _ = s.sim(tlen, ics0)
            d[par + 'trajs']['x1'].append(X[0])
            d[par + 'trajs']['x2'].append(X[1])

    base.save_pickle(d, 'analyzed/ds_' + trial + '.pkl')
