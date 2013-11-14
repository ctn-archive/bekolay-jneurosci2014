"""
My tweaks to python-neo
"""
from __future__ import absolute_import

import numpy as np
import quantities as pq
import scipy.signal

from neo import Segment, AnalogSignal, SpikeTrain, EventArray, EpochArray, Event
from neo.io.baseio import BaseIO
from neo.io.neuroexplorerio import NeuroExplorerIO
from neo.io.hdf5io import NeoHdf5IO


### EpochArray

def epa__eq__(self, other):
    return (np.all(self.times == other.times) and
            np.all(self.durations == other.durations) and
            np.all(self.labels == other.labels))
EpochArray.__eq__ = epa__eq__

def epa__repr__(self):
    return "<EpochArray: %s>" % ", ".join(
        '%s@%s:%s' % item for item in zip(self.labels, self.times,
                                          self.times + self.durations))
EpochArray.__repr__ = epa__repr__

def epa__len__(self):
    return len(self.times)
EpochArray.__len__ = epa__len__

def epa__add__(self, other):
    if not isinstance(other, EpochArray):
        return NotImplemented
    t, d, l = EpochArray._merge_and_sort_epochs(self, other)
    origin = (self.file_origin if
              self.file_origin == other.file_origin else None)
    if isinstance(self.name, str) and isinstance(other.name, str):
        name = self.name + '+' + other.name
    else:
        name = None
    return EpochArray(times=t, durations=d, labels=l,
                      name=name, file_origin=origin)
EpochArray.__add__ = epa__add__

def epa__iadd__(self, other):
    t, d, l = EpochArray._merge_and_sort_epochs(self, other)
    self.times = t
    self.durations = d
    self.labels = l
    return self
EpochArray.__iadd__ = epa__iadd__

def epa_merge_and_sort_epochs(epochs1, epochs2):
    # Rescale times/durations to epochs1's
    if epochs1.times.units != epochs2.times.units:
        epochs2.times.units = epochs1.times.units
    if epochs1.durations.units != epochs2.durations.units:
        epochs2.durations.units = epochs1.durations.units

    merged = []
    merged.extend(zip(epochs1.times, epochs1.durations, epochs1.labels))
    merged.extend(zip(epochs2.times, epochs2.durations, epochs2.labels))
    merged.sort()
    return (map(lambda m: m[0], merged) * epochs1.times.units,
            map(lambda m: m[1], merged) * epochs1.durations.units,
            np.array(map(lambda m: m[2], merged), dtype='S'))
EpochArray._merge_and_sort_epochs = staticmethod(epa_merge_and_sort_epochs)

def epa_filter_for(self, seq, copy=True):
    """Filter the EpochArray so that only the seq `seq` remains."""
    if len(self.times) == 0:
        if copy:
            return EpochArray()
        return self
    seqstarts = np.ones(self.times.shape, dtype=bool)
    for label in reversed(seq):
        seqstarts = np.logical_and(np.roll(seqstarts, -1),
                                   self.labels == label)
    for _ in xrange(len(seq) - 1):
        seqstarts = np.logical_or(seqstarts,
                                  np.roll(seqstarts, 1))

    if copy:
        return EpochArray(self.times[seqstarts],
                          self.durations[seqstarts],
                          self.labels[seqstarts])

    self.times = self.times[seqstarts]
    self.durations = self.durations[seqstarts]
    self.labels = self.labels[seqstarts]
    return self
EpochArray.filter_for = epa_filter_for

def epa_epochs_from(self, seq, name=None):
    """Find the epochs from seq[0] to seq[-1]."""
    if len(self.times) == 0:
        return EpochArray()

    if name is None:
        name = seq[0]

    seqstarts = np.ones(self.times.shape, dtype=bool)
    for label in reversed(seq):
        seqstarts = np.logical_and(np.roll(seqstarts, -1),
                                   self.labels == label)
    seqends = np.roll(seqstarts, len(seq) - 1)

    return EpochArray(times=self.times[seqstarts],
                      durations=self.times[seqends] + self.durations[seqends]
                      - self.times[seqstarts],
                      labels=np.array([name] * len(self.times[seqstarts]),
                                      dtype='S'))
EpochArray.epochs_from = epa_epochs_from

### EventArray

def eva__eq__(self, other):
    return (np.all(self.times == other.times) and
            np.all(self.labels == other.labels))
EventArray.__eq__ = eva__eq__

def eva__len__(self):
    return len(self.times)
EventArray.__len__ = eva__len__

def eva__add__(self, other):
    if not isinstance(other, EventArray):
        return NotImplemented
    t, l = EventArray._merge_and_sort_events(self, other)
    origin = (self.file_origin if
              self.file_origin == other.file_origin else None)
    if isinstance(self.name, str) and isinstance(other.name, str):
        name = self.name + '+' + other.name
    else:
        name = None
    return EventArray(times=t, labels=l,
                      name=name, file_origin=origin)
EventArray.__add__ = eva__add__

def eva__iadd__(self, other):
    t, l = EventArray._merge_and_sort_events(self, other)
    self.times = t
    self.labels = l
    return self
EventArray.__iadd__ = eva__iadd__

def eva__getitem__(self, key):
    return EventArray(self.times[key], self.labels[key])
EventArray.__getitem__ = eva__getitem__

def eva__setitem__(self, key, value):
    if not isinstance(value, Event) and not isinstance(value, tuple):
        raise TypeError("Can only set EventArray with Event or tuple.")

    if isinstance(value, Event):
        self.times[key] = value.time
        self.labels[key] = value.label
    elif isinstance(value, tuple):
        self.times[key] = value[0]
        self.times[key] = value[1]
EventArray.__setitem__ = eva__setitem__

def eva_merge_and_sort_events(events1, events2):
    # Rescale times/durations to epochs1's
    if events1.times.units != events2.times.units:
        events2.times.units = events1.times.units

    merged = []
    merged.extend(zip(events1.times, events1.labels))
    merged.extend(zip(events2.times, events2.labels))
    merged.sort()
    return (map(lambda m: m[0], merged) * events1.times.units,
            np.array(map(lambda m: m[1], merged), dtype='S'))
EventArray._merge_and_sort_events = staticmethod(eva_merge_and_sort_events)

def eva_filter_for(self, seq, copy=True):
    """Filter the EventArray so that only the seq `seq` remains."""
    if len(self.times) == 0:
        if copy:
            return EventArray()
        return self
    seqstarts = np.ones(self.times.shape, dtype=bool)
    for label in reversed(seq):
        seqstarts = np.logical_and(np.roll(seqstarts, -1),
                                   self.labels == label)
    for _ in xrange(len(seq) - 1):
        seqstarts = np.logical_or(seqstarts,
                                  np.roll(seqstarts, 1))

    if copy:
        return EventArray(self.times[seqstarts],
                          self.labels[seqstarts])

    self.times = self.times[seqstarts]
    self.labels = self.labels[seqstarts]
    return self
EventArray.filter_for = eva_filter_for

def eva_epochs_from(self, seq, name=None):
    """Find the epochs from seq[0] to seq[-1]."""
    if len(self.times) == 0:
        return EpochArray()

    if name is None:
        name = seq[0]

    seqstarts = np.ones(self.times.shape, dtype=bool)
    for label in reversed(seq):
        seqstarts = np.logical_and(np.roll(seqstarts, -1),
                                   self.labels == label)
    seqends = np.roll(seqstarts, len(seq) - 1)

    return EpochArray(times=self.times[seqstarts],
                      durations=self.times[seqends] - self.times[seqstarts],
                      labels=np.array([name] * len(self.times[seqstarts]),
                                      dtype='S'))
EventArray.epochs_from = eva_epochs_from

def eva_time_slice(self, t_start, t_stop):
    if len(self) == 0:
        return EventArray()

    iw = np.where(self.times >= t_start)
    i = self.times.shape[0] if iw[0].shape[0] == 0 else iw[0][0]
    jw = np.where(self.times > t_stop)
    j = self.times.shape[0] if jw[0].shape[0] == 0 else jw[0][0]

    return EventArray(self.times[i:j], self.labels[i:j])
EventArray.time_slice = eva_time_slice

def eva_during_epochs(self, epoch_array):
    merged = EventArray()
    for t, d in zip(epoch_array.times, epoch_array.durations):
        merged += self.time_slice(t, t + d)
    return merged
EventArray.during_epochs = eva_during_epochs

### SpikeTrain additions

def st_perievent_slices(self, times, window, align=True):
    slices = []
    for ev_t in times:
        start = ev_t + window[0]
        stop = ev_t + window[1]
        sliced = self.time_slice(start, stop)
        sliced.t_start = start
        sliced.t_stop = stop
        if align:
            sliced = sliced.align(window[0])
        slices.append(sliced)
    return slices
SpikeTrain.perievent_slices = st_perievent_slices

def st_align(self,t_start):
    """
    Shifts the current spike train to a different t_start.
    The duration remains the same, this is just a shift.

    """
    diff = self.t_start - t_start
    new_st = self[:] - diff
    new_st.t_start = t_start
    new_st.t_stop = self.t_stop - diff
    return new_st
SpikeTrain.align = st_align

def st_gaussian_sdf(self, sigma=0.015 * pq.s,
                    binsize=None, bins=None, denoise=True):
    # Note: there are edge artifacts here! You should always
    # give a longer section than you actually need to avoid artifacts.
    if bins is None and binsize is None:
        binsize = self.sampling_period
        binsize.units = self.duration.units
        bins = int(self.duration / binsize)
    elif bins is None:
        binsize.units = self.duration.units
        bins = int(self.duration / binsize)
    elif binsize is None:
        binsize = self.duration / bins
    else:
        warnings.warn("Pass bins or binsize, not both.")

    if isinstance(bins, int):
        bins = np.linspace(self.t_start, self.t_stop, bins + 1)

    sdf, _ = np.histogram(self, bins=bins)

    # Put gaussian in terms of bins
    sigma.units = binsize.units
    sigma = float(sigma / binsize)
    gauss = scipy.signal.gaussian(int(sigma * 7), sigma)  # +/- 3 stddevs
    gauss /= np.sum(gauss)  # normalize to area 1 (important!)
    gauss /= binsize  # then scale to size of bins
    sdf = np.convolve(sdf, gauss, mode='same')

    if denoise:
        sdf = (np.roll(sdf, -1) + sdf + np.roll(sdf, 1)) / 3.0

    return sdf
SpikeTrain.gaussian_sdf = st_gaussian_sdf
