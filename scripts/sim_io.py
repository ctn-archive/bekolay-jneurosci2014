import os
import numpy as np
import quantities as pq
from neo import Segment, AnalogSignal, SpikeTrain, EventArray, EpochArray
from neo.io.baseio import BaseIO
from neo.io.neuroexplorerio import NeuroExplorerIO
from neo.io.hdf5io import NeoHdf5IO


nengoevents = {
    'lightsoff': "lights_off",
    'press': "lever_off",
    'pumpon': "reward_on",
    'release': "lever_on",
    'toneon': "tone_on",
}

nexevents = {
    'lightsoff': "eventHLIGHTOFF",
    'press': "eventPRESS",
    'pumpon': "eventPUMPON",
    'release': "eventRELEASE",
    'toneon': "eventTONEOFF",
}


class NengoSpikeCSVIo(BaseIO):
    is_readable = True
    is_writable = False

    supported_objects = [Segment, AnalogSignal, SpikeTrain,
                         EventArray, EpochArray]

    readable_objects = [Segment]
    writeable_objects = []

    has_header = False
    is_streameable = False

    read_params = {Segment: []}
    write_params = None

    name = "Nengo"
    extensions = ["csv"]

    mode = 'file'

    def __init__(self, filename=None):
        BaseIO.__init__(self)
        self.filename = filename

    def read_segment(self, lazy=False, cascade=True):
        seg = Segment(name=self.filename[-4])
        seg.file_origin = os.path.basename(self.filename)

        with open(self.filename, 'r') as nf:
            for i, l in enumerate(nf):
                if len(l) <= 2:
                    continue
                spikes = map(float, l[:-1].split(", "))
                seg.spiketrains.append(
                    SpikeTrain(spikes, name=str(i),
                               units=pq.s, t_stop=np.amax(spikes)))
        return seg


class NengoIO(BaseIO):
    is_readable = True
    is_writable = False

    supported_objects = [Segment, AnalogSignal, SpikeTrain,
                         EventArray, EpochArray]

    readable_objects = [Segment]
    writeable_objects = []

    has_header = False
    is_streameable = False

    read_params = {Segment: []}
    write_params = None

    name = "Nengo"
    extensions = ["csv"]

    mode = 'file'

    def __init__(self, filename=None):
        BaseIO.__init__(self)
        self.filename = filename

    def read_segment(self, lazy=False, cascade=True):
        seg = Segment(name=self.filename[-4])
        seg.file_origin = os.path.basename(self.filename)

        # Just one pass through the file! Do it live!
        with open(self.filename, 'r') as nf:
            for i, l in enumerate(nf):
                if i == 0:
                    # First line, header. Set up basic data structures.
                    cols = l[:-1].split(",")
                    data = [[] for _ in xrange(len(cols))]
                    lastdata = [None for _ in xrange(len(cols))]
                    types = []
                    for col in cols:
                        if col == "time":
                            types.append("Time")
                        elif col.endswith("_spikes"):
                            types.append("SpikeTrains")
                        elif col.endswith("_event"):
                            types.append("EventArray")
                        elif col.endswith("_states"):
                            types.append("EpochArray")
                        else:
                            types.append("AnalogSignal")
                    continue

                ll = l[:-1].split(",")

                if i == 1:
                    # Second line. Get number of dimensions for everything.
                    for j, t in enumerate(types):
                        if t == "SpikeTrains" or t == "AnalogSignal":
                            dims = len(ll[j].split(';'))
                            data[j] = [[] for _ in xrange(dims)]
                        elif t == "EventArray":
                            data[j] = [[] for _ in xrange(3)]
                    # No continue, keep going

                time = float(ll[0])
                for j, lll in enumerate(ll):
                    if types[j] == "Time":
                        data[j].append(float(lll))
                    elif types[j] == "AnalogSignal":
                        for k, v in enumerate(lll.split(";")):
                            data[j][k].append(float(v))
                    elif types[j] == "EventArray":
                        curdata = float(lll)
                        if curdata == 1.0 and curdata != lastdata[j]:
                            data[j][0].append(time)
                        elif curdata == 0.0 and curdata != lastdata[j]:
                            data[j][1].append(time)
                        elif curdata == -1.0 and curdata != lastdata[j]:
                            data[j][2].append(time)
                        lastdata[j] = curdata
                    elif types[j] == "SpikeTrains":
                        for k in [k for k, s in
                                  enumerate(lll.split(";")) if s == '1']:
                            data[j][k].append(time)

        t_start = data[0][0]
        t_stop = data[0][-1]
        period = pq.s * (t_stop - t_start) / float(len(data[0]))
        # File closed now, process each column and add to segment
        # for col, typ, dat in izip(cols, types, data):
        for col, typ, dat in zip(cols, types, data):
            if typ == "SpikeTrains":
                for i, times in enumerate(dat):
                    seg.spiketrains.append(
                        SpikeTrain(times, name=col + '_' + str(i),
                                   units=pq.s, t_stop=t_stop,
                                   sampling_rate=1.0 / period))
            elif typ == "AnalogSignal":

                for i, v in enumerate(dat):
                    seg.analogsignals.append(
                        AnalogSignal(v, units=pq.mV, sampling_period=period,
                                     name=col + '_' + str(i)))
            elif typ == "EventArray":
                name = col[:-5] + 'on'
                labels = np.array([name] * len(dat[0]), dtype='S')
                seg.eventarrays.append(
                    EventArray(times=dat[0], labels=labels, channel_name=name))
                name = col[:-5] + 'zero'
                labels = np.array([name] * len(dat[0]), dtype='S')
                seg.eventarrays.append(
                    EventArray(times=dat[1], labels=labels, channel_name=name))
                name = col[:-5] + 'off'
                labels = np.array([name] * len(dat[0]), dtype='S')
                seg.eventarrays.append(
                    EventArray(times=dat[2], labels=labels, channel_name=name))

        return seg


def convert_to_h5(filename):
    if not filename.endswith("csv"):
        print "File not a CSV. Returning."
        return

    print "Reading " + filename + " ...",
    r = NengoIO(filename)
    seg = r.read_segment()
    w = NeoHdf5IO(filename[:-3] + "h5")
    w.save(seg)
    w.close()
    os.remove(filename)
    print "done converting."
    return seg, nengoevents


def get_seg(filename):
    if filename.endswith("h5"):
        r = NeoHdf5IO(filename)
        seg = r.read_segment()
        r.close()
        return seg, nengoevents
    elif filename.endswith("nex"):
        r = NeuroExplorerIO(filename)
        seg = r.read_segment()
        return seg, nexevents
    else:
        print "File not H5 or NEX. Returning."
        return None, None
