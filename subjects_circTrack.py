from pathlib import Path
import os
from collections import defaultdict
# from neuropy.io.neuroscopeio import NeuroscopeIO
import neuropy.io.neuroscopeio as neuroscopeio
# from neuropy.io.binarysignalio import BinarysignalIO
import neuropy.io.binarysignalio as binarysignalio
import neuropy.core as core

from WahlbergRepo.event import Event

class ProcessData:
    def __init__(self, basepath=os.getcwd()):
        basepath = Path(basepath)
        self.basepath = basepath
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found fewer/more than one .xml file"

        fp = xml_files[0].with_suffix("")
        self.filePrefix = fp     

        self.probegroup = core.ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))

        # self.recinfo = NeuroscopeIO(xml_files[0])
        self.recinfo = neuroscopeio.NeuroscopeIO(xml_files[0])  

        # txt_files = sorted(basepath.rglob("*.txt"))
        # port_timestamps_file = []
        # block_transitions_file = []
        # for f in txt_files:
        #     if f.match('portTimestamps*'):
        #         port_timestamps_file.append(f)
        #     elif f.match('portTransitions*'):
        #         block_transitions_file.append(f)
        #     else:
        #         continue

        # assert len(port_timestamps_file) == 1, "Found fewer/more than one portTimestamps file"
        # assert len(block_transitions_file) == 1, "Found fewer/more than one portTransitions file"

        #change this to load these after they've been created in the DataAlignment notebook.
      # self.port_timestamps = Event.from_txtfile(port_timestamps_file[0],column_names=['times','port'])
      #  self.block_timestamps = Event.from_txtfile(block_transitions_file[0])

        eegfiles = sorted(basepath.glob("*.eeg"))
        try:
            assert len(eegfiles) == 1, "Fewer/more than one .eeg file detected"
            self.eegfile = binarysignalio.BinarysignalIO(
                eegfiles[0],
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.eeg_sampling_rate,
            )
        except AssertionError:
            print("Fewer/more than one .eeg file detected, no eeg file loaded")
        try:
            self.datfile = binarysignalio.BinarysignalIO(
                eegfiles[0].with_suffix(".dat"),
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.dat_sampling_rate,
            )
        except (FileNotFoundError, IndexError):
            print("No dat file found, not loading")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})"
