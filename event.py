import numpy as np
from neuropy.core.datawriter import DataWriter


class Event(DataWriter):
    def __init__(self, times=None, labels=None,**kwargs) -> None:

        # if metadata is not None:
        #     assert isinstance(metadata, dict), "Only dictionary accepted as metadata"
        #     self._metadata: dict = metadata
        # else:
        #     self._metadata: dict = {}

        self.times = times
        self.labels = labels

    @staticmethod
    def from_txtfile(fi,column_names: list = None):
        #send in column_names as ['time','port'] etc. make sure in order
        # assumes first col is times. can have extra cols after that as long as there's a label per col.

        with open(fi, 'r') as f:
            cols = list(zip(*[line.strip().split() for line in f]))

        if len(cols) != len(column_names):
            raise ValueError("Must have equal number of column names and data columns.")
        
     #   d = dict(zip(column_names, cols))# Create a dictionary with the specified column names
        return Event(times=cols[0],labels=column_names,extracols=cols[1:])
    
    # @classmethod
    # def from_dict(cls, d):
    #     return cls(**d)
    
    # # @property
    # def metadata(self):
    #     return self._metadata
    
    # @metadata.setter
    # def metadata(self, d):
    #     """metadata compatibility"""
    #     if d is not None:
    #         assert isinstance(d, dict), "Only dictionary accepted"
    #         self._metadata = self._metadata | d

    def to_dataframe(self):
        pass

    def add_event(self):
        pass

    def remove_event(self):
        pass