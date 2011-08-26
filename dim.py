# -*- coding: ISO-8859-1 -*-
u"""
dim
========
.. autoclass:: DimBase



"""
import numpy as np


class ValueArrayShapeInfoMismatchError(Exception):
    pass

def flatten(sequence):
    for item in sequence:
        if isinstance(item, (list, tuple)):
            for subitem in flatten(item):
                yield subitem
        else:
            yield item

class DimBase(object):
    sortprio = 0
    def __init__(self, Name, data=None, unit=None, name=None, outputformat=None):
        if isinstance(Name, DimBase):
            dim_data = Name.data
            dim_name = Name.name
            dim_unit = Name.unit
            dim_outputformat = Name.outputformat
        else:
            dim_data = data
            dim_name = Name
            dim_unit = unit
            if outputformat is None:
                dim_outputformat = "%.16e"
            else:
                dim_outputformat = outputformat

        if data is not None:
            dim_data = data
        if unit is not None:
            dim_unit = unit
        if name is not None:
            dim_name = name
        if outputformat is not None:
            dim_outputformat = outputformat

        if isinstance(dim_data, int):
            dim_data = range(dim_data)

        if hasattr(dim_data, "tolist"):
            dim_data = [dim_data.tolist()]
        self._data = tuple(flatten(dim_data))
        self._name = dim_name
        self._unit = dim_unit
        self._outputformat = dim_outputformat

    @property
    def data(self):
        return np.asarray(self._data)

    @property
    def name(self):
        return self._name

    @property
    def unit(self):
        return self._unit

    @property
    def outputformat(self):
        return self._outputformat

    def fullsize(self):
        return self.data.shape[0]

    def __ValueArray__(self):
        return (self.data, (self,))

    def __cmp__(self, other):
        a = (self.sortprio, self.name, self.__class__, self._data, self._unit, self._outputformat)
        try:
            b = (other.sortprio, other.name, other.__class__, other._data, other._unit, self._outputformat)
        except AttributeError:
            a = self.name
            b = other
        return cmp(a, b)

    def __repr__(self):
        return "%s(%r, shape=%r)"%(self.__class__.__name__,
                                   self.name,
                                   self.data.shape)

    def __hash__(self):
        return hash((self.sortprio, self.name, self.__class__, self._data, self._unit, self._outputformat))

    def __getitem__(self, index):
        if isinstance(index, slice) and (index == slice(None, None, None)):
            return self
        elif isinstance(index, slice):
            a = self.__class__(self.name, self.data[index], unit=self.unit, outputformat=self.outputformat)
            return a
        elif isinstance(index, np.ndarray):
            a = self.__class__(self.name, self.data[index], unit=self.unit, outputformat=self.outputformat)
            return a
        else:
            raise IndexError("Must index with slice")

    def copy(self):
        return self


class DimSweep(DimBase):
    pass

class DimRep(DimBase):
    sortprio = 1


class _DimMatrix(DimBase):
    sortprio = 1000



class DimMatrix_i(_DimMatrix):
    sortprio = 1001

class DimMatrix_j(_DimMatrix):
    sortprio = 1001
