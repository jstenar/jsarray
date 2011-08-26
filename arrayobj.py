# -*- coding: ISO-8859-1 -*-
u"""
arrayobj
========

.. autofunction:: change_shape


"""

import pdb, sys, logging
import numpy as np
from numpy import ndarray, array, linspace, ones, arange, newaxis, empty, zeros,\
                  asarray
import numpy.lib.stride_tricks as np_stride_tricks

from dim import DimBase, DimSweep, DimRep, DimMatrix_i, DimMatrix_j

from info import Info, InfoList, replace_dim


class DimensionMismatchError(IndexError):
    pass

class ValueArrayShapeInfoMismatchError(Exception):
    pass
class ValueArrayError(Exception):
    pass

        
def as_strided(x, shape=None, strides=None, offset=0):
    u"""Lågnivå rutin för att omforma en array genom att ange *shape*, *strides
       och *offset*. *shape* är lista som anger hur många element man har i
       varje dimension. *strides* är lista som anger hur långt det är i minnet
       mellan element längs denna dimension. *offset* anger offset från
       startposition i minnesbufferten för första elementet i arrayen dvs när
       alla index är noll.

       Använd endast om du verkligen förstår hur arrayer lagras i minnet. Mer
       information om detta ämne finns i numpys dokumentation.

    """
    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)
    if offset:
        addr, boolflag = interface['data']
        addr += offset
        interface['data'] = addr, boolflag
    return np.asarray(np_stride_tricks.DummyArray(interface, base=x))


def info_union(*a):
    newinfo = InfoList(a[0].info)
    for B in a[1:]:
        for dim in B.info:
            if dim not in newinfo:
                newinfo.append(dim)
    newinfo.sort(key=lambda x: x.sortprio)
    return Info(newinfo)

def change_shape(x, newinfo):
    u"""Try to change shape of *x* to use newinfo.
       Bygger om en array så att den matchar alla Axis objekt i *newinfo*
       listan. Nya dimensioner får längden 1 dvs. understått repetion av
       arrayen i den dimension.

       Om newinfo innehåller någon ComplexAxis så gör vi först om *x* med
       *make_complex_array*.

    """
    newself = x.view()
    newselfshape = []
    neworder = []
    selfinfo = x.info
    for dim in newinfo:
        if dim in x.info:
            newselfshape.append(x.shape[selfinfo.matching_index(dim)])
            neworder.append(selfinfo.matching_index(dim))
        else:
            newselfshape.append(1)
    newself = newself.transpose(*neworder)
    newself.shape = tuple(newselfshape)
    newself.info = Info(newinfo)
    return newself


def make_same_info_list(a):
    newinfo = info_union(*a)
    return [change_shape(x, newinfo) for x in a]


def make_same_info(A, B):
    u"""Anropas med lista med *Arrays*. Returnerar arrayer som har samma *info*
    dvs vi har anropat change_shape med en *newinfo* som innehåller unionen
    av de Axis objekt som finns i *info* av *Arrayerna*.

    """
    if not isinstance(B, _ValueArray):
        B = A.__class__(B.view(), A.info, copy=False)
    return make_same_info_list((A, B))

def remove_tail(x):
    u"""Collapse all dimensions except first and matrix dimensions"""
    
    a = np.array(x)
    shape = x.shape
    if ismatrix(x):
        newshape = (shape[0], np.multiply.reduce(shape[1:-2]))
        newshape = newshape + shape[-2:]
    else:        
        newshape = (shape[0], np.multiply.reduce(shape[1:]))
    outa = a.reshape(newshape)
    if isinstance(x, ValueArray):
        dim = DimRep("Tail", np.arange(outa.shape[1]))
        info = (x.info[0], dim)
        if ismatrix(x):
            info = info + x.info[-2:]
        outa = ValueArray(outa, info=info, unit=x.unit)
    return outa

def remove_rep(data):
    u"""Collapse all DimRep dimensions to single dimension
    """
    order = [(i, dim) for i,dim in enumerate(data.info) if isinstance(dim, DimRep)]
    if order:
        shapes = [data.shape[i]  for i, _ in order]
        repsize = np.multiply.reduce(shapes)
        a = np.array(data)
        a.shape = data.shape[:order[0][0]] + (repsize,) + data.shape[order[-1][0] + 1:]
        info = data.info[:order[0][0]] + (DimRep("AllReps", repsize),) + data.info[order[-1][0] + 1:]
    else:
        info = data.info
        a = data
    return ValueArray(a, info=info, copy=False)


def ismatrix(a):
    i = False
    j = False
    if isinstance(a, _ValueArray):
        for dim in a.info:
            if isinstance(dim, DimMatrix_i):
                i = True
            elif isinstance(dim, DimMatrix_j):
                j = True
            else:
                pass
            if i and j:
                return True
    return False

def check_instance(func):
    def a(self, other):
        try:
            if self.__array_priority__ < other.__array_priority__:
                return NotImplemented
        except AttributeError:
            pass
        a, b = make_same_info(self, ValueArray(other))
        return func(a, b)
    return a


class _ValueArray(ndarray):
    u"""Basklass som ej skall användas direkt
    """
    default_dim = (DimSweep("freq", array(0)), DimRep("rep", array(0)))
    def __new__(subtype, data, info=None, dtype=None, copy=True, order=None, subok=False,
                ndmin=0, unit=None, outputformat=None):

        # Make sure we are working with an array, and copy the data
        # if requested
        subarr = np.array(data, dtype=dtype, copy=copy,
                             order=order, subok=subok,
                             ndmin=ndmin)

        # Transform 'subarr' from an ndarray to our new subclass.
        subarr = subarr.view(subtype)

        # Use the specified 'info' parameter if given
        if info is None:
            if hasattr(data, 'info'):
                info = tuple(data.info)
            elif subarr.ndim == 0:
                info = tuple()
            elif len(subarr.shape) <= len(subtype.default_dim):
                info = tuple(x.__class__(x.name, range(size))
                                    for (x, size) in zip(subtype.default_dim,
                                                         subarr.shape))
            else:
                msg = ("On creation of %s *info* "
                       "must be specified"%subtype.__name__)
                raise DimensionMismatchError(msg)

        subarr.info = Info(info)
        #Check to see that info matches shape
        subarr.verify_dimension()
        if outputformat is not None:
            subarr.outputformat = outputformat
        elif hasattr(data, "outputformat"):
            subarr.outputformat = data.outputformat
        else: 
            subarr.outputformat = "%.16e"
        # Finally, we must return the newly created object:
        if unit is None and hasattr(data, "unit"):
            subarr.__dict__["unit"] = data.unit
        else:
            subarr.__dict__["unit"] = unit
        return subarr

    def __array_finalize__(self, obj):
        self.__dict__["info"] = Info(getattr(obj, "info", Info()))
        self.__dict__["outputformat"] = getattr(obj, "outputformat", "%.16e")
        self.__dict__["unit"] = getattr(obj, "unit", None)


    def verify_dimension(self):
        u"""Internal function that checks to see if the arrays dimensions match
           those of the *info* specification.
        """
        if len(self.info) != self.ndim:
            raise ValueArrayShapeInfoMismatchError

    def info_index(self, name, cls=None):
        u"""Leta upp index för axisobjekt med *name*
        """
        for idx, ax in enumerate(self.info):
            if ax.name == name:
                if cls is None:
                    return idx
                elif isinstance(ax, cls):
                    return idx
        msg = "Can not find AxisObject with name:%r and cls:%s"%(name, cls)
        raise IndexError(msg)


    def replace_dim(self, olddim, newdim):
        self.info = replace_dim(self.info, olddim, newdim)
        return self.info

    def view(self, dtype=None, type=None):
        u"""Return view of *data* i.e. new *ValueArray* object but pointing to
           same data.
        """
        if type is None:
            return self.__class__(ndarray.view(self), self.info, copy=False)
        else:
            return ndarray.view(self, dtype, type)
        
    def reorder_dimensions(self, *order):
        u"""Omorganiserar ordningen på Axis i *info*. Genom att flytta Axis
           objekten som räknas upp i *order* till början.

        """
        infos = list(self.info[:])
        neworder = []
        for dim in order:
            neworder.append(self.info.index(dim))
            del infos[infos.index(dim)]
        for dim in infos:
            neworder.append(self.info.index(dim))
        return self.transpose(*neworder)

    def transpose(self, *order):
        u"""Returnerar ValueArray med dimensionerna omorganiserade i ordning som
           ges av *order*. *order* anger index i *info* listan.

           .. todo:: Ta även emot en lista med Axis objekt.
        """
        if not order:
            order = range(self.ndim)[::-1]
        return self.__class__(ndarray.transpose(self, *order),
                              info=[self.info[i] for i in order], copy=False)

    def squeeze(self):
        u"""Tar bort dimensioner med längden 1.

        """
        newinfo = [ax for idx, ax in enumerate(self.info)
                        if self.shape[idx] != 1]
        return self.__class__(ndarray.squeeze(self), info=newinfo, copy=False)

    def apply_outputformat(fun):
        def __getitem__(self, *x, **kw):
            out = fun(self,  *x, **kw)
            if hasattr(self, "outputformat") and hasattr(out, "outputformat") :
                out.outputformat = self.outputformat
            if hasattr(self, "unit") and hasattr(out, "unit") :
                out.unit = self.unit
            return out
        return __getitem__

    @apply_outputformat
    def __getslice__(self, start, stop):
        return self.__getitem__(slice(start, stop))

    @apply_outputformat
    def __getitem__(self, x):
        if x is newaxis or (isinstance(x, tuple) and newaxis in x):
            return self.view(type=ndarray, dtype=self.dtype)[x]
        if x is Ellipsis:
            return self.view()
        if isinstance(x, tuple):
            indices = x
        else:
            indices = (x,)
        ellipsis_and_ints = True
        orig_indices = indices
        for i in indices:
            if isinstance(i, int) or i is Ellipsis:
                pass
            else:
                ellipsis_and_ints = False

        ellips_count = len([i for i in indices if isinstance(i, type(Ellipsis))])
        if ellips_count==1:
            i = indices.index(Ellipsis)
            indices = indices[:i] + (slice(None),)*(self.ndim-(len(x)-1)) + indices[i+1:]
        elif ellips_count > 1:
            raise IndexError("Can not handle more than one Ellipsis")

        info = self.info
        if len(indices)==1 and isinstance(indices[0], ValueArray) and indices[0].dtype == bool:
            if len(indices[0].info) == 1:
               idx = self.info_index(indices[0].info[0])
               indices = (slice(None),)*idx + (indices[0],)
               olddim = indices[-1].info[0]
               info = list(info)
               info[idx] = olddim.__class__(olddim.name, olddim.data[indices[-1]])
               info = tuple(info)
        try:
            indices = indices + (slice(None),)*(self.ndim - len(indices))
            info = []
            dim_in_indices = dict((x.info[0].name, x.info[0]) for x in indices if isinstance(x, ValueArray))
            for idx, dim in zip(indices, self.info):
                if isinstance(idx, int):
                    continue
                elif isinstance(idx, slice):
                    info.append(dim[idx])
#                    print "X", idx
                elif isinstance(dim, DimBase):
                    info.append(dim_in_indices.get(dim.name, dim))
                else:
                    info.append(dim)
#            print info
            #info = tuple(dim for idx, dim in zip(indices, self.info) if not isinstance(idx, int))
            if ellipsis_and_ints:
                indices = orig_indices
            out = ndarray.__getitem__(self, indices)
            if isinstance(out, ndarray):
                return self.__class__(out, info=info, copy=False)
            else:
                return out
        except ValueArrayShapeInfoMismatchError:
            logging.warn("WARNING mismatch")
            out = ndarray.__getitem__(self, indices)
            return out.view(type=ndarray, dtype=self.dtype)

    @check_instance
    def __add__(self, other):
        return ndarray.__add__(self, other)

    @check_instance
    def __sub__(self, other):
        return ndarray.__sub__(self, other)

    @check_instance
    def __mul__(self, other):
        return ndarray.__mul__(self, other)

    @check_instance
    def __div__(self, other):
        return ndarray.__div__(self, other)

    @check_instance
    def __pow__(self, other):
        return ndarray.__pow__(self, other)

    @check_instance
    def __radd__(self, other):
        return self.__add__(other)

    @check_instance
    def __rsub__(self, other):
        return (-self).__add__(other)

    @check_instance
    def __rmul__(self, other):
        return self.__mul__(other)

    @check_instance
    def __rdiv__(self, other):
        return np.divide(other, self)

    @check_instance
    def __rpow__(self, other):
        return np.power(other, self)

    def __abs__(self):
        return ndarray.__abs__(self)

    def __neg__(self):
        return ndarray.__neg__(self)

    def copy(self):
        u"""Skapa kopia av objekt
        """
        return self.__class__(self)

    def rss(self, axis=None):
        u"""Beräkna kvadratsumma över *axis*. Där *axis* specas av index till
           *info*.

           .. todo:: Ta även emot en lista med Axis objekt.
        """
        return (abs(self)**2).sum(axis)**0.5


    def _reducing_helper(self, methodname, axis=None, **kw):
        out = self
        dim = axis_handler(self, axis)
        if dim is None:
            result = getattr(np.asarray(out), methodname)(axis=None, **kw)
            out = self.__class__(result, copy=False)
            return out

        info = list(self.info)
        axidx = info.index(dim)
        result = getattr(np.asarray(out), methodname)(axis=axidx, **kw)
        del info[axidx]
        out = self.__class__(result, info, copy=False)
        return out

    def _multiple_axis_reducing_helper(self, methodname, axis=None, **kw):
        out = self
        dims = multiple_axis_handler(self, axis)
        if dims is None:
            result = getattr(np.asarray(out), methodname)(axis=None, **kw)
            out = self.__class__(result, copy=False)
            return out

        info = list(self.info)
        for ax in dims:
            axidx = info.index(ax)
            result = getattr(np.asarray(out), methodname)(axis=axidx, **kw)
            del info[axidx]
            out = self.__class__(result, info, copy=False)
        return out

    
    def sum(self, axis=None, dtype=None, out=None):
        u"""Beräkna medelvärde över *axis*. Där *axis* specas av index till
           *info*.

           .. todo:: Ta även emot en lista med Axis objekt.
        """
        
        return self._multiple_axis_reducing_helper("sum", axis, dtype=dtype, out=out)
    
    def mean(self, axis=None, dtype=None, out=None):
        u"""Beräkna medelvärde över *axis*. Där *axis* specas av index till
           *info*.

           .. todo:: Ta även emot en lista med Axis objekt.
        """
        
        return self._multiple_axis_reducing_helper("mean", axis, dtype=dtype, out=out)
        
        
    def std(self, axis=None, dtype=None, out=None, ddof=1):
        u"""Beräkna standardavvikelse över *axis*. Där *axis* specas av index
           till *info*.

           .. todo:: Ta även emot en lista med Axis objekt.
        """
        if np.iscomplexobj(self):
            return (self.real._reducing_helper("std", axis, dtype=dtype, out=out, ddof=ddof) +
                    self.imag._reducing_helper("std", axis, dtype=dtype, out=out, ddof=ddof) * 1j)
        else:
            return self._reducing_helper("std", axis, dtype=dtype, out=out, ddof=ddof)


    def var(self, axis=None, dtype=None, out=None, ddof=1):
        u"""Beräkna standardavvikelse över *axis*. Där *axis* specas av index
           till *info*.

           .. todo:: Ta även emot en lista med Axis objekt.
        """
        return self._reducing_helper("var", axis, dtype=dtype, out=out, ddof=ddof)

    def min(self, axis=None, out=None):
        u"""Beräkna minsta värde över *axis*. Där *axis* specas av index
           till *info*.

           .. todo:: Ta även emot en lista med Axis objekt.
        """
        return self._multiple_axis_reducing_helper("min", axis, out=out)

    def max(self, axis=None, out=None):
        u"""Beräkna största värde över *axis*. Där *axis* specas av index
           till *info*.

           .. todo:: Ta även emot en lista med Axis objekt.
        """
        return self._multiple_axis_reducing_helper("max", axis, out=out)

    def argmin(self, axis=None, out=None):
        u"""Beräkna minsta värde över *axis*. Där *axis* specas av index
           till *info*.

           .. todo:: Ta även emot en lista med Axis objekt.
        """
        return self._reducing_helper("argmin", axis, out=out)

    def argmax(self, axis=None, out=None):
        u"""Beräkna minsta värde över *axis*. Där *axis* specas av index
           till *info*.

           .. todo:: Ta även emot en lista med Axis objekt.
        """
        return self._reducing_helper("argmax", axis, out=out)


    def all(self, axis=None, out=None):
        return self._reducing_helper("all", axis, out=out)

    def any(self, axis=None, out=None):
        return self._reducing_helper("any", axis, out=out)

    def cumprod(self, axis=0, dtype=None, out=None):
        if axis is None:
            raise ValueArrayError("Must choose axis for cumulative product")
        axis = axis_handler(self, axis)
        result = np.asarray(self).cumprod(axis=self.info_index(axis), dtype=dtype, out=out)
        return self.__class__(result, info=self.info, copy=False)

    def cumsum(self, axis=0, dtype=None, out=None):
        if axis is None:
            raise ValueArrayError("Must choose axis for cumulative product")
        axis = axis_handler(self, axis)
        result = np.asarray(self).cumsum(axis=self.info_index(axis), dtype=dtype, out=out)
        return self.__class__(result, info=self.info, copy=False)

    def help(self):
        out = "\n".join(["class: %(_class)s",
                         "dtype: %(dtype)s",
                         "shape: %(shape)r",
                         "info:  (%(info)s)",])
        info = ["%r"%self.info[0]]
        for i in self.info[1:]:
            info.append("        %r"%i)
        out = out%dict(_class=self.__class__.__name__,
                       dtype=self.dtype,
                       shape=self.shape,
                       info=",\n".join(info),
                      )
        return out

def axis_handler(a, axis):
    if axis is None:
        return None
    elif isinstance(axis, int):
        return a.info[axis]
    elif isinstance(axis, type) and issubclass(axis, DimBase):
        outaxis = []
        for dim in a.info:
            if isinstance(dim, axis):
                outaxis.append(dim)
        if len(outaxis) == 0:
            raise IndexError("%r dimension not present in info %r"%(axis.__name__, a.info))
        elif len(outaxis) == 1:
            return outaxis[0]
        else:
            raise IndexError("There are several %r present in %r"%(axis, a.info))
    elif axis in a.info:
        return axis
    else:
        raise IndexError("%r not a valid dimension for %r"%(axis, a.info))

def multiple_axis_handler(a, axis):
    if axis is None:
        return None
    if not isinstance(axis, (tuple, list)):
        axis = [axis]
    outaxis = []
    for ax in axis:
        if isinstance(ax, int):
            outaxis.append(a.info[ax])
        elif isinstance(ax, type) and issubclass(ax, DimBase):
            for dim in a.info:
                if isinstance(dim, ax):
                    outaxis.append(dim)
        elif isinstance(ax, DimBase) and ax in a.info:
            outaxis.append(ax)
        else:
            raise IndexError("%r dimension not present in info %r"%(ax, a.info))
    return outaxis


class ValueArray(_ValueArray):
    def __new__(subtype, data, info=None, dtype=None, copy=True, order=None, subok=False,
                ndmin=0, unit=None, outputformat=None):
        if hasattr(data, "__ValueArray__"):
            data, info = data.__ValueArray__()
            if unit is None and len(info) == 1:
                unit = info[0].unit
            if outputformat is None and len(info) == 1:
                outputformat = info[0].outputformat
        return _ValueArray.__new__(subtype, data, info=info, dtype=dtype, copy=copy,
                                   order=order, subok=subok, ndmin=ndmin, unit=unit, outputformat=outputformat)

def make_matrix(data, info):
    info = info + (DimMatrix_i("i", arange(data.shape[-2])),
                   DimMatrix_j("j", arange(data.shape[-1])))
    return ValueArray(data, info, copy=False)


if __name__ == '__main__':
    freqi = DimSweep("freq", linspace(0, 10e9, 11))
    ri = DimRep("rep", range(10))
    a = ValueArray(zeros((11, 10)), (freqi, ri))
    b = ValueArray(zeros((11, )), (freqi, ))
    c = ValueArray(zeros((10, )), (ri, ))
    

    fi = DimSweep("f", [10, 20, 30])
    gi = DimSweep("g", [100, 200, 300, 400])
    hi = DimSweep("h", [1, 2])

    a = ValueArray(zeros((3, 2)), info=(fi, hi))
    b = ValueArray(zeros((2,)), info=(hi,))
    c = ValueArray(zeros((3,)), info=(fi,))

    AB = a + b 
    AC = a + c 

    FG = ValueArray(fi)+ValueArray(gi)



