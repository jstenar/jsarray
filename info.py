# -*- coding: ISO-8859-1 -*-
u"""
info
========
.. autoclass:: DimBase



"""
from dim import DimBase

class Info(tuple):
    def __contains__(self, value):
        if not isinstance(value, DimBase):
            return False
        for x in self:
            if x.name == value.name and isinstance(x, value.__class__):
                return True
        return False
        
    def get_matching_dim(self, value):
        if isinstance(value, DimBase):
            for x in self:
                if x.name == value.name and isinstance(x, value.__class__):
                    return x
        raise KeyError("No dim matching %r"%value)        
        
    def matching_index(self, value):
        if isinstance(value, DimBase):
            for idx, x in enumerate(self):
                if x.name == value.name and isinstance(x, value.__class__):
                    return idx
        raise KeyError("No dim matching %r"%value)        
        
class InfoList(list):
    def __contains__(self, value):
        if not isinstance(value, DimBase):
            return False
        for x in self:
            if x.name == value.name and isinstance(x, value.__class__):
                return True
        return False
        
    def get_matching_dim(self, value):
        if isinstance(value, DimBase):
            for x in self:
                if x.name == value.name and isinstance(x, value.__class__):
                    return x
        raise KeyError("No dim matching %r"%value)        
        
    def matching_index(self, value):
        if isinstance(value, DimBase):
            for idx, x in enumerate(self):
                if x.name == value.name and isinstance(x, value.__class__):
                    return idx
        raise KeyError("No dim matching %r"%value)        

def replace_dim(info, olddim, newdim):
    out = []
    for d in info:
        if olddim.name == d.name:
            out.append(newdim)
        else:
            out.append(d)
    return Info(out)

