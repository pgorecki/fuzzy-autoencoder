# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:23:38 2016

@author: Przemek
"""

class TriangularMF:
    def __init__(self, abc, name):
        abc.sort()
        self.name = name
        self.a = float(abc[0])
        self.b = float(abc[1])
        self.c = float(abc[2])
        
    def value(self, x):
        if x <= self.a:
            return 0.0
        if self.a <= x <= self.b:
            return (x-self.a)/(self.b-self.a)
        if self.b <= x <= self.c:
            return (self.c-x)/(self.c-self.b)
        return 0
        
    def max(self):
        return self.b
        
        
def test():
    mf = TriangularMF([0,0.5,1])
    assert mf.value(0.5) == 1
    assert mf.value(0.25) == 0.5
    assert mf.value(0.75) == 0.5
    
    mf = TriangularMF([0,0,0.5])
    assert mf.value(0) == 1
    assert mf.value(0.25) == 0.5
    assert mf.value(0.5) == 0
