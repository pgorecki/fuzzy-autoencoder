# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:59:40 2016

@author: Przemek
"""

import numpy as np
from mf import TriangularMF

class CodingRule:
    def __init__(self, num_inputs, mfs):
        self.num_inputs = num_inputs
        self.mfs = mfs
        self.mf_max = np.zeros(len(mfs))
        for i in range(len(mfs)):
            self.mf_max[i] = mfs[i].max()
            
        self.weights = np.zeros((num_inputs, len(mfs)))
        self.firing_str = 0

    def _sanitize_weights(self):
        for i in range(self.weights.shape[0]):
            if self.weights[i,:].sum() == 0:
                self.weights[i,:] = True
        
    def set_or(self, input_idx, value):
        self.weights[input_idx,:] = value
        self._sanitize_weights()
        
    def randomize(self, prob_on = 0.5):
        self.weights = np.random.rand(self.num_inputs, len(self.mfs)) < prob_on
        self._sanitize_weights()
        
    def consequents(self):
        l = (self.weights * 1.0 * self.mf_max).sum(axis=1)
        m = self.weights.sum(axis=1)        
        with np.errstate(divide='ignore'):            
            return l/m
        
    def fire(self, input):
        mf_vals = np.zeros(self.weights.shape)+np.nan
        for i in range(self.num_inputs):
            x = input[i]
            for j in range(len(self.mfs)):
                if self.weights[i,j]:
                    mf_vals[i,j] = self.mfs[j].value(x)            
        op_or = np.nansum(mf_vals,axis=1)
        op_and = np.nanprod(op_or)
        
        return op_and

    def to_string(self, input_names=[]):
        if len(input_names)==0:
            for i in range(self.num_inputs):
               input_names.append("x%i" % i) 
        and_parts = []
        for i in range(self.num_inputs):
            row = self.weights[i]
            or_parts = []
            for j in range(len(self.mfs)):
                if row[j]:
                    or_parts.append(self.mfs[j].name)
                    
            if len(or_parts)==len(self.mfs):
                s = "(x%s IS any)" % i
                and_parts.append(s)
                
            elif len(or_parts)>0:
                s = "(x%s IS %s)" % (i, " OR ".join(or_parts))
                and_parts.append(s)
                
        if len(and_parts) > 0:
            consequent_parts = []
            c = self.consequents()
            for i in range(self.num_inputs):
                if not np.isnan(c[i]):
                    consequent_parts.append("%s' is %.2f" % (input_names[i], c[i]))
            consequent = " AND ".join(consequent_parts)
            return "IF %s THEN %s" % (" AND ".join(and_parts), consequent)
        else:
            return "<empty>"
        
        
    def __repr__(self):
        return "R: " + self.to_string()
        

def null():
    mfs = [
        TriangularMF([-0.1,0,0.5],'lo'),
        TriangularMF([0,0.5,1],'med'),
        TriangularMF([0.5,1,1.1],'hi')
    ]    
            
    r = CodingRule(2, mfs)
    #r.randomize(0.5)
    r.set_or(0,[1,0,0])
    r.set_or(1,[1,0,0])
    
    print(r)
    print(r.fire([0.0, 0.25]))