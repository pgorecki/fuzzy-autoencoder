# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:00:33 2016

@author: Przemek
"""

import numpy as np
from rule import CodingRule as Rule

class FuzzyAutoencoder:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.rules = []
        self.nan_replacer = 0.5
        self.regularization_coeff = 0.1
        
    def set_partition(self, mfs):
        self.mfs = mfs
        
    @property
    def num_rules(self):
        return len(self.rules)

    @property
    def num_mfs(self):
        return len(self.mfs)
        
    def add_random_rules(self, num_rules, prob_on=0.3):
        for i in range(num_rules):        
            rule = Rule(self.num_inputs, self.mfs)
            rule.randomize(prob_on)
            self.rules.append(rule)
    
    def add_rule(self, rule_weights):
        assert self.num_inputs == rule_weights.shape[0]
        assert len(self.mfs) == rule_weights.shape[1]
        
        rule = Rule(self.num_inputs, self.mfs)
        rule.weights = rule_weights
        self.rules.append(rule)
    
    def __repr__(self):
        lines = []
        for i in range(self.num_rules):
            lines.append("R%i: %s" % (i, self.rules[i].to_string()))            
        return "\n".join(lines)
        
    def encode(self, x):
        if len(x.shape) == 1:
            x = np.array([x])
        num_samples = x.shape[0]
        y = np.zeros((num_samples, self.num_rules))
        for s in range(num_samples):
            sample = x[s,:]
            for i in range(self.num_rules):
                y[s,i] = self.rules[i].fire(sample)
        return y
        
    def decode(self, y):
        num_samples = y.shape[0]
        x_hats = np.zeros((num_samples, self.num_inputs))
        for s in range(num_samples):
            code = y[s,:]
            xn = np.zeros((self.num_rules, self.num_inputs))
            xd = np.zeros((self.num_rules, self.num_inputs)) + np.nan
            for i in range(self.num_rules):
                c = self.rules[i].consequents()
                xn[i,:] = c * code[i]
                xd[i,~np.isnan(c)] = code[i]
            xn = np.nansum(xn, axis=0)
            xd = np.nansum(xd, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                x_hat = np.true_divide(xn,xd)
                print x_hat, 'x_h1'
            x_hat[~np.isfinite(x_hat)] = self.nan_replacer
            print x_hat, 'x_h2'
            x_hats[s,:] = x_hat
        return x_hats
        
    def consequents(self):
        result = np.zeros((self.num_rules, self.num_inputs))
        for i in range(self.num_rules):                     
            result[i,:] = self.rules[i].consequents()
        return result
    
    def loss(self, x, x_prime):
        error = ((x - x_prime)**2).sum() / 2.0
        reg = self.get_state().sum() * self.regularization_coeff
        return error + reg 
        
    def get_state(self):
        state = np.zeros((self.num_rules, self.num_inputs, self.num_mfs))
        for i in range(self.num_rules):
            state[i,:,:] = self.rules[i].weights
        return state.ravel()

    def set_state(self, state):
        state = state.reshape((self.num_rules, self.num_inputs, self.num_mfs))
        for i in range(self.num_rules):
            self.rules[i].weights = state[i,:,:]
            self.rules[i]._sanitize_weights()
