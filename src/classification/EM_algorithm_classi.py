#!/usr/bin/env python

import numpy as np
from NR_logistic_regr_classi import logistic_regression
from utils_classi import Utils

"""
Performs the E-step and M-step of EM algorithm
"""
class EM:

    EPS = 1.e-1

    @staticmethod
    def Estep(x, w, a, b):
        p = Utils.logistic_transformation(w, x)
        log_p_a =np.log(p) + np.log(a)
        log_p_ab = np.log(p*a + (1-p)*b)
        log_ycap = log_p_a - log_p_ab
        ycap = np.exp(log_ycap)
        return ycap

    @staticmethod
    def Mstep(x, ycapin, experty):
        try :
            ycap = ycapin.copy()
            ycap_1 = 1-ycap
            ycap[ycap<EM.EPS] = EM.EPS
            ycap_1[ycap_1<EM.EPS] = EM.EPS
            alphacap = np.zeros(len(experty))
            betacap = np.zeros(len(experty))
            for i in range(np.size(alphacap)):
                t1 = np.sum(ycap*experty[i])
                alphacap[i] = np.exp(np.log( EM.EPS if (t1<=0) else t1) - np.log(np.sum(ycap)))
                t2 = np.sum((1-ycap)*(1-experty[i]))
                betacap[i] = np.exp(np.log(EM.EPS if (t2<=0) else t2)- np.log(np.sum(ycap_1)))
            weights = logistic_regression(x[:,1:].T,ycap,verbose=False, MAXIT=10000)
            #print "alpha", alphacap
            return weights, alphacap, betacap
        except Exception ,e :
            raise
