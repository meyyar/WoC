#!/usr/bin/env python

import numpy as np
import random
from EM_algorithm import EM
from utils import Utils
from pprint import pprint
from sys import stdin
import matplotlib.pyplot as plt
from random import randrange
import NR_logistic_regr as NR

"""
Executes the EM algorithm for the given dataset, stores and plots the result
"""
class Crowds_EM:

    def __init__(self, X, y, min_class_label, max_class_label, expert_wrong_percentage, verbose=True, synthetic = False, missing=False, percent_missing = 0):
        #initializations
                #initializations
        if synthetic == False :
            #self.x = data[:, 0:-1] #features
            self.x = X
            self.N = np.shape(self.x)[0] #no of instances
            self.x = np.append(np.ones((self.N,1)), self.x,1) #appending a column of ones as bias (used in logistic regression weights prediction)
            self.F = np.shape(self.x)[1] #no of features+1
    
            #self.y = data[:, -1] #last column in the dataset is the observed output variable
            self.y = y
            #self.y_shape = np.shape(self.Training_y) #dimension of y
            #no of experts
            self.min_class_label = min_class_label
            self.max_class_label = max_class_label
            self.expert_wrong_percentage = expert_wrong_percentage
            #self.experty = self.generate_experty()
            
            
        else :
            #x1 = np.random.normal(-1, 0.1, 5)
            #x2 = np.random.normal(1,0.1,5)
            
            self.x = X
            self.y = y
            self.N = np.shape(self.x)[0]
            #self.x = np.append(x1,x2)
            #self.x = self.x.reshape(10,1)
            
            self.x = np.append(np.ones((self.N,1)), self.x, 1)
            
            self.min_class_label = min_class_label
            self.max_class_label = max_class_label
            self.expert_wrong_percentage = expert_wrong_percentage
            #self.E = 3#len(expert_wrong_percentage)
            self.F = np.shape(self.x)[1]    
                        
        self.E = len(expert_wrong_percentage)
        self.experty = self.generate_experty()
        
        self.percent_missing = percent_missing
        if missing:
            self.generate_random_missing()
            
        self.Testing_instances = self.N
        self.Training_instances = self.N

        self.MAXITER = 10000
        self.CONV_THRESH = 1.e-3

        self.Training_x  = self.x[:self.Training_instances,:]
        self.Testing_x = self.x[self.Training_instances:,:]
        self.Training_y = self.y[:self.Training_instances]
        self.Testing_y = self.y[self.Training_instances:]

        self.y_shape = np.shape(self.Training_y)
        
        self.Training_experty = {}
        for e in xrange(self.E):
        #    self.experty[e][self.experty[e] == -1] = randrange(self.min_class_label,self.max_class_label+1)
            self.Training_experty[e] = self.experty[e][:self.Training_instances]
            

        #results
        self.results = {}
        self.results['weights'] = {}
        self.results['alpha'] = {}
        self.results['beta'] = {}
        self.results['weights_mv'] = {}
        self.results['weights_avg'] = {}
        self.results['weights_at'] = {}
        #debug
        self.verbose = verbose
        

    def generate_experty(self):
        experty = {}
        #print range(self.E)
        shuffled = np.arange(np.size(self.y))
        for e in range(self.E):

            #print "For expert ", e
            newy = self.y.copy()
            no_of_wrongs = self.expert_wrong_percentage[e]*np.size(self.y)
            
            np.random.shuffle(shuffled)
            for i in shuffled[:no_of_wrongs]:
                newy[i] =  (self.y[i]+1)%self.max_class_label#random.randrange(self.min_class_label,self.max_class_label)
            experty[e] = newy
            #print experty[e]
            #ch = stdin.read(1)
            
        #print experty
        return experty

        
    def generate_random_missing(self):
        #experty = {}

        for e in range(self.E):
            newy = self.experty[e]
            no_of_missing = self.percent_missing*np.size(self.experty[e])
            sample = np.arange(np.size(self.experty[e]))
            np.random.shuffle(sample)
            for i in sample[:no_of_missing]:
                newy[i] =  -1
            self.experty[e] = newy
            
        #print "initial experty"
        #pprint (self.experty)
        #ch = stdin.read(1)
        return self.experty

    def binary_y_experty(self, class_no):
        class_id = np.where(self.Training_y > class_no)
        if self.verbose:
            print "Training for target > ", class_no
        y_observed = np.zeros(np.shape(self.Training_y))
        for c in class_id:
            y_observed[c] = 1

        experty_observed = {}
        for e in range(self.E):
            class_id = np.where(self.Training_experty[e] > class_no)
            experty_observed[e] = np.zeros(np.shape(self.Training_experty[e]))
            for c in class_id:
                experty_observed[e][c] = 1
        #print experty_observed
        return y_observed, experty_observed




    def calculate_loglikelihood(self, y, weights, a, b):
        return np.sum( y * -np.logaddexp(0, -1 * np.dot(weights.T, self.Training_x.T)*a) + (1-y) * -np.logaddexp(0, 1 * np.dot(weights.T, self.Training_x.T)*b))


    def run(self):
        try:
            for class_no in range(self.min_class_label, self.max_class_label):
            #for class_no in range(0,3):
        #class_no = 1
                #print class_no
                y_observed, experty_observed = self.binary_y_experty(class_no)
                #random initializations for this class label
                weights = np.random.random(self.F)
                alpha = np.random.random(self.E)  #expert sensitivity
                beta = np.random.random(self.E)   #expert specificity
                l = 0
                iter = 0
                while iter < self.MAXITER:
                    # First iteration
                    if not iter:
                        l_old = 0
                        expertcombined = np.array([])
                        for e in experty_observed:
                            expertcombined = np.append(expertcombined,experty_observed[e], axis=0)

                        expertcombined = np.reshape(expertcombined, (self.E, self.Training_instances))
                        y_average = np.average( expertcombined, axis=0)
                        #y_average = y_predicted.copy()
                        
                        mv_expert_combined =  np.reshape(expertcombined,expertcombined.size,order='F').reshape(np.shape(expertcombined)[1],np.shape(expertcombined)[0])
                        y_predicted = np.array([])
                        
                        for emv in mv_expert_combined:
                            y_predicted = np.append(y_predicted, np.bincount(emv.astype(int)).argmax())
                        
                        

                        acc_MV = np.size(np.where((y_average.round())==y_observed))/float(self.Training_instances)

                        """
                        Classifier with MV as input
                        """
                        self.results['weights_mv'][class_no] = NR.logistic_regression(self.Training_x[:,1:].T,np.asarray(y_predicted).reshape(-1),verbose=False, MAXIT=10000)
                        self.results['weights_avg'][class_no] = NR.logistic_regression(self.Training_x[:,1:].T,np.asarray(y_average).reshape(-1),verbose=False, MAXIT=10000)
                        self.results['weights_at'][class_no] = NR.logistic_regression(self.Training_x[:,1:].T,np.asarray(y_observed).reshape(-1),verbose=False, MAXIT=10000)


                    else :
                        l_old = l
                        w_old = weights
                        alpha_old = alpha
                        beta_old = beta

                        a = Utils.a_calculations(alpha_old, experty_observed,self.y_shape)
                        b = Utils.b_calculations(beta_old, experty_observed, self.y_shape)
                        # E-step
                        y_predicted = EM.Estep(self.Training_x, w_old, a, b)
                        y_predicted = np.asarray(y_predicted).reshape(-1)

                    # M-step
                    weights, alpha, beta = EM.Mstep(self.Training_x, y_predicted, experty_observed)
                    a = Utils.a_calculations(alpha, experty_observed, self.y_shape)
                    b = Utils.b_calculations(beta, experty_observed, self.y_shape)

                    l = self.calculate_loglikelihood(y_predicted, weights, a, b)
                    acc_EM = np.size(np.where(y_observed==y_predicted.round()))/float(self.Training_instances)
                    diff =  np.fabs(l-l_old)
                    if diff <= self.CONV_THRESH and l>=l_old : break
                    iter = iter+1
                    if self.verbose:
                        print "EM algorithm :","diff:",diff,"log:", l, "iteration:", iter

                self.results['weights'][class_no] = weights
                self.results['alpha'][class_no] = alpha.round(1)
                self.results['beta'][class_no] = beta.round(1)
 
                if self.verbose:
                    print "alphacap :"
                    pprint (alpha.round(1))
                    print "betacap :"
                    pprint (beta.round(1))
                    print "weights :"
                    pprint (weights)


                    print "Accuracy of EM approach :"
                    print str(acc_EM)

                    print "Accuracy of majority voting approach :"
                    print str(acc_MV)

                    print "y"
                    print y_observed
                    print "y maj"
                    print y_average.round(2)
                    print "y pred"
                    print y_predicted.round(2)

                    print "Expert wrong percentage"
                    print self.expert_wrong_percentage

                    print '--'*30

        except:
            #print "Running EM again --"
            raise
            #ch = stdin.read(1)
            
    def learn_experty_missing(self, alpha, beta, y):
        experty = np.dot(y.reshape(-1,1), alpha.reshape(-1,1).T) + np.dot((1-y).reshape(-1,1), beta.reshape(-1,1).T)
        #pprint (experty.T)
        return experty.T
      
    def run_EM_missing(self):
        try:
            for class_no in range(self.min_class_label, self.max_class_label):
                y_observed, experty_observed = self.binary_y_experty(class_no)
                #random initializations for this class label
                weights = np.random.random(self.F)
                alpha = np.random.random(self.E)  #expert sensitivity
                beta = np.random.random(self.E)   #expert specificity
                l =0
                iter = 0
                while iter < self.MAXITER:
                    # First iteration
                    if not iter:
                        l_old = 0
                        expertcombined = np.array([])
                        
                        for e in xrange(self.E):
                            experty_observed[e][experty_observed[e] == -1] = randrange(self.min_class_label,self.max_class_label+1)
                            #self.Training_experty[e] = self.experty[e][:self.Training_instances]
  
                        
                        for e in experty_observed:
                            expertcombined = np.append(expertcombined,experty_observed[e], axis=0)
                        
                        
                        
                        expertcombined = np.reshape(expertcombined, (self.E, self.Training_instances))
                        y_predicted = np.average( expertcombined, axis=0)
                        y_average = y_predicted.copy()
                        #acc_MV = np.size(np.where((y_average.round())==y_observed))/float(self.Training_instances)
                        self.results['weights_mv'][class_no] = NR.logistic_regression(self.Training_x[:,1:].T,np.asarray(y_average).reshape(-1),verbose=False, MAXIT=10000)
                        self.results['weights_at'][class_no] = NR.logistic_regression(self.Training_x[:,1:].T,np.asarray(y_observed).reshape(-1),verbose=False, MAXIT=10000)

                    else :
                        l_old = l
                        w_old = weights
                        alpha_old = alpha
                        beta_old = beta
    
                        experty_learnt = self.learn_experty_missing(alpha_old, beta_old, y_observed)

                        for e in experty_observed:
                            missing_ids = np.where(self.experty[e] == -1)
                            for m in missing_ids:
                                experty_observed[e][m] = experty_learnt[e][m]
                        #print "experty :"
                        #pprint(experty_observed)
                        a = Utils.a_calculations(alpha_old, experty_observed,self.y_shape)
                        b = Utils.b_calculations(beta_old, experty_observed, self.y_shape)
                        # E-step
                        y_predicted = EM.Estep(self.Training_x, w_old, a, b)
                        y_predicted = np.asarray(y_predicted).reshape(-1)
    
                    # M-step
                    weights, alpha, beta = EM.Mstep(self.Training_x, y_predicted, experty_observed)
                    a = Utils.a_calculations(alpha, experty_observed, self.y_shape)
                    b = Utils.b_calculations(beta, experty_observed, self.y_shape)
    
                    l = self.calculate_loglikelihood(y_predicted, weights, a, b)
                    #acc_EM = np.size(np.where(y_observed==y_predicted.round()))/float(self.Training_instances)
                    diff =  np.fabs(l-l_old)
                    if diff <= self.CONV_THRESH and l>=l_old : break
                    iter = iter+1
                    if self.verbose:
                        print "EM algorithm :","diff:",diff,"log:", l, "iteration:", iter
    
                self.results['weights'][class_no] = weights
                self.results['alpha'][class_no] = alpha.round(1)
                self.results['beta'][class_no] = beta.round(1)
                """self.results['loglikelihood'][class_no] = l
                self.results['EM_perf']['f1_Score'][class_no] = Utils.calculate_F1score(y_observed, y_predicted)
                self.results['MV_perf']['f1_Score'][class_no] = Utils.calculate_F1score(y_observed, y_average)
                self.results['EM_perf']['rmse'][class_no] = Utils.calculate_RMSE(y_observed, y_predicted)
                self.results['MV_perf']['rmse'][class_no] = Utils.calculate_RMSE(y_observed, y_average)
                self.results['experty'] [class_no] = experty_observed
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title('class :' + str(class_no))
                ax.set_ylim(-1,2)
                ax.plot(y_observed,'ro-',y_predicted,'-b.')"""
                
                if self.verbose:
                    print "alphacap :"
                    pprint (alpha.round(1))
                    print "betacap :"
                    pprint (beta.round(1))
                    print "weights :"
                    pprint (weights)
    
    
                    print "f1_Score of EM approach :"
                    print self.results['EM_perf']['f1_Score'][class_no]
    
                    print "f1_Score of majority voting approach :"
                    print self.results['MV_perf']['f1_Score'][class_no]
    
                    print "y"
                    print y_observed
                    print "y maj"
                    print y_average.round(2)
                    print "y pred"
                    print y_predicted.round(2)
    
                    print "Expert wrong percentage"
                    print self.expert_wrong_percentage
    
                    print '--'*30
     
        except Exception, e:
            raise
      



    """self.x = data[:, 0:-1] #features
        self.N = np.shape(self.x)[0] #no of instances
        self.x = np.append(np.ones((self.N,1)), self.x,1) #appending a column of ones as bias (used in logistic regression weights prediction)
        self.F = np.shape(self.x)[1] #no of features+1

        self.y = data[:, -1] #last column in the dataset is the observed output variable
        #self.y_shape = np.shape(self.Training_y) #dimension of y
        self.E = len(expert_wrong_percentage) #no of experts
        self.min_class_label = min_class_label
        self.max_class_label = max_class_label
        self.expert_wrong_percentage = expert_wrong_percentage
        self.experty = self.generate_experty()

        #print self.experty



        self.N = 20 ###

        self.Testing_instances = self.N#0.2 * self.N
        self.Training_instances = self.N #- self.Testing_instances
        #self.Training_x  = self.x[:self.Training_instances,:]
        #self.Testing_x = self.x[Training_instances:,:]
        #self.Training_y = self.y[:self.Training_instances,:]
        #self.Testing_y = self.y[self.Training_instances:,:]


        self.x = np.random.randint(2, size=self.N)
        self.y = np.random.randint(2, size=self.N)#self.x.copy()
        #print self.y
        self.x = np.reshape(self.x, (self.N,1))
        rand_x = np.random.random(self.N*1)
        rand_x = np.reshape(rand_x, (self.N,1))
        rand_x = np.append(np.ones((self.N,1)), rand_x,1)
        self.x = np.append(rand_x, self.x,1)###
        #self.x = np.load('X.npy')
        #print "x :",  self.x
        #self.y = np.load('Y.npy')
        self.min_class_label = min_class_label
        self.max_class_label = max_class_label
        self.expert_wrong_percentage = expert_wrong_percentage
        self.E = 4#len(expert_wrong_percentage)
 

        
        x1 = np.random.normal(-2, 3,5)
        x2 = np.random.normal(-1, 0.1,5)
        x3 = np.random.normal(1, 3,5)
        x4 = np.random.normal(2, 0.1,5)

        self.x = np.append(x1,x2)
        self.x = np.append(self.x , x3)
        self.x = np.append(self.x , x4)
        self.x = self.x.reshape(20,1)
        
        self.x = np.append(np.ones((self.N,1)), self.x, 1)
        self.F = np.shape(self.x)[1]

        self.y = np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
        print "y :", self.y
        self.experty = {}
#20:20
        self.experty[0] = np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
        self.experty[1] = np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
        self.experty[2] = np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
        self.experty[3] = np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
        

        self.Training_x  = self.x[:self.Training_instances,:]
        self.Testing_x = self.x[self.Training_instances:,:]
        self.Training_y = self.y[:self.Training_instances]
        self.Testing_y = self.y[self.Training_instances:]
        
        self.Training_experty = {}
        for e in xrange(self.E):
            self.Training_experty[e] = self.experty[e][:self.Training_instances]

        #print "training_y",self.Training_y

        self.y_shape = np.shape(self.Training_y)
        #self.experty = self.generate_experty()

        #results
        self.results = {}
        self.results['weights'] = {}
        self.results['alpha'] = {}
        self.results['beta'] = {}
        self.results['loglikelihood'] = {}
        self.results['EM_perf'] = {}
        self.results['MV_perf'] = {}
        self.results['EM_perf']['accuracy'] = {}
        self.results['EM_perf']['rmse'] = {}
        self.results['MV_perf']['accuracy'] = {}
        self.results['MV_perf']['rmse'] = {}

        #debug
        self.verbose = verbose
        
        self.MAXITER = 10000
        self.CONV_THRESH = 1.e-3"""
        
    """def generate_experty(self):
        experty = {}
        for e in range(self.E):
            experty[e] = self.y.copy()
        return experty

    def binary_y_experty(self, class_no):
        if self.verbose:
            print 'Training for class ', class_no

        class_id = np.where(self.Training_y == class_no)
        y_observed = np.zeros(self.y_shape)
        for c in class_id[0]:
            y_observed[c] = 1

        sample = class_id[0]
        experty_observed = {}
        for e in range(self.E):
            no_of_wrongs = self.expert_wrong_percentage[e]*np.size(class_id)
        #print sample
            np.random.shuffle(sample)
        #print sample[:no_of_wrongs]
            experty_observed[e] = y_observed.copy()
            for i in sample[:no_of_wrongs]:
                experty_observed[e][i] = 1 - y_observed[i]

        #print y_observed
        #print 'Experty_observed :',experty_observed
        return y_observed, experty_observed"""
        
        
    """def visualize(self):
        Utils.show_plot('Plot of EM, MV accuracies, ('+ str(self.E) + ' Experts)',[
                  {
                  'x_values':self.results['EM_perf']['accuracy'].keys(),
                  'y_values':self.results['EM_perf']['accuracy'].values(),
                  'xlabel':'y label',
                  'ylabel':'accuracy',
                  'label':'EM accuracy',
                  'xlim':None,
                  'ylim':( min(min(self.results['EM_perf']['accuracy'].values()), min(self.results['MV_perf']['accuracy'].values()))-0.02, max(max(self.results['EM_perf']['accuracy'].values()),max(self.results['MV_perf']['accuracy'].values()))+0.02)},
                  {
                  'x_values':self.results['MV_perf']['accuracy'].keys(),
                  'y_values':self.results['MV_perf']['accuracy'].values(),
                  'xlabel':'y label',
                  'ylabel':'accuracy',
                  'label':'MV accuracy',
                  'xlim':None,
                  'ylim':( min(min(self.results['EM_perf']['accuracy'].values()), min(self.results['MV_perf']['accuracy'].values()))-0.02, max(max(self.results['EM_perf']['accuracy'].values()),max(self.results['MV_perf']['accuracy'].values()))+0.02)}
                    ])
        Utils.show_plot('Plot of EM, MV RMSE, ('+ str(self.E) + ' Experts)',[
                  {
                  'x_values':self.results['EM_perf']['rmse'].keys(),
                  'y_values':self.results['EM_perf']['rmse'].values(),
                  'xlabel':'y label',
                  'ylabel':'RMSE',
                  'label':'EM RMSE',
                  'xlim':None,
                  'ylim':( min(min(self.results['EM_perf']['rmse'].values()), min(self.results['MV_perf']['rmse'].values()))-1, max(max(self.results['EM_perf']['rmse'].values()),max(self.results['MV_perf']['rmse'].values()))+1)},
                  {
                  'x_values':self.results['MV_perf']['rmse'].keys(),
                  'y_values':self.results['MV_perf']['rmse'].values(),
                  'xlabel':'y label',
                  'ylabel':'RMSE',
                  'label':'MV RMSE',
                  'xlim':None,
                  'ylim':( min(min(self.results['EM_perf']['rmse'].values()), min(self.results['MV_perf']['rmse'].values()))-1, max(max(self.results['EM_perf']['rmse'].values()),max(self.results['MV_perf']['rmse'].values()))+1)}
                    ])

        plt.show()"""

    """def predict_EM(self, x, actual_y, weights=None):
        try:
            predicted_y = np.empty(np.shape(actual_y))
            p_old = 1
            prob = []
            if weights == None:
                weights = self.results['weights']
            for weight in weights:
                p = Utils.logistic_transformation( self.results['weights'][weight], x )
                #print p
                prob.append(p_old-p)
                p_old = p
            prob.append(p_old)
            print "prob"
            pprint( prob)
            sum_prob = np.sum(prob,0)
            print "sum"
            pprint (sum_prob)
            max_prob = np.max(prob, 0)
            #print "max_prob", max_prob
            for i in xrange(np.size(max_prob)):
                class_label = np.where(prob == max_prob[i])[0]
                predicted_y[i] = class_label[0]
    
            
            #print "Actual y :", actual_y
            print "EM predicted y:", predicted_y
            acc_EM = np.size(np.where(actual_y==predicted_y))/float(self.N)
            #print "EM accuracy:" , acc_EM
            return acc_EM
        except Exception,e:
            raise
    
    def predict_MV(self, x, actual_y):
    
        try:
            expertcombined = np.array([])
            for e in self.experty:
                expertcombined = np.append(expertcombined,self.experty[e], axis=0)
        
            expertcombined = np.reshape(expertcombined, (self.E, self.N))
            majority_voting_y = np.average( expertcombined, axis=0)
        
            acc_MV = np.size(np.where(actual_y[:self.N]==majority_voting_y.round()))/float(self.N)
            #print np.where(majority_voting_y.round()==actual_y[:self.N])
            #print "actual y :", actual_y
            #print "predicted y:", majority_voting_y.round()
            #print "MV accuracy:" , acc_MV   
            return acc_MV
        except Exception, e:
            raise"""
