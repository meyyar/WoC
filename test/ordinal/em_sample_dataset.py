#!/usr/bin/env python


import numpy as np
from crowds_EM import Crowds_EM
from pprint import pprint
from utils import Utils
from random import random
from chunkify_zoo import getPartitions
from sys import stdin

from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt

np.seterr(all='raise')


no_of_expert_groups = [20,40,60,80,100,120,140,160,180,200]
expert_goods = 0.90
expert_bads = 0.10
dataset_filename = '../../dataset/ordinal/dataset_wine'
delimiter_char = ';'
skiprow_no = 1 #The first row corresponds to feature names, so it can be skipped
min_class_label = 3
max_class_label = 8
verbose_output = False


data = np.loadtxt(dataset_filename, delimiter=delimiter_char, skiprows=(skiprow_no)) #load the dataset


#10 fold cross validation
def k_fold_cross_validation():
    test_id = 0
    learned_params = []
    test_results = {}    
    x = data[:, 0:-1]
    y = data[:,-1]
    kfold = min([ np.size(np.where(y==c)) for c in range(min_class_label,max_class_label+1)])
    print "Folds", kfold
    skf =  StratifiedKFold(y, k=kfold)
    for train_index, test_index in skf:
        print("TRAIN: %s TEST: %s" % (train_index, test_index))
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print "Cross validation set :", test_id
        train_results = train(X_train, y_train, test_id)
        
        """
        For test_1 with average of weights for every class over all cross-validation set:
        """
        test_results[test_id] = {}
        for t in train_results.keys(): 
            test_results[test_id].update({ t : test(X_test, y_test, train_results[t]['weights']) })
        
        test_id += 1
        
    print "Final results :"
    pprint (test_results)
    plot_graph(test_results, kfold)
    #learned_params['weights'] = np.mean(map(lambda k: np.array([train_results[k]['weights'][k1] for k1 in train_results[k]['weights'].keys()]) ,train_results.keys()), 0)
    #learned_params['weights'] = np.mean(map(lambda k: np.array([train_results[k]['weights'][k1] for k1 in train_results[k]['weights'].keys()]) ,train_results.keys()), 0)

def train(X, y, set_id):
    EM_result = {}
    #Utils.initPlot( len(no_of_expert_groups), set_id)
    for ex in xrange(len(no_of_expert_groups)):
        #generate expert #self.Training_instances = self.N - self.Testing_instanceswrong percentage
        #60% good:40 % bad
        print "For group ", ex
        expert_wrong_percentage = []
        for i in xrange(int(no_of_expert_groups[ex]*expert_bads)):
            #bads
            num = 0.60 #((random()%0.5) + 0.5) % 1.0
            expert_wrong_percentage.append(num)
        for i in xrange(int(no_of_expert_groups[ex]*expert_goods)):
            #goods
            num = 0.00 #random()%0.5
            expert_wrong_percentage.append(num)                        
        crowds_EM = None    
        failed = 0
        iterations = 0
        total_iter = 10
        while iterations < total_iter :
            try:
                crowds_EM = Crowds_EM( X, y, min_class_label, max_class_label, expert_wrong_percentage, verbose= verbose_output)
                crowds_EM.run()
            except Exception,e:
                #Rerunning ...
                import traceback
                print traceback.print_exc()
                failed+=1
                try:
                    crowds_EM = Crowds_EM( X, y, min_class_label, max_class_label, expert_wrong_percentage, verbose= verbose_output)
                    crowds_EM.run()
                except Exception,e:
                    failed+=1
                    pass
                else :
                    crowds_EM.results['']
                    EM_result[ex] = crowds_EM.results
                    break
            else :
                EM_result[ex] = crowds_EM.results
                break
            #print "iteration :" , iterations
            iterations+=1
                        
    #print "Final results :\n"
    #pprint(EM_result)
    #for e in xrange(len(no_of_expert_groups)):
        #pprint (EM_result[e])
        #Utils.visualize( EM_result[e], min_class_label, max_class_label, no_of_expert_groups[e] )   
    return EM_result



def test(X, y, learned_params):
    
    N = np.shape(X)[0] #no of instances
    X = np.append(np.ones((N,1)), X,1) #appending a column of ones as bias (used in logistic regression weights prediction)
    #print X
    F = np.shape(X)[1] #no of features+1
    #print learned_params
    
    p_old = 1
    class_prob = []
    for w in learned_params.keys():
        #print w
        #prob = Utils.logistic_transformation(learned_params[w], X)
        #class_prob.append(prob)
        p = Utils.logistic_transformation( learned_params[w], X )
        class_prob.append(p_old-p)
        p_old = p
    class_prob.append(p_old)
    
    #pprint (class_prob)
    max_prob = np.max(class_prob, 0)
    #print max_prob
    
    predicted_y = []
    output_label = range(min_class_label, max_class_label+1)
    for i in xrange(np.size(max_prob)):
            class_label = np.where(class_prob == max_prob[i])[0]
            predicted_y.append(output_label[class_label[0]])
    
    #print "max_prob :", max_prob
    print "predicted y :", predicted_y
    print "Actual y:", y
    accuracy = Utils.calculate_accuracy(np.array(y), np.array(predicted_y))
    print "accuracy for test data :", accuracy
    f_score_mean, f_score_std = Utils.calculate_average_F1score(np.array(y), np.array(predicted_y), min_class_label, max_class_label)
    print "Average f score for test data :", f_score_mean
    #ch = stdin.read(1)
    return (accuracy, f_score_mean, f_score_std)

def plot_graph(test_results, kfold):
    """
    test_1, fixed : expertise of experts, no.of good,bad experts ; varying : total no.of experts
    """
    print kfold
    plot_x = no_of_expert_groups
    plot_accerr = np.std([[test_results[k][g][0] for  g in range(len(no_of_expert_groups)) ]for k in range(kfold) ],axis=0)
    plot_y = np.mean([[test_results[k][g][1] for  g in range(len(no_of_expert_groups)) ] for k in range(kfold) ],axis=0)
    plot_yerr = np.mean([[test_results[k][g][2] for  g in range(len(no_of_expert_groups)) ] for k in range(kfold) ],axis=0)
    plot_acc = np.mean([[test_results[k][g][0] for  g in range(len(no_of_expert_groups))] for k in range(kfold) ],axis=0)
    plt.figure()
    plt.xlim(plot_x[0]-10, plot_x[-1]+10)
    plt.xticks(plot_x)
    plt.errorbar(plot_x, plot_y, yerr=plot_yerr, fmt='ro', ms=10)
    plt.errorbar(plot_x, plot_acc, yerr=plot_accerr, fmt='bs', ms=5)
    plt.legend(['f-score', 'accuracy'])  
    plt.title('F-score, accuracy with varying number of experts') 
    plt.show()
    
    
k_fold_cross_validation()


#Utils.showPlot()
        








"""import sys
sys.path.append('../src/')

import numpy as np
from crowds_EM import Crowds_EM
from pprint import pprint

np.seterr(all='raise')

#alpha  - sensitivity of experts
#beta - specificity of experts

dataset_filename = '../dataset/dataset_car'
delimiter_char = ','
skiprow_no = 0 #The first row corresponds to feature names, so it can be skipped
expert_wrong_percentage = [0,0]#[ 1, 0.5, 0,0,0,0.3,0.8,0.9] # expert y manipulation wrong percentage
min_class_label = 0
max_class_label = 3
verbose_output = False


data = np.loadtxt(dataset_filename, delimiter=delimiter_char, skiprows=(skiprow_no)) #load the dataset
crowds_EM = None

EM_acc = 0
MV_acc = 0
failed = 0

iterations = 0
EM_highest_performance = {}
EM_highest_performance['accuracy'] = 0
total_iter = 1

while iterations < total_iter :

    try:
        
        crowds_EM = Crowds_EM( data, min_class_label, max_class_label, expert_wrong_percentage, verbose= verbose_output)
        crowds_EM.run()
        #pprint(crowds_EM.results)
    except Exception,e:
        #Rerunning ...
        import traceback
        print traceback.print_exc()
        failed+=1
        try:
            crowds_EM = Crowds_EM( data, min_class_label, max_class_label, expert_wrong_percentage, verbose= verbose_output)
            crowds_EM.run()
        except Exception,e:
        #print(crowds_EM.results)
            failed+=1
            continue
    
    else:
        EM_perf = crowds_EM.predict_EM(crowds_EM.x, crowds_EM.y)
        print "EM perf ", EM_perf 
        EM_acc += EM_perf
        if EM_perf > EM_highest_performance['accuracy']:
            EM_highest_performance['accuracy'] = EM_perf
            EM_highest_performance['results'] = crowds_EM.results
            
        MV_acc += crowds_EM.predict_MV(crowds_EM.x, crowds_EM.y)
        #np.save('X.npy', crowds_EM.x)
    print "iteration :" , iterations
    iterations+=1

print "No. of failed iterations : ", failed    
print "Average EM accuracy after ", iterations," iter : ", EM_acc/(total_iter-failed)
print "Average MV accuracy after ", iterations, " iter : ", MV_acc/(total_iter-failed)
print "EM highest accuracy :" , EM_highest_performance['accuracy']
print "EM with highest accuracy deatils:\n", EM_highest_performance['results']"""