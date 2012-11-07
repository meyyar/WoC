#!/usr/bin/env python

import numpy as np
from crowds_EM_classi import Crowds_EM
from pprint import pprint
from utils_classi import Utils
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

np.seterr(all='raise')

no_of_expert_groups = [20,40,60,80,100,120,140,160,180,200]
expert_goods = 0.60
expert_bads = 0.40
dataset_filename = '../../dataset/classification/dataset_zoo'
delimiter_char = ','
skiprow_no = 0 #The first row corresponds to feature names, so it can be skipped
min_class_label = 1
max_class_label = 7
verbose_output = False


data = np.loadtxt(dataset_filename, delimiter=delimiter_char, skiprows=(skiprow_no)) #load the dataset


def k_fold_cross_validation():
    test_id = 0
    test_results = {}
    
    x = data[:, 0:-1]
    y = data[:,-1]
    kfold = min(min([ np.size(np.where(y==c)) for c in range(min_class_label,max_class_label+1)]), 10)
    skf =  StratifiedKFold(y, k=kfold)
    for train_index, test_index in skf:
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        print "Cross validation set :", test_id
        train_results = train(X_train, y_train, test_id)

        test_results[test_id] = {}
        for t in train_results.keys(): 
            test_results[test_id].update({ str(t)+'EM' : test(X_test, y_test, train_results[t]['weights']) })
            test_results[test_id].update({ str(t)+'MV' : test(X_test, y_test, train_results[t]['weights_mv']) })
            test_results[test_id].update({ str(t)+'AT' : test(X_test, y_test, train_results[t]['weights_at']) })
        test_id += 1
        
    print "Final results :"
    pprint (test_results)
    
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_position([0.05,0.15,0.9,0.8])
    ax.set_title('F-score, predictive accuracy')
    plot_graph(ax, test_results, kfold, type='EM', color=['ro', 'bs'], legendpos='lower right')
    plot_graph(ax, test_results, kfold, type='MV', color=['go', 'ys'], legendpos='upper left')
    plot_graph(ax, test_results, kfold, type='AT', color=['ko', 'ms'], legendpos='upper right')
    fontP = FontProperties()
    fontP.set_size('smaller')
    fig.legend(handle, ['F-score EM', "Predictive accuracy EM",'F-score MV', "Predictive accuracy MV",'F-score AT', "Predictive accuracy AT"], 
               loc="lower right", prop=fontP, ncol=3, bbox_to_anchor = (0.02, 0.02, 0.9, 0.1))#bbox_to_anchor=(0.5, -0.05))
    plt.show()
    
def train(X, y, set_id):
    EM_result = {}
    
    for ex in xrange(len(no_of_expert_groups)):
        print "For group ", ex
        expert_wrong_percentage = []
        for i in xrange(int(no_of_expert_groups[ex]*expert_bads)):
            #bads
            num = 0.60 #((random()%0.5) + 0.5) % 1.0
            expert_wrong_percentage.append(num)
        for i in xrange(int(no_of_expert_groups[ex]*expert_goods)):
            #goods
            num = 0.30 #random()%0.5
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
            iterations+=1
                        
    return EM_result


def test(X, y, learned_params):
    
    N = np.shape(X)[0] #no of instances
    X = np.append(np.ones((N,1)), X,1) #appending a column of ones as bias (used in logistic regression weights prediction)
    F = np.shape(X)[1] #no of features+1

    class_prob = []
    for w in learned_params.keys():
        prob = Utils.logistic_transformation(learned_params[w], X)
        class_prob.append(prob)
    
    max_prob = np.max(class_prob, 0)
    
    cs = np.array(class_prob)
    cs = np.reshape(cs,cs.size,order='F').reshape(np.shape(cs)[1],np.shape(cs)[0])
    second_max = (np.where(cs.argsort() == cs.shape[-1]-1)[1]+1)
    cs_prob = cs[np.where(cs.argsort() == cs.shape[-1]-1)] #
    
    predicted_y = []
    arbitrary_answer = []
    output_label = range(min_class_label, max_class_label+1)
    for i in xrange(np.size(max_prob)):
            class_label = np.where(class_prob == max_prob[i])[0]
            predicted_y.append(output_label[class_label[0]])
            arbitrary_answer.append(str(output_label[class_label[0]])+ ' prob '+ str(max_prob[i]) + ' second prob ' + str(cs_prob[i]) + 'second class' + str(second_max[i])  )
    
    
    #print "predicted y :", predicted_y
    #print "Actual y:", y
    accuracy = Utils.calculate_accuracy(np.array(y), np.array(predicted_y))
    f_score_mean, f_score_std = Utils.calculate_average_F1score(np.array(y), np.array(predicted_y), min_class_label, max_class_label)
    return (accuracy, f_score_mean, f_score_std)
    

handle = []

def plot_graph(ax, test_results, kfold, type, color, legendpos):
    
    plot_x = no_of_expert_groups
    plot_accerr = np.std([[test_results[k][str(g)+type][0] for  g in range(len(no_of_expert_groups)) ]for k in range(kfold) ],axis=0)
    plot_y = np.mean([[test_results[k][str(g)+type][1] for  g in range(len(no_of_expert_groups)) ] for k in range(kfold) ],axis=0)
    plot_yerr = np.mean([[test_results[k][str(g)+type][2] for  g in range(len(no_of_expert_groups)) ] for k in range(kfold) ],axis=0)
    plot_acc = np.mean([[test_results[k][str(g)+type][0] for  g in range(len(no_of_expert_groups))] for k in range(kfold) ],axis=0)
    ax.set_xlim(plot_x[0]-10, plot_x[-1]+10)
    ax.set_xticks(plot_x)
    handle.append(ax.errorbar(plot_x, plot_y, yerr=plot_yerr, fmt=color[0], ms=10))
    handle.append(ax.errorbar(plot_x, plot_acc, yerr=plot_accerr, fmt=color[1], ms=5, mec='w'))  
    print type, "fscore",  np.mean(plot_y)
    print type, "acc", np.mean(plot_acc) 
    
    
k_fold_cross_validation()
