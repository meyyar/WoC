#!/usr/bin/env python



import numpy as np
from crowds_EM import Crowds_EM
from pprint import pprint
from utils import Utils
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

np.seterr(all='raise')


no_of_expert_groups = [20]#,40,60,80,100,120,140,160,180,200]
expert_goods = 0.90
expert_bads = 0.10
dataset_filename = '../../dataset/ordinal/dataset_car'
delimiter_char = ','
skiprow_no = 0 #The first row corresponds to feature names, so it can be skipped
min_class_label = 0
max_class_label = 3
verbose_output = False
synthetic_data = False
missing_data = True
pmissing = [0.0, 0.1, 0.2, 0.3, 0.4 ]#, 0.5, 0.6, 0.7, 0.8, 0.9]
if not synthetic_data: 
    data = np.loadtxt(dataset_filename, delimiter=delimiter_char, skiprows=(skiprow_no)) #load the dataset
else:
            x1 = np.random.normal(-1, 0.1, 5)
            x2 = np.random.normal(1,0.1,5)
            
            xapp = np.append(x1,x2)
            xapp = xapp.reshape(10,1)
            
            
            
            #x = np.append(np.ones((self.N,1)), self.x, 1)
            
            
            yapp = np.array([0,0,0,0,0,1,1,1,1,1])
            yapp = yapp.reshape(10,1)
            data = np.append(xapp, yapp, 1)
            min_class_label = 0
            max_class_label = 1
            

handle = []

#10 fold cross validation
def k_fold_cross_validation():
    #partitioned_data = getPartitions()
    #k = len(partitioned_data.keys())
    test_id = 0
    learned_params = []
    test_results = {}
    
    x = data[:, 0:-1]
    y = data[:,-1]
    kfold = min([ np.size(np.where(y==c)) for c in range(min_class_label,max_class_label+1)], 5)
    skf =  StratifiedKFold(y, k=kfold)
    for train_index, test_index in skf:
        #print("TRAIN: %s TEST: %s" % (train_index, test_index))
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        print "Cross validation set :", test_id
        train_results = train(X_train, y_train, test_id)
        
        """
        For test_1 with average of weights for every class over all cross-validation set:
        """

        test_results[test_id] = {}
        for t in train_results.keys(): 
            #test_results[test_id].update({ t : test(X_test, y_test, train_results[t]['weights']) })
            for missing_percent in train_results[t].keys(): 
                test_results[test_id].update({ str(t)+'_'+str(missing_percent)+'EM' : test(X_test, y_test, train_results[t][missing_percent]['weights']) })
                test_results[test_id].update({ str(t)+'_'+str(missing_percent)+'MV' : test(X_test, y_test, train_results[t][missing_percent]['weights_mv']) })
                test_results[test_id].update({ str(t)+'_'+str(missing_percent)+'AT' : test(X_test, y_test, train_results[t][missing_percent]['weights_at']) })
            
        
        test_id += 1
        
    print "Final results :"
    pprint (test_results)
    #plot_graph(test_results, kfold)
    
    
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
    
    
    fig2 = plt.figure()
    ax2 = plt.subplot(111)
    ax2.set_position([0.05,0.15,0.9,0.8])
    ax2.set_title('Damage ratio')
    plot_damage_ratio(ax2, test_results, kfold)
    plt.show()  
    
    
def train(X, y, set_id):
    EM_result = {}
    
    for ex in xrange(len(no_of_expert_groups)):
        #generate expert #self.Training_instances = self.N - self.Testing_instanceswrong percentage
        #60% good:40 % bad
        EM_result[ex] = {}
        for missing_percent in pmissing:    
            print "For group ", ex
            expert_wrong_percentage = []
            for i in xrange(int(no_of_expert_groups[ex]*expert_bads)):
                #bads
                num = 0.90 #((random()%0.5) + 0.5) % 1.0
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
                    crowds_EM = Crowds_EM( X, y, min_class_label, max_class_label, expert_wrong_percentage, verbose= verbose_output, synthetic=synthetic_data, missing=missing_data, percent_missing=missing_percent)
                    #crowds_EM.run_EM_missing()
                    crowds_EM.run()
                except Exception,e:
                    #Rerunning ...
                    import traceback
                    print traceback.print_exc()
                    failed+=1
                    try:
                        crowds_EM = Crowds_EM( X, y, min_class_label, max_class_label, expert_wrong_percentage, verbose= verbose_output, synthetic=synthetic_data, missing=missing_data, percent_missing=missing_percent)
                        #crowds_EM.run_EM_missing()
                        crowds_EM.run()
                    except Exception,e:
                        failed+=1
                        pass
                    else :
                        EM_result[ex][missing_percent] = crowds_EM.results
                        break
                else :
                    EM_result[ex][missing_percent] = crowds_EM.results
                    break
                #print "iteration :" , iterations
                iterations+=1
                            
    return EM_result


def test(X, y, learned_params):
    
    N = np.shape(X)[0] #no of instances
    X = np.append(np.ones((N,1)), X,1) #appending a column of ones as bias (used in logistic regression weights prediction)
    F = np.shape(X)[1] #no of features+1
    
    
    p_old = 1
    class_prob = []
    for w in learned_params.keys():
        
        p = Utils.logistic_transformation( learned_params[w], X )
        class_prob.append(p_old-p)
        p_old = p
    class_prob.append(p_old)
    
    max_prob = np.max(class_prob, 0)
    
    predicted_y = []
    output_label = range(min_class_label, max_class_label+1)
    for i in xrange(np.size(max_prob)):
            class_label = np.where(class_prob == max_prob[i])[0]
            predicted_y.append(output_label[class_label[0]])
    
    #print "predicted y :", predicted_y
    #print "Actual y:", y
    accuracy = Utils.calculate_accuracy(np.array(y), np.array(predicted_y))
    #print "accuracy for test data :", accuracy
    f_score_mean, f_score_std = Utils.calculate_average_F1score(np.array(y), np.array(predicted_y), min_class_label, max_class_label)
    #print "Average f score for test data :", f_score_mean
    
    error_rate = Utils.calculate_error_rate(np.array(y), np.array(predicted_y))
    return (accuracy, f_score_mean, f_score_std, error_rate)

def plot_graph(ax, test_results, kfold, type, color, legendpos):

    plot_x = pmissing
    plot_accerr = np.std([[test_results[k][str(exg)+'_'+str(g)+type][0]  for exg in range(len(no_of_expert_groups)) for  g in pmissing ]for k in range(kfold) ],axis=0)
    plot_y = np.mean([[test_results[k][str(exg)+'_'+str(g)+type][1] for exg in range(len(no_of_expert_groups)) for  g in pmissing ] for k in range(kfold) ] ,axis=0)
    plot_yerr = np.mean([[test_results[k][str(exg)+'_'+str(g)+type][2]  for exg in range(len(no_of_expert_groups)) for  g in pmissing ] for k in range(kfold) ],axis=0)
    plot_acc = np.mean([[test_results[k][str(exg)+'_'+str(g)+type][0]  for exg in range(len(no_of_expert_groups)) for  g in pmissing ] for k in range(kfold) ],axis=0)
    
    ax.set_xlim(plot_x[0]-0.1, plot_x[-1]+0.1)
    ax.set_xticks(plot_x)
    handle.append(ax.errorbar(plot_x, plot_y, yerr=plot_yerr, fmt=color[0], ms=10))
    handle.append(ax.errorbar(plot_x, plot_acc, yerr=plot_accerr, fmt=color[1], ms=5, mec='w'))
      
      
    print type, "fscore",  np.mean(plot_y)
    print type, "acc", np.mean(plot_acc) 

def plot_damage_ratio(ax, test_results, kfold):
    
    plot_x = pmissing
    e1 = np.mean([[test_results[k][str(exg)+'_'+str(g)+'AT'][3]  for exg in range(len(no_of_expert_groups)) for  g in pmissing ]for k in range(kfold) ],axis=0)
    e3 = np.mean([[test_results[k][str(exg)+'_'+str(g)+'EM'][3]  for exg in range(len(no_of_expert_groups)) for  g in pmissing ]for k in range(kfold) ],axis=0)
    e2 = np.mean([[test_results[k][str(exg)+'_'+str(g)+'MV'][3]  for exg in range(len(no_of_expert_groups)) for  g in pmissing ]for k in range(kfold) ],axis=0)
    excess_error_rate = abs((e3-e1) / (e2-e1+1.e-50))
    deviation = np.std(excess_error_rate)
    pprint (excess_error_rate)
    ax.set_xlim(plot_x[0]-0.1, plot_x[-1]+0.1)
    ax.set_xticks(plot_x)
    ax.errorbar(plot_x, excess_error_rate, yerr=deviation, fmt='gs', ms=10, mec='w')
    
    
k_fold_cross_validation()
