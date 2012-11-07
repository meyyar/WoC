#!/usr/bin/env python



import numpy as np
from crowds_EM_classi import Crowds_EM
from pprint import pprint
from utils_classi import Utils
from random import random
from chunkify_zoo import getPartitions
from sys import stdin

np.seterr(all='raise')

#alpha  - sensitivity of experts
#beta - specificity of experts

#input params

no_of_experts = [20]
expert_goods = 0.60
expert_bads = 0.40
dataset_filename = '../../dataset/classification/dataset_zoo'
delimiter_char = ','
skiprow_no = 0 #The first row corresponds to feature names, so it can be skipped
min_class_label = 0
max_class_label = 1
verbose_output = False
synthetic_data = True


data = np.loadtxt(dataset_filename, delimiter=delimiter_char, skiprows=(skiprow_no)) #load the dataset


#10 fold cross validation
def k_fold_cross_validation():
    partitioned_data = getPartitions()
    k = len(partitioned_data.keys())
    test_id = 0
    
    while test_id < k :
        train_data = None
        test_data = None
        print "Cross validation set :", test_id
        for i in range(k):
            if ( i == test_id ):
                test_data = partitioned_data[i]
            else :
                if train_data != None:
                    train_data = np.vstack([train_data,partitioned_data[i]])
                else :
                    train_data = partitioned_data[i]
        
        #Training phase :
        #train_results = train(train_data, test_id)
        test_results = test(test_data)      
        test_id += 1
        
def train(training_data, set_id):
    EM_result = {}
    Utils.initPlot( len(no_of_experts), set_id)
    for ex in xrange(len(no_of_experts)):
        #generate expert #self.Training_instances = self.N - self.Testing_instanceswrong percentage
        #60% good:40 % bad
        print "For group ", ex
        expert_wrong_percentage = []
        for i in xrange(int(no_of_experts[ex]*expert_bads)):
            #bads
            num = 0.90 #((random()%0.5) + 0.5) % 1.0
            expert_wrong_percentage.append(num)
        for i in xrange(int(no_of_experts[ex]*expert_goods)):
            #goods
            num = 0.20 #random()%0.5
            expert_wrong_percentage.append(num)                        
        crowds_EM = None    
        failed = 0
        iterations = 0
        total_iter = 10
        while iterations < total_iter :
            try:
                crowds_EM = Crowds_EM( training_data, min_class_label, max_class_label, expert_wrong_percentage, verbose= verbose_output, synthetic=synthetic_data)
                crowds_EM.run_EM_missing()
            except Exception,e:
                #Rerunning ...
                import traceback
                print traceback.print_exc()
                failed+=1
                try:
                    crowds_EM = Crowds_EM( training_data, min_class_label, max_class_label, expert_wrong_percentage, verbose= verbose_output, synthetic = synthetic_data)
                    crowds_EM.run_EM_missing()
                except Exception,e:
                    failed+=1
                    pass
                else :
                    EM_result[ex] = crowds_EM.results
                    break
            else :
                EM_result[ex] = crowds_EM.results
                break
            #print "iteration :" , iterations
            iterations+=1
                        
    print "Final results :\n"
    pprint(EM_result)
    #for e in xrange(len(no_of_experts)):
    #    Utils.visualize( EM_result[e], min_class_label, max_class_label, no_of_experts[e] )   
    return EM_result
    

def test(test_data):
    print "Test data:"
    pprint (test_data)
    
    
#k_fold_cross_validation()
train(data, 0)
Utils.showPlot()
        
"""else:
            EM_perf = crowds_EM.predict_EM(crowds_EM.x, crowds_EM.y)
            print "EM perf ", EM_perf 
            EM_acc += EM_perf
            if EM_perf > EM_highest_performance['accuracy']:
                EM_highest_performance['accuracy'] = EM_perf
                EM_highest_performance['results'] = crowds_EM.results
                
            MV_acc += crowds_EM.predict_MV(crowds_EM.x, crowds_EM.y)
            #np.save('X.npy', crowds_EM.x)"""
 

"""print "No. of failed iterations : ", failed    
print "Average EM accuracy after ", iterations," iter : ", EM_acc/(total_iter-failed)
print "Average MV accuracy after ", iterations, " iter : ", MV_acc/(total_iter-failed)
print "EM highest accuracy :" , EM_highest_performance['accuracy']
print "EM with highest accuracy deatils:\n", EM_highest_performance['results']"""


"""x1 = np.random.normal(-1, 0.1, 10)
x2 = np.random.normal(1, 0.1, 10)

testing_x = np.append(x1, x2)
testing_x = testing_x.reshape(20,1)
testing_x = np.append(np.ones((20,1)), testing_x, 1)
class_prob = []
for w in crowds_EM.results['weights'].keys():
    prob = Utils.logistic_transformation(crowds_EM.results['weights'][w], testing_x)
    class_prob.append(prob)

pprint (class_prob)
max_prob = np.max(class_prob, 0)
print max_prob

predicted_y = []
for i in xrange(np.size(max_prob)):
        class_label = np.where(class_prob == max_prob[i])[0]
        predicted_y.append(class_label[0])

print "predicted y :", predicted_y"""