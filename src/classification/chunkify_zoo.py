#!/usr/bin/env python

import numpy as np
from pprint import pprint

dataset_filename = '../../dataset/classification/dataset_zoo'
delimiter_char = ','
skiprow_no = 0 
min_class_label = 1
max_class_label = 7
data = np.loadtxt(dataset_filename, delimiter=delimiter_char, skiprows=(skiprow_no))

N = np.shape(data)[0]
F = np.shape(data)[1]-1


def getPartitions():
    classifieds = {}
    for c in range(min_class_label, max_class_label+1):
        classifieds[c] = np.where(data[:,F] == c)
    
        P = N/10 #no of partitions
        partitions = {}
        for p in range(P):
           partitions[p] = None
        #partitions[p] = {}

    
    for c in classifieds:
        for i in range(len(classifieds[c][0])):
            
            if partitions[i%P] == None:
                partitions[i%P] = data[classifieds[c][0][i]]
            else :
                partitions[i%P] = np.vstack([partitions[i%P], data[classifieds[c][0][i]]]) 
    #pprint(partitions)
    return partitions


if __name__ == "__main__":
    getPartitions()


