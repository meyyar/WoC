#!/usr/bin/env python

import sys
sys.path.append('../src/')

import numpy as np
from crowds_EM import Crowds_EM
from pprint import pprint


np.seterr(all='raise')

#alpha  - sensitivity of experts
#beta - specificity of experts

dataset_filename = '../dataset/winequality-red.csv'
delimiter_char = ';'
skiprow_no = 1 #The first row corresponds to feature names, so it can be skipped
#expert_wrong_percentage = [0.10, 0.90, 0.10, 0.20, 0.10] # expert y manipulation wrong percentage
expert_wrong_percentage = [0.90,0.90,0.90,0.30,0.40,0.50,0.60,0.90,0,0,1.0,0,0,0.30,0.70,0.40,0.10,0.20,0.30,0.70,0.60,0.50,0.90,1.0,0,0.30,0.40,0.80,0.80,0.20,0.40,0.50,0.10,0.90,0.20,0.30,0.40,0.50,0.90,0.90,0,0,1.0,0,0,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.50,0.90,1.0,0,0.90,0.90,0.80,0.80,0.90,1.0,0.90,0.10,0.90,0.20,0.30,0.40,0.50,0.60,0.90,0,0,1.0,0,0,0.90,0.90,0.90,0.90,0.90,0.90,0.70,0.90,0.90,0.90,1.0,0,0.90,0.90,0.80,0.80,0.90,0.90,0.90,0.10,1,1,0.90]
min_class_label = 0
max_class_label = 10


data = np.loadtxt(dataset_filename, delimiter=delimiter_char, skiprows=(skiprow_no)) #load the dataset


crowds_EM = Crowds_EM( data, min_class_label, max_class_label, expert_wrong_percentage, verbose= True)
crowds_EM.run()
pprint(crowds_EM.results)
crowds_EM.visualize()
