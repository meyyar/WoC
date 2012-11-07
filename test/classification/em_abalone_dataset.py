#!/usr/bin/env python

import sys
sys.path.append('../src/')

import numpy as np
from crowds_EM import Crowds_EM


np.seterr(all='raise')

#alpha  - sensitivity of experts
#beta - specificity of experts

dataset_filename = '../dataset/dataset_abalone'
delimiter_char = ','
skiprow_no = 0 #The first row corresponds to feature names, so it can be skipped
expert_wrong_percentage = [0.10, 0.90, 0.10, 0.20, 0.10, 0.80, 0.20, 0.30,0.90, 0.90, 0.80] # expert y manipulation wrong percentage
min_class_label = 5
max_class_label = 15


data = np.loadtxt(dataset_filename, delimiter=delimiter_char, skiprows=(skiprow_no)) #load the dataset


crowds_EM = Crowds_EM( data, min_class_label, max_class_label, expert_wrong_percentage, verbose= True)
crowds_EM.run()
