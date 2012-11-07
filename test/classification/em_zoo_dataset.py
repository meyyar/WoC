#!/usr/bin/env python

import sys
sys.path.append('../src/')

import numpy as np
from crowds_EM import Crowds_EM
from pprint import pprint

np.seterr(all='raise')

#alpha  - sensitivity of experts
#beta - specificity of experts

dataset_filename = '../dataset/dataset_zoo'
delimiter_char = ','
skiprow_no = 0 #The first row corresponds to feature names, so it can be skipped
expert_wrong_percentage = [0.90, 0.90, 0.90, 0.90, 0.90, 0.80, 0.90, 0.90,0.90, 0.90, 0.80] # expert y manipulation wrong percentage
min_class_label = 1
max_class_label = 7
verbose_output = False


data = np.loadtxt(dataset_filename, delimiter=delimiter_char, skiprows=(skiprow_no)) #load the dataset


crowds_EM = Crowds_EM( data, min_class_label, max_class_label, expert_wrong_percentage, verbose= verbose_output)
crowds_EM.run()
pprint(crowds_EM.results)
crowds_EM.visualize()
