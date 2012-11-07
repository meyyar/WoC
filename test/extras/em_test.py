#!/usr/bin/env python

from sys import argv, path, exit
path.append('../src/')
import numpy as np
from crowds_EM import Crowds_EM
from pprint import pprint
from optparse import OptionParser


if __name__ == "__main__":

    parser = OptionParser(usage="usage: %prog [options]",version="%prog 1.0")
    parser.add_option("-f", "--filename",
                        dest="filename",
                        help="Path to the dataset file.")
    parser.add_option("-d", "--delimiter",
                        dest="delimiter",
                        default=',',
                        help="Delimiter character used in the dataset file. Defaults to ','.")
    parser.add_option("-s", "--skiprows",
                        dest="skiprows",
                        default=0,
                        help="No. of rows to skip from the beginning in the dataset file. Defaults to 0.")
    parser.add_option("-e", "--expert",
                        dest="expert",
                        help="A list specifying the expert wrong percentage to generate expert answers. Example = [0.10, 0.90, 0.20] generates 3 experts with 10%, 90%, 20% wrong answers.")
    parser.add_option("-b", "--beginclass",
                        dest="beginclass",
                        help=" Beginning integer value of the y class labels.")
    parser.add_option("-l", "--lastclass",
                        dest="lastclass",
                        help="Last integer value of the y class labels.")
    parser.add_option("-v", "--verbose",
                        dest="verbose",
                        default=False,
                        help="Verbose output True/False. Defaults to False.")


    (options, args) = parser.parse_args()

    exp = options.expert.split(',')
    l = []
    for e in exp:
        l.append(float(e))
    options.expert = l
    options.beginclass = int(options.beginclass)
    options.lastclass = int(options.lastclass)
    options.skiprows = int(options.skiprows)

    np.seterr(all='raise')

    data = np.loadtxt(options.filename, delimiter=options.delimiter, skiprows=options.skiprows) #load the dataset


    crowds_EM = Crowds_EM( data, options.beginclass, options.lastclass, options.expert, verbose=options.verbose)
    crowds_EM.run()
    pprint(crowds_EM.results)
    crowds_EM.visualize()


    """
    ./em_test.py  -f ../dataset/dataset_car  -e 0.10,0.90,0.20,0.30,0.40,0.50,0.60,0.90,0,0,1.0,0,0,0.30,0.70,0.40,0.10,0.20,0.30,0.70,0.60,0.50,0.90,1.0,0,0.30,0.40,0.80,0.80,0.20,0.40,0.50,0.10,0.90,0.20,0.30,0.40,0.50,0.90,0.90,0,0,1.0,0,0,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.50,0.90,1.0,0,0.90,0.90,0.80,0.80,0.20,0.40,0.50,0.10,0.90,0.20,0.30,0.40,0.50,0.60,0.90,0,0,1.0,0,0,0.90,0.90,0.90,0.90,0.90,0.90,0.70,0.90,0.90,0.90,1.0,0,0.90,0.90,0.80,0.80,0.90,0.90,0.90,0.10,1,1,0.90   -b 0 -l 3 -s 0 -d ','
    """
