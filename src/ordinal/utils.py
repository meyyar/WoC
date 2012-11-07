#!/usr/bin/env python

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

""" Class Utils
    Provides the utility functions used in the EM algorithm
"""

class Utils:

    EPS = 1.e-1
    subplot = 210
    #figure = plt.figure()
    figure= None

    @staticmethod
    def logistic_transformation(w, x):
        """
        Returns the output of logistic transformation function
        """
        try:
            return (1 / (1 + (np.exp(-1 * np.dot(w.T, x.T) ))))
        except Exception, e:
            raise

    @staticmethod
    def a_calculations(alpha, experty, no_of_wines):
        """
        Returns the product of (alphaj)^yij and (1-alphaj)^(1-yij), i is the
        instance and j is the expert
        """
        try:
            one_alpha = 1-alpha
            alpha[alpha< Utils.EPS] = Utils.EPS
            one_alpha[one_alpha < Utils.EPS] = Utils.EPS
            a = np.zeros(no_of_wines)
            for i in range(np.size(a,0)):
                for j in range(np.size(alpha,0)):
                    a[i] += ((experty[j][i] * np.log(alpha[j])) +
                            ((1-experty[j][i])*np.log(one_alpha[j])))
                a[i] = np.exp(a[i])
            return a
        except Exception,e:
            raise

    @staticmethod
    def b_calculations(beta, experty, no_of_wines):
        """
        Returns the product of (betaj)^(1-yij) and (1-betaj)^yij, where i is the
        instance and j is the expert
        """
        try:
            b = np.zeros(no_of_wines)
            one_beta = 1-beta
            beta[beta < Utils.EPS] = Utils.EPS
            one_beta[one_beta < Utils.EPS] = Utils.EPS
            for i in range(np.size(b,0)):
                for j in range(np.size(beta,0)):
                    b[i] += (((1-experty[j][i]) * np.log(beta[j])) +
                            ((experty[j][i])*np.log(one_beta[j])))
                b[i] = np.exp(b[i])
            return b
        except Exception,e:
            raise

    @staticmethod
    def generate_y(y, wrong_percentage):
        """
        Returns the synthetic expert y value by corrupting the observed y by
        wrong_percentage
        """
        newy = y.copy()
        no_of_wrongs = wrong_percentage*np.size(y)
        sample = np.arange(np.size(y))
        np.random.shuffle(sample)
        for i in sample[:no_of_wrongs]:
            newy[i] =  1-y[i]
        return newy

    @staticmethod
    def calculate_RMSE(y_observed, y_predicted):
        """
        Returns the root mean square error between the actual y values and predicted y values
        """
        error  = (y_observed-y_predicted.round())
        squared_error = np.sum(error**2)
        mean_squared_error = np.mean(squared_error)

        root_mean_squared_error = math.sqrt(mean_squared_error)
        return root_mean_squared_error
    
    
    @staticmethod
    def calculate_accuracy(y_observed, y_predicted):
        """
        Returns the accuracy of the predicted output y values
        """
        return np.size(np.where(y_observed==y_predicted.round()))/float(np.size(y_observed))
    
    @staticmethod
    def calculate_error_rate(y_observed, y_predicted):
        """
        Returns the accuracy of the predicted output y values
        """
        return np.size(np.where(y_observed!=y_predicted.round()))/float(np.size(y_observed))
  
    
    @staticmethod
    def calculate_F1score(y_observed, y_predicted):
        """
        Returns the F1-score for the logistic regression model
        """
        true_positives = np.size(np.intersect1d(np.where(y_observed==1)[0], np.where(y_predicted.round()==1)[0])) + 1.e-50 
        misses = np.size(np.intersect1d(np.where(y_observed==1)[0], np.where(y_predicted.round()==0)[0])) 
        false_alarms = np.size(np.intersect1d(np.where(y_observed==0)[0], np.where(y_predicted.round()==1)[0])) 
        
        recall = true_positives / float(true_positives+misses)
        precision = true_positives / float(true_positives+false_alarms)
        f1_score = 2*precision*recall / float(precision + recall)
        return round(f1_score,2)
        
    @staticmethod
    def calculate_average_F1score(y_observed, y_predicted, min_class_label, max_class_label):
        """
        Returns the average F1 score over all the the output class labels
        """
        f1_avg = 0
        f_score = []
        for c in range(min_class_label, max_class_label+1):
            y_observed_ova = np.zeros(np.shape(y_observed))
            y_predicted_ova = np.zeros(np.shape(y_predicted))
            obs_ind = np.where(y_observed == c)
            pred_ind = np.where(y_predicted == c)
            y_observed_ova[obs_ind] = 1
            y_predicted_ova[pred_ind] = 1
            f = Utils.calculate_F1score(y_observed_ova, y_predicted_ova)
            f_score.append(f)
            #print "intermediate :", f 
            
            f1_avg += f
        
        return f1_avg/float(max_class_label-min_class_label+1), np.std(np.array(f_score))

    
    @staticmethod
    def add_plot(title, plots, min_class_label,max_class_label):
        colors = ['r','y']
        place = [0, 0.35]
        
        Utils.subplot+=1
        #fig = .figure()
        ax = Utils.figure.add_subplot(Utils.subplot)
 
        Utils.figure.subplots_adjust(hspace=0.75, wspace = 0.5)
        ax.set_title(title)
        ax.set_xticks(np.arange(8))
        for i in xrange(len(plots)):
            item = plots[i]
            if item['xlim'] != None:
                ax.set_xlim(item['xlim'])
            if item['ylim'] != None:
                ax.set_ylim(item['ylim'])
            ax.set_xlabel(item['xlabel'])
            ax.set_ylabel(item['ylabel'])
            ax.bar(np.arange(min_class_label, max_class_label+1 )+place[i], item['y_values'], 0.35, color=colors[i],label=item['label'])

        handles, labels = ax.get_legend_handles_labels()
        fontP = FontProperties()
        fontP.set_size('small')
        Utils.figure.legend(handles, labels, loc="upper right", prop=fontP)


    @staticmethod
    def initPlot(no_of_expertgroups, fignum):
        cols = 2
        rows = no_of_expertgroups/2 + (no_of_expertgroups%2)
        Utils.subplot = int(str(rows)+str(cols)+'0')
        Utils.figure = plt.figure(fignum+1)
        Utils.figure.suptitle('F1 score for EM and MV')
        
            
    @staticmethod
    def visualize( results, min_class_label, max_class_label, no_of_experts):
        Utils.add_plot( str(no_of_experts) + ' Experts',[
              {
              'x_values':results['EM_perf']['f1_score'].keys(),
              'y_values':results['EM_perf']['f1_score'].values(),
              'xlabel':'y label',
              'ylabel':'f1_score',
              'label':'EM f1_score',
              'xlim':(min_class_label,max_class_label+1),
              'ylim':( min(min(results['EM_perf']['f1_score'].values()), min(results['MV_perf']['f1_score'].values())), max(max(results['EM_perf']['f1_score'].values()),max(results['MV_perf']['f1_score'].values()))+0.02)},
              {
              'x_values':results['MV_perf']['f1_score'].keys(),
              'y_values':results['MV_perf']['f1_score'].values(),
              'xlabel':'y label',
              'ylabel':'f1_score',
              'label':'MV f1_score',
              'xlim':(min_class_label,max_class_label+1),
              'ylim':(0,1)}
                ], min_class_label, max_class_label)

    @staticmethod
    def showPlot():
        plt.show() 
    
    
    """@staticmethod
    def calculate_F1score(y_observed, y_predicted):
        
        #Returns the F1-score for the logistic regression model
        
        true_positives = np.size(np.intersect1d(np.where(y_observed==1)[0], np.where(y_predicted.round()==1)[0])) + 1.e-50 
        misses = np.size(np.intersect1d(np.where(y_observed==1)[0], np.where(y_predicted.round()==0)[0])) 
        false_alarms = np.size(np.intersect1d(np.where(y_observed==0)[0], np.where(y_predicted.round()==1)[0])) 
        
        recall = true_positives / float(true_positives+misses)
        precision = true_positives / float(true_positives+false_alarms)
        f1_score = 2*precision*recall / float(precision + recall)
        return round(f1_score,2)"""
        
        
    """@staticmethod
    def show_plot(title, plots):

        
        
        colors = ['r','y']
        place = [0, 0.35]
        plt.title(title)
        Utils.subplot+=1
        ax = Utils.figure.add_subplot(Utils.subplot)

        for i in xrange(len(plots)):
            
            item = plots[i]
            if item['xlim'] != None:
                ax.set_xlim(item['xlim'])
            if item['ylim'] != None:
                ax.set_ylim(item['ylim'])
            ax.set_xlabel(item['xlabel'])
            ax.set_ylabel(item['ylabel'])
            
            #plt.title(item['title'])
            #fig_plt = plt.plot(item['x_values'], item['y_values'], 'o-', label=item['label'])
            ax.set_xticks(np.arange(7)+place[i])
            
            ax.bar(np.arange(7)+place[i], item['y_values'], 0.35, color=colors[i])
            fontP = FontProperties()
            fontP.set_size('small')
            ax.legend(loc="lower right", prop=fontP)


    @staticmethod
    def partition_data(data, n):
        return [data[i:i+n] for i in range(0, len(data), n)]"""
