#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from matplotlib import pyplot as plt
from auxiliaryFunctions import AuxiliaryFunctions 
class Graph:    
    
    
    def train_loss_validation_loss_epoch(path_graph,train_loss,validation_loss):
        
        AuxiliaryFunctions.make_dir(path_graph)
        plt.figure()
        plt.plot(train_loss)
        plt.plot(validation_loss)
        plt.title('Train Loss & Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(path_graph+'/graph_loss_train_validation.png')
        plt.show()

    def psnr_test(path_graph,pnsr_1_list,pnsr_2_list,pnsr_3_list):
        plt.hist(pnsr_1_list,  linewidth=1, label ='clean-output')
        plt.hist(pnsr_2_list,  linewidth=1, label = 'clean-dirty')
        plt.hist(pnsr_3_list, linewidth=1, label = 'dirty-output')
        plt.legend(loc='upper right')
        plt.savefig(path_graph+'/graph_psnr.png')
        plt.show()

