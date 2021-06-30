#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from auxiliaryFunctions import AuxiliaryFunctions 
from matplotlib import colors

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
     
    def psnr_test(path_graph,pnsr_clean_output,pnsr_clean_dirty,pnsr_diff):  
        plt.hist(pnsr_clean_output, color='b', edgecolor='b',linewidth=1,label ='clean-output')
        plt.xlabel('psnr output image')
        plt.ylabel('count images')
        plt.savefig(path_graph+'/graph_psnr_clean_output.png')
        plt.show()

        plt.hist(pnsr_clean_dirty,  color='g', edgecolor='g' ,linewidth=1, label = 'clean-dirty')
        plt.legend(loc='upper right')
        plt.xlabel('psnr dirty image')
        plt.ylabel('count images')
        plt.savefig(path_graph+'/graph_psnr_clean_dirty.png')
        plt.show()

        plt.hist(pnsr_diff, linewidth=1,  color='r', edgecolor='r', label = 'diff output-dirty')
        plt.xlabel('diff output pnsr &  dirty pnsr')
        plt.ylabel('count images')
        plt.savefig(path_graph+'/graph_psnr_diff_output_dirty.png')
        plt.show() 
        
    def read_graph(path):
        img = mpimg.imread(path)
        imgplot = plt.imshow(img)
        plt.show()