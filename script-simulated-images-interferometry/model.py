#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

##
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

#sesion save# 
import dill

import cupy as cp
from datasetInterferometry import DatasetInterferometry
from auxiliaryFunctions import AuxiliaryFunctions
from graph import Graph
import numpy as np

# In[ ]:


class Model:
    def __init__(self,size_figure,type_psf,num_epochs,learning_rate,batch_train,criterion,optimizer,start,stop,step = None,path_validation= None,path_test=None,path_graph = None,path_model = None):
        self.size_figure = size_figure
        self.type_psf = type_psf
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_train = batch_train
        self.start = start
        self.stop = stop
        self.step = step
        self.criterion = criterion
        self.optimizer = optimizer
        self.size_data = start-stop
        self.path_test  =  self.init_path_test(path_test)
        self.path_graph =  self.init_path_graph(path_graph)
        self.path_model = self.init_path_model(path_model)
         
    def init_path_test(self,path_test):
        if (path_test == None):
            return '../datasets/images_'+str(self.size_figure )+'x'+str(self.size_figure )+'/convolutions/'+self.type_psf+'/models/model_S'+str(self.size_data)+'_B'+str(self.batch_train)+'_E'+str(self.num_epochs)+'/test'
        else:
            return path_test 
        
    def init_path_graph(self,path_graph):
        if (path_graph == None):
            return '../datasets/images_'+str(self.size_figure )+'x'+str(self.size_figure )+'/convolutions/'+self.type_psf+'/models/model_S'+str(self.size_data)+'_B'+str(self.batch_train)+'_E'+str(self.num_epochs)+'/graph'
        else:
            return path_graph 
        
    def init_path_model(self,path_model):
         if (path_model == None):
            return '../datasets/images_'+str(self.size_figure )+'x'+str(self.size_figure )+'/convolutions/'+self.type_psf+'/models/model_S'+str(self.size_data)+'_B'+str(self.batch_train)+'_E'+str( self.num_epochs)+'/model' 
         else:
            return path_model 
  
    
    def train_data_memory(self,net,device,start,stop,step):
        train_loss = []
        valid_loss = []
        list_index = cp.arange(start,stop,step)
        data = DatasetInterferometry(self.size_figure,self.type_psf,self.batch_train)
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            size_train = 0
            for index in list_index:
                trainLoader = data.create_train_data(index,step)
                size_train = size_train +len(trainLoader)
                for dirty,clean in tqdm((trainLoader)):
                    dirty,clean=dirty.to(device).float(),clean.to(device).float()
                    self.optimizer.zero_grad()            
                    outputs = net(dirty)
                    loss = self.criterion(outputs, clean)
                    #backpropagation
                    loss.backward()
                    #update the parameters
                    self.optimizer.step()
                    running_loss += loss.item()
            loss = running_loss / size_train
            train_loss.append(loss)
        ## validation ## ''
            print('Epoch {} of {}, Train Loss: {:.3f}, Len Train:{}'.format(epoch+1, self.num_epochs, loss,size_train))
            validation_loss = 0.0
            net.eval()
            size_validate = 0  
            for index in list_index:
                validationloader = data.create_validation_data(index,step)
                size_validate = size_validate + len(validationloader)
                for dirty, clean in validationloader:
                    dirty,clean=dirty.to(device).float(),clean.to(device).float()
                    self.optimizer.zero_grad()    ## TODO BORRRAR ??         
                    outputs = net(dirty)
                    loss = self.criterion(outputs, clean)
                    validation_loss += loss.item()
            loss = validation_loss / size_validate
            valid_loss.append(loss)
            print('Epoch {} of {}, Validate Loss: {:.3f} Len Validate:{}'.format(epoch+1, self.num_epochs, loss,size_validate))
        return net,train_loss,valid_loss
    
    def test_data_memory(self,net,device,start,stop,step):
        list_index = cp.arange(start,stop,step)
        data = DatasetInterferometry(self.size_figure,self.type_psf,self.batch_train)
        pnsr_1_list = []
        pnsr_2_list = []
        pnsr_3_list = []
        i = 0
        for index in list_index:
            testloader = data.create_test_data(index,step)
            for dirty,clean in tqdm((testloader)):
                dirty,clean=dirty.to(device).float(),clean.float()
                outputs = net(dirty)
                dirty  = dirty.cpu().data
                outputs = outputs.cpu().data
                psnr_1 = cv2.PSNR(np.array(clean), np.array(outputs))
                psnr_2 = cv2.PSNR(np.array(clean), np.array(dirty))
                psnr_3 = cv2.PSNR(np.array(dirty), np.array(outputs))
                pnsr_1_list.append(psnr_1)
                pnsr_2_list.append(psnr_2)
                pnsr_3_list.append(psnr_3)

                AuxiliaryFunctions.save_decoded_image(dirty, self.size_figure,name=self.path_test+'/noisy{}.png'.format(i))
                AuxiliaryFunctions.save_decoded_image(outputs, self.size_figure,
                                                      name=self.path_test+'/denoised{}.png'.format(i))
                AuxiliaryFunctions.save_decoded_image(clean, self.size_figure,
                                                      name=self.path_test+'/clean{}.png'.format(i))
                i = i +1
        print(' Len test:{}'.format(i))
        return net,pnsr_1_list,pnsr_2_list,pnsr_3_list
    
    
    ### TODO: CARGADO POR EPOCA  PODRIA SER UNA OPCION!! ##
    def train_data(self,net,device,start,stop):
        train_loss = []
        valid_loss = []
        data = DatasetInterferometry(self.size_figure,self.type_psf,self.batch_train)
        trainLoader = data.read_train_data(start,stop)
        validationloader = data.read_validation_data(start,stop)
        for epoch in range(self.num_epochs):
            running_loss = 0.0           
            for dirty,clean in tqdm((trainLoader)):
                dirty,clean=dirty.to(device).float(),clean.to(device).float()
                self.optimizer.zero_grad()            
                outputs =net(dirty)
                loss = self.criterion(outputs, clean)
                #backpropagation
                loss.backward()
                #update the parameters
                self.optimizer.step()
                running_loss += loss.item()
            loss = running_loss / len(trainLoader)
            train_loss.append(loss)
            print('Epoch {} of {}, Train Loss: {:.3f}, Len Train:{}'.format(epoch+1, self.num_epochs, loss,len(trainLoader)))
            validation_loss = 0.0
            net.eval()
            size_validate = 0  
            for dirty, clean in validationloader:
                dirty,clean=dirty.to(device).float(),clean.to(device).float()
                self.optimizer.zero_grad()         
                outputs = net(dirty)
                loss = self.criterion(outputs, clean)
                validation_loss += loss.item()
            loss = validation_loss / len(validationloader)
            valid_loss.append(loss)
            print('Epoch {} of {}, Validate Loss: {:.3f} Len Validate:{}'.format(epoch+1, self.num_epochs, loss,
                                                                                 len(validationloader)))
        return net,train_loss,valid_loss
    
    def test_data(self,net,device,start,stop):
        data = DatasetInterferometry(self.size_figure,self.type_psf,self.batch_train)
        testloader = data.read_test_data(start,stop)
        pnsr_1_list = []
        pnsr_2_list = []
        pnsr_3_list = []
        i = 0
        for dirty,clean in tqdm((testloader)):
            dirty,clean=dirty.to(device).float(),clean.float()
            outputs = net(dirty)
            dirty  = dirty.cpu().data
            outputs = outputs.cpu().data
            psnr_1 = cv2.PSNR(np.array(clean), np.array(outputs))
            psnr_2 = cv2.PSNR(np.array(clean), np.array(dirty))
            psnr_3 = cv2.PSNR(np.array(dirty), np.array(outputs))
            pnsr_1_list.append(psnr_1)
            pnsr_2_list.append(psnr_2)
            pnsr_3_list.append(psnr_3)

            AuxiliaryFunctions.save_decoded_image(dirty, self.size_figure,name=self.path_test+'/noisy{}.png'.format(i))
            AuxiliaryFunctions.save_decoded_image(outputs, self.size_figure,
                                                  name=self.path_test+'/denoised{}.png'.format(i))
            AuxiliaryFunctions.save_decoded_image(clean, self.size_figure,
                                                  name=self.path_test+'/clean{}.png'.format(i))
            i = i +1
        print(' Len test:{}'.format(i))
        return net,pnsr_1_list,pnsr_2_list,pnsr_3_list
    
    
    def run_data_memory(self,net,start,stop,step):
        AuxiliaryFunctions.make_dir(self.path_test)
        AuxiliaryFunctions.make_dir(self.path_graph)
        AuxiliaryFunctions.make_dir(self.path_model)
        
        torch.cuda.empty_cache()
        device = AuxiliaryFunctions.get_device()
        net.to(device)
        net,train_loss,validation_loss = self.train_data_memory(net,device,start,stop,step)
        AuxiliaryFunctions.save_model(net,self.path_model)
  
        Graph.train_loss_validation_loss_epoch(path_graph =self.path_graph ,train_loss = train_loss,
                                               validation_loss = validation_loss)        
        net,pnsr_1_list,pnsr_2_list,pnsr_3_list = self.test_data_memory(net,device,start,stop,step)
        Graph.psnr_test(self.path_graph,pnsr_1_list,pnsr_2_list,pnsr_3_list)
        return net


        
        
    def run_data(self,net,start,stop):
        AuxiliaryFunctions.make_dir(self.path_test)
        AuxiliaryFunctions.make_dir(self.path_graph)
        AuxiliaryFunctions.make_dir(self.path_model)
        
        torch.cuda.empty_cache()
        device = AuxiliaryFunctions.get_device()
        net.to(device)
        net,train_loss,validation_loss = self.train_data(net,device,start,stop)
        AuxiliaryFunctions.save_model(net,self.path_model)
        Graph.train_loss_validation_loss_epoch(path_graph =self.path_graph ,train_loss = train_loss,
                                               validation_loss = validation_loss)        
        net,pnsr_1_list,pnsr_2_list,pnsr_3_list = self.test_data(net,device,start,stop)
        Graph.psnr_test(self.path_graph,pnsr_1_list,pnsr_2_list,pnsr_3_list)
        return net
        
    def get_resumen(self,net = None):
        print("Image size: "+str(self.size_figure))
        print("Len dataset: "+str(self.stop-self.start))
        print("Type psf: "+str(self.type_psf))
        print("Number epoch: " +str(self.num_epochs))
        print("Learning rate: " + str(self.learning_rate))
        print("Batch train: " + str(self.batch_train))
        print("\n")
        print("Net: ",net)
        print("\n")
        img = mpimg.imread(self.path_graph+'/graph_loss_train_validation.png')
        imgplot = plt.imshow(img)
        plt.show()
        img = mpimg.imread(self.path_graph+'/graph_psnr.png')
        imgplot = plt.imshow(img)
        plt.show()
        clean = cv2.imread(self.path_test+'/clean'+str(0)+'.png')
        dirty = cv2.imread(self.path_test+'/noisy'+str(0)+'.png')
        output =  cv2.imread(self.path_test+'/denoised'+str(0)+'.png')
        AuxiliaryFunctions.display(clean,'clean')
        AuxiliaryFunctions.display(dirty,'dirty')
        AuxiliaryFunctions.display(output,'output')
        
