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

#sesion save# 
import dill

import cupy as cp
from datasetInterferometry import DatasetInterferometry
from auxiliartFunction import AuxiliaryFunctions
from graph import Graph


# In[ ]:


class model:
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
            return '../datasets/images_'+str(self.size_figure )+'x'+str(self.size_figure )+'/convolutions/'+self.type_psf+'/models/model_S'+str(self.size_data)+'_B'+str(BATCH_TRAIN)+'_N'+str(self.num_epochs)+'/test'
        else:
            return path_test 
        
    def init_path_graph(self,path_graph):
        if (path_test == None):
            return '../datasets/images_'+str(self.size_figure )+'x'+str(self.size_figure )+'/convolutions/'+self.type_psf+'/models/model_S'+str(self.size_data)+'_B'+str(BATCH_TRAIN)+'_N'+str(self.num_epochs)+'/graph'
        else:
            return path_graph 
        
    def init_path_model(self,path_model):
         if (path_model == None):
            return '../datasets/images_'+str(self.size_figure )+'x'+str(self.size_figure )+'/convolutions/'+self.type_psf+'/models/model_S'+str(self.size_data)+'_B'+str(BATCH_TRAIN)+'_N'+str( self.num_epochs)+'/model' 
         else:
            return path_model 
  
    
    def train_data_memory(self,net,start,stop,step):
        train_loss = []
        valid_loss = []
        list_index = cp.arange(start,stop,step)
        data = Dataset(self.size_figure,self.type_psf)
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            size_train = 0
            for index in list_index:
                trainLoader = data.create_train_data(index,step)
                size_train = size_train +len(trainLoader)
                for dirty,clean in tqdm((trainLoader)):
                    dirty,clean=dirty.to(device).float(),clean.to(device).float()
                    self.optimizer.zero_grad()            
                    outputs = self.net(dirty)
                    loss = self.criterion(outputs, clean)
                    #backpropagation
                    loss.backward()
                    #update the parameters
                    self.optimizer.step()
                    running_loss += loss.item()
            loss = running_loss / size_train
            train_loss.append(loss)
        ## validation ## ''
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(epoch+1, self.num_epochs, loss))
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
                loss = criterion(outputs, clean)
                validation_loss += loss.item()
                loss = validation_loss / size_validate
                valid_loss.append(loss)
        print('Epoch {} of {}, Validate Loss: {:.3f}'.format(epoch+1, self.num_epochs, loss))
        return net,train_loss,valid_loss
    
    def test_data_memory(self,net,start,stop,step):
        list_index = cp.arange(start,stop,step)
        data = Dataset(self.size_figure,self.type_psf)
        pnsr_1_list = []
        pnsr_2_list = []
        pnsr_3_list = []
        i = 0
        for index in list_index:
            testloader = data.create_validation_data(index,step)
            for dirty,clean in tqdm((testloader)):
                dirty=dirty.to(device)
                outputs = net(dirty)
                dirty  = dirty.cpu().data
                outputs = outputs.cpu().data
        # psnr #
        #psnr_1 = cv2.PSNR(clean.detach().numpy(), outputs.detach().numpy())
        #psnr_2 = cv2.PSNR(clean.detach().numpy(),dirty.detach().numpy())
        #psnr_3 = cv2.PSNR(dirty.detach().numpy(), outputs.detach().numpy())
                psnr_1 = cv2.PSNR(np.array(clean), np.array(outputs))
                psnr_2 = cv2.PSNR(np.array(clean), np.array(dirty))
                psnr_3 = cv2.PSNR(np.array(dirty), np.array(outputs))
                pnsr_1_list.append(psnr_1)
                pnsr_2_list.append(psnr_2)
                pnsr_3_list.append(psnr_3)
                AuxiliaryFunctions.save_decoded_image(dirty, name=self.path_test+'/noisy{}.png'.format(i))
                AuxiliaryFunctions.save_decoded_image(outputs, name=self.path_test+'/denoised{}.png'.format(i))
                AuxiliaryFunctions.save_decoded_image(clean, name=self.path_test+'/clean{}.png'.format(i))
                i = i +1
        return net,pnsr_1_list,pnsr_2_list,pnsr_3_list
    
    
      
    def train_data(self,net,start,stop):
        train_loss = []
        valid_loss = []
        list_index = cp.arange(start,stop,step)
        data = Dataset(self.size_figure,self.type_psf)
        for epoch in range(self.num_epochs):
            running_loss = 0.0           
            trainLoader = data.read_train_data(start,stop):
            for dirty,clean in tqdm((trainLoader)):
                dirty,clean=dirty.to(device).float(),clean.to(device).float()
                self.optimizer.zero_grad()            
                outputs = self.net(dirty)
                loss = self.criterion(outputs, clean)
                #backpropagation
                loss.backward()
                #update the parameters
                self.optimizer.step()
                running_loss += loss.item()
            loss = running_loss / len(trainLoader)
            train_loss.append(loss)
        ## validation ## ''
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(epoch+1, self.num_epochs, loss))
        validation_loss = 0.0
        net.eval()
        size_validate = 0  
        for index in list_index:
            validationloader = data.read_validation_data(start,stop)
            for dirty, clean in validationloader:
                dirty,clean=dirty.to(device).float(),clean.to(device).float()
                self.optimizer.zero_grad()    ## TODO BORRRAR ??         
                outputs = net(dirty)
                loss = criterion(outputs, clean)
                validation_loss += loss.item()
                loss = validation_loss / len(validationloader)
                valid_loss.append(loss)
        print('Epoch {} of {}, validate Loss: {:.3f}'.format(epoch+1, self.num_epochs, loss))
        return net,train_loss,valid_loss
    
    def test_data_memory(self,net,start,stop,step):
        list_index = cp.arange(start,stop,step)
        data = Dataset(self.size_figure,self.type_psf)
        pnsr_1_list = []
        pnsr_2_list = []
        pnsr_3_list = []
        i = 0
        for index in list_index:
            testloader = data.create_validation_data(index,step)
            for dirty,clean in tqdm((testloader)):
                dirty=dirty.to(device)
                outputs = net(dirty)
                dirty  = dirty.cpu().data
                outputs = outputs.cpu().data
        # psnr #
        #psnr_1 = cv2.PSNR(clean.detach().numpy(), outputs.detach().numpy())
        #psnr_2 = cv2.PSNR(clean.detach().numpy(),dirty.detach().numpy())
        #psnr_3 = cv2.PSNR(dirty.detach().numpy(), outputs.detach().numpy())
                psnr_1 = cv2.PSNR(np.array(clean), np.array(outputs))
                psnr_2 = cv2.PSNR(np.array(clean), np.array(dirty))
                psnr_3 = cv2.PSNR(np.array(dirty), np.array(outputs))
                pnsr_1_list.append(psnr_1)
                pnsr_2_list.append(psnr_2)
                pnsr_3_list.append(psnr_3)
                AuxiliaryFunctions.save_decoded_image(dirty, name=self.path_test+'/noisy{}.png'.format(i))
                AuxiliaryFunctions.save_decoded_image(outputs, name=self.path_test+'/denoised{}.png'.format(i))
                AuxiliaryFunctions.save_decoded_image(clean, name=self.path_test+'/clean{}.png'.format(i))
                i = i +1
        return net,pnsr_1_list,pnsr_2_list,pnsr_3_list
    
    
    
    def run_data_memory(self,net,start,stop,step):
        torch.cuda.empty_cache()
        device = AuxiliaryFunctions.get_device() #### 
        print(device)
        net.to(device)
        AuxiliaryFunctions.make_dir(self.path_test)
        net,train_loss,validation_loss = self.train_data_memory(net,start,stop,step)
        Graph.train_loss_validation_loss_epoch(self.path_graphs,train_loss,validation_loss)        
        net,pnsr_1_list,pnsr_2_list,pnsr_3_list = self.test_data_memory(self,self.net,start,stop,step):
        Graph.psnr_test(self.path_graphs,pnsr_1_list,pnsr_2_list,pnsr_3_list)

        
        
    def run_data(self,net,start,stop):
        torch.cuda.empty_cache()
        device = AuxiliaryFunctions.get_device() #### 
        print(device)
        net.to(device)
        AuxiliaryFunctions.make_dir(self.path_test)
        net,train_loss,validation_loss = self.train_data(net,start,stop)
        Graph.train_loss_validation_loss_epoch(self.path_graphs,train_loss,validation_loss)        
        net,pnsr_1_list,pnsr_2_list,pnsr_3_list = self.test_data(self,self.net,start,stop):
        Graph.psnr_test(self.path_graphs,pnsr_1_list,pnsr_2_list,pnsr_3_list)

