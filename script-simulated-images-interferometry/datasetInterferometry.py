#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from interferometryData import TestData
from interferometryData import TrainData

from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from paramsEllipses import ParamsEllipses
from datasetImages import DatasetImages
from datasetPSF import DatasetPSF
from datasetDirty import DatasetDirty
import math
from matplotlib import pyplot as plt
import torch
from torch.utils.data.dataloader import default_collate
from auxiliaryFunctions import AuxiliaryFunctions



# In[ ]:


class DatasetInterferometry:
    def __init__(self,
                    size_figure,
                    type_psf,
                    batch_train,
                    device,
                    perc_train = 0.7,
                    perc_validation = 0.2,
                    perc_test = 0.1,
                    batch_test = 1,
                    batch_validation =  None,
                    path_images = None,
                    path_convolution = None,
                    path_psf = None,
                ):
        self.size_figure = size_figure
        self.type_psf = type_psf
        self.path_images = self.init_path_images(path_images)
        self.path_convolution = self.init_path_convolution(path_convolution)
        self.path_psf = self.init_path_psf(path_psf)
        self.perc_train = perc_train
        self.perc_validation = perc_validation
        self.perc_test = perc_test
        self.batch_train = batch_train
        self.batch_validation = batch_train
        self.batch_test = batch_test
        self.device = device
        
        
    def init_path_images(self,path_save):
        if (path_save == None):
            return'../../datasets/images_'+str(self.size_figure)+'x'+str(self.size_figure)+'/images'
        else:
            return path_save  
    def init_path_convolution(self,path_convolution):
        if (path_convolution == None):
            return '../../datasets/images_'+str(self.size_figure)+'x'+str(self.size_figure)+'/convolutions/'+self.type_psf+'/conv'
        else: 
            return path_convolution
    
    def init_path_psf (self,path_psf):
        if (path_psf == None):
            return '../../datasets/images_'+str(self.size_figure)+'x'+str(self.size_figure)+'/convolutions/'+self.type_psf+'/psf'
        else:
            return path_psf

                      
    def tsfms(self):
        return transforms.Compose([transforms.ToTensor()])
     
        
        
    def create_train_data(self,start,step):
        size_train = math.trunc(step*self.perc_train)
        params = ParamsEllipses(self.size_figure)

        data_image = DatasetImages(self.size_figure,self.device)
        data_image.create(size_image=self.size_figure,params = params , start = start,stop = start +size_train)

        data_psf = DatasetPSF(self.size_figure,self.type_psf,self.device)
        data_psf.create(self.size_figure,self.type_psf)
        data_dirty = DatasetDirty(self.size_figure,self.type_psf,data_psf.image,self.device)
        data_dirty.create(images = data_image.images,size_image = self.size_figure, type_psf = self.type_psf,
                          psf =data_psf.image)
        trainSet=TrainData(data_dirty.dirtys,data_image.images
                                    ,self.tsfms(), self.device)
        trainLoader=DataLoader(trainSet,self.batch_train,shuffle=False)
        return trainLoader
    
    
    def create_validation_data(self,start,step):
        start = start + math.trunc(step*self.perc_train)
        size_validation =  math.trunc(step*self.perc_validation)
       
        params = ParamsEllipses(self.size_figure)
        data_image = DatasetImages(self.size_figure,self.device)
        data_image.create(size_image=self.size_figure,params = params , start = start,stop = start +size_validation)

        data_psf = DatasetPSF(self.size_figure,self.type_psf,self.device)
        data_psf.create(self.size_figure,self.type_psf)

        data_dirty = DatasetDirty(self.size_figure,self.type_psf,data_psf.image,self.device)
        data_dirty.create(images = data_image.images,size_image = self.size_figure, type_psf = self.type_psf,
                          psf =data_psf.image)
        validationSet =TrainData(data_dirty.dirtys,data_image.images,
                                          self.tsfms(), self.device)
        validationLoader=DataLoader(validationSet,self.batch_validation,shuffle=False)
        return validationLoader

    def create_test_data(self,start,step):
    
        start = start + math.trunc(step*self.perc_train) + math.trunc(step*self.perc_validation)
        size_test =  math.trunc(step*self.perc_test)
                
        params = ParamsEllipses(self.size_figure)
        data_image = DatasetImages(self.size_figure,self.device)
        data_image.create(size_image=self.size_figure,params = params , start = start,stop = start +size_test)

        data_psf = DatasetPSF(self.size_figure,self.type_psf,self.device)
        data_psf.create(self.size_figure,self.type_psf)

        data_dirty = DatasetDirty(self.size_figure,self.type_psf,data_psf.image,self.device)
        data_dirty.create(images = data_image.images,size_image = self.size_figure, type_psf = self.type_psf, 
                          psf =data_psf.image)
        testSet = TestData(data_dirty.dirtys,data_image.images,data_image.masks,
                                     self.tsfms(), self.device)
        testLoader=DataLoader(testSet,self.batch_test,shuffle=False)
        return testLoader
    
    
    def read_train_data(self,start,stop):
        
        size =stop-start  #size of lot of the dataset
        size_train = math.trunc(size*self.perc_train)
        
        data_image = DatasetImages(self.size_figure,self.device)
        data_image.read(size_image=self.size_figure, start = start,stop = start+size_train)
        data_dirty = DatasetDirty(self.size_figure,self.type_psf,self.device)
        data_dirty.read(size_image = self.size_figure, type_psf = self.type_psf,start = start, stop = (start + size_train) )
        trainSet = TrainData(data_dirty.dirtys,data_image.images,self.tsfms(), self.device)
        trainLoader=DataLoader(trainSet,self.batch_train,shuffle=False)
        return trainLoader
    
    def read_validation_data(self,start,stop):
        size = stop-start  
     
        size_validation = math.trunc(size*self.perc_validation)
        size_train = math.trunc(size*self.perc_train)
        start = start + size_train
        data_image = DatasetImages(self.size_figure,self.device)
        data_image.read(size_image=self.size_figure, start = start,stop = start+size_validation)
        data_dirty = DatasetDirty(self.size_figure,self.type_psf,self.device)
        data_dirty.read(size_image = self.size_figure, type_psf = self.type_psf,start = start, stop = start + size_validation)
        validationSet= TrainData(data_dirty.dirtys,data_image.images,self.tsfms(), self.device)
        validationLoader=DataLoader(validationSet,self.batch_validation,shuffle=False)
        return validationLoader
    

    def read_test_data(self,start,stop):
        size = stop -start
        size_test = math.trunc(size*self.perc_test)
        size_train = math.trunc(size*self.perc_train)
        size_validation = math.trunc(size*self.perc_validation)
        start = start + size_train +size_validation
        data_image = DatasetImages(self.size_figure,self.device)
        data_image.read(size_image=self.size_figure, start = start,stop = start+size_test)
        data_dirty = DatasetDirty(self.size_figure,self.type_psf,self.device)
        data_dirty.read(size_image = self.size_figure, type_psf = self.type_psf,start = start, stop = start + size_test)
        testSet= TestData(data_dirty.dirtys,data_image.images,data_image.masks,
                                     self.tsfms(), self.device)
        testLoader=DataLoader(testSet,self.batch_test,shuffle=False)
        return testLoader
    

    def view_data(self,data):
        i = 1
        size = len(data)
        fig = plt.figure(figsize=(5,5))
        for d in data:
            dirty,clean,mask = d               
            ax = fig.add_subplot(size,2, i, xticks=[], yticks=[])
            plt.imshow(dirty.squeeze().numpy())
            i = i+1
            ax = fig.add_subplot(size,2, i, xticks=[], yticks=[])
            plt.imshow(clean.squeeze().numpy())
            i = i+1

    