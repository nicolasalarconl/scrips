# %%
from matplotlib import pyplot as plt
from auxiliaryFunctions import AuxiliaryFunctions
from datasetImages import DatasetImages
from datasetPSF import DatasetPSF
from astropy.io import fits
import cupy as cp
import numpy as np
import time
from cupyx.scipy import ndimage #as ndcupy


class DatasetDirty:
    def __init__(self,size_image,type_psf,path_save = None,path_read = None): 
        self.size_image = size_image
        self.type_psf = type_psf
        self.psf = []
        self.path_save = self.init_path_save(path_save)
        self.path_read = self.init_path_read(path_read)
        self.dirtys = []
        self.times = []

    def init_path_save(self,path_save):
        if (path_save is None):
            return '../datasets/images_'+str(self.size_image)+'x'+str(self.size_image)+'/convolutions/'+self.type_psf+'/conv/'
        else:
            return path_save
    def init_path_read(self,path_read):
        if (path_read == None):
            return'../datasets/images_'+str(self.size_image)+'x'+str(self.size_image)+'/convolutions/'+self.type_psf+'/conv/'
        else:
            return path_read
    def time_averange(self):
        a = np.array(self.times)
        return cp.sum(a)/(self.len_dirtys())

    def len_dirtys(self):
        return len(self.dirtys)
    
    def save(self,images,size_image,type_psf,psf,start,finish ,path = None):
        self.size_image = size_image
        self.type_psf = type_psf
        self.psf = psf 
        if(path != None):   
            self.path_save = path
        AuxiliaryFunctions.make_dir(self.path_save)
        index = start
        self.times = []
        for image in images:
            start_time = time.time()
            image = cp.array(image)
            psf = cp.array(psf)
            conv = ndimage.convolve(image,psf,mode='constant', cval=0.0)
            hdu_image =fits.PrimaryHDU(cp.asnumpy(conv))            hdu_image.writeto(self.path_save+'/conv_'+str(self.size_image)+'x'+str(self.size_image)+'_'+str(index)+'.fits',clobber=True)
            index = index + 1
            stop_time = time.time()
            self.times.append(stop_time-start_time)
    

    def create(self,images,size_image,type_psf,psf):
        self.size_image = size_image
        self.type_psf = type_psf
        self.psf = psf 
        self.dirtys = []
        convs = []
        self.times = []
        for image in images:
            start_time = time.time()
            conv = ndimage.convolve(image,psf,mode='constant', cval=0.0)
            convs.append(conv)
            stop_time = time.time()
            self.times.append(stop_time-start_time)
        self.dirtys = convs
      
    def read(self,size_image,type_psf,start,stop,path = None):
        self.size_image  = size_image
        self.type_psf = type_psf
        if(path != None):  
            self.path_read = path
        AuxiliaryFunctions.make_dir(self.path_read)
        self.dirtys = []
        for index in range(start,stop):
            path_file = self.path_read+'/conv_'+str(self.size_image)+'x'+str(self.size_image)+'_'+str(index)+'.fits'
            hdul=fits.open(path_file)
            data = hdul[0].data.astype(cp.float32)
            image = cp.reshape(data,[self.size_image,self.size_image])
            image = cp.array(image)
            self.dirtys.append(image)
     
    def view(self,index = None):
        if  (index == None):
            index = 1
        if (self.len_dirtys() <= index):
            print("index out of bounds, index max: "+str(self.len_dirtys()-1))
        else:
            plt.imshow(cp.asnumpy(self.dirtys[index]))
            