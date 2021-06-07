# %%
#import cupy as cp
import cupy as cp
#import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file
from matplotlib import pyplot as plt
import cv2
from auxiliaryFunctions import AuxiliaryFunctions

# %%
class DatasetPSF:
    def __init__(self,size_image,type_psf,path_save = None,path_read = None):
        self.size_image = size_image
        self.type_psf = type_psf
        self.image = []
        self.path_save = self.init_path_save(path_save)
        self.path_read = self.init_path_read(path_read)
    
    def init_size_image(self,size_image):
        if(size_image == None):
            return None
        else:
            return size_image  
    def init_path_save(self,path_save):
        if (path_save == None):
            return '../datasets/images_'+str(self.size_image)+'x'+str(self.size_image)+'/convolutions/'+str(self.type_psf)+'/psf' 
        else:
            return path_save
        
    def init_path_read(self,path_read):
        if (path_read == None):
            return '../datasets/images_'+str(self.size_image)+'x'+str(self.size_image)+'/convolutions/'+str(self.type_psf)+'/psf' 
        else:
            return path_read   
    def len_image(self):
        return len(self.image)
        
    def save(self,size_image,type_psf,psf,path = None):
        self.size_image = size_image
        self.type_psf =  type_psf
        psf = cp.asnumpy(psf)
        if(path != None):   
            self.path_save = path
        AuxiliaryFunctions.make_dir(self.path_save)
        hdu_image =fits.PrimaryHDU(psf)
        hdu_image.writeto(self.path_save+'/psf_'+self.type_psf+'.fits',clobber=True)
        self.image = psf

    def read(self,size_image,type_psf,path = None):
        self.type_psf =  type_psf
        self.size_image = size_image
        if(path != None):
            self.path_read = path
        hdul=fits.open(self.path_read+'/psf_'+self.type_psf+'.fits')
        hdr = hdul[0].header
        size = hdr[3]
        data = hdul[0].data.astype(cp.float32)
        self.image =  cp.reshape(data,[size,size])
        self.image 
        return self.image
    def read_url(self,size_image,type_psf,url):
        self.type_psf =  type_psf
        self.size_image = size_image
        image_link = download_file(url, cache=True )
        image = fits.getdata(image_link).astype(cp.float32)
        image = cp.reshape(image,[image.shape[2],image.shape[3]])
        self.image = cp.array(image)
        return self.image
      
    def resize(self,psf,size):
        image = cv2.resize(cp.asnumpy(psf), dsize=(size, size), interpolation=cv2.INTER_CUBIC)
        #image = cv2.resize(psf, dsize=(size, size), interpolation=cv2.INTER_CUBIC)        
        self.image = cp.array(image)
        return self.image
       
    def view(self):
            plt.imshow(cp.asnumpy(self.image))
            #plt.imshow(self.image)
    