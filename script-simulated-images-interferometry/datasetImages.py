# %%
from matplotlib import pyplot as plt
from auxiliaryFunctions import AuxiliaryFunctions
from listEllipses import ListEllipses
from randomImage import RandomImage
from astropy.io import fits
import cupy as cp
#import numpy as cp
import numpy as np
import time


# %%
class DatasetImages:  
    def __init__(self,size_image,path_save = None,path_read = None): 
        self.size_image =size_image
        self.path_save = self.init_path_save(path_save)
        self.path_read = self.init_path_read(path_read)
        self.params = []       
        self.images = []
        self.start = 0
        self.stop = 0
        self.recursions = []

    def init_path_save(self,path_save):
        if (path_save == None):
            return'../datasets/images_'+str(self.size_image)+'x'+str(self.size_image)+'/images'
        else:
            return path_save
    def init_path_read(self,path_read):
        if (path_read == None):
            return'../datasets/images_'+str(self.size_image)+'x'+str(self.size_image)+'/images'
        else:
            return path_read

    def recursion_average(self):
        a = np.array(self.recursions)
        return cp.sum(a)/(self.start-self.stop)

    def time_averange(self):
        a = np.array(self.times)
        return cp.sum(a)/(self.start-self.stop)
    
    def len_images(self):
        return len(self.images)
    
    def save(self,size_image,params,stop,start = None,path = None):
        self.size_image  = size_image
        self.params = params
        if (start is None):
            start = 0
        if(path is not None):   
            self.path_save = path
        self.start = start
        self.stop = stop
        AuxiliaryFunctions.make_dir(self.path_save)        
        list_figure_random = ListEllipses(params,start)
        self.recursions = []
        self.times = []
        for index in cp.arange(int(start),int(stop),1):
            start_time = time.time()
            image = RandomImage(list_figure_random,index)
            self.recursions.append(image.recursion)
            hdu_image =fits.PrimaryHDU(cp.asnumpy(image.image))
            #hdu_image = fits.PrimaryHDU(image.image)
            hdu_image.writeto(self.path_save+'/image_'+str(self.size_image)+'x'+str(self.size_image)+'_'+str(index)+'.fits',overwrite=True)
            stop_time = time.time()
            self.times.append(stop_time-start_time)       

        
    def create(self,size_image,params,stop,start = None):
        self.size_image  = size_image
        self.params = params
        if (start is None):
            start = 0
        self.start = start
        self.stop = stop
        list_figure_random = ListEllipses(params,start)
        self.recursions = []
        self.times = []
        images = []
        for index in cp.arange(int(start),int(stop),1):
            start_time = time.time()
            image = RandomImage(list_figure_random,index)
            images.append(image)
            self.recursions.append(image.recursion)
            stop_time = time.time()
            self.times.append(stop_time-start_time)
        self.images = images

        
    def read(self,size_image,stop,path = None,start = None,):
        self.size_image  = size_image
        if (start is None):
            start = 0
        if(path != None):   
            self.path_read= path
        AuxiliaryFunctions.make_dir(self.path_read)
        images = []
        for index in cp.arange(int(start),int(stop)):
            path_file = self.path_read+'/image_'+str(self.size_image)+'x'+str(self.size_image)+'_'+str(index)+'.fits'
            hdul=fits.open(path_file)
            data = hdul[0].data.astype(cp.float32)
            image = cp.reshape(data,[self.size_image,self.size_image])
            image = cp.array(image)
            images.append(image)
        self.images = images
        return self.images

    def get_images(self):
        if (len(self.images) == 0):
            return self.read_dataset()
        else:
            return self.images
     
    def view(self,index = None):
        if  (index is None):
            index = 1
        if (self.len_images() <= index):
            print("index out of bounds, index max: "+str(self.len_images()-1))
        else:
            plt.imshow(cp.asnumpy(self.images[index]))
            #plt.imshow(self.images[index])


# %%
#from paramsEllipses import ParamsEllipses
#params = ParamsEllipses(128)
#dataset = DatasetImages(128) 
#dataset.save(size_image = 128,params = params,stop =10)
#x = dataset.read(size_image = 128,stop = 10)


# %%


# %%
