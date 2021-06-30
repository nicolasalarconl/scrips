# %%
import cupy as cp 
#import numpy as cp
import random 
class ParamsEllipses:
    def __init__(self
                ,size_figure
                ,device 
                ,min_value_intensity = 0.01
                ,max_value_intensity = 0.99
                ,step_axis_minor= 0.01
                ,step_axis_major = 0.01
                ,step_mov_x= 1
                ,step_mov_y = 1
                ,min_angle = -cp.pi
                ,max_angle = cp.pi 
                ,step_angle = 0.1
                ,min_sigma = 0.1
                ,max_sigma = 1 
                ,step_sigma =0.15
                ,size_sample =  100
                ,n_operation = 10
                ,percentage_info = 0.01
                ,min_value_axis_major= None
                ,max_value_axis_major= None
                ,min_value_axis_minor= None
                ,max_value_axis_minor= None
                ,mov_x = None
                ,mov_y = None
                ):
        
        self.size_figure = size_figure
        self.min_value_intensity = min_value_intensity
        self.max_value_intensity = max_value_intensity
        
        self.min_value_axis_minor = self.init_min_value_axis_minor(min_value_axis_minor)
        self.max_value_axis_minor = self.init_max_value_axis_minor(max_value_axis_minor)
        self.step_axis_minor = step_axis_minor
        
        self.min_value_axis_major = self.init_min_value_axis_major(min_value_axis_major)
        self.max_value_axis_major = self.init_max_value_axis_major(max_value_axis_major)
        self.step_axis_major = step_axis_major
        
        self.mov_x = self.init_mov_x(mov_x)
        self.step_mov_x = step_mov_x
        
        self.mov_y = self.init_mov_y(mov_y)
        self.step_mov_y = step_mov_y
        
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.step_angle = step_angle  
        
        self.min_sigma = min_sigma 
        self.max_sigma = max_sigma
        self.step_sigma = step_sigma
        
        self.size_sample  = size_sample
        self.n_operation=  n_operation
        
        self.percentage_info = percentage_info
        self.device = self.init_device(device)
  
    def init_device(self,device):
        cp.cuda.Device(device).use()
         
    def init_min_value_axis_minor(self,min_value_axis_minor):
        if(min_value_axis_minor == None):
            return self.size_figure*0.05
        else:
            return self.size_figure*min_value_axis_minor  
        
    def init_max_value_axis_minor(self,max_value_axis_minor):
        if(max_value_axis_minor == None):
            return self.size_figure*0.15
        else:
            return self.size_figure*max_value_axis_minor
        
    def init_min_value_axis_major(self,min_value_axis_major):
         if(min_value_axis_major == None):
            return self.size_figure*0.05
         else:
            return self.size_figure*min_value_axis_major  
        
    def init_max_value_axis_major(self,max_value_axis_major):
        if(max_value_axis_major == None):
            return self.size_figure*0.15
        else:
            return self.size_figure*max_value_axis_major  
    def init_mov_x(self,mov_x):
        if(mov_x == None):
            return self.size_figure*0.1
        else:
            return self.size_figure*mov_x  
        
    def init_mov_y(self,mov_y):
        if(mov_y == None):
            return self.size_figure*0.1
        else:
            return self.size_figure*mov_y  
        
    def create_list_params(self):
        axis_minor_list = cp.arange(self.min_value_axis_minor,self.max_value_axis_minor,self.step_axis_minor)
        axis_major_list = cp.arange(self.min_value_axis_major,self.max_value_axis_major,self.step_axis_major)
        mov_x_list = cp.arange(-self.mov_x,self.mov_x,self.step_mov_x)
        mov_y_list = cp.arange(-self.mov_y,self.mov_y,self.step_mov_y)
        angle_list=  cp.arange(self.min_angle,self.max_angle,self.step_angle)
        sigma_list = cp.arange(self.min_sigma,self.max_sigma,self.step_sigma)
        return axis_minor_list,axis_major_list,mov_x_list,mov_y_list,angle_list,sigma_list
    
    def get_params_random(self,index_random):
        axis_minor_list,axis_major_list,mov_x_list,mov_y_list,angle_list,sigma_list = self.create_list_params()
        random.seed(cp.asnumpy(index_random).item(0))
        #random.seed(index_random)
        axis_minor  = random.choice(axis_minor_list)
        axis_major  = random.choice(axis_major_list)
        mov_x = random.choice(mov_x_list)
        moy_y= random.choice(mov_y_list)
        angle = random.choice(angle_list) 
        sigma = random.choice(sigma_list)
        return self.size_figure,axis_minor,axis_major,self.min_value_intensity, self.max_value_intensity,mov_x,moy_y,angle,sigma

# %%
