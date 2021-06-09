# %%
from ellipse import Ellipse
import cupy as cp
#import numpy as cp

# %%
class ListEllipses:
    def __init__(self,params,index_random):
        self.params = params
        self.data =self.create_list_ellipses(index_random)
        self.recursion = 0
    def len_list_params(self):
        return self.params.size_sample
    
    def len_list(self):
        return len(self.data)

    def get_percentage_info(self,figure):
        n = len(figure)
        c = cp.sum(figure>0)  
        percentage=(c)/(n*n)
        return percentage
    
    def is_null(self,figure):
        if (self.get_percentage_info(figure) < self.params.percentage_info):
            return True
        return False

    def create_ellipse(self,parameter):
        size_figure,axis_minor,axis_major,min_value_intensity, max_value_intensity,mov_x,moy_y,angle,sigma = parameter
        return Ellipse(size_figure,axis_minor,axis_major,min_value_intensity, max_value_intensity,mov_x,moy_y,angle,sigma)
        
    def sample_params(self,index_random):
            params = self.params.get_params_random(index_random)
            ellipse = self.create_ellipse(params)
            if(self.is_null(ellipse.data) == False):
                return ellipse
            else:
                self.recursion = self.recursion+1
                return self.sample_params(index_random+1)
     
    def create_list_ellipses(self,index_random):
        ellipses = []
        self.recursion = 0
        for i in range(0,self.len_list_params()):
            ellipse = self.sample_params(index_random)
            index_random = index_random+i
            ellipses.append(ellipse)
        return ellipses
  
                 
    def view(self,index = None):
        if  (index == None):
            index = 1
        if (self.len_list() <= index):
            print("index out of bounds, index max: "+str(self.len_list()-1))
        else:
            self.data[index].view()


# %%
#from paramsEllipses import ParamsEllipses
#list_Ellipses= ListEllipses(ParamsEllipses(500),1000000)
#list_Ellipses.view(99)

# %%


# %%
