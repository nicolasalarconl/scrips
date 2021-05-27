# %%
from ellipse import Ellipse
import numpy as cp

# %%
class ListEllipses:
    def __init__(self,params):
        self.params = params
        self.list_sample_params = self.create_list_sample_params() 
        self.data =self.create_list_ellipses()

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
    
    def create_list_sample_params(self):
        list_sample_params = []
        for i in range (0,self.len_list_params()):
            params = self.params.get_params_random()
            list_sample_params.append(params) 
        return list_sample_params
    
    def create_list_ellipses(self):
        ellipses = []
        for i in range(0,self.len_list_params()):
            ellipse = self.create_ellipse(self.list_sample_params[i])
            if(self.is_null(ellipse.data) == False):
                ellipses.append(ellipse)
        return ellipses
    
    def view(self):
        if(self.len_list() > 0):
            index = random.randrange(0,self.len_list()-1,1)
            ellipse_random = self.data[index]
            ellipse_random.view()
        else:
            print("error")
                 
    def view(self,index = None):
        if  (index == None):
            index = 1
        if (self.len_list() <= index):
            print("index out of bounds, index max: "+str(self.len_list()-1))
        else:
            self.data[index].view()


# %%
#from paramsEllipses import ParamsEllipses
#list_Ellipses= ListEllipses(ParamsEllipses(120))
#list_Ellipses.view()