# %%
import cupy as cp
from matplotlib import pyplot as plt
import time

# %%
class PSNR: 
    def __init__(self,clean,dirty,mask,device):
        self.init_device(device)
        self.get = self.get_psnr(clean ,dirty,mask) #. squeeze(),dirty.squeeze())
  
    def init_device(self,device):
         cp.cuda.Device(device).use()
            
 
    def get_values(self,image,mask): 
        mask = mask[0]
        values = []
        mask = mask[1:len(mask)]
        for point in mask:
            idx = point[0].numpy()
            idy = point[1].numpy()
            values.append(image[idx][idy])
        values = cp.array(values)
        return values
    
    def get2(self,image,mask):
        mask = mask[0]
        values = []
        mask = mask[1:len(mask)]
        for point in mask:
            idx = point[0].numpy()
            idy = point[1].numpy()
            values.append(image[idx][idy])
        values = cp.array(values)
        return values
    
        
    
    def get_psnr(self,clean,dirty,mask):
        clean = clean.squeeze()
        clean = cp.array(clean)
        dirty = dirty.squeeze() 
        dirty = cp.array(dirty)
        #start_time = time.time()
        values = self.get_values(dirty,mask)
        #stop_time = time.time()
        #final_time = stop_time -start_time
        #print('time: '+str(final_time))
        #start_time = time.time()
        #values = self.get2(dirty,mask)
        #stop_time = time.time()
        #final_time = stop_time -start_time
        idx = mask[0][0][0].numpy()
        idy = mask[0][0][1].numpy()
        max_value = dirty[idx][idy]
        std_value = cp.std(values)
        psnr = cp.asnumpy((20*cp.log10(max_value/std_value))).item(0)   
        return psnr
     


