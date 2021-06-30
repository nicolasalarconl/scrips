# %%
import os
import torch 
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import cv2
from astropy.io import fits
import cupy as cp
import pandas as pd



# %%
class AuxiliaryFunctions:
    
    
    
    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_fit_image(img, path):
        hdu_image =fits.PrimaryHDU(img)
        hdu_image.writeto(path,overwrite=True)
        
    def display_fit_image(title,path):
        hdul=fits.open(path)
        data = hdul[0].data.astype(cp.float32)
        size = data.shape[2]
        image = cp.reshape(data,[size,size])
        plt.imshow(image)
        plt.show()

    def get_device(id):
        if torch.cuda.is_available():
            device = 'cuda:'+str(id)
            print('device: '+device)
        else:
            device = 'cpu'
        return device
    
    def load_mode(path,file = None):
        if ( file is None):
            return torch.load(path+'/model.pkl')
        else:
            return torch.load(path+file)
        
    def save_model(model,path,file = None):
        if (file is None):
            torch.save(model, path+'/model.pkl')
        else:
            torch.save(model, path+file)

            
    def display(a, title1 = "Original"):
        plt.imshow(a), plt.title(title1)
        plt.show()
    
    def write_csv(path,data):
        d = {'idx': cp.asnumpy(data[0]), 'idy': cp.asnumpy(data[1])}
        df = pd.DataFrame(d)
        df.to_csv(path,index=False)
              
    def read_csv(path):
        df = pd.read_csv(path).to_numpy()
        return df
    
    def write_log(path,data):
        file1 = open(path,"w+")
        file1.writelines(data)
        file1.close() 
        
    def send_alert(path_log,psnr,data_model):
        path_avg = path_log+'/avg.csv'
        path_alert = path_log+'/alert.txt'
        avg_psnr = cp.asnumpy(cp.average(psnr))
        

        if not os.path.isfile(path_avg):
            df = pd.DataFrame([avg_psnr])
            df.to_csv(path_avg,index=False)
            file = open(path_alert,"w+")
            file.writelines(['model: '+data_model,' avg:' ,str(avg_psnr),'\n'])
            file.close()
        else:
            avg = pd.read_csv(path_avg).to_numpy()
            maximum = max(avg)[0]
            if(avg_psnr > maximum):
                df = pd.DataFrame([avg_psnr])
                df.to_csv(path_avg,index=False)
                file = open(path_alert,"a")
                file.writelines(['model: '+data_model,' avg:' ,str(avg_psnr),'\n'])
                file.close()
                    
             
            
    def log(path_log,data_model,status,time = None):
        path_log = path_log+'/log.txt'
        if not os.path.isfile(path_log):
            file = open(path_log,"a+")
        else:
            file = open(path_log,"a")
        if (time == None):
            file.writelines(['model: '+data_model,'| status:' ,status,'\n'])
        else:
            file.writelines(['model: '+data_model,'| status:' ,status, '| time:',str(round(time, 3)),'\n'])
        file.close()
        
                           
        
    