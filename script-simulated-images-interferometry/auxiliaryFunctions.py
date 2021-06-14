# %%
import os
import torch 
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import cv2

# %%
class AuxiliaryFunctions:
    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_decoded_image(img,size_figure, name):
        img = img.view(img.size(0), 1, size_figure, size_figure)
        save_image(img, name)
    

    def get_device():
        if torch.cuda.is_available():
            device = 'cuda:1'
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