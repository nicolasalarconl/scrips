# %%
import os
import torch 

# %%
class AuxiliaryFunctions:
    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_decoded_image(img, name):
        img = img.view(img.size(0), 1, N, N)
        save_image(img, name)
    

    def get_device():
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device
