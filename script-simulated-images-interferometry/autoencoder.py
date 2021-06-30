# %%
import torch.nn as nn
import torch.nn.functional as F
from auxiliaryFunctions import AuxiliaryFunctions



class Autoencoder(nn.Module):
    def __init__(self,**kwargs):
        super(Autoencoder, self).__init__()
        #self.out_shape = round(kwargs["input_shape"]*0.1)
        self.enc1 = nn.Conv2d(1, kwargs["input_shape"], kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(kwargs["input_shape"], round(kwargs["input_shape"]*0.5), kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(round(kwargs["input_shape"]*0.5), round(kwargs["input_shape"]*0.25), kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(round(kwargs["input_shape"]*0.25), round(kwargs["input_shape"]*0.10), kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(round(kwargs["input_shape"]*0.10), round(kwargs["input_shape"]*0.10), 
                                       kernel_size=3, stride=2)  
        self.dec2 = nn.ConvTranspose2d(round(kwargs["input_shape"]*0.10), round(kwargs["input_shape"]*0.10)*2, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(round(kwargs["input_shape"]*0.10)*2, round(kwargs["input_shape"]*0.10)*4, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(round(kwargs["input_shape"]*0.10)*4, kwargs["input_shape"], kernel_size=2, stride=2)
        self.out = nn.Conv2d(kwargs["input_shape"], 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x) # the latent space representation
        
        # decode
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.sigmoid(self.out(x))
        return x
    
    
class Autoencoder24(nn.Module):
    def __init__(self,size,device):
        self.device = AuxiliaryFunctions.get_device(device)
        self.initial = 0
        self.p  = 0
        self.k = 2
        self.size = size
        super(Autoencoder24, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv2d(1, size, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(size, round(size*0.75), kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(round(size*0.75), round(size*0.50), kernel_size=3, padding=1)
        #self.enc4 = nn.Conv2d(round(size*0.50), round(size*0.25), kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
     
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(round(size*0.50), round(size*0.50), kernel_size=3, stride=2)  
        self.dec2 = nn.ConvTranspose2d(round(size*0.50), round(size*0.75), kernel_size=3, stride=2)
        #self.dec3 = nn.ConvTranspose2d(round(size*0.75), size, kernel_size=self.k, stride=2)
        #self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(size, 1, kernel_size=3, padding=1)
    
   
    def set_padding_kernel(self,x):
        self.k = 2
        s = round(self.size*0.75)
        p = (((x.shape[2]-1)*2 -self.size)+self.k )/2
        if (p % 1 == 0):
            self.p = int(p)
            self.k = 2
            self.dec3 = nn.ConvTranspose2d(s, self.size, kernel_size=self.k, padding=self.p ,stride=2).to(self.device)
            return 
        else:
            self.k = 3
            self.p =  int((((x.shape[2]-1)*2 -self.size)+self.k)/2)
            self.dec3 = nn.ConvTranspose2d(s, self.size, kernel_size=self.k, padding=self.p ,stride=2).to(self.device)
            return
            
    def forward(self, x):
        # encode
        if (self.initial == 0):
            print('initial : '+str(x.shape))
        x = F.relu(self.enc1(x))
        if (self.initial == 0):
            print('enc1:'+str(x.shape))
        x = self.pool(x)
        if (self.initial == 0):
            print('pool:'+str(x.shape))
       
        x = F.relu(self.enc2(x))
        if (self.initial == 0):
            print('enc2:'+str(x.shape))
        x = self.pool(x)
        if (self.initial == 0):
            print('pool:'+str(x.shape))
        x = F.relu(self.enc3(x))
        if (self.initial == 0):
            print('enc3:'+str(x.shape))
        x = self.pool(x)
        if (self.initial == 0):
            print('pool:'+str(x.shape))
        #x = F.relu(self.enc4(x))
        #if (self.initial == 0):
       #     print('enc4:'+str(x.shape))
       # x = self.pool(x) # the latent space representation
       # if (self.initial == 0):
       #     print('pool:'+str(x.shape))
        
        # decode
        x = F.relu(self.dec1(x))
        if (self.initial == 0):
            print('dec1:'+str(x.shape))
        x = F.relu(self.dec2(x))
        if (self.initial == 0):
            print('dec2:'+str(x.shape))
        self.set_padding_kernel(x)
        x = F.relu(self.dec3(x))
        if (self.initial == 0):
            print('dec3:'+str(x.shape))
        #x = F.relu(self.dec4(x))
        #if (self.initial == 0):
           # print(x.shape)
        x = F.sigmoid(self.out(x))
        if (self.initial == 0):
            print('out:'+str(x.shape))
        self.initial = 1
        return x
    
class Autoencoder64(nn.Module):
    def __init__(self,size):
        super(Autoencoder64, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)  
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=1)
        self.out = nn.Conv2d(64, 1, kernel_size=2, padding=13)
    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x) # the latent space representation
        
        # decode
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.sigmoid(self.out(x))
        return x