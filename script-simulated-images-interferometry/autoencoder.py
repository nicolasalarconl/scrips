# %%
import torch.nn as nn
import torch.nn.functional as F
from auxiliaryFunctions import AuxiliaryFunctions
import  sys

    
class Autoencoder1(nn.Module):
    def __init__(self,size,device,
                 out_in,
                 k1,p1,
                 k2,p2,
                 k3,p3,
                 k4,p4,
                 k5,s1,
                 k6,s2,
                 k7,s3,
                 k8,s4):
        super(Autoencoder1, self).__init__()
     
        self.size = size
        self.out_in = out_in
        self.device = AuxiliaryFunctions.get_device(device)
        # encoder layers
        self.enc1 = nn.Conv2d(1, out_in, kernel_size=k1[0], padding=p1[0]).to('cuda:1')
        self.enc2 = nn.Conv2d(out_in, int(out_in/2), kernel_size=k2[0], padding=p2[0])
        self.enc3 = nn.Conv2d(int(out_in/2), int(out_in/4),kernel_size=k3[0], padding=p3[0])
        self.enc4 = nn.Conv2d(int(out_in/4), int(out_in/8), kernel_size=k4[0], padding=p4[0])
        self.pool = nn.MaxPool2d(2, 2)
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(int(out_in/8),int(out_in/8), kernel_size=k5[0], stride=s1[0])  
        self.dec2 = nn.ConvTranspose2d(int(out_in/8),int(out_in/4), kernel_size=k6[0], stride=s2[0])
        self.dec3 = nn.ConvTranspose2d(int(out_in/4),int(out_in/2), kernel_size=k7[0], stride=s3[0])
        #self.dec4 = nn.ConvTranspose2d(int(out_in/2),int(out_in), kernel_size=k8[0], stride=s4[0])
        self.out = nn.Conv2d(out_in, 1, kernel_size=3, padding=1)
        self.initial = 0
        
    def set_padding_kernel(self,x):
        k  = 2
        s =  2
        p = int((((x.shape[2]-1)*s -self.size)+k)/2)
        if (p % 1 != 0):
            k = 3
            p =  int((((x.shape[2]-1)*s - self.size)+k)/2)
        self.dec4 = nn.ConvTranspose2d(int(self.out_in/2), self.out_in, kernel_size=k, padding=p ,stride=s).to(self.device)
            
    def forward(self, x):
        # encode
        count = 0
        if (self.initial == 0):
            print('initial : '+str(x.shape))
            count = count + x.element_size() * x.nelement()*1e-6
            print((x.element_size() * x.nelement())*1e-6)
        x = F.relu(self.enc1(x))
        if (self.initial == 0):
            print('enc1:'+str(x.shape))
            count = count + x.element_size() * x.nelement()*1e-6
            print((x.element_size() * x.nelement())*1e-6)

        x = self.pool(x)
        if (self.initial == 0):
            print('pool:'+str(x.shape))
            count = count + x.element_size() * x.nelement()*1e-6
            print((x.element_size() * x.nelement())*1e-6)
      
        x = F.relu(self.enc2(x))
        if (self.initial == 0):
            print('enc2:'+str(x.shape))
            count = count + x.element_size() * x.nelement()*1e-6
            print((x.element_size() * x.nelement())*1e-6)

        x = self.pool(x)
        if (self.initial == 0):
            print('pool:'+str(x.shape))
            count = count + x.element_size() * x.nelement()*1e-6
            print((x.element_size() * x.nelement())*1e-6)

        x = F.relu(self.enc3(x))
        if (self.initial == 0):
            print('enc3:'+str(x.shape))
            print((x.element_size() * x.nelement())*1e-6)
            count = count + x.element_size() * x.nelement()*1e-6

        x = self.pool(x)
        if (self.initial == 0):
            print('pool:'+str(x.shape))
            print((x.element_size() * x.nelement())*1e-6)

        x = F.relu(self.enc4(x))
        if (self.initial == 0):
            print('enc4:'+str(x.shape))
            print((x.element_size() * x.nelement())*1e-6)
            count = count + x.element_size() * x.nelement()*1e-6

        x = self.pool(x) # the latent space representation
        if (self.initial == 0):
            print('pool:'+str(x.shape))
            print((x.element_size() * x.nelement())*1e-6)
            count = count + x.element_size() * x.nelement()*1e-6

        
        # decode
        x = F.relu(self.dec1(x))
        if (self.initial == 0):
            print('dec1:'+str(x.shape))
            print((x.element_size() * x.nelement())*1e-6)
            count = count + x.element_size() * x.nelement()*1e-6

        x = F.relu(self.dec2(x))
        if (self.initial == 0):
            print('dec2:'+str(x.shape))
            print((x.element_size() * x.nelement())*1e-6)
            count = count + x.element_size() * x.nelement()*1e-6

        x = F.relu(self.dec3(x))
        if (self.initial == 0):
            print('dec3:'+str(x.shape))
            print((x.element_size() * x.nelement())*1e-6)
            count = count + x.element_size() * x.nelement()*1e-6

        self.set_padding_kernel(x)
        x = F.relu(self.dec4(x))
        if (self.initial == 0):
            print(x.shape)
            print((x.element_size() * x.nelement())*1e-6)
            count = count + x.element_size() * x.nelement()*1e-6

        x = F.sigmoid(self.out(x))
        if (self.initial == 0):
            print('out:'+str(x.shape))
            print((x.element_size() * x.nelement())*1e-6)
            count = count + x.element_size() * x.nelement()*1e-6
            print('size final:'+str(count))
        self.initial = 1
        return x

class Autoencoder2(nn.Module):
    def __init__(self,size,device):
        self.device = AuxiliaryFunctions.get_device(device)
        self.initial = 0
        self.p  = 0
        self.k = 2
        self.size = size
        super(Autoencoder2, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv2d(1, size*3, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(size*3, round(size*1.5), kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(round(size*1.5), round(size*0.75), kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(round(size*0.75), round(size*0.375), kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
     
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(round(size*0.375), round(size*0.375), kernel_size=3, stride=2)  
        self.dec2 = nn.ConvTranspose2d(round(size*0.375), round(size*0.75), kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(round(size*0.75), round(size*1.5), kernel_size=3, stride=2)
        #self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(size, 1, kernel_size=3, padding=1)
    
   
    def set_padding_kernel(self,x):
        self.k = 2
        s = round(self.size*1.5)
        p = (((x.shape[2]-1)*2 -self.size)+self.k )/2
        if (p % 1 == 0):
            self.p = int(p)
            self.k = 2
            self.dec4 = nn.ConvTranspose2d(s, self.size, kernel_size=self.k, padding=self.p ,stride=2).to(self.device)
            return 
        else:
            self.k = 3
            self.p =  int((((x.shape[2]-1)*2 -self.size)+self.k)/2)
            self.dec4 = nn.ConvTranspose2d(s, self.size, kernel_size=self.k, padding=self.p ,stride=2).to(self.device)
            return
            
    def forward(self, x):
        # encode
        #if (self.initial == 0):
        #    print('initial : '+str(x.shape))
        x = F.relu(self.enc1(x))
        #if (self.initial == 0):
        #    print('enc1:'+str(x.shape))
        x = self.pool(x)
        #if (self.initial == 0):
        #    print('pool:'+str(x.shape))
       
        x = F.relu(self.enc2(x))
        #if (self.initial == 0):
        #    print('enc2:'+str(x.shape))
        x = self.pool(x)
        #if (self.initial == 0):
        #    print('pool:'+str(x.shape))
        x = F.relu(self.enc3(x))
        #if (self.initial == 0):
        #    print('enc3:'+str(x.shape))
        x = self.pool(x)
        #if (self.initial == 0):
        #    print('pool:'+str(x.shape))
        x = F.relu(self.enc4(x))
        #if (self.initial == 0):
          #  print('enc4:'+str(x.shape))
        x = self.pool(x) # the latent space representation
        #if (self.initial == 0):
           # print('pool:'+str(x.shape))
        
        # decode
        x = F.relu(self.dec1(x))
        #if (self.initial == 0):
            #print('dec1:'+str(x.shape))
        x = F.relu(self.dec2(x))
        #if (self.initial == 0):
            #print('dec2:'+str(x.shape))
        x = F.relu(self.dec3(x))
        #if (self.initial == 0):
            #print('dec3:'+str(x.shape))
        self.set_padding_kernel(x)
        x = F.relu(self.dec4(x))
        #if (self.initial == 0):
            #print(x.shape)
        x = F.sigmoid(self.out(x))
        #if (self.initial == 0):
            #print('out:'+str(x.shape))
        self.initial = 1
        return x
    
    
class Autoencoder3(nn.Module):
    def __init__(self,size,device):
        self.device = AuxiliaryFunctions.get_device(device)
        self.initial = 0
        self.p  = 0
        self.k = 2
        self.size = size
        super(Autoencoder3, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv2d(1,size*3, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(size*3, round(size*1.5), kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(round(size*1.5), round(size*0.75), kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(round(size*0.75), round(size*0.375), kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(round(size*0.375), round(size*0.375), kernel_size=3, stride=2)  
        self.dec2 = nn.ConvTranspose2d(round(size*0.375), round(size*0.75), kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(round(size*0.75), round(size*1.5), kernel_size=3, stride=2)
        #self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(size, 1, kernel_size=3, padding=1)
    
   
    def set_padding_kernel(self,x):
        self.k = 2
        s = round(self.size*1.5)
        p = (((x.shape[2]-1)*2 -self.size)+self.k )/2
        if (p % 1 == 0):
            self.p = int(p)
            self.k = 2
            self.dec4 = nn.ConvTranspose2d(s, self.size, kernel_size=self.k, padding=self.p ,stride=2).to(self.device)
            return 
        else:
            self.k = 3
            self.p =  int((((x.shape[2]-1)*2 -self.size)+self.k)/2)
            self.dec4 = nn.ConvTranspose2d(s, self.size, kernel_size=self.k, padding=self.p ,stride=2).to(self.device)
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
        x = F.relu(self.enc4(x))
        if (self.initial == 0):
            print('enc4:'+str(x.shape))
        x = self.pool(x) # the latent space representation
        if (self.initial == 0):
            print('pool:'+str(x.shape))
        
        # decode
        x = F.relu(self.dec1(x))
        if (self.initial == 0):
            print('dec1:'+str(x.shape))
        x = F.relu(self.dec2(x))
        if (self.initial == 0):
            print('dec2:'+str(x.shape))
        x = F.relu(self.dec3(x))
        if (self.initial == 0):
            print('dec3:'+str(x.shape))
        self.set_padding_kernel(x)
        x = F.relu(self.dec4(x))
        if (self.initial == 0):
            print(x.shape)
        x = F.sigmoid(self.out(x))
        if (self.initial == 0):
            print('out:'+str(x.shape))
        self.initial = 1
        return x
    
'''class Autoencoder64(nn.Module):
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
        return x'''