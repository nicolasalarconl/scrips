from torch.utils.data import Dataset
import cupy as cp

class TrainData(Dataset):
    def __init__(self,datasetnoised,datasetclean,transform,device):
        self.noise=datasetnoised
        self.clean=datasetclean
        self.transform=transform
        self.init_device(device)

    def init_device(self,device):
        cp.cuda.Device(device).use()
            
    def __len__(self):
        return len(self.noise)
  
    def __getitem__(self,idx):
        noise=self.noise[idx]
        clean=self.clean[idx]
        if self.transform != None:
            noise=self.transform(cp.asnumpy(noise))
            clean=self.transform(cp.asnumpy(clean))
        return (noise,clean)

class TestData(Dataset):
    def __init__(self,datasetnoised,datasetclean,mask,transform,device):
        self.noise=datasetnoised
        self.clean=datasetclean
        self.transform=transform
        self.mask = mask
        self.init_device(device)
  
    def init_device(self,device):
        cp.cuda.Device(device).use()
        
    def __len__(self):
        return len(self.noise)
  
    def __getitem__(self,idx):
        noise=self.noise[idx]
        clean=self.clean[idx]
        mask=self.mask[idx]
        if self.transform != None:
            noise=self.transform(cp.asnumpy(noise))
            clean=self.transform(cp.asnumpy(clean))
        return (noise,clean,mask)


