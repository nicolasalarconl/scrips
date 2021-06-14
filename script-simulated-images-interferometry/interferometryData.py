from torch.utils.data import Dataset
import cupy as cp

class InterferometryData(Dataset):
  def __init__(self,datasetnoised,datasetclean,transform):
    self.noise=datasetnoised
    self.clean=datasetclean
    self.transform=transform
  
  def __len__(self):
    return len(self.noise)
  
  def __getitem__(self,idx):
    xNoise=self.noise[idx]
    xClean=self.clean[idx]
    if self.transform != None:
      xNoise=self.transform(cp.asnumpy(xNoise))
      xClean=self.transform(cp.asnumpy(xClean))
    return (xNoise,xClean)

