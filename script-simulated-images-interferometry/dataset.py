#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class InterferometryDataset(Dataset):
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


# In[ ]:


class Dataset:
    def __init__(self,
                    size_figure,
                    type_psf,
                    perc_train = 0.7,
                    perc_validation = 0.25,
                    perc_test = 0.05,
                    batch_train = 10,
                    batch_validation = 10,
                    batch_test = 1
                    path_images = None,
                    path_convolution = None,
                    path_psf = None,
                ):
        self.size_figure = size_figure
        self.type_psf = type_psf
        self.path_images = self.init_path_images(path_images)
        self.path_convolution  self.init_path_convolution(path_convolution)
        self.path_psf = self.init_path_psf(path_psf)
        self.perc_train = perc_train
        self.perc_validation = perc_validation
        self.perc_test = perc_test
        self.batch_train = batch_train
        self.batch_test = batch_test
        self.batch_validation = batch_validation
        
    def init_path_images(self,path_save):
        if (path_save == None):
            return'../datasets/images_'+str(self.size_figure)+'x'+str(self.size_figure)+'/images'
        else:
            return path_save  
    def init_path_convolution(self,path_convolution):
        if (path_convolution == None):
            return '../datasets/images_'+str(self.size_figure)+'x'+str(self.size_figure)+'/convolutions/'+self.type_psf+'/conv'
        else 
            return path_convolution
    
    def init_path_psf (self,path_psf):
        if (path_psf == None):
            return '../datasets/images_'+str(self.size_figure)+'x'+str(self.size_figure)+'/convolutions/'+self.type_psf+'/psf'
        else:
            return path_psf

    def tsfms(self):
        return transforms.Compose([transforms.ToTensor()])
        
    def create_train_data(start,step):
        size_train = round(step*self.perc_train)
        batch_train_size=  BATCH_TRAIN 
        params = ParamsEllipses(self.size_figure)

        data_image = DatasetImages(self.size_figure)
        data_image.create(size_image=self.size_figure,params = params , start = start,stop = start +step)

        data_psf = DatasetPSF(self.size_figure,self.type_psf)
        data_psf.create(N,TYPE_PSF)

        data_dirty = DatasetDirty(self.size_figure,self.type_psf,data_psf.image)
        data_dirty.create(images = data_image.images,size_image = self.size_figure, type_psf = self.type_psf, psf =data_psf.image)
        trainSet=interferometryDataset(data_dirty.dirtys,data_image.images,self.tsfms())
        trainLoader=DataLoader(trainSet,self.batch_train,shuffle=True)
        return trainLoader
    
    
    def create_validation_data(start,step):
        start = start + round(step*self.perc_train)
        size_validation =  round(step*self.perc_validation)

        params = ParamsEllipses(self.size_figure)
        data_image = DatasetImages(self.size_figure)
        data_image.create(size_image=self.size_figure,params = params , start = start,stop = start +size_validation)

        data_psf = DatasetPSF(self.size_figure,self.type_psf)
        data_psf.create(self.size_figure,self.type_psf)

        data_dirty = DatasetDirty(self.size_figure,self.type_psf,data_psf.image)
        data_dirty.create(images = data_image.images,size_image = self.size, type_psf = self.type_psf, psf =data_psf.image)
        validationSet = interferometryDataset(data_dirty.dirtys,data_image.images,self.tsfms())
        validationLoader=DataLoader(validationSet,self.batch_validation,shuffle=True)
        return validationLoader

     def create_test_data(start,step):
        start = start + round(step*self.perc_train) + round(step*self.perc_validation)
        size_test =  round(step*self.perc_test)

        params = ParamsEllipses(self.size_figure)
        data_image = DatasetImages(self.size_figure)
        data_image.create(size_image=self.size_figure,params = params , start = start,stop = start +size_test)

        data_psf = DatasetPSF(self.size_figure,self.type_psf)
        data_psf.create(self.size_figure,self.type_psf)

        data_dirty = DatasetDirty(self.size_figure,self.type_psf,data_psf.image)
        data_dirty.create(images = data_image.images,size_image = self.size, type_psf = self.type_psf, psf =data_psf.image)
        testSet = interferometryDataset(data_dirty.dirtys,data_image.images,self.tsfms())
        testLoader=DataLoader(testSet,self.batch_test,shuffle=True)
        return testLoader
    
    
    def read_train_data(start,stop):
        size = start- stop  #size of lot of the dataset
        size_train = round(size*self.perc_train)
        data_image = DatasetImages(self.size_figure)
        data_image.read(size_image=self.size_figure, start = start,stop = start+size_train)
        data_dirty = DatasetDirty(self.size_figure,self.type_psf)
        data_dirty.read(size_image = self.size_figure, type_psf = self.type_psf,start = start, stop = start + size_train)
        trainSet= interferometryDataset(data_dirty.dirtys,data_image.images,self.tsfms())
        trainLoader=DataLoader(trainSet,self.batch_train,shuffle=True)
        return trainLoader
    
    def read_validation_data(start,stop):
        size = start- stop  
        size_validation = round(size*self.perc_validation)
        size_train = round(size*self.perc_train)
        start = start + size_train
    
        data_image = DatasetImages(self.size_figure)
        data_image.read(size_image=self.size_figure, start = start,stop = start+size_validation)
        data_dirty = DatasetDirty(self.size_figure,self.type_psf)
        data_dirty.read(size_image = self.size_figure, type_psf = self.type_psf,start = start, stop = start + size_validation)
        validationSet= interferometryDataset(data_dirty.dirtys,data_image.images,self.tsfms())
        validationLoader=DataLoader(validationSet,self.batch_validation,shuffle=True)
        return validationLoader
    

    def read_test_data(start,stop):
        size = start- stop 
        size_test = round(size*self.perc_test)
        size_train = round(size*self.perc_train)
        size_validation = round(size*self.perc_train)
        start = start + size_train +size_validation
        data_image = DatasetImages(self.size_figure)
        data_image.read(size_image=self.size_figure, start = start,stop = start+size_test)
        data_dirty = DatasetDirty(self.size_figure,self.type_psf)
        data_dirty.read(size_image = self.size_figure, type_psf = self.type_psf,start = start, stop = start + size_test)
        testSet= interferometryDataset(data_dirty.dirtys,data_image.images,self.tsfms())
        testLoader=DataLoader(testSet,self.batch_validation,shuffle=True)
        return testLoader


# In[ ]:


NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    size_data= FINAL_DATASET - INITIAL_DATASET
    PATH_VALIDATION =  '../datasets/images_'+str(N)+'x'+str(N)+'/convolutions/'+TYPE_PSF+'/models/model_S'+str(size_data)+'_B'+str(BATCH_TRAIN)+'_N'+str(NUM_EPOCHS)+'/validation'
    PATH_TEST =  '../datasets/images_'+str(N)+'x'+str(N)+'/convolutions/'+TYPE_PSF+'/models/model_S'+str(size_data)+'_B'+str(BATCH_TRAIN)+'_N'+str(NUM_EPOCHS)+'/test'
    PATH_GRAPH ='../datasets/images_'+str(N)+'x'+str(N)+'/convolutions/'+TYPE_PSF+'/models/model_S'+str(size_data)+'_B'+str(BATCH_TRAIN)+'_N'+str(NUM_EPOCHS)+'/graph'
    PATH_MODEL_SAVE ='../datasets/images_'+str(N)+'x'+str(N)+'/convolutions/'+TYPE_PSF+'/models/model_S'+str(size_data)+'_B'+str(BATCH_TRAIN)+'_N'+str(NUM_EPOCHS)+'/model' #path where the dataset convolution is saved
    

