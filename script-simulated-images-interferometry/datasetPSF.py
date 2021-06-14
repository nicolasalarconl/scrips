# %%
#import cupy as cp
import cupy as cp
from astropy.io import fits
from astropy.utils.data import download_file
from matplotlib import pyplot as plt
import cv2
from auxiliaryFunctions import AuxiliaryFunctions

# %%
class DatasetPSF:
    def __init__(self,size_image,type_psf,path_save = None,path_read = None):
        self.size_image = size_image
        self.type_psf = type_psf
        self.image = []
        self.path_save = self.init_path_save(path_save)
        self.path_read = self.init_path_read(path_read)
    
    
    def init_path_save(self,path_save):
        if (path_save is None):
            return '../datasets/images_'+str(self.size_image)+'x'+str(self.size_image)+'/convolutions/'+str(self.type_psf)+'/psf' 
        else:
            return path_save
        
    def init_path_read(self,path_read):
        if (path_read == None):
            return '../datasets/images_'+str(self.size_image)+'x'+str(self.size_image)+'/convolutions/'+str(self.type_psf)+'/psf' 
        else:
            return path_read 
        
    def len_image(self):
        return len(self.image)
        
    def save(self,size_image,type_psf,psf,path = None):
        self.size_image = size_image
        self.type_psf =  type_psf
        psf = cp.asnumpy(psf)
        if(path != None):   
            self.path_save = path
        AuxiliaryFunctions.make_dir(self.path_save)
        hdu_image =fits.PrimaryHDU(psf)
        hdu_image.writeto(self.path_save+'/psf_'+self.type_psf+'.fits',clobber=True)


    def psf_gauss(self,tamX,tamY):
        x, y = cp.meshgrid(cp.linspace(-1,1,tamX), cp.linspace(-1,1,tamY))
        d = cp.sqrt(x*x+y*y)
        sigma, mu = 1/(tamX/2), 0.0
        gauss = cp.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        return gauss


    def psf_real(self,size):
        type_psf = 'psf_real_'+str(size)+'x'+str(size)
        url = 'https://github.com/nicolasalarconl/InterferometryDeepLearning/blob/main/4_hd142_128x128_08.psf.fits?raw=true'
        psf = DatasetPSF(size,type_psf).read_url(size,type_psf,url)
        return DatasetPSF(size,type_psf).resize(psf,size)
        
    def create(self,N,type_psf):
        if (type_psf == 'psf_gauss_'+str(N)+'x'+str(N)):
          self.image = self.psf_gauss(N,N)
        if (type_psf == 'psf_real_'+str(N)+'x'+str(N)):
          self.image = self.psf_real(N)
        
        
    ## TODO: arreglar ## 
    def read(self,size_image,type_psf,path = None):
        '''self.type_psf =  type_psf
        self.size_image = size_image
        if(path != None):
            self.path_read = path
        hdul=fits.open(self.path_read+'/psf_'+self.type_psf+'.fits')
        hdr = hdul[0].header
        size = hdr[3]
        data = hdul[0].data
        data=  cp.reshape(data,[size,size])[0]
        data  = cp.array(data)
        print(type(data))'''

    def read_url(self,size_image,type_psf,url):
        self.type_psf =  type_psf
        self.size_image = size_image
        image_link = download_file(url, cache=True )
        image = fits.getdata(image_link).astype(cp.float32)
        image = cp.reshape(image,[image.shape[2],image.shape[3]])
        self.image = cp.array(image)
        return self.image
      
    def resize(self,psf,size):
        return  cv2.resize(cp.asnumpy(psf), dsize=(size, size), interpolation=cv2.INTER_CUBIC)
       
    def view(self):
            plt.imshow(cp.asnumpy(self.image))
    