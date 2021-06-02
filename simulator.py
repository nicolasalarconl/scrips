# %%
from datasetImages import DatasetImages
from datasetPSF import DatasetPSF
from datasetDirty import DatasetDirty
from paramsEllipses import ParamsEllipses
#import cupy as cp
import numpy as cp

class Simulator:
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
        
    
    def create_all(self,size,start,finish,step):
        list_index = cp.arange(int(start),int(finish),int(step))
        for index in list_index:    
            params = ParamsEllipses(size,index)
            dataset = DatasetImages(size)
            dataset.save(size_image =params.size_figure, params = params,start = index,finish =index+step)
            
            type_psf = 'psf_gauss_'+str(params.size_figure)+'x'+str(params.size_figure)
            psf_gauss = self.psf_gauss(params.size_figure,params.size_figure)
            psf = DatasetPSF(size,type_psf)
            psf.save(size_image=params.size_figure,type_psf=type_psf,psf= psf_gauss)
            psf_gauss = psf.image
        
            images = dataset.read(params.size_figure,start = index,finish =index+step)
            dirty_gauss = DatasetDirty(size,type_psf)
            dirty_gauss.save(images,params.size_figure,type_psf,psf_gauss,start = index ,finish = index+step)
        
            type_psf = 'psf_real_'+str(params.size_figure)+'x'+str(params.size_figure)
            psf_real = self.psf_real(params.size_figure)
            psf = DatasetPSF(size,type_psf)
            psf.save(size_image=params.size_figure,type_psf=type_psf,psf= psf_real)
            psf_real = psf.image 
         
            dirty_gauss = DatasetDirty(size,type_psf)
            dirty_gauss.save(images,params.size_figure,type_psf,psf_gauss,start = index ,finish = index+step)

        
           
        
        

# %%
#x = Simulator().create_all(28,0,1000,10)
#images = DatasetImages(500)
#images.read(size_image=500, start = 0,finish = 10)
#images.view()

# %%
