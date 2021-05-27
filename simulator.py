# %%
from datasetImages import DatasetImages
from datasetPSF import DatasetPSF
from datasetDirty import DatasetDirty
from paramsEllipses import ParamsEllipses
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
        psf = DatasetPSF(size).read_url(size,type_psf,url)
        return DatasetPSF(size).resize(psf,size)
        
    
    def create_all(self,size,start,finish,step):
    
        list_index = cp.arange(int(start),int(finish),int(step))
        for index in list_index[0:len(list_index)-2]:
            params = ParamsEllipses(size)
            dataset = DatasetImages(size)
            dataset.save(size_image =params.size_figure, params = params,start = index,finish =index+step-1)
        type_psf = 'psf_gauss_'+str(params.size_figure)+'x'+str(params.size_figure)
        psf_gauss = self.psf_gauss(params.size_figure,params.size_figure)
        psf = DatasetPSF(size)
        psf.save(size_image=params.size_figure,type_psf=type_psf,psf= psf_gauss)
        psf_gauss = psf.image
        
        list_index = cp.arange(start,finish,step)
        for index in list_index:
            if (index != list_index[len(list_index)-1]):
                dataset = DatasetImages(size)
                images = dataset.read(params.size_figure,start = index,finish =index+step)
                dirty_gauss = DatasetDirty(size)
                dirty_gauss.save(images,params.size_figure,type_psf,psf_gauss,start = index ,finish = index+step-1)
        type_psf = 'psf_real_'+str(params.size_figure)+'x'+str(params.size_figure)
        psf_real = self.psf_real(params.size_figure)
        psf = DatasetPSF(size)
        psf.save(size_image=params.size_figure,type_psf=type_psf,psf= psf_real)
        psf_real = psf.image 
        list_index = cp.arange(start,finish,step)
        for index in list_index:
            if (index != list_index[len(list_index)-1]):
                dataset = DatasetImages(size)
                images = dataset.read(params.size_figure,start = index,finish =index+step)
                dirty_gauss = DatasetDirty(size)
                dirty_gauss.save(images,params.size_figure,type_psf,psf_gauss,start = index ,finish = index+step-1)

        

        
        

# %%
#x = Simulator().create_all(28,0,10)

# %%
#images = DatasetImages(11)
#images.read(size_image=11, start = 0,finish = 10)
#images.view()

# %%
#psf_gauss = DatasetPSF()
#psf_gauss.read(11,'psf_gauss_11x11')
#psf_gauss.view()

# %%
#psf_real = DatasetPSF()
#psf_real.read(11,'psf_real_11x11')
#psf_real.view()
#psf_real.view()

# %%
#dirty = DatasetDirty(11,'psf_real_11x11')
#dirty.read(11,'psf_real_11x11',start = 0, finish= 10)
#dirty.view_dirty()


# %%
