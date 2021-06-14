
from datasetImages import DatasetImages
from datasetPSF import DatasetPSF
from datasetDirty import DatasetDirty
from paramsEllipses import ParamsEllipses
import cupy as cp

class Simulator:
         
    def create_save(size,start,stop,step):
        list_index = cp.arange(int(start),int(stop),int(step))
        for index in list_index:    
            params = ParamsEllipses(size)
            dataset = DatasetImages(size)
            dataset.save(size_image =params.size_figure, params = params,start = index,stop =index+step)
            
            type_psf = 'psf_gauss_'+str(params.size_figure)+'x'+str(params.size_figure)
            psf = DatasetPSF(size,type_psf)
            psf_gauss = psf.psf_gauss(params.size_figure,params.size_figure)
            psf.save(size_image=params.size_figure,type_psf=type_psf,psf= psf_gauss)
            # TODO: arreglar
            ##psf.read(size_image=params.size_figure,type_psf=type_psf)
        
            dataset.read(params.size_figure,start = index,stop =index+step)
            dirty_gauss = DatasetDirty(size,type_psf)
            dirty_gauss.save(dataset.images,params.size_figure,type_psf,psf_gauss,start = index ,finish = index+step)
        
            type_psf = 'psf_real_'+str(params.size_figure)+'x'+str(params.size_figure)
            psf = DatasetPSF(size,type_psf)
            psf_real = psf.psf_real(params.size_figure)
            psf.save(size_image=params.size_figure,type_psf=type_psf,psf= psf_real)
            psf_real = psf.image 
         
            dirty_real = DatasetDirty(size,type_psf)
            dirty_real.save(dataset.images,params.size_figure,type_psf,psf_gauss,start = index ,finish = index+step)
        