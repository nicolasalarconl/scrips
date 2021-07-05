from model import Model
from autoencoder import Autoencoder1 as net1
import cupy as cp
from auxiliaryFunctions import AuxiliaryFunctions
#import torch 


#from pytorch_modelsize import SizeEstimator

architecture_name = 'autoencoder1'
device = 0
size_figure = 640
type_psf_gauss = 'psf_gauss_'+str(size_figure)+'x'+str(size_figure)
type_psf_real = 'psf_real_'+str(size_figure)+'x'+str(size_figure)
types =  [type_psf_gauss,type_psf_real]

k1 = 3,
p1 = 1,
k2 = 3,
p2 = 1,
k3 = 3,
p3 = 1,
k4 = 3,
p4 = 1,
k5 = 3,
s1 = 2,
k6 = 3,
s2 = 2,
k7 = 3,
s3 = 2,
k8 = 3,
s4 = 2
out_in = 32



def get_parameter_sizes(model):
        '''Get sizes of all parameters in `model`'''
        mods = list(model.modules())
        sizes = []
        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(cp.array(p[j].size()))
        return sizes
        
def calc_param_bits(model):
        total_bits = 0
        bits = 32
        params_sizes =  get_parameter_sizes(model)
        for i in range(len(params_sizes)):
            s = params_sizes[i]
            bits = cp.prod(cp.array(s))*32
            total_bits += bits
        total_megabytes = (total_bits*1.25e-7)
        print(total_megabytes)
    
net = net1(size = size_figure,
           device = device,
           out_in = out_in,
           k1=k1,p1=p1,k2=k2,p2=p2,k3=k3,p3=p3,k4=k4,p4=p4,
           k5=k5,s1=s1,k6=k6,s2=s2,k7=k7,s3=s3,k8=k8,s4=s4
        )  
calc_param_bits(net)
#batchs = 3#,1,128]
#se = SizeEstimator(net, input_size=(batchs,1,size_figure,size_figure))
##print(se.estimate_size())






# %%

'''datasets = [1000]
batchs = [3]#,1,128]
epochs = [10]
learning_rates = [1e-3]#,1e-4,1e-5]
combinations = len(datasets)*len(batchs)*len(epochs)*len(learning_rates)
# agregar mas capas por que el in_out no puede ser muy grande, calcular esto..#
index = 1

for type_psf in types:
    for learning_rate in learning_rates:
        for stop in datasets:
            for batch_train in batchs:
                for num_epochs in epochs:
                    print('start model :'+str(index)+'/'+str(combinations))
                    #torch.cuda.empty_cache()
                    m = Model(size_figure = size_figure,
                                        type_psf = type_psf,
                                        num_epochs = num_epochs,
                                        learning_rate =  learning_rate , 
                                        batch_train = batch_train ,
                                        start = 0,
                                        stop = stop,
                                        device = device,
                                        architecture_name = architecture_name,
                                        net = net           
                    ) 
                    print('...train model')
                    #net =m.run_train_data(net = net,start = 0 , stop = stop)
                    #print('...test model')
                    #net = m.run_test_data(net = net,start = 0, stop = stop,size_save = 1)
                    #print('finish model:'+str(index)+'/'+str(combinations))
                    #index = index +1
                    #m.get_resumen(net= net,idx = 0)'''

# %%
#https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
#device multiple model..

# %%
# calcular el tamaño y una formula para saber.



# %%
# agregar mas capas para imagens mas grandes..#

# %%


# %%
