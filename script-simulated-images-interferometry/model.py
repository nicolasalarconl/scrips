

from tqdm import tqdm
import torch 
from datasetInterferometry import DatasetInterferometry
from auxiliaryFunctions import AuxiliaryFunctions
from graph import Graph
from psnr import PSNR
import time


class Model:
    def __init__(self,size_figure,type_psf,num_epochs,learning_rate,batch_train,criterion,optimizer,device,start,stop, step = None,path_validation= None,path_test=None,path_graph = None,path_model = None,path_log= None):
        self.size_figure = size_figure
        self.type_psf = type_psf
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_train = batch_train
        self.start = start
        self.stop = stop
        self.step = step
        self.criterion = criterion
        self.optimizer = optimizer
        self.size_data = start-stop
        self.path_test  =  self.init_path_test(path_test)
        self.path_graph =  self.init_path_graph(path_graph)
        self.path_model = self.init_path_model(path_model)
        self.path_log =  self.init_path_log(path_log)
        self.model_data = self.type_psf+'_S'+str(self.size_data)+'_B'+str(self.batch_train)+'_E'+str(self.num_epochs)
        self.device = device
         
    def init_path_test(self,path_test):
        if (path_test == None):
            return '../../datasets/images_'+str(self.size_figure )+'x'+str(self.size_figure )+'/convolutions/'+self.type_psf+'/models/model_S'+str(self.size_data)+'_B'+str(self.batch_train)+'_E'+str(self.num_epochs)+'/test'
        else:
            return path_test 
        
    def init_path_graph(self,path_graph):
        if (path_graph == None):
            return '../../datasets/images_'+str(self.size_figure )+'x'+str(self.size_figure )+'/convolutions/'+self.type_psf+'/models/model_S'+str(self.size_data)+'_B'+str(self.batch_train)+'_E'+str(self.num_epochs)+'/graph'
        else:
            return path_graph 
        
    def init_path_model(self,path_model):
         if (path_model == None):
            return '../../datasets/images_'+str(self.size_figure )+'x'+str(self.size_figure )+'/convolutions/'+self.type_psf+'/models/model_S'+str(self.size_data)+'_B'+str(self.batch_train)+'_E'+str( self.num_epochs)+'/model' 
         else:
            return path_model 

        
    def init_path_log(self,path_test):
        if (path_test == None):
            return '../../datasets/images_'+str(self.size_figure )+'x'+str(self.size_figure )+'/log'
        else:
            return path_test 
    
   
    def train_data_memory(self,net,device,start,stop,step):
        train_loss = []
        valid_loss = []
        list_index = cp.arange(start,stop,step)
        data = DatasetInterferometry(size_figure = self.size_figure,
                                     type_psf = self.type_psf,
                                     batch_train = self.batch_train,
                                     device = self.device)
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            size_train = 0
            for index in list_index:
                trainLoader = data.create_train_data(index,step)
                size_train = size_train +len(trainLoader)
                for dirty,clean in tqdm((trainLoader)):
                    dirty,clean=dirty.to(device).float(),clean.to(device).float()
                    self.optimizer.zero_grad()            
                    outputs = net(dirty)
                    loss = self.criterion(outputs, clean)
                    #backpropagation
                    loss.backward()
                    #update the parameters
                    self.optimizer.step()
                    running_loss += loss.item()
            loss = running_loss / size_train
            train_loss.append(loss)
        ## validation ## ''
            print('Epoch {} of {}, Train Loss: {:.3f}, Len Train:{}'.format(epoch+1, self.num_epochs, loss,size_train))
            validation_loss = 0.0
            net.eval()
            size_validate = 0  
            for index in list_index:
                validationloader = data.create_validation_data(index,step)
                size_validate = size_validate + len(validationloader)
                for dirty, clean in validationloader:
                    dirty,clean=dirty.to(device).float(),clean.to(device).float()
                    self.optimizer.zero_grad()    ## TODO BORRRAR ??         
                    outputs = net(dirty)
                    loss = self.criterion(outputs, clean)
                    validation_loss += loss.item()
            loss = validation_loss / size_validate
            valid_loss.append(loss)
            print('Epoch {} of {}, Validate Loss: {:.3f} Len Validate:{}'.format(epoch+1, self.num_epochs, loss,size_validate))
        return net,train_loss,valid_loss
    
    def test_data_memory(self,net,device,start,stop,step):
        list_index = cp.arange(start,stop,step)
        data = DatasetInterferometry(size_figure = self.size_figure,
                                     type_psf = self.type_psf,
                                     batch_train = self.batch_train,
                                     device = self.device)
        pnsr = []
        i = 0
        for index in list_index:
            testloader = data.create_test_data(index,step)
            for dirty,clean in tqdm((testloader)):
                dirty,clean=dirty.to(device).float(),clean.float()
                outputs = net(dirty)
                dirty  = dirty.cpu().data
                outputs = outputs.cpu().data
                value_psnr = PSNR(clean,dirty,self.device).get
                pnsr.append(value_psnr)

                AuxiliaryFunctions.save_fit_image(dirty, self.size_figure,name=self.path_test+'/noisy{}.png'.format(i))
                AuxiliaryFunctions.save_fit_image(outputs, self.size_figure,
                                                      name=self.path_test+'/denoised{}.png'.format(i))
                AuxiliaryFunctions.save_decoded_image(clean, self.size_figure,
                                                      name=self.path_test+'/clean{}.png'.format(i))
                i = i +1
        print(' Len test:{}'.format(i))
        return net,pnsr
    
    
    ### TODO: CARGADO POR EPOCA  PODRIA SER UNA OPCION!! ##
    def train_data(self,net,device,start,stop):
        train_loss = []
        valid_loss = []
        data = DatasetInterferometry(size_figure = self.size_figure,
                                     type_psf = self.type_psf,
                                     batch_train = self.batch_train,
                                     device = self.device)
        trainLoader = data.read_train_data(start,stop)
        validationloader = data.read_validation_data(start,stop)
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            net.train()
            for dirty,clean in tqdm((trainLoader)):
                dirty,clean=dirty.to(device).float(),clean.to(device).float()
                self.optimizer.zero_grad()            
                outputs =net(dirty)
                loss = self.criterion(outputs, clean)
                #backpropagation
                loss.backward()
                #update the parameters
                self.optimizer.step()
                running_loss += loss.item()
            loss = running_loss / len(trainLoader)
            train_loss.append(loss)
            print('Epoch {} of {}, Train Loss: {:.3f}, Len Train:{}'.format(epoch+1, self.num_epochs, loss,len(trainLoader)))
            validation_loss = 0.0
            net.eval()
            size_validate = 0  
            for dirty, clean in validationloader:
                dirty,clean=dirty.to(device).float(),clean.to(device).float()
                self.optimizer.zero_grad()         
                outputs = net(dirty)
                loss = self.criterion(outputs, clean)
                validation_loss += loss.item()
            loss = validation_loss / len(validationloader)
            valid_loss.append(loss)
            print('Epoch {} of {}, Validate Loss: {:.3f} Len Validate:{}'.format(epoch+1, self.num_epochs, loss,
                                                                                 len(validationloader)))
        return net,train_loss,valid_loss
    
    def save_images(self,index,dirty,output,clean):
        AuxiliaryFunctions.save_fit_image(dirty,path=self.path_test+'/noisy{}.png'.format(index))
        AuxiliaryFunctions.save_fit_image(output,path=self.path_test+'/denoised{}.png'.format(index))
        AuxiliaryFunctions.save_fit_image(clean,path=self.path_test+'/clean{}.png'.format(index))
         
    
    def test_data(self,net,device,start,stop,size_save):
        data = DatasetInterferometry(size_figure = self.size_figure,
                                     type_psf = self.type_psf,
                                     batch_train = self.batch_train,
                                     device = self.device)
        testloader = data.read_test_data(start,stop)
        index = 0
        pnsr_clean_output = []
        pnsr_clean_dirty = []
        pnsr_diff = []        
        
        for dirty,clean,mask in tqdm((testloader)):
            dirty=dirty.to(device).float()
            output = net(dirty).cpu().data
            dirty = dirty.cpu()
            psnr_c_o = PSNR(clean,output,mask,self.device).get
            psnr_c_d = PSNR(clean,dirty,mask,self.device).get
            pnsr_d  = psnr_c_o - psnr_c_d
            pnsr_clean_output.append(psnr_c_o)
            pnsr_clean_dirty.append(psnr_c_d)
            pnsr_diff.append(pnsr_d)
            if (index < size_save ): 
                self.save_images(index,dirty,output,clean)
            index = index + 1  
         
        print('Len test:{}'.format(index))
        return net,pnsr_clean_output,pnsr_clean_dirty,pnsr_diff
    
    
    def run_data_memory(self,net,start,stop,step):
        AuxiliaryFunctions.make_dir(self.path_test)
        AuxiliaryFunctions.make_dir(self.path_graph)
        AuxiliaryFunctions.make_dir(self.path_model)
        
        torch.cuda.empty_cache()
        device = AuxiliaryFunctions.get_device(self.device)
        net.to(device)
        net,train_loss,validation_loss = self.train_data_memory(net,device,start,stop,step)
        AuxiliaryFunctions.save_mfodel(net,self.path_model)
  
        Graph.train_loss_validation_loss_epoch(path_graph =self.path_graph ,train_loss = train_loss,
                                               validation_loss = validation_loss)        
        net,pnsr= self.test_data_memory(net,device,start,stop,step)
        Graph.psnr_test(self.path_graph,pnsr)
        return net

        
    def run_train_data(self,net,start,stop):
        AuxiliaryFunctions.make_dir(self.path_graph)
        AuxiliaryFunctions.make_dir(self.path_model)
        AuxiliaryFunctions.make_dir(self.path_log)     
        
        start_time = time.time()
        AuxiliaryFunctions.log(self.path_log,self.model_data,'start train data')
        torch.cuda.empty_cache()
        device = AuxiliaryFunctions.get_device(self.device)
        net.to(device)
        net,train_loss,validation_loss = self.train_data(net,device,start,stop)
        AuxiliaryFunctions.save_model(net,self.path_model)
        Graph.train_loss_validation_loss_epoch(path_graph =self.path_graph ,train_loss = train_loss,
                                               validation_loss = validation_loss)
        stop_time = time.time()
        AuxiliaryFunctions.log(self.path_log,self.model_data,'stop train data')
        time_final = stop_time-start_time
        AuxiliaryFunctions.log(self.path_log,self.model_data,'total time train data',time_final)
        return net
    
    def run_test_data(self,net,start,stop,size_save):
        AuxiliaryFunctions.make_dir(self.path_test)
        AuxiliaryFunctions.make_dir(self.path_log) 
        AuxiliaryFunctions.log(self.path_log,self.model_data,'start test data')
        start_time = time.time()
        torch.cuda.empty_cache()
        device = AuxiliaryFunctions.get_device(self.device)
        net.to(device) 
        net,pnsr_clean_output,pnsr_clean_dirty,pnsr_diff = self.test_data(net,device,start,stop,size_save)
        Graph.psnr_test(self.path_graph,pnsr_clean_output,pnsr_clean_dirty,pnsr_diff)
        AuxiliaryFunctions.send_alert(self.path_log,pnsr_diff,self.model_data)
        stop_time = time.time()
        AuxiliaryFunctions.log(self.path_log,self.model_data,'stop test data')
        time_final = stop_time-start_time
        AuxiliaryFunctions.log(self.path_log,self.model_data,'total time test data',time_final)
        return net
       
        
    def view_data(self,start,stop):
        data = DatasetInterferometry(size_figure = self.size_figure,
                                     type_psf = self.type_psf,
                                     batch_train = self.batch_train,
                                     device = self.device)
        testloader = data.read_test_data(start,stop)
        data.view_data(testloader)
        
    def get_resumen(self,net = None, idx = None):
        print("Image size: "+str(self.size_figure))
        print("Len dataset: "+str(self.stop-self.start))
        print("Type psf: "+str(self.type_psf))
        print("Number epoch: " +str(self.num_epochs))
        print("Learning rate: " + str(self.learning_rate))
        print("Batch train: " + str(self.batch_train))
        print("\n")
        print("Net: ",net)
        print("\n")
        Graph.read_graph(self.path_graph+'/graph_loss_train_validation.png')
        Graph.read_graph(self.path_graph+'/graph_psnr_clean_output.png')
        Graph.read_graph(self.path_graph+'/graph_psnr_clean_dirty.png')
        Graph.read_graph(self.path_graph+'/graph_psnr_diff_output_dirty.png')
        if (idx == None):
            idx = 0       
        AuxiliaryFunctions.display_fit_image('clean',self.path_test+'/clean'+str(idx)+'.png')
        AuxiliaryFunctions.display_fit_image('dirty',self.path_test+'/noisy'+str(idx)+'.png')
        AuxiliaryFunctions.display_fit_image('output',self.path_test+'/denoised'+str(idx)+'.png')
        
