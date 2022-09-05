from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torchvision
from torch.autograd import Variable

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import os
import time
import itertools
import cv2
import tqdm
from skimage import io
from multiprocessing import Process, freeze_support, set_start_method
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

class custom_Dataset(Dataset):
    def __init__(self, root_dir, classname, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []
        self.file_len = 0
        classcount = -1
        self.class_list = []
        for root, dir, file in os.walk(root_dir):
            if classcount == -1:
                classcount += 1
                continue
            if len(dir) == 0:
                temp = []
                n = 1
                for i in range(len(file)):
                    p = str(n) + ".png"
                    temp.append(os.path.join(root, p))
                    n += 1
                self.file_list.append(temp)
                self.class_list.append(classname[classcount])
                del p, n, temp
            elif len(file) == 0:
                self.file_len += len(dir)
                if len(self.file_list) != 0:
                    classcount += 1
                    
    def __len__(self):
        return self.file_len
    
    def __getitem__(self, index):
        image = []
        t = 0
        pre = []
        for im in self.file_list[index]:
            if t == 0:
                t += 1
                pre = self.transform(Image.open(im).convert('L'))
            else:
                temp = []
                now = self.transform(Image.open(im).convert('L'))
                temp = torch.cat((pre,now),0)
                pre = now
                image.append(temp)
                t += 1
        image = torch.stack(image)
        label_one_hot = []
        for step in range(t-1):
            temp = []
            for l in range(10):
                if str(l) == self.class_list[index]:
                    temp.append(1.0)
                else:
                    temp.append(0.0)
            label_one_hot.append(temp)
        label_one_hot = torch.tensor(label_one_hot)
        label = torch.tensor(int(self.class_list[index]))
            
        return (image, label, label_one_hot)
    
class DECOLLE_Neuron(nn.Module):
    def __init__(
        self, 
        layer_block,
        spike_grad=surrogate.fast_sigmoid(slope=10),
        alpha=0.9, 
        beta=0.85, 
        alpharp=0.65, 
        wrp=1.0, 
        do_detach=True, 
        gain=1
        ):
        super(DECOLLE_Neuron, self).__init__()
        self.conv_layer = layer_block
        self.spike_grad = spike_grad

        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.beta = torch.tensor(beta, requires_grad=False)
        self.alpharp = torch.tensor(alpharp, requires_grad=False)
        self.wrp = torch.tensor(wrp, requires_grad=False)
        self.gain = torch.tensor(gain, requires_grad=False)
            
        self.tau_m = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad=False)
        self.tau_s = torch.nn.Parameter(1. / (1 - self.beta), requires_grad=False)
        
        self.do_detach = do_detach
        self.state = None
        self.P = torch.zeros(1)
        self.Q = torch.zeros(1)
        self.R = torch.zeros(1)
        self.S = torch.zeros(1)
        
    def forward(self, x, init_state=False):
        if init_state:
            self.state = None
        if self.state == None:
            P = torch.zeros_like(x)
            Q = torch.zeros_like(x)
            Q = self.beta * Q + self.tau_s * self.gain * x
            P = self.alpha * P + self.tau_m * Q
            U = self.conv_layer(P)
            R = torch.zeros_like(U)
            S = torch.zeros_like(U)
            R = self.alpharp * R - S * self.wrp
            U = U + R
            S = self.spike_grad(U)
            self.state = True
        else:
            P = self.P
            Q = self.Q
            R = self.R
            S = self.S
            Q = self.beta * Q + self.tau_s * self.gain * x
            P = self.alpha * P + self.tau_m * Q
            R = self.alpharp * R - S * self.wrp
            U = self.conv_layer(P) + R
            S = self.spike_grad(U)
            
        self.P = P
        self.Q = Q
        self.R = R
        self.S = S
        
        if self.do_detach:
            self.P.detach_()
            self.Q.detach_()
            self.R.detach_()
            self.S.detach_()
            
        return U

class snnmodel(nn.Module):
    def __init__(self, kernel_size, beta, spike_grad):
        super(snnmodel, self).__init__()
        
        # DECOLLE_Neuron can be replaced by nn.Conv2d or nn.Sequential
        self.deco1 = DECOLLE_Neuron(nn.Conv2d(2, 4, kernel_size),alpha=0.4, beta=0.3,alpharp=0.6)
        self.deco2 = DECOLLE_Neuron(nn.Conv2d(4, 8, kernel_size),alpha=0.4, beta=0.3,alpharp=0.6)
        self.deco3 = DECOLLE_Neuron(nn.Conv2d(8, 8, kernel_size),alpha=0.4, beta=0.3,alpharp=0.6)
        

        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)
        
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()
        
        self.fc1 = nn.Linear(170*127*4, 10)
        self.fc2 = nn.Linear(82*60*8, 10)
        self.fc3 = nn.Linear(38*27*8, 10)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        
    def forward(self, x):
        
        spk_out = []
        r_out = []
        mem_out = []
        
        # stage 1
        
        mem = self.lif1.init_leaky()
        cur = self.deco1(x)
        pool = self.maxpool1(cur)
        spk, mem = self.lif1(pool, mem)
        dspk = self.dropout1(spk)
        flat = self.flatten1(dspk)
        r = self.fc1(flat)
               
        spk_out.append(spk.cpu())
        r_out.append(r.cpu())
        mem_out.append(mem.cpu())
        
        # stage 2
        
        mem = self.lif2.init_leaky()
        x = spk.detach_()
        cur = self.deco2(x)
        pool = self.maxpool2(cur)
        spk, mem = self.lif2(pool, mem)
        dspk = self.dropout2(spk)
        flat = self.flatten2(dspk)
        r = self.fc2(flat)
        
        spk_out.append(spk.cpu())
        r_out.append(r.cpu())
        mem_out.append(mem.cpu())
        
        # stage 3
        
        mem = self.lif3.init_leaky()
        x = spk.detach_()
        cur = self.deco3(x)
        pool = self.maxpool3(cur)
        spk, mem = self.lif3(pool, mem)
        dspk = self.dropout3(spk)
        flat = self.flatten3(dspk)
        r = self.fc3(flat)
        
        spk_out.append(spk.cpu())
        r_out.append(r.cpu())
        mem_out.append(mem.cpu())

        return spk_out, r_out, mem_out
   
def set_datalog(net, data_type, epoch, iter, label):
    global store_path, epoch_path, test
    l = label.numpy()
    epoch_path = store_path + str(epoch) + str(data_type) + '/' + str(l) + '/' + str(iter) + '/'
    os.makedirs(epoch_path, exist_ok=True)
    
    if test:
        with open(epoch_path + 'snnAction_training_cur3.csv', mode='wt',encoding='UTF-8') as outfile:
            with open(epoch_path + 'snnAction_training_spk3.csv', mode='wt',encoding='UTF-8') as outfile:
                with open(epoch_path + 'snnAction_training_mem3.csv', mode='wt',encoding='UTF-8') as outfile:
                    with open(epoch_path + 'snnAction_training_cur4.csv', mode='wt',encoding='UTF-8') as outfile:
                        with open(epoch_path + 'snnAction_training_spk4.csv', mode='wt',encoding='UTF-8') as outfile:
                            with open(epoch_path + 'snnAction_training_mem4.csv', mode='wt',encoding='UTF-8') as outfile:
                                utils.reset(net)
    else:
        with open(epoch_path + 'snnAction_training_cur4.csv', mode='wt',encoding='UTF-8') as outfile:
            with open(epoch_path + 'snnAction_training_spk4.csv', mode='wt',encoding='UTF-8') as outfile:
                with open(epoch_path + 'snnAction_training_mem4.csv', mode='wt',encoding='UTF-8') as outfile:
                    utils.reset(net)
    
def show_train_history(value1,value2,title,value_type, legend=[]):
    plt.figure(figsize=(20, 10))
    plt.plot(value1)
    plt.plot(value2)
    plt.xticks([i for i in range(0, len(value1))])
    plt.title(title)
    plt.ylabel(value_type)
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper left')
    plt.savefig(store_path + 'snnAction_' + title + '_' + str(pkl_version) +'.png')
    
if __name__ == '__main__':
    num_epochs = 60
    batch_size = 1
    num_workers = 8
    pkl_updates = True
    training = True
    last_epoch = 0
    best_los = 10000
    best_acc = 0
    
    learning_rate = 1e-3
    num_classes = 10
    file_len = 0
    trigger_times = 0
    patience = 10
    last_validLos = 100
    pkl_version = 1 
    epoch_path = ''
    test = False
    
    class_dict = {'0':'arm_crossing', '1':'get-up', '2':'jumping', '3':'kicking', '4':'picking_up', '5':'sit_down', '6':'throwing', '7':'turning_around', '8':'walking', '9':'waving'}
    classlabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    datalog = "snnAction_DECOLLE Training DataLog\n"
    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []

    main_path = "./snn_event/event_dataset/Action_Recognition_Dataset/"
    train_img_path = main_path + "train/"
    valid_img_path = main_path + "valid/"
    store_path = "./snn_event/snnDECOLLE_test/" + str(pkl_version) + '/'

    freeze_support()
    mp.set_start_method('spawn', force=True)
    
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name())
    print(torch.cuda.get_device_capability())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    
    while(1):
        if not os.path.exists(store_path): 
            if pkl_updates:
                os.makedirs(store_path, exist_ok=True)
                break
            else:
                pkl_version -= 1
                store_path = "./snn_event/snnDECOLLE_test/" + str(pkl_version) + '/'
                break
        pkl_version += 1
        store_path = "./snn_event/snnDECOLLE_test/" + str(pkl_version) + '/'
    print("pkl_version: %d" % pkl_version)
    
    datalog += 'use:'+str(device)+'\n'
 
#################################################################################################################################################################### 
    
    transform = transforms.Compose([
            #transforms.Resize((32, 64)),
            #transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
            ])
    
    train_set = custom_Dataset(root_dir = train_img_path, classname = classlabels, transform = transform)
    valid_set = custom_Dataset(root_dir = valid_img_path, classname = classlabels, transform = transform)
    train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True, num_workers=num_workers)
    valid_loader = DataLoader(dataset = valid_set, batch_size = batch_size, shuffle = True, num_workers=num_workers)
    
    test_img_path = main_path + "test/"
    Testset = custom_Dataset(root_dir = test_img_path, classname = classlabels, transform = transform)
    test_loader = DataLoader(dataset = Testset, batch_size = batch_size, shuffle = True, num_workers=num_workers)
    
    print("Loader set")
    datalog += "Loader set"+'\n'

    torch.cuda.empty_cache()
    beta = 0.3
    spike_grad = surrogate.fast_sigmoid()
    
    cnn = snnmodel(7, beta, spike_grad)
    
    print(cnn)
    print("cnn set")
    datalog += str(cnn)+'\n'
    datalog += "cnn set"+'\n'

    criterion = [nn.SmoothL1Loss(), nn.SmoothL1Loss(), nn.SmoothL1Loss()]
    print("criterion set: [nn.SmoothL1Loss(), nn.SmoothL1Loss(), nn.SmoothL1Loss()]")
    datalog += "criterion set: [nn.SmoothL1Loss(), nn.SmoothL1Loss(), nn.SmoothL1Loss()]"+'\n'
    optimizer = torch.optim.Adamax(cnn.parameters(), lr = learning_rate, betas=(0.9, 0.999))
    print("optimizer set: Adamax")
    datalog += "optimizer set: Adamax"+'\n'
    datalog += "learning_rate: "+ str(learning_rate) +'\n'
    
    cnn = cnn.to(device)
    
    best_los_epoch = 0
    best_acc_epoch = 0
    
    if training:
        test = False
        last_epoch = 0
        best_los = 10000
        best_acc = 0
        with open(store_path + 'snnAction_'+ str(pkl_version) +'.txt', mode='wt',encoding='UTF-8') as outfile:
            print(datalog,file=outfile)
        with open(store_path + 'snnAction_history_'+ str(pkl_version) +'.csv', mode='wt',encoding='UTF-8') as outfile:
            print("time,trainLos_0,trainLos_1,trainLos_2,trainAcc,validLos_0,validLos_1,validLos_2,validAcc,validpre,validrec,validf1c",file=outfile)
        with open(store_path + 'snnAction_test_'+ str(pkl_version) +'.csv', mode='wt',encoding='UTF-8') as outfile:
            print("testLos_0,testLos_1,testLos_2,testAcc,testPre,testRec,testF1c",file=outfile)
        
        print("start training:")
        datalog += "start training:"+'\n'  
        with torch.cuda.device(0):
            
            for epoch in range(last_epoch, num_epochs):

                cnn.train()
                utils.reset(cnn)
                test = False
                loss_total = 0
                acc = 0
                total = 0
                loss_tv = torch.tensor(0.).to(device) 
                st=time.time()
                for (data, label, label_one_hot) in tqdm.tqdm(iter(train_loader), desc='Train Epoch {}'.format(epoch+1)):
                    data_time_step = data.shape[1]
                    for time_step in range(data_time_step):
                        spk_out, read_out, mem_out = cnn(data[:,time_step,:,:,:].to(device))
                        loss_train = []
                        for n in range(3):
                            loss_train.append(criterion[n](read_out[n], label_one_hot[:,time_step,:]))
                            
                        loss_total += (torch.tensor(loss_train)).detach().cpu().numpy() 
                        loss_tv += sum(loss_train) 

                        _, predicted = torch.max(read_out[2], 1)
                        predicted = predicted.detach().cpu().numpy()
                        acc += (predicted == label.cpu().numpy()).sum()
                        total += 1
                        
                        loss_tv.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        loss_tv = torch.tensor(0.).to(device) 
                    
                t = time.time()-st
                trainAcc = acc/total
                trainLos = loss_total/len(train_loader)
                torch.cuda.empty_cache()

                cnn.eval()
                utils.reset(cnn)
                test = False
                true = []
                pred = []
                loss_total = 0
                acc = 0
                total = 0
                with torch.no_grad():
                    for (data, label, label_one_hot) in tqdm.tqdm(iter(valid_loader), desc='Valid Epoch {}'.format(epoch+1)):
                        label = label.detach().cpu().numpy()

                        data_time_step = data.shape[1]
                        for time_step in range(data_time_step):
                            spk_out, read_out, mem_out = cnn(data[:,time_step,:,:,:].to(device))
                            
                            loss_valid = []
                            for n in range(3):
                                loss_valid.append(criterion[n](read_out[n], label_one_hot[:,time_step,:]))
                                
                            loss_total += (torch.tensor(loss_valid)).detach().cpu().numpy() 

                            _, predicted = torch.max(read_out[2], 1)
                            predicted = predicted.detach().cpu().numpy()
                            true.append(label)
                            pred.append(predicted) 
                            acc += (predicted == label).sum()
                            total += 1
                    
                validAcc = acc/total
                validLos = loss_total/len(valid_loader)
                torch.cuda.empty_cache()
                
                cnn.eval()
                utils.reset(cnn)
                test = False
                loss_total = 0
                acc = 0
                total = 0
                ttrue = []
                tpred = []
                with torch.no_grad():
                    for (data, label, label_one_hot) in tqdm.tqdm(iter(test_loader), desc='Test Epoch {}'.format(epoch+1)):
                        label = label.detach().cpu().numpy()
                        
                        data_time_step = data.shape[1]
                        for time_step in range(data_time_step):
                            spk_out, read_out, mem_out = cnn(data[:,time_step,:,:,:].to(device))
                            
                            loss_test = []
                            for n in range(3):
                                loss_test.append(criterion[n](read_out[n], label_one_hot[:,time_step,:]))
                                
                            loss_total += (torch.tensor(loss_test)).detach().cpu().numpy() 

                            _, predicted = torch.max(read_out[2], 1)
                            predicted = predicted.detach().cpu().numpy()
                            ttrue.append(label)
                            tpred.append(predicted) 
                            acc += (predicted == label).sum()
                            total += 1     
                                
                testAcc = acc/total
                testLos = loss_total/len(test_loader)
                torch.cuda.empty_cache()
                
                if validLos[2] > last_validLos:
                    trigger_times += 1
                elif trigger_times > 0:
                    trigger_times = 0
                
                print("=================================================================================================================================================")
                print('epoch [%3d/%d]'%(epoch+1, num_epochs))
                print('time used:%.4f'%t)
                print('train_Loss: ', trainLos)
                print('train_Acc: ', trainAcc)
                print('valid_Loss: ', validLos)
                print('valid_Acc: ', validAcc)
                print("testLos: ", testLos)
                print("testAcc: ", testAcc)
                print('trigger_times: ', trigger_times)
                print("=================================================================================================================================================")
                
                with open(store_path + 'snnAction_'+ str(pkl_version) +'.txt', mode='at',encoding='UTF-8') as outfile:
                    print("=================================================================================================================================================",file=outfile)
                    print('epoch [%3d/%d]'%(epoch+1, num_epochs),file=outfile)
                    print('time used:%.4f'%t,file=outfile)
                    print('train_Loss: ', trainLos,file=outfile)
                    print('train_Acc: ', trainAcc,file=outfile)
                    print('valid_Loss: ', validLos,file=outfile)
                    print('valid_Acc: ', validAcc,file=outfile)
                    print("testLos: ", testLos,file=outfile)
                    print("testAcc: ", testAcc,file=outfile)
                    print("=================================================================================================================================================",file=outfile)
                
                train_loss_history.append(trainLos)
                train_acc_history.append(trainAcc)
                valid_loss_history.append(validLos)
                valid_acc_history.append(validAcc)
                
                with open(store_path + 'snnAction_history_'+ str(pkl_version) +'.csv', mode='at',encoding='UTF-8') as outfile:
                    print("%.4f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f"%(t,trainLos[0],trainLos[1],trainLos[2],trainAcc,validLos[0],validLos[1],validLos[2],validAcc,precision_score(true, pred, average='macro',zero_division=0),recall_score(true, pred, average='macro',zero_division=0),f1_score(true, pred, average='macro',zero_division=0)), file=outfile)
                    
                with open(store_path + 'snnAction_pre_micro_history_'+ str(pkl_version) +'.csv', mode='at',encoding='UTF-8') as outfile:
                    sc = precision_score(true, pred, average=None,zero_division=0)
                    scstr = ""
                    for i in sc:
                        scstr += str(i) + ","
                    print(scstr, file=outfile)
                    
                with open(store_path + 'snnAction_rec_micro_history_'+ str(pkl_version) +'.csv', mode='at',encoding='UTF-8') as outfile:
                    sc = recall_score(true, pred, average=None,zero_division=0)
                    scstr = ""
                    for i in sc:
                        scstr += str(i) + ","
                    print(scstr, file=outfile)
                    
                with open(store_path + 'snnAction_f1c_micro_history_'+ str(pkl_version) +'.csv', mode='at',encoding='UTF-8') as outfile:
                    sc = f1_score(true, pred, average=None,zero_division=0)
                    scstr = ""
                    for i in sc:
                        scstr += str(i) + ","
                    print(scstr, file=outfile)
                    
                    
                    
                with open(store_path + 'snnAction_test_'+ str(pkl_version) +'.csv', mode='at',encoding='UTF-8') as outfile:
                    print("%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f"%(testLos[0],testLos[1],testLos[2],testAcc,precision_score(ttrue, tpred, average='macro',zero_division=0),recall_score(ttrue, tpred, average='macro',zero_division=0),f1_score(ttrue, tpred, average='macro',zero_division=0)),file=outfile)
                    
                with open(store_path + 'snnAction_test_pre_micro_history_'+ str(pkl_version) +'.csv', mode='at',encoding='UTF-8') as outfile:
                    sc = precision_score(ttrue, tpred, average=None,zero_division=0)
                    scstr = ""
                    for i in sc:
                        scstr += str(i) + ","
                    print(scstr, file=outfile)
                    
                with open(store_path + 'snnAction_test_rec_micro_history_'+ str(pkl_version) +'.csv', mode='at',encoding='UTF-8') as outfile:
                    sc = recall_score(ttrue, tpred, average=None,zero_division=0)
                    scstr = ""
                    for i in sc:
                        scstr += str(i) + ","
                    print(scstr, file=outfile)
                    
                with open(store_path + 'snnAction_test_f1c_micro_history_'+ str(pkl_version) +'.csv', mode='at',encoding='UTF-8') as outfile:
                    sc = f1_score(ttrue, tpred, average=None,zero_division=0)
                    scstr = ""
                    for i in sc:
                        scstr += str(i) + ","
                    print(scstr, file=outfile)
                    
                if best_los > validLos[2]:
                    best_los = validLos[2]
                    best_los_epoch = epoch
                    
                if  best_acc < validAcc:
                    best_acc = validAcc
                    best_acc_epoch = epoch
                    '''
                    if best_acc > 0.7:
                        test = True
                        cnn.eval()
                        with torch.cuda.device(0):
                            for i, (data, label) in enumerate(iter(valid_loader)):
                                spk_out, read_out, mem_out = cnn(data[time_step,:,:,:].to(device))
                        
                        cnn.eval()
                        with torch.cuda.device(0):
                            for i, (data, label) in enumerate(iter(test_loader)):
                                spk_out, read_out, mem_out = cnn(data[time_step,:,:,:].to(device))
                    '''
                    
                if trigger_times >= patience:
                    print("Early Stopping")
                    datalog += "Early Stopping"+'\n'
                    break
                    
                last_validLos = validLos[2]
                with open(store_path + 'snnAction_best_'+ str(pkl_version) +'.txt', mode='wt',encoding='UTF-8') as outfile:
                    print("%d best_los:%.10f\n%d best_acc:%.10f"%(best_los_epoch+1,best_los,best_acc_epoch+1,best_acc),file=outfile)
                    

        print("=================================================================================================================================================") 
        with open(store_path + 'snnAction_'+ str(pkl_version) +'.txt', mode='at',encoding='UTF-8') as outfile:
            print("=================================================================================================================================================",file=outfile)
  
    
        show_train_history(train_loss_history,valid_loss_history,'loss_History','loss',['train_0', 'train_1', 'train_2', 'validation_0', 'validation_1', 'validation_2'])
        show_train_history(train_acc_history,valid_acc_history,'Acc_History','acc',['train', 'validation'])

        print('training_log_output')
        
    print('process_done')
