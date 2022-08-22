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
                self.class_list.append(str(classname[classcount]))
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
                t = 1
                pre = self.transform(Image.open(im).convert('L'))
            else:
                temp = []
                now = self.transform(Image.open(im).convert('L'))
                temp = torch.cat((pre,now),0)
                pre = now
                image.append(temp)
        image = torch.stack(image)
        label = torch.tensor(int(self.class_list[index])) 
            
        return (image, label)

class snnmodel(nn.Module):
    def __init__(self, kernel_size, beta, spike_grad):
        super(snnmodel, self).__init__()
        #32*64
        self.conv1 = nn.Conv2d(2, 4, kernel_size) #26*58
        self.maxpool1 = nn.MaxPool2d(2) #13*29
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(4, 8, kernel_size) #7*23
        self.maxpool2 = nn.MaxPool2d(2) #3*11
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(82*60*8, 512)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(512, 10)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def forward(self, x):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()
        self.mem4 = self.lif4.init_leaky()

        self.cur1 = self.conv1(x)
        self.pool1 = self.maxpool1(self.cur1)
        self.spk1, self.mem1 = self.lif1(self.pool1, self.mem1)
        self.cur2 = self.conv2(self.spk1)
        self.pool2 = self.maxpool2(self.cur2)
        self.spk2, self.mem2 = self.lif2(self.pool2, self.mem2)
        self.spk_flat = self.flatten(self.spk2)
        self.cur3 = self.fc1(self.spk_flat)
        self.spk3, self.mem3 = self.lif3(self.cur3, self.mem3)
        self.cur4 = self.fc2(self.spk3)
        self.spk4, self.mem4 = self.lif3(self.cur4, self.mem4)
        
        global test
        if test:
            self.output_data()

        return self.spk4
    
    def output_data(self):
        global epoch_path
        
        temp = self.cur1.cpu().detach().numpy()
        count = 0
        step = 0
        os.makedirs(epoch_path + "cur1/", exist_ok=True)
        for index in temp:
            os.makedirs(epoch_path + "cur1/" + str(step) + "/" , exist_ok=True)
            for im in index:
                im = Image.fromarray(np.uint8(im*255))
                im.save(epoch_path + "cur1/" + str(step) + "/cur1_" + str(count) +".png")
                count += 1
            step += 1
        
        temp = self.pool1.cpu().detach().numpy()
        os.makedirs(epoch_path + "pool1/", exist_ok=True)
        count = 0
        step = 0
        for index in temp:
            os.makedirs(epoch_path + "pool1/" + str(step) + "/" , exist_ok=True)
            for im in index:
                im = Image.fromarray(np.uint8(im*255))
                im.save(epoch_path + "pool1/" + str(step) + "/pool1_" + str(count) +".png")
                count += 1
            step += 1
            
        temp = self.spk1.cpu().detach().numpy()
        os.makedirs(epoch_path + "spk1/", exist_ok=True)
        count = 0
        step = 0
        for index in temp:
            os.makedirs(epoch_path + "spk1/" + str(step) + "/" , exist_ok=True)
            for im in index:
                im = Image.fromarray(np.uint8(im*255))
                im.save(epoch_path + "spk1/" + str(step) + "/spk1_" + str(count) +".png")
                count += 1
            step += 1
            
        temp = self.mem1.cpu().detach().numpy()
        os.makedirs(epoch_path + "mem1/", exist_ok=True)
        count = 0
        step = 0
        for index in temp:
            os.makedirs(epoch_path + "mem1/" + str(step) + "/" , exist_ok=True)
            for im in index:
                im = Image.fromarray(np.uint8(im*255))
                im.save(epoch_path + "mem1/" + str(step) + "/mem1_" + str(count) +".png")
                count += 1
            step += 1
            
        temp = self.cur2.cpu().detach().numpy()
        os.makedirs(epoch_path + "cur2/", exist_ok=True)
        count = 0
        step = 0
        for index in temp:
            os.makedirs(epoch_path + "cur2/" + str(step) + "/" , exist_ok=True)
            for im in index:
                im = Image.fromarray(np.uint8(im*255))
                im.save(epoch_path + "cur2/" + str(step) + "/cur2_" + str(count) +".png")
                count += 1
            step += 1
        
        temp = self.pool2.cpu().detach().numpy()
        os.makedirs(epoch_path + "pool2/", exist_ok=True)
        count = 0
        step = 0
        for index in temp:
            os.makedirs(epoch_path + "pool2/" + str(step) + "/" , exist_ok=True)
            for im in index:
                im = Image.fromarray(np.uint8(im*255))
                im.save(epoch_path + "pool2/" + str(step) + "/pool2_" + str(count) +".png")
                count += 1
            step += 1
            
        temp = self.spk2.cpu().detach().numpy()
        os.makedirs(epoch_path + "spk2/", exist_ok=True)
        count = 0
        step = 0
        for index in temp:
            os.makedirs(epoch_path + "spk2/" + str(step) + "/" , exist_ok=True)
            for im in index:
                im = Image.fromarray(np.uint8(im*255))
                im.save(epoch_path + "spk2/" + str(step) + "/spk2_" + str(count) +".png")
                count += 1
            step += 1
            
        temp = self.mem2.cpu().detach().numpy()
        os.makedirs(epoch_path + "mem2/", exist_ok=True)
        count = 0
        step = 0
        for index in temp:
            os.makedirs(epoch_path + "mem2/" + str(step) + "/" , exist_ok=True)
            for im in index:
                im = Image.fromarray(np.uint8(im*255))
                im.save(epoch_path + "mem2/" + str(step) + "/mem2_" + str(count) +".png")
                count += 1
            step += 1
            
        del count
        
        with open(epoch_path + 'snnAction_training_spk_flat.csv', mode='wt',encoding='UTF-8') as outfile:
            temp = self.spk_flat.cpu().detach().numpy()
            for time_step in temp:
                s = ""
                for index in time_step:
                    s += str(index) + ","
                print(s, file=outfile)
        
        with open(epoch_path + 'snnAction_training_cur3.csv', mode='wt',encoding='UTF-8') as outfile:
            temp = self.cur3.cpu().detach().numpy()
            for time_step in temp:
                s = ""
                for index in time_step:
                    s += str(index) + ","
                print(s, file=outfile)
                
        with open(epoch_path + 'snnAction_training_spk3.csv', mode='wt',encoding='UTF-8') as outfile:
            temp = self.spk3.cpu().detach().numpy()
            for time_step in temp:
                s = ""
                for index in time_step:
                    s += str(index) + ","
                print(s, file=outfile)
                
        with open(epoch_path + 'snnAction_training_mem3.csv', mode='wt',encoding='UTF-8') as outfile:
            temp = self.mem3.cpu().detach().numpy()
            for time_step in temp:
                s = ""
                for index in time_step:
                    s += str(index) + ","
                print(s, file=outfile)
                
        with open(epoch_path + 'snnAction_training_cur4.csv', mode='wt',encoding='UTF-8') as outfile:
            temp = self.cur4.cpu().detach().numpy()
            for time_step in temp:
                s = ""
                for index in time_step:
                    s += str(index) + ","
                print(s, file=outfile)
                
        with open(epoch_path + 'snnAction_training_spk4.csv', mode='wt',encoding='UTF-8') as outfile:
            temp = self.spk4.cpu().detach().numpy()
            for time_step in temp:
                s = ""
                for index in time_step:
                    s += str(index) + ","
                print(s, file=outfile)
                
        with open(epoch_path + 'snnAction_training_mem4.csv', mode='wt',encoding='UTF-8') as outfile:
            temp = self.mem4.cpu().detach().numpy()
            for time_step in temp:
                s = ""
                for index in time_step:
                    s += str(index) + ","
                print(s, file=outfile)
        
        del temp


   
def forward_pass(net, data, device, data_type, epoch, iter, label):
    global store_path, epoch_path, test
    if test:
        l = label.numpy()
        epoch_path = store_path + str(epoch) + '/' + str(data_type) + '/' + str(l) + '/' + str(iter) + '/'
        os.makedirs(epoch_path, exist_ok=True)
        
        with open(epoch_path + 'snnAction_training_cur3.csv', mode='wt',encoding='UTF-8') as outfile:
            with open(epoch_path + 'snnAction_training_spk3.csv', mode='wt',encoding='UTF-8') as outfile:
                with open(epoch_path + 'snnAction_training_mem3.csv', mode='wt',encoding='UTF-8') as outfile:
                    with open(epoch_path + 'snnAction_training_cur4.csv', mode='wt',encoding='UTF-8') as outfile:
                        with open(epoch_path + 'snnAction_training_spk4.csv', mode='wt',encoding='UTF-8') as outfile:
                            with open(epoch_path + 'snnAction_training_mem4.csv', mode='wt',encoding='UTF-8') as outfile:
                                utils.reset(net)
                                spk_rec = []
    utils.reset(net)
    spk_rec = []
    for step in range(data.size(0)):
        temp_data = data[step].to(device)
        spk_out = net(temp_data)
        spk_rec.append(spk_out.cpu())
        
    return torch.stack(spk_rec)
    
def show_train_history(value1,value2,title,value_type):
    plt.figure(figsize=(5, 5))
    plt.plot(value1)
    plt.plot(value2)
    plt.xticks([i for i in range(0, len(value1))])
    plt.title(title)
    plt.ylabel(value_type)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(store_path + 'snnAction_' + title + '_' + str(pkl_version) +'.png')
    
if __name__ == '__main__':
    num_epochs = 100
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
    
    class_dict = {'0':'arm_crossing', '1':'get-up', '2':'jumping', '3':'kicking', '4':'picking_up', '5':'sit_down', '6':'throwing', '7':'turning_around', '8':'walking', '9':'waving'}
    classlabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    datalog = "snnAction_model Training DataLog\n"
    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []

    main_path = "./snn_event/event_dataset/Action_Recognition_Dataset/"
    train_img_path = main_path + "train/"
    valid_img_path = main_path + "valid/"
    store_path = "./snn_event/snnpkl_test/" + str(pkl_version) + '/'

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
                store_path = "./snn_event/snnpkl_test/" + str(pkl_version) + '/'
                break
        pkl_version += 1
        store_path = "./snn_event/snnpkl_test/" + str(pkl_version) + '/'
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
    #spike_grad = surrogate.LeakySpikeOperator.apply
    '''
    cnn = nn.Sequential(
            #346*260
            nn.Conv2d(2, 4, 3), #344*258
            nn.MaxPool2d(2), #172*129
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(4, 8, 3), #170*127
            nn.MaxPool2d(2), #85*63
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(85*63*8, 512),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(512, 10),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
            )
    
    cnn = nn.Sequential(
            #346*260
            nn.Conv2d(2, 4, 5), #342*256
            nn.MaxPool2d(2), #171*128
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(4, 8, 5), #167*124
            nn.MaxPool2d(2), #83*62
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(83*62*8, 512),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(512, 10),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
            )
    
    cnn = nn.Sequential(
            #346*260
            nn.Conv2d(2, 4, 7), #340*254
            nn.AvgPool2d(2), #170*127
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(4, 8, 7), #164*121
            nn.AvgPool2d(2), #82*60
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(82*60*8, 512),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(512, 10),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
            )
    
    cnn = nn.Sequential(
            #346*260
            nn.Conv2d(2, 4, 9), #338*252
            nn.MaxPool2d(2), #169*126
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(4, 8, 9), #161*118
            nn.MaxPool2d(2), #80*59
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(80*59*8, 512),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(512, 10),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
            )
    '''
    
    cnn = snnmodel(7, beta, spike_grad)
    
    print(cnn)
    print("cnn set")
    datalog += str(cnn)+'\n'
    datalog += "cnn set"+'\n'

    criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    print("criterion set: SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)")
    datalog += "criterion set: SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)"+'\n'
    optimizer = torch.optim.AdamW(cnn.parameters(), lr = learning_rate, betas=(0.9, 0.999))
    print("optimizer set: AdamW")
    datalog += "optimizer set: AdamW"+'\n'
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
            print("time,trainLos,trainAcc,validLos,validAcc,validpre,validrec,validf1c",file=outfile)
        with open(store_path + 'snnAction_test_'+ str(pkl_version) +'.csv', mode='wt',encoding='UTF-8') as outfile:
            print("testLos,testAcc,testPre,testRec,testF1c",file=outfile)
        
        print("start training:")
        datalog += "start training:"+'\n'  
        with torch.cuda.device(0):
            
            for epoch in range(last_epoch, num_epochs):

                cnn.train()
                test = False
                loss_total = 0
                acc = 0
                total = 0
                st=time.time()
                for i, (data, label) in enumerate(iter(train_loader)):
                    spk_rec = forward_pass(cnn, data, device, "train", epoch, i, label)
                    batch, channel, target = spk_rec.size()
                    spk_rec = spk_rec.view(channel, batch, target)
                    loss_train = criterion(spk_rec, label)
                    
                    loss_total += loss_train.item()
                    _, idx = spk_rec.detach().sum(dim=0).max(1)
                    acc += (label == idx).detach().numpy().sum()
                    total += label.size(0)
                    
                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()
                    
                t = time.time()-st
                trainAcc = acc/total
                trainLos = loss_total/len(train_loader)

                cnn.eval()
                test = False
                true = []
                pred = []
                loss_total = 0
                acc = 0
                total = 0
                with torch.cuda.device(0):
                    for i, (data, label) in enumerate(iter(valid_loader)):
                        true.extend(label)

                        spk_rec = forward_pass(cnn, data, device, "valid", epoch, i, label)
                        batch, channel, target = spk_rec.size()
                        spk_rec = spk_rec.view(channel, batch, target)
                        loss_valid = criterion(spk_rec, label)
                        
                        loss_total += loss_valid.item()
                        _, idx = spk_rec.detach().sum(dim=0).max(1)
                
                        pred.extend(idx.detach().numpy())
                        acc += (label == idx).detach().numpy().sum()
                        total += label.size(0)
                    
                validAcc = acc/total
                validLos = loss_total/len(valid_loader)
                
                cnn.eval()
                test = False
                loss_total = 0
                acc = 0
                total = 0
                ttrue = []
                tpred = []
                with torch.cuda.device(0):
                    for i, (data, label) in enumerate(iter(test_loader)):
                        ttrue.extend(label)

                        spk_rec = forward_pass(cnn, data, device, "test", epoch, i, label)
                        batch, channel, target = spk_rec.size()
                        spk_rec = spk_rec.view(channel, batch, target)
                        loss_test = criterion(spk_rec, label)
                        
                        loss_total += loss_test.item()
                        _, idx = spk_rec.detach().sum(dim=0).max(1)
                
                        tpred.extend(idx.detach().numpy())
                        acc += (label == idx).detach().numpy().sum()
                        total += label.size(0)
                                
                                
                testAcc = acc/total
                testLos = loss_total/len(test_loader)
                
                if validLos > last_validLos:
                    trigger_times += 1
                elif trigger_times > 0:
                    trigger_times = 0
                
                print("=================================================================================================================================================")
                print('epoch [%3d/%d], time used:%.4f  train_Loss:%.10f  train_Acc:%.10f  Val_Loss:%.10f  Val_Acc:%.10f  trigger_times:%d'%(epoch+1, num_epochs, t, trainLos, trainAcc, validLos, validAcc, trigger_times))
                with open(store_path + 'snnAction_'+ str(pkl_version) +'.txt', mode='at',encoding='UTF-8') as outfile:
                    print("================================================================================================================================================="+'\n'+'epoch [%3d/%d], time used:%.4f  train_Loss:%.10f  train_Acc:%.10f  Val_Loss:%.10f  Val_Acc:%.10f'%(epoch+1, num_epochs, t, trainLos, trainAcc, validLos, validAcc),file=outfile)
                
                train_loss_history.append(trainLos)
                train_acc_history.append(trainAcc)
                valid_loss_history.append(validLos)
                valid_acc_history.append(validAcc)
                
                
                
                with open(store_path + 'snnAction_history_'+ str(pkl_version) +'.csv', mode='at',encoding='UTF-8') as outfile:
                    print("%.4f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f"%(t,trainLos,trainAcc,validLos,validAcc,precision_score(true, pred, average='macro',zero_division=0),recall_score(true, pred, average='macro',zero_division=0),f1_score(true, pred, average='macro',zero_division=0)), file=outfile)
                    
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
                    print("%.10f,%.10f,%.10f,%.10f,%.10f"%(testLos,testAcc,precision_score(ttrue, tpred, average='macro',zero_division=0),recall_score(ttrue, tpred, average='macro',zero_division=0),f1_score(ttrue, tpred, average='macro',zero_division=0)),file=outfile)
                    
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
                    
                if best_los > validLos:
                    best_los = validLos
                    best_los_epoch = epoch
                    
                if  best_acc < validAcc:
                    best_acc = validAcc
                    best_acc_epoch = epoch
                    if best_acc > 0.65:
                        cnn.eval()
                        test = True
                        with torch.cuda.device(0):
                            for i, (data, label) in enumerate(iter(valid_loader)):
                                spk_rec = forward_pass(cnn, data, device, "valid", epoch, i, label)
                        
                        cnn.eval()
                        test = True
                        with torch.cuda.device(0):
                            for i, (data, label) in enumerate(iter(test_loader)):
                                spk_rec = forward_pass(cnn, data, device, "test", epoch, i, label)
                    
                if trigger_times >= patience:
                    print("Early Stopping")
                    datalog += "Early Stopping"+'\n'
                    break
                    
                last_validLos = validLos
                with open(store_path + 'snnAction_best_'+ str(pkl_version) +'.txt', mode='wt',encoding='UTF-8') as outfile:
                    print("%d best_los:%.10f\n%d best_acc:%.10f"%(best_los_epoch+1,best_los,best_acc_epoch+1,best_acc),file=outfile)
                    

        print("=================================================================================================================================================") 
        with open(store_path + 'snnAction_'+ str(pkl_version) +'.txt', mode='at',encoding='UTF-8') as outfile:
            print("=================================================================================================================================================",file=outfile)
  
    
        show_train_history(train_loss_history,valid_loss_history,'loss_History','loss')
        show_train_history(train_acc_history,valid_acc_history,'Acc_History','acc')

        print('training_log_output')
        
    print('process_done')
