from pcc_dataset import TextData, get_dataset, _init_rawdata, Embedder
from networks import RecogNet_gru, RecogNet_lstm


import os, pickle, time

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader



#A small tool for evaluating processing time of a function
def FunctionTimer(f,args):    
   
    if not callable(f):
        raise TypeError("It is not callable!!")
    t1=time.time()
    result=f(*args)
    t2=time.time()
    print("{n} takes : {t:.10f} s".format(n=f.__name__ ,t=t2-t1))
   
    return result
#F1-score
#https://blog.csdn.net/matrix_space/article/details/50384518

_init_rawdata(multiworking=True)








Train_data = TextData( get_dataset(os.path.join('.','Dataset','TrainingSet.csv')) )
Val_data = TextData( get_dataset(os.path.join('.','Dataset','ValidationSet.csv')))
Test_data = TextData( get_dataset(os.path.join('.','Dataset','TestingSet.csv')))





model = RecogNet_gru(Embedder.get_dim())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()


max_epoch=20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  #https://zhuanlan.zhihu.com/p/31936740

embedding_layer = nn.Embedding(Embedder.get_vocabulary_size(),Embedder.get_dim())
embedding_layer.weight = torch.nn.Parameter(Embedder.vectors)


for epoch in range(max_epoch):
    #continue

    #Training Procedure:
    loss_total=0
    model.train(True)#https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch

    dataloader = DataLoader( dataset=Train_data,
                        batch_size=32,
                        shuffle=True,
                        collate_fn=Train_data.collate_fn,
                        num_workers=4)
    
    for i, ( batch_data, batch_target ) in enumerate(dataloader):

        x = embedding_layer( batch_data ) 
        training_data = x.to(device)
    
        labels = batch_target.to(device)


        prediction = model(training_data)
        batch_loss = criterion(prediction, labels)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()


        loss_total += batch_loss.item()#https://blog.csdn.net/moshiyaofei/article/details/90519430
    print("epoch :", epoch," ; loss_total :",loss_total)   

    #============================================================================================================================================
    #Validation: 
    validation_loss_total=0
    model.train(False)
    dataloader = DataLoader( dataset=Val_data ,
                        batch_size=32,
                        shuffle=False,
                        collate_fn=Val_data.collate_fn,
                        num_workers=4)


    for i, ( batch_data, batch_target ) in enumerate( dataloader ):

        x=embedding_layer( batch_data ) 
        Validation_data= x.to(device)

        Validation_labels=batch_target.to(device)

        prediction=model(Validation_data)

        batch_loss=criterion(prediction, Validation_labels)

        validation_loss_total+=batch_loss.item()

    print("epoch :", epoch," ; validation_loss_total :",validation_loss_total)

#============================================================================================================================================
    #Saving Current Model:

    if not os.path.exists('model'):
        os.makedirs('model')
    torch.save(model.state_dict(), os.path.join('.','model','model.pkl.'))
    
    












































        

