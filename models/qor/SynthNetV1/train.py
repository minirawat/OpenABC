import os

import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from model import *
from utils import *
from netlistDataset import *

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split
import os.path as osp
import pickle
import sys
#import multiprocessing
#multiprocessing.set_start_method('fork')

datasetDict =  {
    'set1' : ["train_data_set1.csv","test_data_set1.csv"],
    'set2' : ["train_data_set2.csv","test_data_set2.csv"],
    'set3' : ["train_data_mixmatch_v1.csv","test_data_mixmatch_v1.csv"]
}

DUMP_DIR = None
criterion = torch.nn.MSELoss()

def mape(y_pred, y_true, epsilon=1e-8):
  y_true += epsilon
  per_error = torch.abs((y_true - y_pred) / y_true)
  return torch.mean(per_error, dim=0) * 100

def plotChart(x,y,xlabel,ylabel,leg_label,title):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x,y, label=leg_label)
    leg = plt.legend(loc='best', ncol=2, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel(xlabel, weight='bold')
    plt.ylabel(ylabel, weight='bold')
    plt.title(title,weight='bold')
    plt.savefig(osp.join(DUMP_DIR,title+'.png'), format='png', bbox_inches='tight')

def train(model,device,dataloader,optimizer, maxTargetDict, batch_size):
    epochLoss = AverageMeter()
    total_mape = 0
    #total_mape2 = 0
    total_mape3 = 0
    model.train()
    for _, batch in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
        batch = batch.to(device)
        desName = batch.desName[0][0]
        #print(desName)
        lbl = batch.target.reshape(-1, 1)
        optimizer.zero_grad()
        pred = model(batch)
        #print(pred * maxTargetDict[desName][1], lbl * maxTargetDict[desName][1])
        #print(type(pred))
        #print(type(lbl))
        err_mape = mape(pred * maxTargetDict[desName][1], lbl * maxTargetDict[desName][1])
        #err_mape2 = mean_absolute_percentage_error( (pred * maxTargetDict[desName][1]).tolist(), (lbl * maxTargetDict[desName][1]).tolist()) * 100
        err_mape3 = mape(pred * maxTargetDict[desName][1], lbl * maxTargetDict[desName][1]).detach().cpu().numpy()
        total_mape += err_mape.detach().cpu().numpy() * batch_size
        #total_mape2 += err_mape2 * batch_size
        total_mape3 += err_mape3 * batch_size
        loss = criterion(pred,lbl)
        loss.backward()
        optimizer.step()
        numInputs = pred.view(-1,1).size(0)
        epochLoss.update(loss.detach().item(),numInputs)
    return epochLoss.avg, total_mape/len(dataloader.dataset), total_mape3/len(dataloader.dataset)


def evaluate(model, device, dataloader, maxTargetDict, batch_size):
    model.eval()
    validLoss = AverageMeter()
    total_mape = 0
    #total_mape2 = 0
    total_mape3 = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
            batch = batch.to(device)
            desName = batch.desName[0][0]
            pred = model(batch)
            lbl = batch.target.reshape(-1, 1)
            err_mape = mape(pred * maxTargetDict[desName][1], lbl * maxTargetDict[desName][1])
            #err_mape2 = mean_absolute_percentage_error( (pred * maxTargetDict[desName][1]).tolist(), (lbl * maxTargetDict[desName][1]).tolist()) * 100
            err_mape3 = mape(pred * maxTargetDict[desName][1], lbl * maxTargetDict[desName][1]).detach().cpu().numpy()
            total_mape += err_mape.detach().cpu().numpy() * batch_size
            #total_mape2 += err_mape2 * batch_size
            total_mape3 += err_mape3 * batch_size
            #mseVal = mse(pred, lbl)
            mseVal = criterion(pred, lbl)
            numInputs = pred.view(-1,1).size(0)
            validLoss.update(mseVal.detach().item(),numInputs)
    return validLoss.avg, total_mape/len(dataloader.dataset), total_mape3/len(dataloader.dataset)

def evaluate_plot(model, device, dataloader):
    model.eval()
    totalMSE = AverageMeter()
    batchData = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
            batch = batch.to(device)
            pred = model(batch)
            lbl = batch.target.reshape(-1, 1)
            desName = batch.desName
            synID = batch.synID
            predArray = pred.view(-1,1).detach().cpu().numpy()
            actualArray = lbl.view(-1,1).detach().cpu().numpy()
            batchData.append([predArray,actualArray,desName,synID])
            #mseVal = mse(pred, lbl)
            mseVal = criterion(pred, lbl)
            numInputs = pred.view(-1,1).size(0)
            totalMSE.update(mseVal,numInputs)

    return totalMSE.avg,batchData



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on Synthesis Task Pytorch Geometric')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lp', type=int, default=1,
                        help='Learning problem (QoR prediction: 1,Classification: 2)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--dataset', type=str, default="set1",
                        help='Split strategy (set1/set2/set3 default: set1)')
    parser.add_argument('--rundir', type=str, required=True,default="",
                        help='Output directory path to store result')
    parser.add_argument('--datadir', type=str, required=True, default="",
                        help='Dataset directory containing processed dataset, train test split file csvs')
    parser.add_argument('--target', type=str, required=True, default="nodes",
                        help='Target label (nodes/area/delay), default:"nodes"')
    args = parser.parse_args()

    datasetChoice = args.dataset
    #RUN_DIR = args.rundir

    # Hyperparameters
    batchSize = args.batch_size #64
    num_epochs = args.epochs #80
    learning_rate = args.lr #0.001
    learningProblem = args.lp
    targetLbl = args.target
    #nodeEmbeddingDim = 3
    nodeEmbeddingDim = 32
    synthEncodingDim = 3

    IS_STATS_AVAILABLE = True
    ROOT_DIR = args.datadir #'/scratch/abc586/OPENABC_DATASET'
    global DUMP_DIR
    DUMP_DIR = args.rundir #osp.join('/scratch/abc586/OpenABC-dataset/SynthV9_AND',RUN_DIR)

    if not osp.exists(DUMP_DIR):
        os.mkdir(DUMP_DIR)


    # Load train and test datasets
    trainDS = NetlistGraphDataset(root=osp.join(ROOT_DIR,"lp"+str(learningProblem)),filePath=datasetDict[datasetChoice][0])
    testDS = NetlistGraphDataset(root=osp.join(ROOT_DIR,"lp"+str(learningProblem)),filePath=datasetDict[datasetChoice][1])

    if IS_STATS_AVAILABLE:
        with open(osp.join(ROOT_DIR,'synthesisStatistics.pickle'),'rb') as f:
            targetStats = pickle.load(f)
    else:
        print("\nNo pickle file found for number of gates")
        exit(0)

    #meanVarTargetDict = computeMeanAndVarianceOfTargets(targetStats,targetVar=targetLbl)
    maxTargetDict = computeMaxOfTargets(targetStats,targetVar=targetLbl)
    #print(maxTargetDict)

    trainDS.transform = transforms.Compose([lambda data: addNormalizedTargetsMax(data,targetStats,maxTargetDict,targetVar=targetLbl)])
    testDS.transform = transforms.Compose([lambda data: addNormalizedTargetsMax(data,targetStats,maxTargetDict,targetVar=targetLbl)])

    num_classes = 1


    # Define the model
    synthFlowEncodingDim = trainDS[0].synVec.size()[0]*synthEncodingDim
    node_encoder = NodeEncoder(emb_dim=nodeEmbeddingDim)
    synthesis_encoder = SynthFlowEncoder(emb_dim=synthEncodingDim)

    model = SynthNet(node_encoder=node_encoder,synth_encoder=synthesis_encoder,n_classes=num_classes,synth_input_dim=synthFlowEncodingDim,node_input_dim=nodeEmbeddingDim)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    device = getDevice()
    model = model.to(device)

    # Split the training data into training and validation dataset
    training_validation_samples = [int(0.8*len(trainDS)),len(trainDS)-int(0.8*len(trainDS))]
    train_DS,valid_DS = random_split(trainDS,training_validation_samples)


    # Initialize the dataloaders
    train_dl = DataLoader(train_DS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=0)
    valid_dl = DataLoader(valid_DS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=0)
    test_dl = DataLoader(testDS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=0)


    # Monitor the loss parameters
    valid_curve = []
    train_loss = []
    validLossOpt = 0
    bestValEpoch = 1
    train_mapes = None
    val_mapes = None

    for ep in range(1, num_epochs + 1):
        print("\nEpoch [{}/{}]".format(ep, num_epochs))
        print("\nTraining..")
        trainLoss, totalTrainMape, totalTrainMapeK = train(model, device, train_dl, optimizer, maxTargetDict, batchSize)
        print({'Train loss': trainLoss,'Total Training MAPE': {totalTrainMape.item()},'Total Training MAPE(K)': {totalTrainMapeK.item()}})
        train_mapes = totalTrainMape if train_mapes is None else np.vstack((train_mapes, totalTrainMape))
        print("\nEvaluation..")
        validLoss, totalValMape, totalValMapeK = evaluate(model, device, valid_dl, maxTargetDict, batchSize)
        if ep > 1:
            if validLossOpt > validLoss:
                validLossOpt = validLoss
                bestValEpoch = ep
                torch.save(model.state_dict(), osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(bestValEpoch, validLossOpt)))
        else:
            validLossOpt = validLoss
            torch.save(model.state_dict(), osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(bestValEpoch, validLossOpt)))
        print({'Validation loss': validLoss,'Total Validation MAPE': {totalValMape.item()},'Total Validation MAPE(K)': {totalValMapeK.item()}})
        val_mapes = totalValMape if val_mapes is None else np.vstack((val_mapes, totalValMape))
        valid_curve.append(validLoss)
        train_loss.append(trainLoss)
        #scheduler.step(validLoss)
        scheduler.step()

    # Loading best validation model
    model.load_state_dict(torch.load(osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(bestValEpoch, validLossOpt))))


    # Save training data for future plots
    with open(osp.join(DUMP_DIR,'valid_curve.pkl'),'wb') as f:
        pickle.dump(valid_curve,f)

    with open(osp.join(DUMP_DIR,'train_loss.pkl'),'wb') as f:
        pickle.dump(train_loss,f)

    ##### EVALUATION ######
    plotChart([i+1 for i in range(len(valid_curve))],valid_curve,"# Epochs","Loss","test_acc","Validation loss")
    plotChart([i+1 for i in range(len(train_loss))],train_loss,"# Epochs","Loss","train_loss","Training loss")

    # Evaluate on train data
    trainMSE,trainBatchData = evaluate_plot(model, device, train_dl)
    NUM_BATCHES_TRAIN = len(train_dl)
    doScatterAndTopKRanking(NUM_BATCHES_TRAIN,batchSize,trainBatchData,DUMP_DIR,"train",maxTargetDict)

    # Evaluate on validation data
    validMSE,validBatchData = evaluate_plot(model, device, valid_dl)
    NUM_BATCHES_VALID = len(valid_dl)
    doScatterAndTopKRanking(NUM_BATCHES_VALID,batchSize,validBatchData,DUMP_DIR,"valid",maxTargetDict)

    # Evaluate on test data
    testMSE,testBatchData = evaluate_plot(model, device, test_dl)
    NUM_BATCHES_TEST = len(test_dl)
    doScatterAndTopKRanking(NUM_BATCHES_TEST,batchSize,testBatchData,DUMP_DIR,"test",maxTargetDict)
    
    num_params = sum(p.numel() for p in model.parameters())
    
    print("********************")
    print("Final run statistics")
    print("********************")
    print(f'Total Params: {num_params}')
    print("Training loss per sample:{}".format(trainMSE))
    print("Validation loss per sample:{}".format(validMSE))
    print("Test loss per sample:{}".format(testMSE))
    print("********************")
    

if __name__ == "__main__":
    main()
