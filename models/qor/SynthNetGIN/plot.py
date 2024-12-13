from torch_geometric.data import DataLoader
from netlistDataset import *
import matplotlib.pyplot as plt
import pickle
from utils import *
from model import *
from train import *
from torch.utils.data import random_split

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

DUMP_DIR='/Users/minirawat/GitRepos/NETV1_set1'
ROOT_DIR='/Users/minirawat/GitRepos/OPENABC-D'
learningProblem=1
num_classes=1
batchSize=64
IS_STATS_AVAILABLE = True
targetLbl = 'nodes'

datasetChoice='set1'
datasetDict =  {
    'set1' : ["train_data_set1.csv","test_data_set1.csv"],
    'set2' : ["train_data_set2.csv","test_data_set2.csv"],
    'set3' : ["train_data_mixmatch_v1.csv","test_data_mixmatch_v1.csv"]
}

# Load train and test datasets
trainDS = NetlistGraphDataset(root=osp.join(ROOT_DIR,"lp"+str(learningProblem)),filePath=datasetDict[datasetChoice][0])
testDS = NetlistGraphDataset(root=osp.join(ROOT_DIR,"lp"+str(learningProblem)),filePath=datasetDict[datasetChoice][1])

if IS_STATS_AVAILABLE:
  with open(osp.join(ROOT_DIR,'synthesisStatistics.pickle'),'rb') as f:
    targetStats = pickle.load(f)
else:
  print("\nNo pickle file found for number of gates")
  exit(0)

meanVarTargetDict = computeMeanAndVarianceOfTargets(targetStats,targetVar=targetLbl)

trainDS.transform = transforms.Compose([lambda data: addNormalizedTargets(data,targetStats,meanVarTargetDict,targetVar=targetLbl)])
testDS.transform = transforms.Compose([lambda data: addNormalizedTargets(data,targetStats,meanVarTargetDict,targetVar=targetLbl)])

synthEncodingDim = 3
nodeEmbeddingDim = 3

# Define the model
synthFlowEncodingDim = trainDS[0].synVec.size()[0]*synthEncodingDim
node_encoder = NodeEncoder(emb_dim=nodeEmbeddingDim)
synthesis_encoder = SynthFlowEncoder(emb_dim=synthEncodingDim)

model = SynthNet(node_encoder=node_encoder,synth_encoder=synthesis_encoder,n_classes=num_classes,synth_input_dim=synthFlowEncodingDim,node_input_dim=nodeEmbeddingDim)
model.load_state_dict(torch.load(osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(54, 0.537))))

# Split the training data into training and validation dataset
training_validation_samples = [int(0.8*len(trainDS)),len(trainDS)-int(0.8*len(trainDS))]
train_DS,valid_DS = random_split(trainDS,training_validation_samples)

# Initialize the dataloaders
train_dl = DataLoader(train_DS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=4)
valid_dl = DataLoader(valid_DS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=4)
test_dl = DataLoader(testDS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=4)

valid_curve=[]
train_loss=[]

with open(osp.join(DUMP_DIR,'valid_curve.pkl'),'rb') as f:
  valid_curve=pickle.load(f)

with open(osp.join(DUMP_DIR,'train_loss.pkl'),'rb') as f:
  train_loss=pickle.load(f)

##### EVALUATION ######
plotChart([i+1 for i in range(len(valid_curve))],valid_curve,"# Epochs","Loss","test_acc","Validation loss")
plotChart([i+1 for i in range(len(train_loss))],train_loss,"# Epochs","Loss","train_loss","Training loss")

device=getDevice()

# Evaluate on train data
trainMSE,trainBatchData = evaluate_plot(model, device, train_dl)
NUM_BATCHES_TRAIN = len(train_dl)
doScatterAndTopKRanking(NUM_BATCHES_TRAIN,batchSize,trainBatchData,DUMP_DIR,"train")

# Evaluate on validation data
validMSE,validBatchData = evaluate_plot(model, device, valid_dl)
NUM_BATCHES_VALID = len(valid_dl)
doScatterAndTopKRanking(NUM_BATCHES_VALID,batchSize,validBatchData,DUMP_DIR,"valid")

# Evaluate on test data
testMSE,testBatchData = evaluate_plot(model, device, test_dl)
NUM_BATCHES_TEST = len(test_dl)
doScatterAndTopKRanking(NUM_BATCHES_TEST,batchSize,testBatchData,DUMP_DIR,"test")

num_params = sum(p.numel() for p in model.parameters())

print("********************")
print("Final run statistics")
print("********************")
print(f'Total Params: {num_params}')
print("Training loss per sample:{}".format(trainMSE))
print("Validation loss per sample:{}".format(validMSE))
print("Test loss per sample:{}".format(testMSE))
print("********************")
