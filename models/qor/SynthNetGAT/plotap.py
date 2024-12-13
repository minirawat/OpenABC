from torch_geometric.data import DataLoader
from netlistDataset import *
import matplotlib.pyplot as plt
import pickle
from utils import *
from model import *
from train import *
from torch.utils.data import random_split
import seaborn as sns
import numpy as np

def plotChart(x,y,xlabel,ylabel,leg_label,title):
    fig = plt.figure(figsize=(13,5))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x,y, label=leg_label)
    leg = plt.legend(loc='best', ncol=2, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel(xlabel, weight='bold')
    plt.ylabel(ylabel, weight='bold')
    plt.title(title,weight='bold')
    plt.savefig(osp.join(DUMP_DIR,title+'.png'), format='png', bbox_inches='tight')

def plotActualPredicted(df,xlabel,ylabel,title):
    #plt.rcParams['figure.figsize'] = [13,5]
   
    n = 500
    df_subset = df.head(n)

    plt.scatter(df_subset.index, df_subset.prediction, label='Predicted', marker='o')
    plt.scatter(df_subset.index, df_subset.actual, label='Actual', marker='x')
    plt.title(title,weight='bold')
    plt.xlabel('Index')  # Since there's no Date column, use the index as the X label
    plt.ylabel('Values')
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show() 
    fig1.savefig(osp.join(DUMP_DIR,title+'.png'), format='png', bbox_inches='tight')

def plotResidual(df,xlabel,ylabel,title):
    #plt.rcParams['figure.figsize'] = [13,5]
   
    residuals = [a - p for a, p in zip(df.actual, df.prediction)]

    plt.scatter(df.actual, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
    plt.title('Residual Plot')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(osp.join(DUMP_DIR,title+'.png'), format='png', bbox_inches='tight')

def plotHeatmap(df, xlabel, ylabel, title):

    #plt.rcParams['figure.figsize'] = [13,5]
    sns.histplot(x=df.actual, y=df.prediction, bins=50, cmap='viridis')
    plt.title('Density Plot of Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(osp.join(DUMP_DIR,title+'.png'), format='png', bbox_inches='tight')

def plotParityWithErrorBars(df, xlabel, ylabel, title):

    residuals = [a - p for a, p in zip(df.actual, df.prediction)]

    #plt.rcParams['figure.figsize'] = [13,5]
    plt.errorbar(df.actual, df.prediction, yerr=np.abs(residuals), fmt='o', ecolor='gray', alpha=0.7, label='Predicted ± Residuals')
    plt.plot([min(df.actual), max(df.actual)], [min(df.actual), max(df.actual)], 'r--', label='Perfect Fit')
    plt.title('Actual vs Predicted with Residuals')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(osp.join(DUMP_DIR,title+'.png'), format='png', bbox_inches='tight')

DUMP_DIR='/Users/minirawat/GitRepos/GATRun'
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


state_dict = torch.load(osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(40, 0.452)))
model.load_state_dict(state_dict)

# Split the training data into training and validation dataset
training_validation_samples = [int(0.8*len(trainDS)),len(trainDS)-int(0.8*len(trainDS))]
train_DS,valid_DS = random_split(trainDS,training_validation_samples)

# Initialize the dataloaders
train_dl = DataLoader(train_DS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=4)
valid_dl = DataLoader(valid_DS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=4)
test_dl = DataLoader(testDS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=4)

device=getDevice()

# Evaluate on test data
testMSE,testBatchData = evaluate_plot(model, device, test_dl)
NUM_BATCHES_TEST = len(test_dl)
#doScatterAndTopKRanking(NUM_BATCHES_TEST,batchSize,testBatchData,DUMP_DIR,"test")

predList = []
actualList = []
for i in range(NUM_BATCHES_TEST):
        numElemsInBatch = len(testBatchData[i][0])
        for batchID in range(numElemsInBatch):
            predList.append(testBatchData[i][0][batchID][0])
            actualList.append(testBatchData[i][1][batchID][0])

df = pd.DataFrame({'prediction': predList,
                   'actual': actualList})


plotActualPredicted(df,'Series','Acutal/Predicted','Actual vs Predicted nodes')
plotResidual(df,'Series','Actual Residual','Actual Residual')
plotHeatmap(df,'Series','Heatmap','Heatmap')
plotParityWithErrorBars(df, 'Series', 'Parity With Error', 'Parity with Error')
