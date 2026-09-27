import esm
import os
from sklearn.linear_model import LogisticRegression
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# Input: Files
inFileNameActive = 'Mpro2-Register_Q@R4-R6'
inFileNameInactive = 'Mpro2-Init'
inPathDir = 'Data/Train/' # Path to directory

# Input: ESM
inESM = 'esm2_t36_3B_UR50D' # 'esm2_t48_15B_UR50D' # Model size

# Input: Model
inModelName = 'Mpro2' #


# ========================================================================================
class TrainClassifier:
    def __init__(self, modelName, directory, filePos, fileNeg, esmSize='esm2_t36_3B_UR50D'):
        """
            :param modelName: Save the model with this name

            :param directory: Path to the directory with positive and negative substrates

            :param filePos: File name for active substrates

            :param fileNeg: File name for inactive substrates

            :param esmSize: Model size
                Ex: esm2_t48_15B_UR50D, esm2_t36_3B_UR50D
        """
        self.device = self.setTrainingDevice()

        # Load files
        self.directory = directory
        self.positive = self.loadData(filePos, setClass='Pos')
        self.negative = self.loadData(fileNeg, setClass='Neg')

        # Model
        self.modelName = modelName
        self.pathModel = 'Models'
        if not os.path.exists(self.pathModel):
            os.makedirs(self.pathModel)
        self.train(esmSize=esmSize)


    def loadData(self, fileName, setClass):
        print('================================= Loading Data '
              '==================================')
        path = os.path.join(self.directory, f'substrates_{setClass}_{fileName}.txt')
        print(f'Loading file: {path}')
        subs = []
        with open(path, 'r') as f:
            subs = list(dict.fromkeys(f.read().splitlines()))
        print(f'Loaded: {len(subs):,} substrates\n\n')
        return subs


    def setTrainingDevice(self):
        print('============================== Set Training Device '
              '==============================')
        try:
            deviceName = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f'GPU Name: {deviceName}')
        except:
            pass

        # Select device
        if torch.cuda.is_available():
            device = torch.device('cuda') # NVIDIA GPU
        elif torch.backends.mps.is_available():
            device = torch.device('mps') # Apple GPU (Metal Performance Shaders)
        else:
            device = torch.device('cpu')
        print(f'Training device: {device}\n\n')
        return device


    def esmEmbeddings(self, sequences, esmSize):
        print('============================== Get ESM Embeddings '
              '===============================')

        # Step 2: Load the ESM model and batch converter
        if esmSize == 'esm2_t36_3B_UR50D':
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        batch_converter = alphabet.get_batch_converter()


        # Step 3: Convert substrates to ESM model format and generate embeddings
        try:
            batchLabels, batchSubs, batchTokens = batch_converter(subs)
        except Exception as exc:
            print(f'ERROR: The ESM has failed to evaluate your substrates\n\n'
                  f'Exception:\n{exc}\n\n'
                  f'Suggestion:'
                  f'     Try replacing: esm.pretrained.esm2_t36_3B_UR50D()'
                  f'\n'
                  f'     With: esm.pretrained.esm2_t33_650M_UR50D()'
                  f'\n')
            sys.exit(1)

        print(f'Batch Tokens:{greenLight} {batchTokens.shape}\n'
              f'{greenLight}{batchTokens}\n\n')
        slicedTokens = pd.DataFrame(batchTokens[:, 1:-1],
                                    index=batchSubs,
                                    columns=subLabel)
        if useSubCounts:
            slicedTokens['Counts'] = counts
        print(f'Sliced Tokens:\n'
              f'{greenLight}{slicedTokens}\n\n')

        return slicedTokens, batchSubs, sampleSize


    def train(esmSize='esm2_t36_3B_UR50D'):
        print('========================== Training Binary Classifier '
              '===========================')
        print('Positive Substrates:')
        for i, s in enumerate(positive):
            print(f'* {s}')
            if i >= 5:
                break
        print('\nNegative Substrates:')
        for i, s in enumerate(negative):
            print(f'* {s}')
            if i >= 5:
                break




# ========================================================================================

# Train model
classifier = TrainClassifier(
    directory=inPathDir, filePos=inFileNameActive, fileNeg=inFileNameInactive,
    esmSize='esm2_t36_3B_UR50D'
)
