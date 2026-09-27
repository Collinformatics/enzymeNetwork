from math import trunc

import esm
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
from tqdm import tqdm


"""
    :param inESMModel: Model size
        Options: 
            esm2_t48_15B_UR50D, esm2_t36_3B_UR50D, 
            esm2_t33_650M_UR50D, or esm2_t30_150M_UR50D
"""


# Input: Files
inFileNameActive = 'Mpro2-Register_Q@R4-R6'
inFileNameInactive = 'Mpro2-Init'
inPathDir = 'Data/Train/' # Path to directory

# Input: ESM
inESMModel = 'esm2_t33_650M_UR50D' # 'esm2_t36_3B_UR50D'  # Model size

# Input: Model
inModelName = 'Mpro2' #
inBatchSize = 512


# ========================================================================================
class TrainClassifier:
    def __init__(self, modelName, directory, filePos, fileNeg,
                 batchSize, esmSize='esm2_t36_3B_UR50D'):
        """
            :param modelName: Save the model with this name

            :param directory: Path to the directory with positive and negative substrates

            :param filePos: File name for active substrates

            :param fileNeg: File name for inactive substrates

            :param esmSize: Model size
                Ex: esm2_t48_15B_UR50D, esm2_t36_3B_UR50D,
                    esm2_t33_650M_UR50D, or esm2_t30_150M_UR50D
        """
        self.setTrainingDevice()

        # Load files
        self.directory = directory
        self.embPos, self.positive, self.embNeg, self.negative = None, None, None, None
        pathPosSubs, pathPosEmb = self.getPaths(filePos, setClass='Pos', esm=esmSize)
        pathNegSubs, pathNegEmb = self.getPaths(fileNeg, setClass='Neg', esm=esmSize)
        if os.path.exists(pathPosEmb):
            self.embPos = self.loadData(pathPosEmb, tag='Positive Substrates')
        else:
            self.positive = self.loadData(pathPosSubs, tag='Positive Substrates')
            self.generateEmbeddings(path=pathPosEmb, sequences=self.positive,
                                    batch=batchSize, modelSize=esmSize)

        if os.path.exists(pathNegEmb):
            self.embNeg = self.loadData(pathNegEmb, tag='Negative Substrates')
        else:
            self.negative = self.loadData(pathNegSubs, tag='Negative Substrates')
            self.generateEmbeddings(path=pathNegEmb, sequences=self.negative,
                                    batch=batchSize, modelSize=esmSize)

        # Model
        self.modelName = modelName
        self.pathModel = 'Models'
        if not os.path.exists(self.pathModel):
            os.makedirs(self.pathModel)

        self.train()


    def getPaths(self, fileName, setClass, esm):
        pathSubs = os.path.join(self.directory, f'substrates_{setClass}_{fileName}.txt')
        pathEmb = os.path.join(self.directory, f'embeddings_{setClass}_{fileName}_{esm}.pt')
        return pathSubs, pathEmb


    def loadData(self, path, tag):
        print('================================= Loading Data '
              '==================================')
        print(f'Loading file: {path}')
        if path.endswith('.txt'):
            with open(path, 'r') as f:
                data = list(dict.fromkeys(f.read().splitlines()))
            print(f'Loaded: {len(data):,} substrates\n')
            print(f'{tag}:')
            for i, s in enumerate(data):
                print(f'* {s}')
                if i >= 10:
                    break
        elif path.endswith('.pt'):
            data = torch.load(path, map_location=self.device)
            print(data)
        else:
            sys.stdout.flush()
            raise ValueError(f'\n\tThe file path "{path}" is not recognized.\n'
                             f'\tExpected: ".txt" or ".pt".')
        print('\n')

        return data


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
            self.device = torch.device('cuda') # NVIDIA GPU
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps') # Apple GPU (Metal Performance Shaders)
        else:
            self.device = torch.device('cpu')
        print(f'Training device: {self.device}\n\n')


    def generateEmbeddings(self, path, sequences, batch, modelSize):
        print(f'========================== Generating ESM Embeddings '
              f'===========================')
        # Step 1: Load the ESM model and batch converter
        layer = None
        if modelSize == 'esm2_t48_15B_UR50D':
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            layer = 48
        elif modelSize == 'esm2_t36_3B_UR50D':
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            layer = 36
        elif modelSize == 'esm2_t33_650M_UR50D':
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            layer = 33
        elif modelSize == 'esm2_t30_150M_UR50D':
            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            layer = 30
        else:
            sys.stdout.flush()
            raise ValueError(f'\n\tThe ESM model "{modelSize}" is not available.\n'
                             f'\tUse: esm2_t36_3B_UR50D, esm2_t36_3B_UR50D, '
                             f'esm2_t33_650M_UR50D, or esm2_t30_150M_UR50D')
        model.to(self.device)
        batch_converter = alphabet.get_batch_converter()


        # Step 2: Convert substrates to ESM model format
        try:
            batchLabels, batchSubs, batchTokens = batch_converter(
                [('', seq) for seq in sequences]
            ) # Use empty str as dummy variable in lieu of activity scores
        except Exception as exc:
            print(f'ERROR: The ESM has failed to evaluate your substrates\n\n'
                  f'Exception:\n{exc}\n\n')
            sys.exit(1)
        slicedTokens = pd.DataFrame(batchTokens[:, 1:-1], index=batchSubs,
                                    columns=[f'R{i+1}' for i in range(len(sequences[0]))])
        print(f'Tokens:\n{slicedTokens}\n\n')
        batchTokens = batchTokens.to(self.device) # Move tokens to device

        # Step 3: Generate embeddings
        embed, numTokens = [], len(batchTokens)
        numIterations = (numTokens + batch - 1) // batch
        print(f'Generating Embeddings:')
        for i in tqdm(range(0, numTokens, batch), total=numIterations, desc='Embeddings'):
            chunk = batchTokens[i:i+batch]
            with torch.no_grad():
                results = model(chunk, repr_layers=[layer])
            e = results['representations'][layer].mean(dim=1).cpu()
            embed.append(e)
        embeddings = torch.cat(embed, dim=0)
        print(f'Embeddings shape: {embeddings.shape}')
        print(f'Sample (first 3, first 8 dims):\n{embeddings[:3, :8]}\n')

        # Save embeddings
        print(f'Saving Embeddings:\n{path}\n\n')
        torch.save(embeddings, path)

        return embeddings


    def train(self):
        print('========================== Training Binary Classifier '
              '===========================')
        embPos = 0





# ========================================================================================

# Train model
classifier = TrainClassifier(
    modelName=inModelName, directory=inPathDir,
    filePos=inFileNameActive, fileNeg=inFileNameInactive,
    batchSize=inBatchSize, esmSize=inESMModel
)
