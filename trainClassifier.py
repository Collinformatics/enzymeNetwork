import esm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# Input: Files
inEnzymeName = 'Mpro2'
inFileNameActive = 'Mpro2-Register_Q@R4-R6'
inFileNameInactive = 'Mpro2-Init'
inPathDir = 'Data/Train/' # Path to directory

# Input: ESM
inESMModel = 'esm2_t36_3B_UR50D' # Model size
inBatchSize = 512

# Input: Training
inNumEpochs = 10

# Input: Testing Substrates
inTestSubstates = [
    'ATLQSGVE', 'DVILQCAW', 'FICLIQAG', 'CVILHSAG', 'LPVADFCG',
    'ACVSDEDR', 'IVDERNFS', 'FGHERYHG', 'IPLKASVC', 'TIVSDWAH'
] # Active and inactive sequences
inFileNameTestEmbeddings = f'{inEnzymeName}_N-{len(inTestSubstates)}'


# ========================================================================================
class TrainClassifier:
    def __init__(self, modelName, epochs, directory, filePos, fileNeg,
                 testSubstrates, fileTest, batchSize, esmSize='esm2_t36_3B_UR50D'):
        """
            :param modelName:
                Save the model with this name

            :param epochs:
                Number of training epochs

            :param directory:
                Path to the directory with positive and negative substrates

            :param filePos:
                File name for active substrates

            :param fileNeg:
                File name for inactive substrates

            :param testSubstrates:
                A list of active and inactive substrates used to test the model

            :param fileTest:
                File name for testing substrates

            :param batchSize:
                Split substrates into batches for generating embeddings

            :param esmSize:
                Model size
                Ex: esm2_t48_15B_UR50D, esm2_t36_3B_UR50D,
                    esm2_t33_650M_UR50D, or esm2_t30_150M_UR50D
        """
        self.device = None
        self.setTrainingDevice()

        # Data
        self.directory = directory
        self.embPos, self.positive = None, None
        self.embNeg, self.negative = None, None
        self.embTest = None
        pathPosSubs, pathPosEmb = self.getPaths(filePos, setClass='Pos', esm=esmSize)
        pathNegSubs, pathNegEmb = self.getPaths(fileNeg, setClass='Neg', esm=esmSize)
        _, pathTestEmb = self.getPaths(fileTest, setClass='Test', esm=esmSize)
        pathModel = os.path.join('Models', f'{modelName}.pt')

        # Load: Positive
        if os.path.exists(pathPosEmb):
            self.embPos = self.loadData(
                pathPosEmb, tag='Positive Substrates', loadEmb=True
            )
        else:
            self.positive = self.loadData(pathPosSubs, tag='Positive Substrates')
            self.embPos = self.generateEmbeddings(
                path=pathPosEmb, sequences=self.positive, batch=batchSize,
                modelSize=esmSize, tag='Positive Substrates'
            )

        # Load: Negative
        if os.path.exists(pathNegEmb):
            self.embNeg = self.loadData(
                pathNegEmb, tag='Negative Substrates', loadEmb=True
            )
        else:
            self.negative = self.loadData(pathNegSubs, tag='Negative Substrates')
            self.embNeg = self.generateEmbeddings(
                path=pathNegEmb, sequences=self.negative, batch=batchSize,
                modelSize=esmSize, tag='Negative Substrates'
            )

        # Load: Testing
        if os.path.exists(pathTestEmb):
            self.embTest = self.loadData(
                pathTestEmb, tag='Testing Substrates', loadEmb=True
            )
        else:
            self.embTest = self.generateEmbeddings(
                path=pathTestEmb, sequences=testSubstrates, batch=batchSize,
                modelSize=esmSize, tag='Testing Substrates'
            )

        # Train model
        modelDir = 'Models'
        pathModel = os.path.join(modelDir, f'{modelName}.pt')
        if not os.path.exists(modelDir):
            os.makedirs(modelDir)
        if os.path.exists(pathModel):
            model = self.loadModel(pathModel)
        else:
            model = self.train(epochs, pathModel)

        # Test model
        self.test(model)


    def getPaths(self, fileName, setClass, esm):
        pathSubs = os.path.join(self.directory, f'substrates_{setClass}_{fileName}.txt')
        pathEmb = os.path.join(self.directory, f'embeddings_{setClass}_{fileName}_{esm}.pt')
        return pathSubs, pathEmb


    def loadData(self, path, tag, loadEmb=False):
        print('================================= Loading Data '
              '==================================')
        print(f'Loading: {tag}\n\t{path}\n')
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

        if loadEmb:
            data = data.to(self.device)

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


    def generateEmbeddings(self, path, sequences, batch, modelSize, tag):
        print(f'========================== Generating ESM Embeddings '
              f'===========================')
        print(f'Dataset: {tag}\n')

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
        for i in tqdm(range(0, numTokens, batch), total=numIterations, desc='Progress:'):
            chunk = batchTokens[i:i+batch]
            with torch.no_grad():
                results = model(chunk, repr_layers=[layer])
            e = results['representations'][layer].mean(dim=1).cpu()
            embed.append(e)
        embeddings = torch.cat(embed, dim=0)
        print(f'\nEmbeddings shape: {embeddings.shape}')
        print(f'Sample: 3x{len(sequences[0])}\n{embeddings[:3, :len(sequences[0])]}\n')

        # Save embeddings
        print(f'Saving Embeddings:\n\t{path}\n\n')
        torch.save(embeddings, path)

        return embeddings


    def train(self, epochs, modelPath):
        print('========================== Training Binary Classifier '
              '===========================')

        # Step 1: Build tensors
        X = torch.cat([self.embPos, self.embNeg], dim=0)  # (N_total, 1280)
        y = torch.cat([
            torch.ones(len(self.embPos), dtype=torch.long),
            torch.zeros(len(self.embNeg), dtype=torch.long),
        ])  # (N_total,)

        print(f'Positive: {len(self.embPos):,}, Negative: {len(self.embNeg):,}')
        print(f'Embedding dim: {X.shape[1]:,}')

        # Step 2: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=256)

        # Step 3: Model (linear probe)
        model = nn.Sequential(
            nn.Linear(X.shape[1], 2),
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Step 4: Training loop
        print('\nTraining Model:')
        model.train()
        for epoch in range(epochs):
            total_loss, correct, total = 0, 0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
            print(f'  Epoch {epoch+1:2d}/{epochs:,} — '
                  f'loss: {total_loss/len(train_dl):.4f}, '
                  f'acc: {correct/total:.4f}', flush=True)

        # Step 5: Evaluate
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(self.device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=-1)
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                all_labels.extend(yb.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        print(f'\nTest AUC: {roc_auc_score(all_labels, all_probs):.4f}')
        print(classification_report(all_labels, all_preds,
                                    target_names=['Negative', 'Positive']))

        # Step 6: Save model
        torch.save(model.state_dict(), modelPath)
        print(f'Saved model: {modelPath}\n')

        return model


    def loadModel(self, path):
        print('================================= Loading Model '
              '=================================')
        print(f'Loading:\n\t{path}\n')
        model = nn.Sequential(nn.Linear(self.embPos.shape[1], 2),)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model


    def test(self, model):
        print('Test substrates:')
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(self.embTest), dim=-1)[:, 1].cpu()

        for seq, p in zip(self.embTest, probs.numpy()):
            label = 'Active' if p > 0.5 else 'Inactive'
            print(f'  {seq}  →  {p:.4f}  ({label})')


# ========================================================================================

# Train model
classifier = TrainClassifier(
    modelName=inEnzymeName, epochs=inNumEpochs, directory=inPathDir,
    filePos=inFileNameActive, fileNeg=inFileNameInactive,
    testSubstrates=inTestSubstates, fileTest=inFileNameTestEmbeddings,
    batchSize=inBatchSize, esmSize=inESMModel
)
