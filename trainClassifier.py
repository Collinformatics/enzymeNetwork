import esm
import os
from sklearn.linear_model import LogisticRegression
import sys
import torch


# Input: Files
inFileNameActive = 'Mpro2-Reg_R4-R6'
inFileNameInactive = 'Mpro2-Init'
inPathDir = 'Data/Train/' # Path to directory

# Input: ESM
inESM = 'esm2_t36_3B_UR50D' # 'esm2_t48_15B_UR50D' # Model size

# Input: Model
inModelName = 'Mpro2'


# ========================================================================================
def loadData(directory, fileName, setClass):
    print('================================= Loading Data '
          '==================================')
    path = os.path.join(directory, f'substrates_{setClass}_{fileName}.txt')
    print(f'Loading file: {path}')

    return path


def setTrainingDevice():
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


def trainBinaryClassifier(device, esmSize='esm2_t36_3B_UR50D'):
    """
        :param device: Hardware used to train the model
            Ex: cuda, mps, or cpu
        
        :param esmSize: Model size
            Ex: esm2_t48_15B_UR50D, esm2_t36_3B_UR50D
    """

    print('========================== Training Binary Classifier '
          '===========================')
    model, alphabet = esm.pretrained.load_model_and_alphabet(esmSize)
    batch_converter = alphabet.get_batch_converter()

    def embed(seq):
        _, _, toks = batch_converter([("x", seq)])
        toks = toks.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            rep = model(toks, repr_layers=[33], return_contacts=False)
        return rep["representations"][33].mean(dim=1).cpu().numpy()

    # X = np.array([embed(s) for s in seqs]);  y = np.array(labels)
    clf = LogisticRegression(max_iter=1000).fit(X, y)


# ========================================================================================
# File location
subsActive = loadData(directory=inPathDir, fileName=inFileNameActive, setClass='Pos')
subsInactive = loadData(directory=inPathDir, fileName=inFileNameInactive, setClass='Neg')

# Train model
trainBinaryClassifier(device=setTrainingDevice(), esmSize='esm2_t36_3B_UR50D')
