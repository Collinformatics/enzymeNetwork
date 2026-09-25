import esm
import os
import torch
from sklearn.linear_model import LogisticRegression


# Input: Files
inSubstratesActive = ''
inSubstratesInactive = ''
inFilePath = '' # Path to directory

# Input: ESM
inESM = 'esm2_t36_3B_UR50D' # 'esm2_t48_15B_UR50D' # Model size

# Input: Model
inModelName = 'Mpro2'


# ========================================================================================
# File location
filePathActive = os.path.join(inFilePath, inSubstratesActive)
filePathInactive = os.path.join(inFilePath, inSubstratesInactive)



#
model, alphabet = esm.pretrained.load_model_and_alphabet(inESM)
batch_converter = alphabet.get_batch_converter()

def embed(seq):
    _, _, toks = batch_converter([("x", seq)])
    toks = toks.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        rep = model(toks, repr_layers=[33], return_contacts=False)
    return rep["representations"][33].mean(dim=1).cpu().numpy()

# X = np.array([embed(s) for s in seqs]);  y = np.array(labels)
clf = LogisticRegression(max_iter=1000).fit(X, y)