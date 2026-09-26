import os
import networkx as nx
import matplotlib.pyplot as plt

# ========================================================================================
inFileName = 'net'
inSaveDir = 'Networks'

data = {
    'A': ['B', 'D'],
    'B': ['C'],
    'C': ['A'],
    'D': ['A', 'E'],
    'E': ['F'],
    'F': ['D'],
}


# ========================================================================================
if not inFileName.endswith('.png') or inFileName.endswith('.jpeg'):
    inFileName += '.png'
savePath = os.path.join(inSaveDir, inFileName)
if not os.path.exists(inSaveDir):
    os.mkdir(inSaveDir)


# Graph network
G = nx.from_dict_of_lists(data, create_using=nx.DiGraph())
nx.draw(G, with_labels=True, node_size=700, font_size=12, node_color='green')
plt.savefig(savePath, bbox_inches='tight')
print(f'Graph was saved at path:\n\t{savePath}')
os.system(f'xdg-open {savePath}')
