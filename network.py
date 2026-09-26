import os
import networkx as nx
import matplotlib.pyplot as plt

# ========================================================================================
# Input: Figures
inNodeColor = 'green'
inNodeSize = 700
inFontSize = 12
inArrowSize = 25
inArc = 0.3 # Arrow curve

# Input: Save Params
inFileName = 'net'
inSaveDir = 'Networks'


# ========================================================================================
# Input: Data
data = {
    'A': ['B', 'D'],
    'B': ['C'],
    'C': ['A'],
    'D': ['A', 'E'],
    'E': ['F'],
    'F': ['D'],
}


# ========================================================================================
# Save params
if not inFileName.endswith('.png') or inFileName.endswith('.jpeg'):
    inFileName += '.png'
savePath = os.path.join(inSaveDir, inFileName)
if not os.path.exists(inSaveDir):
    os.mkdir(inSaveDir)


# Graph network
G = nx.from_dict_of_lists(data, create_using=nx.DiGraph())
nx.draw(G, with_labels=True, node_size=inNodeSize, font_size=inFontSize,
        node_color=inNodeColor,
        arrowsize=inArrowSize, # arrow head size
        arrowstyle='->', # '->', '-|>', '-['
        connectionstyle=f'arc3,rad={inArc}' # curve for parallel edges
)

# Save figure
plt.savefig(savePath, bbox_inches='tight')
print(f'Graph was saved at path:\n\t{savePath}')
os.system(f'xdg-open {savePath}')
