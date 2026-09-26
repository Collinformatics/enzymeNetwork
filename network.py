import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import subprocess


# ========================================================================================
# Input: Figures
inNodeColor = 'green'
inNodeSize = 700
inFontSize = 12
inLineWidth = 1.5
inArrowSize = 18
inArc = 0.3 # Arrow curve

# Input: Save Params
inFileName = 'net' # Image name, save as .png or .jpeg
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
def plotNetwork(network):
    """
    :param network: A dictionary containing the names of each node as keys, and a list of
        nodes a given node makes a connection to.
    """
    print('==================================== Network '
          '====================================')
    df = pd.DataFrame(0, index=network.keys(), columns=network.keys())
    for node in df.columns:
        for connNode in network[node]:
            df.loc[connNode, node] = 1
    print(f'{df}\n')

    # Plot network
    G = nx.from_dict_of_lists(network, create_using=nx.DiGraph())
    nx.draw(
        G, with_labels=True, node_size=inNodeSize, node_color=inNodeColor,
        edge_color='black', min_source_margin=15, min_target_margin=15,
        font_size=inFontSize, font_weight='bold',
        width=inLineWidth, arrowsize=inArrowSize,  # arrow head size
        arrowstyle='-|>',  # '->', '-|>', '-['
        connectionstyle=f'arc3,rad={inArc}'  # curve for parallel edges
    )

    # Save figure
    plt.savefig(savePath, bbox_inches='tight')
    print(f'Graph was saved at path:\n\t{savePath}\n\n')
    subprocess.run(['xdg-open', savePath])  # Open fig in web browser


# ========================================================================================
if __name__ == '__main__':
    # Save params
    if not inFileName.endswith('.png') or inFileName.endswith('.jpeg'):
        inFileName += '.png'
    savePath = os.path.join(inSaveDir, inFileName)
    if not os.path.exists(inSaveDir):
        os.mkdir(inSaveDir)


    # Graph network
    plotNetwork(data)
