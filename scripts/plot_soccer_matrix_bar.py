import sys
import matplotlib
matplotlib.use('PDF')
text_size = 20
matplotlib.rcParams['xtick.labelsize'] = text_size
matplotlib.rcParams['ytick.labelsize'] = text_size
matplotlib.rcParams['axes.labelsize'] = text_size
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

if __name__ == '__main__':
    output = sys.argv[1]
    fig, ax = plt.subplots()
    width = 0.2
    ind = np.arange(5)
    xticks = ('DQN-o', 'DQN-d', 'DQN-m', 'DRON-concat', 'DRON-world')
    labels = ['offensive', 'defensive']
    colors = ['r', 'b']
    rewards = {}
    rewards['offensive'] = [0.897, -0.272, 0.811, 0.875, 0.870]
    rewards['defensive'] = [0.480, 0.504, 0.498, 0.493, 0.486]
    for i,name in enumerate(labels):
        ax.bar(ind+i*width, rewards[name], width, label=labels[i], color=colors[i])

    ax.set_ylabel('Reward')
    ax.set_xticks(ind+2*width)
    ax.set_xticklabels(xticks)
    ax.legend(loc='lower right', fontsize=text_size)
    fig.tight_layout()
    plt.savefig(output)
