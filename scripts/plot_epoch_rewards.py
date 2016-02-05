import sys
import matplotlib
matplotlib.use('PDF')
text_size = 20
matplotlib.rcParams['xtick.labelsize'] = text_size
matplotlib.rcParams['ytick.labelsize'] = text_size
matplotlib.rcParams['axes.labelsize'] = text_size
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) % 2 != 0:
        print 'Usage: <output> <reward file> <model name> ...'
        sys.exit()

    output = sys.argv[1]
    reward_files = sys.argv[2::2]
    model_names = sys.argv[3::2]
    assert(len(reward_files) == len(model_names))
    rewards = {}
    for name, f in zip(model_names, reward_files):
        with open(f, 'r') as fin:
            rewards[name] = [float(line.strip()) for line in fin]

    line_style = {'DQN-self':'k--', 'DQN-world':'g--', 'DRON-concat':'b-', 'DRON-MoE':'r-'}
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Average reward')
    for name, reward in rewards.items():
        print name, max(reward), np.mean(reward[-10:])
        x = range(1,len(reward)+1)
        if name in line_style:
            ax.plot(x, reward, line_style[name], linewidth=2, label=name)
        else:
            ax.plot(x, reward, linewidth=2, label=name)
    ax.legend(loc='lower right', fontsize=text_size)
    plt.tight_layout()
    fig.savefig(output)
