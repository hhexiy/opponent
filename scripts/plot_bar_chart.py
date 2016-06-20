import sys
import matplotlib
text_size = 20
matplotlib.rcParams['xtick.labelsize'] = text_size
matplotlib.rcParams['ytick.labelsize'] = text_size
matplotlib.rcParams['axes.labelsize'] = text_size
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict

def bs_avg(sample, stat=np.mean):
    n = len(sample)
    sample_stats = []
    for i in range(16):
        bs_sample = np.random.choice(sample, n, True)
        sample_stats.append(stat(bs_sample))
    sample_stats.sort()
    return np.mean(sample_stats), sample_stats[1], sample_stats[-2]

def get_stats(filename, stats):
    with open(filename, 'r') as fin:
        rewards = np.array([float(line.strip().split(',')[-1]) for line in fin])
    mean, lo, hi = bs_avg(rewards)
    stats['lo'].append(mean-lo)
    stats['hi'].append(hi-mean)
    stats['mean'].append(mean)
    #stats['mean'].append(np.mean(rewards))

if __name__ == '__main__':
    output = sys.argv[1]
    game = sys.argv[2]
    if game == 'qb':
        baseline = 'results/QBNeuralQLearner.log'
        basename = 'results/QBONeuralQLearner'
    else:
        baseline = 'results/SoccerQ-H-0.5.log'
        basename = 'results/SoccerQO-H-0.5'

    # compare four models
    models = {'DQN':defaultdict(list), 'R':defaultdict(list), 'R+action':defaultdict(list), 'R+group':defaultdict(list)}
    N = len(models)

    reward_id = -2 if game == 'qb' else -1
    get_stats(baseline, models['DQN'])
    for k in models['DQN']:
        models['DQN'][k] = models['DQN'][k]*3

    for n in [2,3,4]:
        if game == 'qb':
            get_stats(basename+'_e%d_2'%n+'.log', models['R'])
            get_stats(basename+'_e%d_2'%n+'_ma.log', models['R+action'])
            get_stats(basename+'_e%d_2'%n+'_mg.log', models['R+group'])
        else:
            get_stats(basename+'_e%d'%n+'.log', models['R'])
            get_stats(basename+'_e%d'%n+'_ma.log', models['R+action'])
            get_stats(basename+'_e%d'%n+'_mg.log', models['R+group'])

    fig, ax = plt.subplots()
    width = 0.2
    rects = {}
    ind = np.arange(3)
    names = ['DQN', 'R', 'R+action', 'R+group']
    labels = ['DQN-world', 'DRON-MoE(R)', 'DRON-MoE(R+action)', 'DRON-MoE(R+type)']
    #colors = ['b', 'r', 'g', 'y']
    colors = [cm.rainbow(i) for i in (0.1, 0.4, 0.7, 1.0)]
    for i,name in enumerate(names):
        stat = models[name]
        yerr = np.array([stat['lo'], stat['hi']])
        rects[i] = ax.bar(ind+i*width, stat['mean'], width, label=labels[i], color=colors[i], yerr=yerr)

    ax.set_ylabel('Reward')
    ax.set_xticks(ind+2*width)
    ax.set_xticklabels(('K=2', 'K=3', 'K=4'))
    ax.legend(loc='lower right', fontsize=text_size)
    fig.tight_layout()
    plt.savefig(output)
