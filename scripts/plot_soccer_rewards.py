import sys
import matplotlib
matplotlib.use('PDF')
text_size = 20
matplotlib.rcParams['xtick.labelsize'] = text_size
matplotlib.rcParams['ytick.labelsize'] = text_size
matplotlib.rcParams['axes.labelsize'] = text_size
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 4 or len(sys.argv) % 2 != 1:
        print 'Usage: <output> <exp> <reward file> <model name> ...'
        sys.exit()

    output = sys.argv[1]
    exp = sys.argv[2]
    reward_files = sys.argv[3::2]
    model_names = sys.argv[4::2]
    assert(len(reward_files) == len(model_names))
    rewards = {}
    #defensive = (0.1, 0.3, 0.5, 0.7, 0.9)
    defensive = (0.5,)
    for name, f in zip(model_names, reward_files):
        max_r = []
        mean_r = []
        # percentage of defensive agents
        for d in defensive:
            fd = f.replace('DEF', str(d))
            with open(fd, 'r') as fin:
                r = [float(line.strip()) for line in fin]
                max_r.append(np.max(r))
                mean_r.append(np.mean(r[-10:]))
        rewards[name] = {'max':max_r, 'mean':mean_r}
    for name, v in rewards.items():
        print name, v['max'], v['mean']
    sys.exit()


    if exp == 'model':
        line_style_max = {'DQN-world':'g-', 'DRON-concat':'b-', 'DRON-MoE':'r-'}
        line_style_mean = {'DQN-world':'g--', 'DRON-concat':'b--', 'DRON-MoE':'r--'}
    elif exp == 'moe_sp':
        line_style_max = {'DRON-MoE':'g-', 'DRON-MoE+action':'b-', 'DRON-MoE+type':'r-'}
        line_style_mean = {'DRON-MoE':'g--', 'DRON-MoE+action':'b--', 'DRON-MoE+type':'r--'}
    fig, ax = plt.subplots()
    ax.set_xlabel('Percentage of defensive opponents')
    ax.set_ylabel('Average reward')
    line_max = []
    line_mean = []
    for name, reward in rewards.items():
        #print name, max(reward), np.mean(reward[-10:])
        x = defensive
        l, = plt.plot(x, reward['max'], line_style_max[name], linewidth=2, label=name)
        line_max.append(l)
        l, = plt.plot(x, reward['mean'], line_style_mean[name], linewidth=2, label=name)
        line_mean.append(l)
    legend1 = plt.legend(handles=line_max, loc='upper right', fontsize=text_size)
    ax = plt.gca().add_artist(legend1)
    plt.legend(handles=line_mean, loc='lower left', fontsize=text_size)
    plt.tight_layout()
    fig.savefig(output)
