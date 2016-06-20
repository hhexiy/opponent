import sys
import matplotlib
text_size = 20
matplotlib.rcParams['xtick.labelsize'] = text_size
matplotlib.rcParams['ytick.labelsize'] = text_size
matplotlib.rcParams['axes.labelsize'] = text_size
import matplotlib.pyplot as plt
import numpy as np

def bs_avg(sample, stat=np.mean):
    n = len(sample)
    sample_stats = []
    for i in range(16):
        bs_sample = np.random.choice(sample, n, True)
        sample_stats.append(stat(bs_sample))
    sample_stats.sort()
    return np.mean(sample_stats), sample_stats[1], sample_stats[-2]

if __name__ == '__main__':
    predict_log = sys.argv[1]
    acc = []
    with open(predict_log, 'r') as fin:
        for line in fin:
            correct = [int(x) for x in line.strip().split()]
            if len(correct) >= 20:
                acc.append(bs_avg(correct, np.mean))
    x = np.array(range(len(acc)))
    y = np.array([a[0] for a in acc])
    err_lo = np.array([a[0]-a[1] for a in acc])
    err_hi = np.array([a[2]-a[0] for a in acc])
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of words revealed')
    ax.set_ylabel('Accuracy')
    #ax.set_xlim([0,120])
    #ax.errorbar(range(len(acc)), [x[0] for x in acc], yerr=[err_lo, err_hi])
    #ax.fill_between(x, y-err_lo, y+err_hi, alpha=0.5, interpolate=True)
    ax.plot(x, y, linewidth=2)
    ax.plot(x, y-err_lo, 'g--', label='95\% confidence interval')
    ax.plot(x, y+err_hi, 'g--')
    ax.legend(loc='lower right', fontsize=text_size)
    plt.tight_layout()
    fig.savefig('figures/content_acc.pdf')
