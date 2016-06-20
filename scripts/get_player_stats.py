import sys, csv
import matplotlib
text_size = 20
matplotlib.rcParams['xtick.labelsize'] = text_size
matplotlib.rcParams['ytick.labelsize'] = text_size
matplotlib.rcParams['axes.labelsize'] = text_size
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    buzz_file = sys.argv[1]
    users = {}
    with open(buzz_file, 'r') as fin:
        # header
        fields = {v: k for k, v in enumerate(fin.readline().strip().split(','))}
        qid_field = fields['Question ID']
        uid_field = fields['User ID']
        pos_field = fields['Buzz Position']
        correct_field = fields['Correct']
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            qid = int(row[qid_field])
            user_id = row[uid_field]
            position = int(float(row[pos_field]))
            correct = int(row[correct_field])
            if user_id not in users:
                users[user_id] = {}
                for k in ['total', 'correct', 'position']:
                    users[user_id][k] = 0
                users[user_id]['questions'] = set()
            users[user_id]['total'] += 1
            users[user_id]['correct'] += correct
            users[user_id]['position'] += position
            users[user_id]['questions'].add(qid)

    sorted_users = sorted(users.items(), key=lambda x: x[1]['total'], reverse=True)
    cutoff = 200
    sorted_users = filter(lambda x: x[1]['total'] >= cutoff, sorted_users)
    all_answered_questions = [x for user in sorted_users for x in user[1]['questions']]
    answered_questions = set(all_answered_questions)

    # overall stats
    print 'after removing users who answered fewer than %d questions' % cutoff
    print 'number of users:', len(users)
    print 'number of questions answered:', len(answered_questions)

    # sorted stats
    print '{:<10}{:<10}{:<7}{:<7}{:<7}'.format('id', 'tot', '%', 'acc', 'pos')
    str_format = '{:<10}{:<10}{:<7.2f}{:<7.2f}{:<7.2f}'
    total = float(len(all_answered_questions))
    for i, user in enumerate(sorted_users):
        tot = user[1]['total']
        user[1]['position'] /= float(tot)
        user[1]['correct'] /= float(tot)
        if i < 10:
            print str_format.format(user[0][:6], tot, tot / total,  user[1]['correct'], user[1]['position'])

    # scatter plot
    acc = [u[1]['correct'] for u in sorted_users]
    pos = [u[1]['position'] for u in sorted_users]
    s = [(10*u[1]['total']/total*100)**2 for u in sorted_users]
    num_answered = np.array([u[1]['total'] for u in sorted_users])
    fig, ax = plt.subplots()
    cs = ax.scatter(pos, acc, s=s, c=num_answered, cmap=plt.cm.seismic, alpha=0.8)
    fig.colorbar(cs, ax=ax)
    ax.set_xlabel('Number of words revealed', fontsize=20)
    ax.set_ylabel('Accuracy', fontsize=20)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('figures/acc_pos_scatter.pdf')
