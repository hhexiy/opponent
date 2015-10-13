import sys
from collections import defaultdict
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    buzz_file = sys.argv[1]
    users = {}
    with open(buzz_file, 'r') as fin:
        for line in fin:
            ss = line.split(',')
            user_id = int(ss[1])
            position = int(ss[2])
            correct = int(ss[3])
            if user_id not in users:
                users[user_id] = {}
                for k in ['total', 'correct', 'position']:
                    users[user_id][k] = 0
                users[user_id]['questions'] = set()
            users[user_id]['total'] += 1
            users[user_id]['correct'] += correct
            users[user_id]['position'] += position
            users[user_id]['questions'].add(int(ss[0]))

    sorted_users = sorted(users.items(), key=lambda x: x[1]['total'], reverse=True)
    cutoff = 10
    sorted_users = filter(lambda x: x[1]['total'] >= cutoff, sorted_users)
    answered_questions = set([x for user in sorted_users for x in user[1]['questions']])

    # overall stats
    print 'after removing users who answered fewer than %d questions' % cutoff
    print 'number of users:', len(users)
    print 'number of questions answered:', len(answered_questions)

    # sorted stats
    print '{:<6}{:<6}{:<7}{:<7}'.format('id', 'tot', 'acc', 'pos')
    str_format = '{:<6}{:<6}{:<7.2f}{:<7.2f}'
    for i, user in enumerate(sorted_users):
        tot = user[1]['total']
        user[1]['position'] /= float(tot)
        user[1]['correct'] /= float(tot)
        if i < 10:
            print str_format.format(user[0], tot, user[1]['correct'], user[1]['position'])

    # scatter plot
    acc = [u[1]['correct'] for u in sorted_users]
    pos = [u[1]['position'] for u in sorted_users]
    s = [0.5*2**(u[1]['total']/100.0) for u in sorted_users]
    plt.scatter(pos, acc, s=s)
    plt.savefig('acc_pos_scatter.pdf')
