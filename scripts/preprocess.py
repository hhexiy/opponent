import argparse, csv, regex, re, sys
from collections import defaultdict
import random
import numpy as np
from itertools import izip

def remove_punctuation(text):
    return regex.sub(ur"\p{P}+", " ", text)

FTP = ' FTP '
ftp = ["for 10 points", "for ten points", 'ftp']
def clean_text(q):
    # remove pronunciation guides and other formatting extras
    q = q.replace(' (*) ', ' ')
    q = q.replace('\n', '')
    q = q.replace('mt. ', 'mt ')
    q = q.replace('{', '')
    q = q.replace('}', '')
    q = q.replace('~', '')
    q = q.replace('(*)', '')
    q = q.replace('*', '')
    q = re.sub(r'\[.*?\]', '', q)
    q = re.sub(r'\(.*?\)', '', q)
    q = re.sub(r'\?+', '', q)

    for phrase in ftp:
        q = q.replace(phrase, FTP)

    # remove punctuation
    q = remove_punctuation(q)

    # simple ner (replace answers w/ concatenated versions)
    if ners:
        for ner in ners:
            q = q.replace(ner, ner.replace(' ', '_'))

    q = re.sub(r'[ ]+', ' ', q).strip()

    return q

def map_buzz_pos(buzz_pos, q):
    '''
    map buzz pos relative to original text to cleaned text
    '''
    words = q.split()
    # buzz_pos starts from 1
    words[buzz_pos-1] = words[buzz_pos-1] + ' BUZZ'
    q = clean_text(' '.join(words))
    # get new buzz position
    ss = q.split()
    buzz_pos = None
    for i, s in enumerate(ss):
        if s == 'BUZZ':
            buzz_pos = i
            break
    assert buzz_pos is not None
    return buzz_pos

def assign_fold(probs):
    # probs = p(train), p(dev), p(test)
    p = np.cumsum(probs)
    while True:
        r = random.uniform(0, 1)
        if r <= p[0]:
            yield 'train'
        elif r > p[0] and r <= p[1]:
            yield 'dev'
        else:
            yield 'test'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buzz', help='buzzes.csv')
    parser.add_argument('--question', help='questions.csv')
    parser.add_argument('--users', help='file of user ids')
    parser.add_argument('--ners', help='file of ners (answers)')
    parser.add_argument('--fake', action='store_true', help='generate artifical data')
    parser.add_argument('--maxlen', type=int, default=10, help='generate artifical data')
    parser.add_argument('--ans_cutoff', type=int, default=10, help='remove answers who have fewer than 10 questions')
    parser.add_argument('--user_cutoff', type=int, default=10, help='remove users who have answered fewer than 10 questions')
    parser.add_argument('--train_frac', type=float, default=0.8, help='training fraction')
    parser.add_argument('--dev_frac', type=float, default=0.1, help='validation fraction')
    parser.add_argument('--output', help='path of output file')
    args = parser.parse_args()
    random.seed(100)

    # load ners
    ners = []
    if args.ners:
        with open(args.ners, 'r') as fin:
            ners = [l.strip().lower() for l in fin]

    # load questions
    questions = {}
    ans_question = defaultdict(list)
    with open(args.question, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            id_ = int(row[0])
            ans = row[1].lower().replace(' ', '_')
            questions[id_] = [ans, row[4]]
            # make sure that questions.csv has unique qids
            ans_question[ans].append(id_)
    print 'load quesitons from', args.question
    print 'number of answers:', len(ans_question.keys())
    print 'number of questions:', len(questions)

    # filter answers
    for ans, q in ans_question.items():
        if len(q) < args.ans_cutoff:
            for qid in q:
                if qid in questions:
                    del questions[qid]
            del ans_question[ans]
    num_questions = sum([len(x) for x in ans_question.values()])
    assert(len(questions) == num_questions)
    print 'after removing answers with fewer than %d questions:' % args.ans_cutoff
    print 'number of answers:', len(ans_question.keys())
    print 'number of questions:', num_questions

    # load buzzes
    user_buzzes = defaultdict(dict)
    with open(args.buzz, 'r') as fin:
        for line in fin:
            ss = line.split(',')
            qid = int(ss[0])
            uid = int(ss[1])
            position = int(ss[2])
            correct = int(ss[3])
            # remove duplicated questions answered by the same user
            # use the one with a later buzz
            if qid not in user_buzzes[uid] or \
                    position > user_buzzes[uid][qid][0]:
                user_buzzes[uid][qid] = (position, correct)
    print 'load buzzes from', args.buzz
    print 'number of users:', len(user_buzzes)
    print 'number of questions answered by at least one user:', len(set([x for qlist in user_buzzes.values() for x in qlist]))

    # filter buzzes
    for uid, qdict in user_buzzes.items():
        if len(qdict) < args.user_cutoff:
            del user_buzzes[uid]
    buzzes = defaultdict(list)
    for uid, qdict in user_buzzes.items():
        for qid, buzz in qdict.items():
            position, correct = buzz
            buzzes[qid].append([uid, position, correct])
    print 'after removing users answered fewer than %d questions:' % args.user_cutoff
    print 'number of users:', len(user_buzzes)
    print 'number of questions answered by at least one user:', len(buzzes)
    print 'average user per question:', sum([len(x) for x in buzzes.values()]) / float(len(buzzes))

    # split to train, dev, test
    # make sure each answer has questions in each set
    probs = [args.train_frac, args.dev_frac, 1.0-args.dev_frac-args.train_frac]
    fold_count = defaultdict(int)
    for ans, q in ans_question.items():
        for qid, fold in izip(q, assign_fold(probs)):
            questions[qid].append(fold)
            fold_count[fold] += 1
            qtext = questions[qid][1].strip().lower()
            # update buzz position relative to cleaned text
            for i, buzz in enumerate(buzzes[qid]):
                uid, buzz_pos, correct = buzz
                if qid == 117294:
                    new_buzz_pos = map_buzz_pos(buzz_pos, qtext)
                    buzz[1] = new_buzz_pos
                    assert(buzzes[qid][i][1] == new_buzz_pos)
            questions[qid][1] = clean_text(qtext)
    for fold, count in fold_count.items():
        print '%s: %d' % (fold, count)

    # print
    qids = questions.keys()
    # sort questions by length (for minimum padding)
    qids = sorted(qids, key=lambda i: len(questions[i][1].split()))
    #random.shuffle(qids)
    num_nobuzz = 0
    with open(args.output, 'w') as fout:
        for qid in qids:
            ans, qtext, fold = questions[qid]
            if len(buzzes[qid]) == 0:
                num_nobuzz += 1
            buzz = '|'.join(['-'.join([str(x) for x in buzz]) for buzz in buzzes[qid]])
            fout.write('%d,%s,%s,%s,%s\n' % (qid, ans, fold, qtext, buzz))
            #sys.stdout.write('%d,%s,%s,%s,%s\n' % (qid, ans, fold, qtext, buzz))
            #break
    print 'number of questions without buzz:', num_nobuzz
