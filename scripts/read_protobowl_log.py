import argparse, json, re, math, sqlite3, sys
from subprocess import Popen, PIPE
import os.path
import datetime
import cPickle as pickle

def zcat(path):
    if path.endswith('.gz'):
        p = Popen(['zcat', path], stdout=PIPE)
    else:
        p = Popen(['cat', path], stdout=PIPE)
    for line in p.stdout:
       yield line

def get_col_names(c, table_name):
    cols = {}
    for row in c.execute('PRAGMA table_info(%s)' % table_name):
        cols[row[1]] = row[0]
    return cols

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True, help='protobowl log file')
    parser.add_argument('--output_dir', default='./', help='protobowl log file')
    parser.add_argument('--db', default=None, required=True, help='protobowl database')
    args = parser.parse_args()

    assert os.path.isfile(args.db)
    conn = sqlite3.connect(args.db)
    c = conn.cursor()
    questions_cols = get_col_names(c, 'questions')
    text_cols = get_col_names(c, 'text')

    # qid, category, answer, wiki_page, text
    questions = {}
    # qid, uid, pos, correct, guess
    buzzes = []
    num_lines = 0
    missing_protobowl_qids = set()
    missing_ans = set()
    num_missing = 0
    num_no_ans = 0
    total_buzz_pos = 0
    for i, line in enumerate(zcat(args.log)):
        if i % 1000 == 0:
            print i, datetime.datetime.now()
            print len(missing_protobowl_qids), len(missing_ans), len(questions), len(buzzes)

        num_lines += 1
        item = json.loads(line.strip())
        # get question
        protobowl_qid = item['object']['qid']
        if protobowl_qid not in questions:
            row = c.execute('select * from questions where protobowl == "%s"' % (protobowl_qid)).fetchall()
            if not row:
                missing_protobowl_qids.add(protobowl_qid)
                num_missing += 1
                continue
            row = row[0]
            answer = row[questions_cols['page']]
            if not answer:
                missing_ans.add(protobowl_qid)
                num_no_ans += 1
                continue
            qid = row[questions_cols['id']]
            category = row[questions_cols['category']]

            sents = []
            for row in c.execute('select * from text where question == %d' % qid):
                sents.append(row[text_cols['raw']])
            questions[protobowl_qid] = (qid, category, answer, sents)
        else:
            qid, category, answer, sents = questions[protobowl_qid]

        # get buzz
        uid = item['object']['user']['id']
        guess = item['object']['guess']
        correct = item['object']['ruling']
        # approximate buzz position
        time_elapsed = item['object']['time_elapsed']
        time_remaining = item['object']['time_remaining']
        text = ' '.join(sents)
        question_len = len(text.split())
        buzz_pos = int(max(1, math.floor(float(time_elapsed) / (time_elapsed + time_remaining) * question_len)))
        total_buzz_pos += buzz_pos
        assert buzz_pos <= question_len
        buzzes.append((qid, uid, buzz_pos, guess, correct))

    print '# processed:', num_lines
    print 'missing:', len(missing_protobowl_qids), num_missing
    print 'no answer:', len(missing_ans), num_no_ans
    print '# questions:', len(questions)
    print '# buzzes:', len(buzzes)
    print 'avg buzz position:', float(total_buzz_pos) / len(buzzes)

    logname = os.path.basename(args.log)
    #with open('%s/%s.questions.pkl' % (args.output_dir, logname), 'wb') as fout:
    #    pickle.dump(questions, fout)
    with open('%s/%s.buzzes.pkl' % (args.output_dir, logname), 'wb') as fout:
        pickle.dump(buzzes, fout)

