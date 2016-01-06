from os import listdir
import os.path, sys
import argparse
import csv
import cPickle as pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True, help='protobowl processed file(.pkl) directory')
    parser.add_argument('--output_dir', default='./', help='protobowl log file')
    args = parser.parse_args()

    question_files = []
    buzz_files = []
    for f in listdir(args.log):
        f = os.path.join(args.log, f)
        if os.path.isfile(f):
            if f.endswith('questions.pkl'):
                question_files.append(f)
            elif f.endswith('buzzes.pkl'):
                buzz_files.append(f)
    print 'question files:', question_files
    print 'buzz files:', buzz_files

    questions = {}
    for f in question_files:
        with open(f, 'rb') as fin:
            q = pickle.load(fin)
            questions.update(q)
    print '# questions:', len(questions)

    buzzes = []
    for f in buzz_files:
        with open(f, 'rb') as fin:
            b = pickle.load(fin)
            buzzes.extend(b)
    print '# buzzes:', len(buzzes)

    if len(questions) > 0:
        with open('%s/questions.csv' % args.output_dir, 'w') as fout:
            writer = csv.DictWriter(fout, delimiter=',', fieldnames=['Question ID','Fold','Category','Original Answer','Wikipedia Page','Text'])
            writer.writeheader()
            for i, qst in questions.items():
                writer.writerow({'Question ID':qst[0], 'Fold':'', 'Category':qst[1], 'Original Answer':'', 'Wikipedia Page':qst[2].encode('utf8'), 'Text':' ||| '.join([x.encode('utf8') for x in qst[3]])})

    if len(buzzes) > 0:
        with open('%s/buzzes.csv' % args.output_dir, 'w') as fout:
            writer = csv.DictWriter(fout, delimiter=',', fieldnames=['Question ID','User ID','Buzz Position', 'Correct', 'Guess'])
            writer.writeheader()
            for buzz in buzzes:
                writer.writerow({'Question ID':buzz[0], 'User ID':buzz[1], 'Buzz Position':buzz[2], 'Correct':1 if buzz[4] else 0, 'Guess':buzz[3].encode('utf8')})

