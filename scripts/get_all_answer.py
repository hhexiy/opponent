import csv, argparse, json, re, sys
from subprocess import Popen, PIPE

def zcat(path):
    if path.endswith('.gz'):
        p = Popen(['zcat', path], stdout=PIPE)
    else:
        p = Popen(['cat', path], stdout=PIPE)
    for line in p.stdout:
       yield line

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True, help='protobowl log file')
    parser.add_argument('--questions', required=True, help='processed naqt questions')
    parser.add_argument('--output_dir', default='./', help='protobowl log file')
    args = parser.parse_args()

    ans_mapping = {}
    with open(args.questions, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        fields = {v: k for k, v in enumerate(fin.readline().strip().split(','))}
        wiki_ans_field = fields['Wikipedia Page']
        ans_field = fields['Original Answer']
        for row in reader:
            ans = row[ans_field]
            wiki_ans = row[wiki_ans_field]
            ans_mapping[ans.lower()] = wiki_ans

    known_answers = {}
    unknown_answers = set()
    for line in zcat(args.log):
        item = json.loads(line.strip())
        category = item['object']['question_info']['category']
        if category.lower() == 'math':
            continue
        answer = item['object']['answer']
        answer = answer.split(' or ')[0]
        answer = answer.split(' (')[0]
        answer = answer.split(' [')[0]
        if len(answer.split()) > 15:
            continue
        print answer
        #if '[' in answer or '(' in answer:
        if len(answer) > 100:
        #if '^' in answer:
            print 'cannot find answer'
            print item
            print 'cat:', category
            print 'ans:', item['object']['answer']
            print 'text:', item['object']['question_text']
            sys.exit(0)
        guess = item['object']['guess']
        for ans in [answer, guess]:
            if ans.lower() in ans_mapping:
                known_answers[ans] = ans_mapping[ans.lower()]
            else:
                unknown_answers.add(ans)

    with open(args.output_dir + '/known_answers.txt', 'w') as fout:
        for k, v in known_answers.items():
            fout.write('%s %s\n' % (k, v))

    with open(args.output_dir + '/unknown_answers.txt', 'w') as fout:
        for a in unknown_answers:
            try:
                fout.write('%s\n' % a)
            except UnicodeEncodeError:
                print a


