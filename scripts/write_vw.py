import sys

if __name__ == '__main__':
    file_in = sys.argv[1]
    fold = sys.argv[2]
    ans_mapping = {}
    with open(file_in, 'r') as fin:
        for line in fin:
            ss = line.strip().split(',')
            f = ss[2]
            if f == fold:
                ans = ss[1]
                if ans not in ans_mapping:
                    ans_mapping[ans] = len(ans_mapping) + 1
                text = ss[3]
                toks = text.split()
                for i, tok in enumerate(toks):
                    print '%d |f %s' % (ans_mapping[ans], ' '.join(toks[:i+1]))


