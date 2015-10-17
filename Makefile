SHELL=/bin/bash

.SECONDORY:

dat/glove/glove.840B.300d.txt:
	if [ ! -d dat/glove ]; then mkdir -p dat/glove; fi
	wget http://www-nlp.stanford.edu/data/glove.840B.300d.txt.gz -P dat/glove
	gunzip $@.gz

dat/glove/vocab.txt: dat/glove/glove.840B.300d.txt
	th scripts/convert_word2vec.lua $< $@ dat/glove/vec.t7
dat/glove/vec.t7: dat/glove/vocab.txt

dat/qb/buzzes.csv:
	wget http://terpconnect.umd.edu/~ying/downloads/qb_emnlp_2012.tar.gz 
	tar xzf qb_emnlp_2012.tar.gz
	if [ ! -d dat/qb ]; then mkdir -p dat/qb; fi
	mv release/* dat/qb
	rm -rf release
dat/qb/questions.csv: dat/qb/buzzes.csv

dat/qb/question_buzz.txt: dat/qb/buzzes.csv dat/qb/questions.csv
	python scripts/preprocess.py --buzz dat/qb/buzzes.csv --question dat/qb/questions.csv --ans_cutoff 6 --output $@
