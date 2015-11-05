SHELL=/bin/bash

.SECONDORY:

dat/glove/glove.840B.300d.txt:
	if [ ! -d dat/glove ]; then mkdir -p dat/glove; fi
	wget http://www-nlp.stanford.edu/data/glove.840B.300d.txt.gz -P dat/glove
	gunzip $@.gz

dat/glove/vocab.txt: dat/glove/glove.840B.300d.txt
	th scripts/convert_word2vec.lua $< $@ dat/glove/vec.t7
dat/glove/vec.t7: dat/glove/vocab.txt

dat/qb1/buzzes.csv:
	wget http://terpconnect.umd.edu/~ying/downloads/qb_emnlp_2012.tar.gz 
	tar xzf qb_emnlp_2012.tar.gz
	if [ ! -d dat/qb1 ]; then mkdir -p dat/qb1; fi
	mv release/* dat/qb1
	rm -rf release
dat/qb1/questions.csv: dat/qb1/buzzes.csv

dat/qb1/question_buzz.txt: dat/qb1/buzzes.csv dat/qb1/questions.csv
	python scripts/preprocess.py --buzz dat/qb1/buzzes.csv --question dat/qb1/questions.csv --ans_cutoff 6 --output $@ --format qb1
