SHELL=/bin/bash

.SECONDORY:

dat/glove/glove.840B.300d.txt:
	if [ ! -d dat/glove ]; then mkdir -p dat/glove; fi
	wget http://www-nlp.stanford.edu/data/glove.840B.300d.txt.gz -P dat/glove
	gunzip $@.gz

dat/glove/vocab.txt: dat/glove/glove.840B.300d.txt
	th scripts/convert_word2vec.lua $< $@ dat/glove/vec.t7
dat/glove/vec.t7: dat/glove/vocab.txt

