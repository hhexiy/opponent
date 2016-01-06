SHELL=/bin/bash

.SECONDORY:

#================= preprocess ===================
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

dat/qb/buzz_data.txt: dat/qb1/buzzes.csv dat/qb/naqt_questions.csv
	python scripts/preprocess.py --buzz $< --question $(word 2, $^) --ans_cutoff 5 --output dat/qb --content_train_frac 0.9
dat/qb/content_data.txt: dat/qb/buzz_data.txt
#================================================

#================= run_cpu ===================
agent="QBNeuralQLearner"
update_freq=4
discount=0.9
learn_start=1000
eps=0.3
eps_end=0.1
eps_endt=500000
lr=0.0005
replay_memory=100000
batch_size=64

agent_params="lr="$(lr)",ep="$(eps)",ep_end="$(eps_end)",ep_endt="$(eps_endt)",discount="$(discount)",learn_start="$(learn_start)",update_freq="$(update_freq)",minibatch_size=$(batch_size),rescale_r=1,bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,replay_memory=$(replay_memory)"

network=""
data_dir=dat/protobowl
input_file="buzz_data.txt"
content_model=content_gru_epoch20.00_3.8778.t7
test=0
max_epochs=50
hist_len=2
num_threads=8
eval_freq=0  # default to eval after each epoch
prog_freq=500
gpu=-1
simulate=0
supervise=0
savefile=""

args="-data_dir $(data_dir) -input_file $(input_file) -batch_size $(batch_size) -init_content $(content_model) -agent $(agent) -agent_params $(agent_params) -max_epochs $(max_epochs) -hist_len $(hist_len) -eval_freq $(eval_freq) -prog_freq $(prog_freq) -gpuid $(gpu) -threads $(num_threads) -simulate $(simulate) -test $(test)"

run:  
	th train_buzz_agent.lua -data_dir $(data_dir) -input_file $(input_file) -batch_size $(batch_size) -init_content $(content_model) -agent $(agent) -agent_params $(agent_params) -max_epochs $(max_epochs) -hist_len $(hist_len) -eval_freq $(eval_freq) -prog_freq $(prog_freq) -gpuid $(gpu) -threads $(num_threads) -simulate $(simulate) -test $(test) -network $(network) -best -supervise $(supervise) -savefile $(savefile)
#=============================================

#================= rnn buzzer ===================
train_buzz_rnn:
	th train_buzz_rnn.lua -data_dir $(data_dir) -input_file $(input_file) -batch_size $(batch_size) -init_content $(content_model) -print_every $(prog_freq) -gpuid $(gpu) -threads $(num_threads) -test $(test) -best -eval_val_every $(eval_freq) -max_epochs $(max_epochs)

test_buzz_rnn:
	th train_buzz_rnn.lua -data_dir $(data_dir) -input_file $(input_file) -batch_size $(batch_size) -init_content $(content_model) -print_every $(prog_freq) -gpuid $(gpu) -threads $(num_threads) -test $(test) -best -eval_val_every $(eval_freq) -max_epochs $(max_epochs) -init_from $(init_from)
#=============================================
