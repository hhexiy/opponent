#!/bin/bash

agent="QBNeuralQLearner"
update_freq=4
discount=0.9
learn_start=1000
eps=0.3
eps_end=0.1
eps_endt=500000
lr=0.0005
replay_memory=100000

agent_params="lr="$(lr)",ep="$(eps)",ep_end="$(eps_end)",ep_endt="$(eps_endt)",discount="$(discount)",learn_start="$(learn_start)",update_freq="$(update_freq)",minibatch_size=16,rescale_r=1,bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,replay_memory=$(replay_memory)"

network=""
input_file="buzz_data.txt"
content_model=content_gru_epoch22.00_4.3936.t7
test=0
max_epochs=50
hist_len=2
num_threads=8
eval_freq=0  # default to eval after each epoch
prog_freq=500
gpu=-1
simulate=0

args="-data_dir dat/qb -input_file $(input_file) -batch_size 16 -init_content $(content_model) -agent $(agent) -agent_params $(agent_params) -max_epochs $(max_epochs) -hist_len $(hist_len) -eval_freq $(eval_freq) -prog_freq $(prog_freq) -gpuid $(gpu) -threads $(num_threads) -simulate $(simulate) -test $(test)"

run:  
	th train_buzz_agent.lua -data_dir dat/qb -input_file $(input_file) -batch_size 16 -init_content $(content_model) -agent $(agent) -agent_params $(agent_params) -max_epochs $(max_epochs) -hist_len $(hist_len) -eval_freq $(eval_freq) -prog_freq $(prog_freq) -gpuid $(gpu) -threads $(num_threads) -simulate $(simulate) -test $(test) -network $(network)
