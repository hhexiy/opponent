--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Buzz Agent in Quizbowl Environment:')
cmd:text()
cmd:text('Options:')

-- data
cmd:option('-data_dir','dat/qb','data directory. Should contain the file input.txt with input data')
cmd:option('-input_file','input.txt','data file name')
cmd:option('-batch_size',50,'number of questions to process in parallel')
-- model params
cmd:option('-embedding', 'dat/glove', 'directory of pretrained word embeddings')
cmd:option('-init_content', '', 'pretrained content model parameters from checkpoint at this path')
cmd:option('-hist_len', 1, 'history length of state features')
-- dqn
cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', 'QBNeuralQLearner', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 0, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')

cmd:text()

opt = cmd:parse(arg)
-- dqn package use gpu
opt.gpu = opt.gpuid

--- General env setup based on opt
require 'setup'
env_setup()
local agent, game_env = dqn_setup()
local actions = agent.actions

function test_framework()
    local state, terminal, reward = game_env:new_game(1)
    for i=1,2 do
        if not terminal then
            state, terminal, reward = game_env:step(qb.WAIT)
        end
    end
    if not terminal then
        game_env:step(qb.BUZZ)
    end
end

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

---------------------------------------------------

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward = 0
local total_length = 0
local nepisodes = 0
local episode_reward = 0
local episode_length = 0

function reset_stats()
    total_reward = 0
    total_length = 0
    nepisodes = 0
    episode_reward = 0
    episode_length = 0
end

function eval_split(split_index, test)
    -- reset seed for sample test players
    torch.manualSeed(opt.seed)
    print(string.format('============= eval split %d =============', split_index))
    test = test or false
    reset_stats()
    local eval_time = sys.clock()

    local n = game_env.num_examples[split_index]
    for i=1,n do
        state, terminal, reward = game_env:new_game(split_index, true)
        while true do
            local action_index = agent:perceive(reward, state, terminal, true, 0.0)
            if not terminal then
                state, terminal, reward = game_env:step(actions[action_index])
                -- record every reward
                episode_reward = episode_reward + reward
                episode_length = episode_length + 1
            else break end
        end
        total_reward = total_reward + episode_reward
        total_length = total_length + episode_length
        episode_reward = 0
        episode_length = 0
    end
    eval_time = sys.clock() - eval_time
    total_reward = total_reward / n
    total_length = total_length / n

    if not test then 
        start_time = start_time + eval_time
        agent:compute_validation_statistics()
        local ind = #reward_history+1
        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            'epsilon: %.2f, lr: %G\n' ..
            'reward: %.2f, episode length: %.2f\n' ..
            'training time: %ds, training rate: %dfps, eval time: %ds, ' ..
            'eval rate: %dfps,  num. ep.: %d',
            agent.ep, agent.lr, total_reward, total_length, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            n))
    else
        print(string.format(
            '\nreward: %.2f, epsilon: %.2f, ' ..
            'eval time: %ds, eval rate: %dfps,  num. ep.: %d',
            total_reward, agent.ep, 
            eval_time, opt.actrep*opt.eval_steps/eval_time, n))
    end
end


--------------------------- training --------------------------------
local ntrain = game_env.num_examples[1]
if opt.eval_freq == 0 then
    opt.eval_freq = ntrain
end
local max_num_games = opt.max_epochs * ntrain 
local epoch = 0
local state, terminal, reward
for i=1,max_num_games do
    epoch = i / ntrain
    state, terminal, reward = game_env:new_game(1) 
    while true do
        local priority = reward ~= 0 and true or false 
        local action_index = agent:perceive(reward, state, terminal, false, nil, priority)
        if not terminal then
            state, terminal, reward = game_env:step(actions[action_index])
            episode_reward = episode_reward + reward
            episode_length = episode_length + 1
        else break end
    end
    total_reward = total_reward + episode_reward
    total_length = total_length + episode_length
    episode_reward = 0
    episode_length = 0
    nepisodes = nepisodes + 1
    --if i == 3 then
    --    os.exit()
    --end

    -- progress report
    if i % opt.prog_freq == 0 then
        print(string.format("%d/%d (epoch %.3f), lr: %G, epsilon: %.2f\n" .. 
        "reward: %.2f, episode length: %.2f, num. ep.: %d", 
        i, max_num_games, epoch, agent.lr, agent.ep,
        total_reward/nepisodes, total_length/nepisodes, nepisodes))
        --agent:report()
        reset_stats()
        collectgarbage()
    end

    -- evaluation
    if i % opt.eval_freq == 0 then
        --game_env.debug = true
        eval_split(2)
        --game_env.debug = false
    end

    if i % opt.save_freq == 0 or i == max_num_games then
    end

    if i % 1000 == 0 then
        collectgarbage()
    end
end



