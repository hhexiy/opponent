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
cmd:option('-embedding', '', 'directory of pretrained word embeddings')
cmd:option('-init_content', '', 'pretrained content model parameters from checkpoint at this path')
cmd:option('-hist_len', 1, 'history length of state features')
cmd:option('-best', false, 'load best model or current model')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',20,'in number of epochs, when to start decaying the learning rate')
-- dqn
cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-simulate', 0, 'simulate players or not')
cmd:option('-supervise', 0, 'using supervised signal as reward during training')
cmd:option('-checkpoint_dir', '/fs/clip-scratch/hhe/opponent/cv', 'output directory where checkpoints get written')
cmd:option('-savefile','','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', 'QBNeuralQLearner', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 0, 'frequency of greedy evaluation')

cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 0, 'number of BLAS threads')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')
cmd:option('-test',0,'evaluate on test set')

cmd:text()

opt = cmd:parse(arg)
-- dqn package use gpu
opt.gpu = opt.gpuid
if opt.savefile == '' then opt.savefile = opt.agent end
opt.supervise = opt.supervise == 1 and true or false

--- General env setup based on opt
require 'setup'
env_setup()
local agent, game_env = qb_dqn_setup()
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
-- this is just training reward
local reward_history = {}
local eval_reward_history = {}
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
    game_env:reset(split_index) 
    print(string.format('============= eval split %d =============', split_index))
    test = test or false
    local total_reward = 0
    local total_length = 0
    local nepisodes = 0
    local episode_reward = 0
    local episode_length = 0
    local group_reward = torch.zeros(game_env.num_player_groups)
    local group_length = torch.zeros(game_env.num_player_groups)
    local group_nepisodes = torch.zeros(game_env.num_player_groups)
    local gating = (string.find(opt.agent, 'QBO', 1) and (agent.n_experts > 0)) and true or false
    local group_weights = gating and torch.FloatTensor(game_env.num_player_groups, agent.n_experts):zero() or nil
    local log = opt.test == 1 and io.open(opt.savefile .. '.log', 'w') or nil
    local eval_time = sys.clock()

    local n = game_env.num_buzzes[split_index]
    for i=1,n do
        state, terminal, reward = game_env:new_game(split_index, true)
        local episode_gating_weights
        if test and gating then
            episode_gating_weights = torch.FloatTensor(agent.n_experts):zero()
        end
        while true do
            local action_index = agent:perceive(reward, state, terminal, true, 0.0)
            if not terminal then
                if test and gating then
                    episode_gating_weights:add(agent.gating_weights)
                end
                state, terminal, reward = game_env:step(actions[action_index])
                -- record every reward
                episode_reward = episode_reward + reward
                episode_length = episode_length + 1
            else break end
        end
        local group = game_env.player_group
        if test and gating then 
            episode_gating_weights:div(episode_length)
            group_weights[group]:add(episode_gating_weights)
        end
        -- group stats
        if game_env.num_player_groups > 1 then
            group_reward[group] = group_reward[group] + episode_reward
            group_length[group] = group_length[group] + episode_length
            group_nepisodes[group] = group_nepisodes[group] + 1
        end
        -- write log
        if log ~= nil then
            log:write(string.format('%d,%d,%d,%.2f,%d\n', game_env.qid, game_env.player_id, group, episode_reward, episode_length)) 
        end
        -- overall stats
        total_reward = total_reward + episode_reward
        total_length = total_length + episode_length
        episode_reward = 0
        episode_length = 0
        nepisodes = nepisodes + 1
    end
    assert(nepisodes == n)
    eval_time = sys.clock() - eval_time
    total_reward = total_reward / nepisodes
    total_length = total_length / nepisodes
    if game_env.num_player_groups > 1 then
        for g=1,game_env.num_player_groups do
            group_reward[g] = group_reward[g] / group_nepisodes[g]
            group_length[g] = group_length[g] / group_nepisodes[g]
            if group_weights ~= nil then
                group_weights[g]:div(group_nepisodes[g])
            end
        end
    end
    if log ~= nil then io.close(log) end

    if not test then 
        start_time = start_time + eval_time
        agent:compute_validation_statistics()
        local ind = #reward_history+1
        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            print('new best network on dev set')
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

        print(string.format(
            'epsilon: %.2f, lr: %G\n' ..
            'reward: %.2f, episode length: %.2f\n' ..
            'training time: %ds, eval time: %ds, ' ..
            'num. ep.: %d',
            agent.ep, agent.lr, total_reward, total_length, time_dif,
            eval_time, 
            n))
        game_env:report_error_analysis()
    else
        print(string.format(
            '\nreward: %.2f, episode length: %.2f, epsilon: %.2f, ' ..
            'eval time: %ds, num. ep.: %d',
            total_reward, total_length, agent.ep, 
            eval_time, n))
        game_env:report_error_analysis()
        if game_env.num_player_groups > 1 then
            for g=1,game_env.num_player_groups do
                print('------ player group ------' .. g)
                print(string.format(
                    'reward: %.2f, episode length: %.2f,  num. ep.: %d',
                    group_reward[g], group_length[g], group_nepisodes[g]))
                if group_weights ~= nil then
                    for e=1,agent.n_experts do
                        io.write(string.format('%.4f ', group_weights[g][e]))
                    end
                    print('')
                end
                game_env:report_error_analysis(g)
            end
        end
        -- record eval reward history
        -- TODO: remove split_index
        if eval_reward_history[split_index] == nil then
            eval_reward_history[split_index] = {}
        end
        local ind = #eval_reward_history[split_index] + 1
        eval_reward_history[split_index][ind] = total_reward
    end
    print('=========================================')
end

--------------------------- testing --------------------------------
if opt.test == 1 then
    eval_split(3, true)
    os.exit()
end

--------------------------- training --------------------------------
-- use num_questions instead of num_buzzes because 
-- only one buzz is sampled for each question during training
local ntrain = game_env.num_questions[1]
--local ntrain = game_env.num_buzzes[1]
if opt.eval_freq == 0 then
    opt.eval_freq = ntrain
end
opt.save_freq = 5*opt.eval_freq
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

    -- learning rate decay
    if i % ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            agent.lr_start = agent.lr_start * opt.learning_rate_decay
            agent.lr_end = agent.lr_start
            print('decayed learning rate by a factor ' .. opt.learning_rate_decay .. ' to ' .. agent.lr_start)
        end
    end

    -- evaluation
    if i % opt.eval_freq == 0 then
        --game_env.debug = true
        eval_split(2)
        --eval_split(3, true)
        --game_env.debug = false
    end
    if i % ntrain == 0 then
        eval_split(3, true)
    end

    if i % opt.save_freq == 0 or i == max_num_games then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filename = string.format('%s/%s_lr%.6f_disc%.2f_epoch%.2f.t7', opt.checkpoint_dir, opt.savefile, agent.lr, agent.discount, epoch)
        print('saving checkpoint to ' .. filename)
        torch.save(filename, {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                reward_history = reward_history,
                                eval_reward_history = eval_reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
        if opt.saveNetworkParams then
            local nets = {network=w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, r, s2, term
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
        io.flush()
        collectgarbage()
    end

    if i % 1000 == 0 then
        collectgarbage()
    end
end



