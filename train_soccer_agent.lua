--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Soccer Agent in Grid World:')
cmd:text()
cmd:text('Options:')

cmd:option('-height', 4, 'height of the playground')
cmd:option('-width', 7, 'width of the playground')
cmd:option('-defend', 0.5, 'percentage of defensive agents')
cmd:option('-framework', '', 'name of training framework')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-checkpoint_dir', '/fs/clip-scratch/hhe/opponent/cv', 'output directory where checkpoints get written')
cmd:option('-savefile','','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-hist_len', 1, 'history length of state features')
cmd:option('-best', false, 'load best model or current model')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-opponent', '', 'name of opponent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-games', 5000, 'number of games to play')
cmd:option('-eval_games', 5000, 'number of evaluation games')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to state')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpuid', -1, 'gpu flag')
cmd:option('-test',0,'evaluate on test set')

cmd:text()

opt = cmd:parse(arg)
opt.gpu = opt.gpuid
if opt.savefile == '' then opt.savefile = opt.agent end

--- General setup.
require 'setup'
env_setup()
local agent, opponent, game_env = soccer_dqn_setup()
local actions = agent.actions

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
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

-- TODO: percentage of winning
function eval(test)
    -- use different seed for eval and test
    game_env:reset(test and opt.seed+1 or opt.seed+2)
    if not test then
        print(string.format('============= eval ============='))
    else
        print(string.format('============= test ============='))
    end
    test = test or false
    local total_reward = 0
    local total_length = 0
    local nepisodes = 0
    local num_win = 0
    local num_tie = 0
    local episode_reward = 0
    local episode_length = 0
    local eval_time = sys.clock()
    local n = opt.eval_games
    local log = opt.test == 1 and io.open(opt.savefile .. '.log', 'w') or nil
    for i=1,n do
        state, terminal, reward = game_env:new_game(true)
        while true do
            local action_index = agent:perceive(reward, state, terminal, true, 0.0)
            -- zero sum game; opponent reward = -reward
            local action2_index = opponent:perceive(-1*reward, state, terminal, true, 0.0)
            if not terminal then
                state, terminal, reward = game_env:step(actions[action_index], actions[action2_index])
                episode_reward = episode_reward + reward
                episode_length = episode_length + 1
            else break end
        end
        -- write log
        if log ~= nil then
            log:write(string.format('%.4f\n', episode_reward)) 
        end
        -- overall stats
        if episode_reward > 0 then 
            num_win = num_win + 1 
        elseif episode_reward == 0 then
            num_tie = num_tie + 1
        end
        total_reward = total_reward + episode_reward
        total_length = total_length + episode_length
        episode_reward = 0
        episode_length = 0
        nepisodes = nepisodes + 1
    end
    eval_time = sys.clock() - eval_time
    total_reward = total_reward / nepisodes
    total_length = total_length / nepisodes
    num_win = num_win / nepisodes
    num_tie = num_tie / nepisodes

    -- make sure it's a learning agent
    if agent.w ~= nil and not test then 
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

        if agent.lr ~= nil then
            print(string.format('epsilon: %.2f, lr: %G', agent.ep, agent.lr))
        end
        print(string.format(
            'reward: %.4f, win: %.4f, tie: %.4f, episode length: %.2f\n' ..
            'training time: %ds, eval time: %ds, ' ..
            'num. ep.: %d',
            total_reward, num_win, num_tie, total_length, time_dif,
            eval_time, 
            n))
    else
        if agent.ep ~= nil then
            print(string.format('epsilon: %.2f', agent.ep))
        end
        print(string.format(
            'reward: %.4f, win: %.4f, tie: %.4f, episode length: %.2f, ' ..
            'eval time: %ds, num. ep.: %d',
            total_reward, num_win, num_tie, total_length,
            eval_time, n))
        -- record eval reward history
        eval_reward_history[#eval_reward_history+1] = total_reward
    end
    print('=========================================')
end

--------------------------- testing --------------------------------
if opt.test == 1 then
    eval(true)
    os.exit()
end

--------------------------- training --------------------------------
local state, reward, terminal
local max_num_games = opt.games
opt.save_freq = 5*opt.eval_freq
for i=1,max_num_games do
    epoch = i / opt.eval_freq
    state, terminal, reward = game_env:new_game(false) 
    while true do
        local priority = reward ~= 0 and true or false 
        local action_index = agent:perceive(reward, state, terminal, false, nil, priority)
        -- zero sum game; opponent reward = -reward
        local action2_index = opponent:perceive(-1*reward, state, terminal, false, nil, priority)
        if not terminal then
            state, terminal, reward = game_env:step(actions[action_index], actions[action2_index])
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
        i, max_num_games, epoch, agent.lr==nil and 0 or agent.lr, agent.ep==nil and 0 or agent.ep,
        total_reward/nepisodes, total_length/nepisodes, nepisodes))
        --agent:report()
        reset_stats()
        collectgarbage()
    end

    -- evaluation
    if i % opt.eval_freq == 0 then
        eval()
        --eval(true)
    end

    -- agent.w == nil means this is not a learner
    --if agent.w ~= nil and (i % opt.save_freq == 0 or i == max_num_games) then
    if agent.w ~= nil and (i == max_num_games) then
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
