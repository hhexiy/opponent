require 'torch'
require 'nn'

savefile = arg[1]
game = arg[2]

if not dqn then
    dqn = {}
    if game == 'qb' then
        require 'dqn.QBNeuralQLearner'
        require 'dqn.QBONeuralQLearner'
        require 'dqn.QBONeuralQLearner_multitask_group'
        require 'dqn.QBONeuralQLearner_multitask_action'
    elseif game == 'soccer' then
        require 'dqn.RandomAgent'
        require 'dqn.SoccerRuleAgent'
        require 'dqn.SoccerNeuralQLearner'
        require 'dqn.SoccerNeuralQLearner_multitask'
        require 'dqn.SoccerONeuralQLearner'
        require 'dqn.SoccerONeuralQLearner_multitask_group'
        require 'dqn.SoccerONeuralQLearner_multitask_action'
    end
    require 'dqn.Rectifier'
    require 'dqn.TransitionTable'
end

function write_rewards(filename, rewards)
    print(filename)
    local fout = io.open(filename, 'w')
    for i, r in ipairs(rewards) do
        fout:write(r)
        fout:write('\n')
    end
    fout:close()
end

local data = torch.load(savefile)
local eval_rewards, test_rewards
eval_rewards = data.reward_history
-- TODO: fix this: make it same for qb and soccer
if game == 'qb' then
    test_rewards = data.eval_reward_history[3]
else
    test_rewards = data.eval_reward_history
end
write_rewards(savefile .. '.eval_rewards', eval_rewards)
write_rewards(savefile .. '.test_rewards', test_rewards)
