local nql = torch.class('dqn.QBONeuralQLearner', 'dqn.QBNeuralQLearner')

function nql:__init(args)
    self.feat_groups = args.feat_groups
    self.use_words = self.feat_groups.words ~= nil and true or false
    if self.use_words then
        self.word_embedding = qb.word_embedding  
    end
    if args.model == 'fc2' then
        self.n_experts = 0
        self.createNetwork = self.createNetwork_fc2
    else
        self.n_experts = args.n_experts
        self.createNetwork = self.createNetwork_moe
    end
    print('number of experts: ', self.n_experts)
    dqn.QBNeuralQLearner.__init(self, args)
    self.gating_weights = nil
    --print(self.network)
    --os.exit()
end

function nql:state_to_input(state)
    --return state
    if self.use_words then
        return {
                state:narrow(2, self.feat_groups.opponent.offset, self.feat_groups.opponent.size):clone(),
                state:narrow(2, self.feat_groups.pred.offset, self.feat_groups.pred.size),
                state:narrow(2, self.feat_groups.words.offset, self.feat_groups.words.size):clone()
            }
    else
        return {
                state:narrow(2, self.feat_groups.opponent.offset, self.feat_groups.opponent.size):clone(),
                state:narrow(2, self.feat_groups.pred.offset, self.feat_groups.pred.size)
            }
    end
end

-- recored gater's output
function nql:greedy(args)
    local result = dqn.QBNeuralQLearner.greedy(self, args)
    if self.n_experts > 0 then
        -- share state
        self.gating_weights = self.network.modules[2].modules[1].modules[4].output
        -- share hid
        --self.gating_weights = self.network.modules[3].modules[1].modules[4].output
    end
    return result
end

-- fully connected 2 parts
function nql:createNetwork_fc2()
    local n_hid = 128
    local n_hid_opp = 10
    local mlp = nn.Sequential()
    local parallel = nn.ParallelTable()
    parallel:add(nn.Sequential():add(nn.Linear(self.feat_groups.opponent.size, n_hid_opp)):add(nn.Rectifier()))
    parallel:add(nn.Sequential():add(nn.Linear(self.feat_groups.pred.size, n_hid)):add(nn.Rectifier()))
    mlp:add(parallel):add(nn.JoinTable(2))
    mlp:add(nn.Linear(n_hid+n_hid_opp, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end

-- fully connected 2 parts
function nql:createNetwork_fc2_words()
    local n_hid = 128
    local n_hid_opp = 10
    local mlp = nn.Sequential()
    local parallel = nn.ParallelTable()
    parallel:add(nn.Sequential():add(nn.Linear(self.feat_groups.opponent.size, n_hid_opp)):add(nn.Rectifier()))
    parallel:add(nn.Sequential():add(nn.Linear(self.feat_groups.pred.size, n_hid)):add(nn.Rectifier()))
    parallel:add(nn.Sequential():add(self.word_embedding):add(nn.Max(2)))
    mlp:add(parallel):add(nn.JoinTable(2))
    mlp:add(nn.Linear(n_hid+n_hid_opp+qb.emb_size, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end

-- fully connected
function nql:createNetwork_fc()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Linear(self.feat_groups.pred.size+self.feat_groups.opponent.size, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end

-- add state to opp
function nql:createNetwork_moe()
    local n_hid = 128
    local mlp = nn.Sequential()

    local mlp_default = nn.Sequential()
    mlp_default:add(nn.Linear(self.feat_groups.pred.size, n_hid))
    mlp_default:add(nn.Rectifier())
    mlp_default:add(nn.Linear(n_hid, n_hid))
    mlp_default:add(nn.Rectifier())

    mlp:add(mlp_default)

    experts = nn.ConcatTable()
    local n_experts = self.n_experts
    for i = 1,n_experts do
       local expert = nn.Sequential()
       expert:add(nn.Linear(n_hid, self.n_actions))
       experts:add(expert)
    end

    mlp:add(experts)

    -- get mixture weights from input
    gater = nn.Sequential()
    gater:add(nn.Linear(self.feat_groups.pred.size+self.feat_groups.opponent.size, n_hid))
    gater:add(nn.Rectifier())
    gater:add(nn.Linear(n_hid, n_experts))
    gater:add(nn.SoftMax())

    -- mixture of experts
    local network = nn.Sequential()
    local split = nn.ConcatTable()
    split:add(nn.JoinTable(2))
    split:add(nn.SelectTable(2))
    network:add(split)
    local parallel = nn.ParallelTable()
    parallel:add(gater)
    parallel:add(mlp)
    network:add(parallel)
    network:add(nn.MixtureTable())
    
    return network
end

-- add state hid to opp
function nql:createNetwork3()
    local n_hid = 128
    local n_hid_opp = 10
    local mlp = nn.Sequential()

    local network = nn.Sequential()
    local parallel = nn.ParallelTable()
    -- opponent feat
    parallel:add(nn.Sequential():add(nn.Linear(self.feat_groups.opponent.size, n_hid_opp)):add(nn.Rectifier()))
    -- map state feat
    parallel:add(nn.Sequential():add(nn.Linear(self.feat_groups.pred.size, n_hid)):add(nn.Rectifier()))
    network:add(parallel)
    -- divide to opponent+state and state
    local split = nn.ConcatTable()
    split:add(nn.JoinTable(2))
    split:add(nn.SelectTable(2))
    network:add(split)

    -- gater and experts
    local parallel2 = nn.ParallelTable()
    local n_experts = self.n_experts

    -- get mixture weights from input
    gater = nn.Sequential()
    gater:add(nn.Linear(n_hid_opp+n_hid, n_hid))
    gater:add(nn.Rectifier())
    gater:add(nn.Linear(n_hid, n_experts))
    gater:add(nn.SoftMax())

    experts = nn.ConcatTable()
    for i = 1,n_experts do
       local expert = nn.Sequential()
       expert:add(nn.Linear(n_hid, self.n_actions))
       experts:add(expert)
    end

    parallel2:add(gater)
    parallel2:add(experts)
    network:add(parallel2)
    network:add(nn.MixtureTable())

    return network
end
