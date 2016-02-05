local nql = torch.class('dqn.SoccerONeuralQLearner', 'dqn.SoccerNeuralQLearner')

function nql:__init(args)
    self.feat_groups = args.feat_groups
    print('network model:', args.model)
    if args.model == 'fc2' then
        self.n_experts = 0
        self.createNetwork = self.createNetwork_fc2
    else
        self.n_experts = args.n_experts
        self.createNetwork = self.createNetwork_moe
    end
    print('number of experts: ', self.n_experts)
    dqn.SoccerNeuralQLearner.__init(self, args)
end

function nql:state_to_input(state)
    return {
            state:narrow(2, self.feat_groups.opponent.offset, self.feat_groups.opponent.size):clone(),
            state:narrow(2, self.feat_groups.state.offset, self.feat_groups.state.size)
        }
    --return state:narrow(2, self.feat_groups.state.offset, self.feat_groups.state.size)
end

-- fully connected 2 parts
function nql:createNetwork_fc2()
    local n_hid = 50
    local n_hid_opp = 50 
    local mlp = nn.Sequential()
    --mlp:add(nn.JoinTable(2)):add(nn.Linear(self.feat_groups.opponent.size+self.feat_groups.state.size, n_hid)):add(nn.Rectifier()):add(nn.Linear(n_hid, n_hid))
    local parallel = nn.ParallelTable()
    parallel:add(nn.Sequential():add(nn.Linear(self.feat_groups.opponent.size, n_hid_opp)):add(nn.Rectifier()))
    parallel:add(nn.Sequential():add(nn.Linear(self.feat_groups.state.size, n_hid)):add(nn.Rectifier()))
    mlp:add(parallel):add(nn.JoinTable(2))
    mlp:add(nn.Linear(n_hid+n_hid_opp, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end

function nql:createNetwork_moe()
    local n_hid = 50
    local mlp = nn.Sequential()

    local mlp_default = nn.Sequential()
    mlp_default:add(nn.Linear(self.feat_groups.state.size, n_hid))
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
    gater:add(nn.Linear(self.feat_groups.state.size+self.feat_groups.opponent.size, n_hid))
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

-- use same feature
function nql:createNetwork_same()
    local n_hid = 50
    local mlp = nn.Sequential()

    local mlp_default = nn.Sequential()
    mlp_default:add(nn.Linear(self.feat_groups.state.size, n_hid))
    mlp_default:add(nn.Rectifier())
    --mlp_default:add(nn.Linear(n_hid, n_hid))
    --mlp_default:add(nn.Rectifier())

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
    gater:add(nn.Linear(self.feat_groups.state.size, n_hid))
    gater:add(nn.Rectifier())
    gater:add(nn.Linear(n_hid, n_experts))
    gater:add(nn.SoftMax())

    -- mixture of experts
    local network = nn.Sequential()
    network:add(nn.ConcatTable():add(gater):add(mlp))
    network:add(nn.MixtureTable())
    
    return network
end
