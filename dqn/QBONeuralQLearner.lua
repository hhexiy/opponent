local nql = torch.class('dqn.QBONeuralQLearner', 'dqn.QBNeuralQLearner')

function nql:__init(args)
    dqn.QBNeuralQLearner.__init(self, args)
    self.feat_groups = args.feat_groups
end

-- TODO: fix state features: add to default using parallel
function nql:state_to_input(state)
    --return state
    return {
            state:narrow(2, self.feat_groups.opponent.offset, self.feat_groups.opponent.size):clone(),
            state:narrow(2, self.feat_groups.pred.offset, self.feat_groups.pred.size)
        }
end

-- fully connected
function nql:createNetwork2()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Linear(self.feat_groups.pred.size+self.feat_groups.opponent.size, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end

function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()

    local mlp_default = nn.Sequential()
    mlp_default:add(nn.Linear(self.feat_groups.pred.size, n_hid))
    mlp_default:add(nn.Rectifier())
    --mlp_default:add(nn.Linear(n_hid, n_hid))
    --mlp_default:add(nn.Rectifier())

    mlp:add(mlp_default)

    experts = nn.ConcatTable()
    local n_experts = 3
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

