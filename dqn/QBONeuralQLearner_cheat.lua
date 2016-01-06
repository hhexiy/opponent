local nql = torch.class('dqn.QBONeuralQLearner_cheat', 'dqn.QBNeuralQLearner')

function nql:__init(args)
    dqn.QBNeuralQLearner.__init(self, args)
    self.feat_groups = args.feat_groups
end

-- TODO: fix state features: add to default using parallel
function nql:state_to_input(state)
    --return state
    return {
            state:narrow(2, self.feat_groups.cheat.offset, self.feat_groups.cheat.size):clone(),
            state:narrow(2, self.feat_groups.pred.offset, self.feat_groups.pred.size)
            --state:narrow(2, self.feat_groups.state.offset, self.feat_groups.state.size)
        }
end

function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()

    local mlp_default = nn.Sequential()
    mlp_default:add(nn.Linear(self.feat_groups.pred.size, n_hid))
    mlp_default:add(nn.Rectifier())
    mlp_default:add(nn.Linear(n_hid, n_hid))
    mlp_default:add(nn.Rectifier())

    mlp:add(mlp_default)

    experts = nn.ConcatTable()
    -- TODO: set this as a variable
    local n = 4
    for i = 1, n do
       local expert = nn.Sequential()
       expert:add(nn.Linear(n_hid, self.n_actions))
       experts:add(expert)
    end

    mlp:add(experts)

    -- get mixture weights from input
    gater = nn.Identity()

    -- mixture of experts
    local network = nn.Sequential()
    local parallel = nn.ParallelTable()
    parallel:add(gater)
    parallel:add(mlp)
    network:add(parallel)
    network:add(nn.MixtureTable())

    return network
end

function nql:createNetwork_cheat()
    local n_hid = 128
    local parallel = nn.ParallelTable()
    local mlp = nn.Sequential()
    local mlp_default = nn.Sequential()
    mlp_default:add(nn.Linear(self.feat_groups.pred.size, n_hid))
    mlp_default:add(nn.Rectifier())
    parallel:add(mlp_default)
    parallel:add(nn.Identity())
    mlp:add(parallel)
    mlp:add(nn.JoinTable(2))
    mlp:add(nn.Linear(n_hid+self.feat_groups.cheat.size, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))
    return mlp
end

function nql:createNetwork2()
    local n_hid = 128
    local pred_nhid = 50
    local inputs = {}
    table.insert(inputs, nn.Identity()())
    --table.insert(inputs, nn.Identity()())
    --table.insert(inputs, nn.Identity()())
    --local pred_emb = nn.LookupTable(self.ans_size, pred_nhid)(inputs[2])
    --local join = nn.JoinTable(2)({inputs[1], inputs[2]})
    local h = nn.Linear(self.feat_groups.default.size+self.feat_groups.cheat.size, n_hid)(inputs[1])
    h = nn.Rectifier()(h)
    h = nn.Rectifier()(nn.Linear(n_hid, n_hid)(h))
    local logsoft = nn.LogSoftMax()(nn.Linear(n_hid, self.n_actions)(h))
    local outputs = {}
    table.insert(outputs, logsoft)
    return nn.gModule(inputs, outputs)
end

function nql:createNetwork2()
    local n_hid = 128
    local mlp = nn.Sequential()
    -- get different parts of the input
    local concat = nn.ConcatTable()
    concat:add(nn.Narrow(2, self.feat_groups.default.offset, self.feat_groups.default.size))
    concat:add(nn.Narrow(2, self.feat_groups.pred.offset, self.feat_groups.pred.size))
    concat:add(nn.Narrow(2, self.feat_groups.cheat.offset, self.feat_groups.cheat.size))
    mlp:add(concat)
    -- transformation for each group
    local parallel = nn.ParallelTable()
    parallel:add(nn.Identity())
    local pred_nhid = 50
    --local pred_mlp = nn.Sequential()
    --pred_mlp:add(MultiHot(self.ans_size))
    --pred_mlp:add(nn.Linear(self.ans_size, pred_nhid))
    --parallel:add(pred_mlp)
    parallel:add(nn.LookupTable(self.ans_size, pred_nhid))
    parallel:add(nn.Identity())
    mlp:add(parallel)
    -- join two parts
    mlp:add(nn.JoinTable(2))
    mlp:add(nn.Linear(self.feat_groups.default.size+pred_nhid+self.feat_groups.cheat.size, n_hid))
    --mlp:add(nn.Rectifier())
    --mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))
    return mlp
end
