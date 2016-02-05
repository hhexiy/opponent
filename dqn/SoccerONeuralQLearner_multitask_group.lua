local nql = torch.class('dqn.SoccerONeuralQLearner_multitask_group', 'dqn.SoccerNeuralQLearner_multitask')

function nql:__init(args)
    self.n_experts = args.n_experts
    self.feat_groups = args.feat_groups
    print('number of experts: ', self.n_experts)
    self.criterion = nn.ClassNLLCriterion() 
    self.num_classes = 2 
    print('network model:', args.model)
    if args.model == 'fc2' then
        self.createNetwork = self.createNetwork_fc2
    else
        self.createNetwork = self.createNetwork_moe
    end
    dqn.SoccerNeuralQLearner_multitask.__init(self, args)
end

function nql:state_to_input(state)
    return {
            state:narrow(2, self.feat_groups.opponent.offset, self.feat_groups.opponent.size):clone(),
            state:narrow(2, self.feat_groups.state.offset, self.feat_groups.state.size)
        },
        state:narrow(2, self.feat_groups.supervision.offset, self.feat_groups.supervision.size)
end

-- fully connected 2 parts
function nql:createNetwork_fc2()
    local n_hid = 50
    local n_hid_opp = 50 
    local mlp = nn.Sequential()
    local parallel = nn.ParallelTable()
    parallel:add(nn.Sequential():add(nn.Linear(self.feat_groups.opponent.size, n_hid_opp)):add(nn.Rectifier()))
    parallel:add(nn.Sequential():add(nn.Linear(self.feat_groups.state.size, n_hid)):add(nn.Rectifier()))
    mlp:add(parallel)
    local multitask = nn.ConcatTable()
    multitask:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.Linear(n_hid_opp, self.num_classes)):add(nn.LogSoftMax()))
    multitask:add(nn.Sequential():add(nn.JoinTable(2)):add(nn.Linear(n_hid+n_hid_opp, n_hid)):add(nn.Rectifier()):add(nn.Linear(n_hid, self.n_actions)))
    mlp:add(multitask)
    mlp:add(nn.JoinTable(2))

    return mlp
end


function nql:createNetwork_moe()
    local n_hid = 50
    local network = nn.Sequential()
    -- {opponent}, {state}
    local split_inputs = nn.ConcatTable():add(nn.JoinTable(2)):add(nn.SelectTable(2))
    -- {opponent + state}, {state}
    network:add(split_inputs)
    local parallel = nn.ParallelTable()

    -- state to predict q-values
    local mlp = nn.Sequential()
    local mlp_state = nn.Sequential()
    mlp_state:add(nn.Linear(self.feat_groups.state.size, n_hid))
    mlp_state:add(nn.Rectifier())
    mlp_state:add(nn.Linear(n_hid, n_hid))
    mlp_state:add(nn.Rectifier())
    mlp:add(mlp_state)
    -- multiple experts
    local experts = nn.ConcatTable()
    local n_experts = self.n_experts
    for i = 1,n_experts do
       local expert = nn.Sequential()
       expert:add(nn.Linear(n_hid, self.n_actions))
       experts:add(expert)
    end
    mlp:add(experts)

    -- multitask: 1. get mixture weights from input; 2. opponent prediction
    local multitask = nn.Sequential()
    multitask:add(nn.Linear(self.feat_groups.state.size+self.feat_groups.opponent.size, n_hid))
    multitask:add(nn.Rectifier())
    local concat = nn.ConcatTable() 
    -- opponent prediction
    local opp_pred = nn.Sequential():add(nn.Linear(n_hid, self.num_classes)):add(nn.LogSoftMax())
    concat:add(opp_pred)
    -- mixture weights
    local gater = nn.Sequential():add(nn.Linear(n_hid, n_experts)):add(nn.SoftMax())
    concat:add(gater)
    multitask:add(concat)

    parallel:add(multitask):add(mlp)
    network:add(parallel)
    -- {{opp pred}, {gater weights}}, {{experts1} ... {expertsn}}
    local outputs = nn.ConcatTable()
    outputs:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(1)))
    outputs:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(2)))
    outputs:add(nn.SelectTable(2))
    -- {opp pred}, {gater weights}, {{experts1} ... {expertsn}}
    network:add(outputs)
    network:add(nn.ConcatTable():add(nn.SelectTable(1)):add(nn.NarrowTable(2, 2)))
    -- two tables: {opp pred}, {gater weights, experts}

    local parallel_outputs = nn.ParallelTable():add(nn.Identity()):add(nn.MixtureTable())
    -- two tables: {opp pred}, {q values}
    network:add(parallel_outputs):add(nn.JoinTable(2))

    return network
end

