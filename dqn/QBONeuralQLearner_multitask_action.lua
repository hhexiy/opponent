local nql = torch.class('dqn.QBONeuralQLearner_multitask_action', 'dqn.QBNeuralQLearner')

function nql:__init(args)
    self.criterion = nn.MSECriterion() 
    self.num_classes = 1
    self.gating_weights = nil
    if args.model == 'fc2' then
        self.createNetwork = self.createNetwork_fc2
        self.n_experts = 0
    else
        self.createNetwork = self.createNetwork_moe
        self.n_experts = args.n_experts
    end
    print('number of experts: ', self.n_experts)
    dqn.QBNeuralQLearner.__init(self, args)
    --print(self.network)
    --os.exit()
end

function nql:greedy(state)
    if self.gpu >= 0 then
        state = state:cuda()
    end
   
    local s, _ = self:state_to_input(state)
    local outputs = self.network:forward(s):float()
    local q = self:split_output(outputs)[2]:squeeze()
    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestq = maxq

    local r = torch.random(1, #besta)

    self.lastAction = besta[r]

    if self.n_experts > 0 then
        -- concat
        --self.gating_weights = self.network.modules[4].modules[2].modules[4].output
        --print(self.network.modules[4].modules[2].modules[3].output)
        --print(self.network.modules[4].modules[2].modules[4].output)
        --print(self.network.modules[2].modules[1].modules[2].modules[1].output)
        -- share
        self.gating_weights = self.network.modules[2].modules[1].modules[3].modules[2].modules[2].output
    end
    return besta[r]
end

function nql:getQUpdate(args)
    local s, a, r, s2, term, delta
    local q, q2, q2_max

    s, opp_a = self:state_to_input(args.s)
    a = args.a
    r = args.r
    s2, _ = self:state_to_input(args.s2)
    term = args.term
    local minibatch_size = a:size(1)

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

    -- Compute max_a Q(s_2, a).
    local outputs2 = target_q_net:forward(s2):float()
    q2_max = self:split_output(outputs2)[2]:max(2)

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(self.discount):cmul(term)

    delta = r:clone():float()

    if self.rescale_r then
        delta:div(self.r_max)
    end
    delta:add(q2)

    local outputs = self.network:forward(s)
    local opp_pred, q_all = unpack(self:split_output(outputs))
    local opp_targets = self.criterion:backward(opp_pred:double(), opp_a:squeeze():double()):mul(-1)
    --print('pred:', opp_pred:narrow(1,1,1):squeeze(), 'target:', opp_a:narrow(1,1,1):squeeze(), 'grad:', opp_targets:narrow(1,1,1):squeeze())

    -- q = Q(s,a)
    --q_all = q_all:float()
    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
        --opp_targets:clamp(-self.clip_delta, self.clip_delta)
    end

    -- using minibatch_size is wrong for validation (valid_size != minibatch_size)
    --local targets = torch.zeros(self.minibatch_size, self.n_actions)
    --for i=1,math.min(self.minibatch_size,a:size(1)) do
    --    targets[i][a[i]] = delta[i]
    --end
    local targets = torch.zeros(minibatch_size, self.n_actions)
    for i=1,minibatch_size do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then 
        targets = targets:cuda() 
        opp_targets = opp_targets:cuda() 
    end

    --return targets, delta, q2_max
    --print('opp:', opp_targets)
    --print('tgt:', targets)
    return torch.cat(opp_targets, targets, 2), delta, q2_max
end

-- fully connected 2 parts
function nql:createNetwork_fc2()
    local n_hid = 128
    local n_hid_opp = 10
    local mlp = nn.Sequential()
    local parallel = nn.ParallelTable()
    parallel:add(nn.Sequential():add(nn.Linear(self.feat_groups.opponent.size, n_hid_opp)):add(nn.Rectifier()))
    parallel:add(nn.Sequential():add(nn.Linear(self.feat_groups.pred.size, n_hid)):add(nn.Rectifier()))
    mlp:add(parallel)
    local multitask = nn.ConcatTable()
    multitask:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.Linear(n_hid_opp, 1)))
    --multitask:add(nn.Sequential():add(nn.JoinTable(2)):add(nn.Linear(n_hid+n_hid_opp, 1)))
    multitask:add(nn.Sequential():add(nn.JoinTable(2)):add(nn.Linear(n_hid+n_hid_opp, n_hid)):add(nn.Rectifier()):add(nn.Linear(n_hid, self.n_actions)))
    mlp:add(multitask)
    mlp:add(nn.JoinTable(2))

    return mlp
end

function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)
    --print(s:narrow(2, 1, 10), a[1], r[1], s2:narrow(2, 1, 10), term[1])

    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,
        term=term, update_qmax=true}

    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    s, _ = self:state_to_input(s)
    self.network:backward(s, targets)

    -- clip gradients 
    --self.dw:clamp(-5, 5)

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()

    --rmsprop
    --local smoothing_value = 1e-8
    --self.tmp:cmul(self.dw, self.dw)
    --self.g:mul(0.9):add(0.1, self.tmp)
    --self.tmp = torch.sqrt(self.g)
    --self.tmp:add(smoothing_value)  --negative learning rate

    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    self.w:add(self.deltas)
end


-- TODO: fix state features: add to default using parallel
function nql:state_to_input(state)
    --return state
    return {
            state:narrow(2, self.feat_groups.opponent.offset, self.feat_groups.opponent.size):clone(),
            state:narrow(2, self.feat_groups.pred.offset, self.feat_groups.pred.size)
        },
        state:narrow(2, self.feat_groups.opp_action.offset, self.feat_groups.opp_action.size)
end

-- concat multitask hid state
function nql:createNetwork_concat()
    local n_hid = 128

    local network = nn.Sequential()
    -- {opponent}, {state}
    local split_inputs = nn.ConcatTable():add(nn.JoinTable(2)):add(nn.JoinTable(2)):add(nn.SelectTable(2))
    -- {opponent + state}, {opponent + state}, {state}
    network:add(split_inputs)
    local parallel = nn.ParallelTable()

    -- state to predict q-values
    local mlp = nn.Sequential()
    local mlp_state = nn.Sequential()
    mlp_state:add(nn.Linear(self.feat_groups.pred.size, n_hid))
    mlp_state:add(nn.Rectifier())
    --mlp_state:add(nn.Linear(n_hid, n_hid))
    --mlp_state:add(nn.Rectifier())
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

    local multitask = nn.Sequential()
    -- opponent model
    local opponent = nn.Sequential()
    opponent:add(nn.Linear(self.feat_groups.pred.size+self.feat_groups.opponent.size, n_hid))
    opponent:add(nn.Rectifier())
    multitask:add(opponent)
    multitask:add(nn.ConcatTable():add(nn.Linear(n_hid, 1)):add(nn.Identity()))
    -- {opp pred}, {hid}

    parallel:add(multitask):add(nn.Sequential():add(nn.Linear(self.feat_groups.pred.size+self.feat_groups.opponent.size, n_hid)):add(nn.Rectifier())):add(mlp)
    network:add(parallel):add(nn.FlattenTable())
    -- {{opp pred}, {hid1}}, {hid2}, {{experts1} ... {expertsn}}

    local outputs = nn.ConcatTable()
    -- opp pred
    outputs:add(nn.SelectTable(1))
    outputs:add(nn.Sequential():add(nn.NarrowTable(2, 2)):add(nn.JoinTable(2)):add(nn.Linear(n_hid*2, n_experts)):add(nn.SoftMax()))
    outputs:add(nn.NarrowTable(4, n_experts))
    network:add(outputs)
    -- {opp pred}, {gater}, {experts}

    local results = nn.ConcatTable()
    results:add(nn.SelectTable(1))
    results:add(nn.Sequential():add(nn.NarrowTable(2, 2)):add(nn.MixtureTable()))
    -- two tables: {opp pred}, {q values}
    network:add(results):add(nn.JoinTable(2))

    return network
end

function nql:split_output(outputs)
    return {outputs:narrow(2, 1, self.num_classes),
            outputs:narrow(2, self.num_classes+1, self.n_actions)
        }
end

-- share multitask hid state
function nql:createNetwork_moe()
    local n_hid = 128

    local network = nn.Sequential()
    -- {opponent}, {state}
    --local split_inputs = nn.ConcatTable():add(nn.JoinTable(2)):add(nn.JoinTable(2))
    local split_inputs = nn.ConcatTable():add(nn.JoinTable(2)):add(nn.SelectTable(2))
    -- {opponent + state}, {state}
    network:add(split_inputs)
    local parallel = nn.ParallelTable()

    -- state to predict q-values
    local mlp = nn.Sequential()
    local mlp_state = nn.Sequential()
    --mlp_state:add(nn.Linear(self.feat_groups.pred.size+self.feat_groups.opponent.size, n_hid))
    mlp_state:add(nn.Linear(self.feat_groups.pred.size, n_hid))
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
    multitask:add(nn.Linear(self.feat_groups.pred.size+self.feat_groups.opponent.size, n_hid))
    multitask:add(nn.Rectifier())
    local concat = nn.ConcatTable() 
    -- opponent prediction
    local opp_pred = nn.Linear(n_hid, 1)
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

function nql:split_output(outputs)
    return {outputs:narrow(2, 1, self.num_classes),
            outputs:narrow(2, self.num_classes+1, self.n_actions)
        }
end
