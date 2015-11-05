local model_utils = require 'util.model_utils'
local recurrent = require 'model.recurrent'

local Content = torch.class('qb.Content')

function Content:__init(rnn_type, rnn_size, n, batch_size, dropout)
    local net_params = {}
    net_params.rnn_type = rnn_type
    net_params.rnn_size = rnn_size
    net_params.num_layers = n
    net_params.batch_size = batch_size
    net_params.dropout = dropout or 0
    self.net_params = net_params
    self.num_classes = qb.ans_size
    self.module = self:create_rnn_module(net_params)
    self.criterion = nn.ClassNLLCriterion()
end

function Content:create_rnn_module(net_params)
    local inputs = {}
    -- input words
    table.insert(inputs, nn.Identity()()) 
    local nin = 1
    -- previous hidden states
    for L = 1,net_params.num_layers do
        table.insert(inputs, nn.Identity()()) -- prev_h[L]
    end
    local x, input_size_L
    local outputs = {}
    for L = 1,net_params.num_layers do
        local prev_h = inputs[L+nin]
        -- the input to this layer
        if L == 1 then
            x = qb.word_embedding(inputs[1])
            input_size_L = qb.emb_size
        else 
            x = outputs[(L-1)]
            -- apply dropout, if any
            if net_params.dropout > 0 then x = nn.Dropout(net_params.dropout)(x) end 
            input_size_L = net_params.rnn_size
        end
        -- rnn tick
        local next_h = recurrent[net_params.rnn_type](input_size_L, net_params.rnn_size, x, prev_h)
        table.insert(outputs, next_h)
    end

    -- set up the decoder
    local top_h = outputs[#outputs]
    if net_params.dropout > 0 then top_h = nn.Dropout(net_params.dropout)(top_h) end
    local proj = nn.Linear(net_params.rnn_size, self.num_classes)(top_h)
    local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, logsoft)

    return nn.gModule(inputs, outputs)
end

local BuzzCorrect = torch.class('qb.BuzzCorrect')    

function BuzzCorrect:__init(input_sizes, hidden_size, threshold)
    local net_params = {}
    net_params.input_sizes = input_sizes
    net_params.hidden_size = hidden_size
    net_params.batch_size = batch_size
    self.net_params = net_params
    self.num_classes = 3
    self.num_inputs = #input_sizes
    self.threshold = threshold or 0
    -- create network
    self.network = self:create_network(net_params)
    self.params, self.grad_params = model_utils.combine_all_parameters(self.network) 
    self.params:uniform(-0.08, 0.08)
    -- criterion
    local weights = torch.Tensor(self.num_classes):fill(1)
    weights[qb.WAIT] = 5 
    self.criterion = nn.ClassNLLCriterion(weights)
end

-- logistic regression
function BuzzCorrect:create_network2(net_params)
    local inputs = {}
    local size = 0
    --local hidden_nodes = {}
    for k, input_size in ipairs(net_params.input_sizes) do
        table.insert(inputs, nn.Identity()())
        --table.insert(hidden_nodes, nn.Linear(input_size, net_params.hidden_size)(inputs[#inputs]))
        size = size + input_size
    end
    local h
    if #inputs > 1 then
        h = nn.JoinTable(2)(inputs)
    else
        h = inputs 
    end
    local outputs = {}
    --local logsoft = nn.LogSoftMax()(nn.Linear(net_params.hidden_size, self.num_classes)(h))
    local logsoft = nn.LogSoftMax()(nn.Linear(size, self.num_classes)(h))
    table.insert(outputs, logsoft)
    return nn.gModule(inputs, outputs)
end

function BuzzCorrect:create_network(net_params)
    local inputs = {}
    local hidden_nodes = {}
    --local all_input_size = 0
    for k, input_size in ipairs(net_params.input_sizes) do
        table.insert(inputs, nn.Identity()())
        table.insert(hidden_nodes, nn.Linear(input_size, net_params.hidden_size)(inputs[#inputs]))
        --all_input_size = all_input_size + input_size
    end
    local h
    if #inputs > 1 then
        h = nn.Tanh()(nn.CAddTable()(hidden_nodes))
        --h = nn.Tanh()(nn.Linear(all_input_size, net_params.hidden_size)(nn.JoinTable(2)(inputs)))
    else
        h = nn.Tanh()(hidden_nodes[1])
    end
    local outputs = {}
    local logsoft = nn.LogSoftMax()(nn.Linear(net_params.hidden_size, self.num_classes)(h))
    table.insert(outputs, logsoft)
    return nn.gModule(inputs, outputs)
end

function BuzzCorrect:forward(inputs, target, eval)
    eval = eval or false
    if eval then
        self.network:evaluate()
    else
        self.network:training()
    end
    -- inputs is a table of input tensors
    local outputs = self.network:forward(inputs)
    local loss = self.criterion:forward(outputs[1], target)
    return outputs, loss
end

function BuzzCorrect:backward(inputs, target, output)
    local doutput = self.criterion:backward(output, target)
    self.network:backward(inputs, doutput)
end

function BuzzCorrect:oracle_buzz(ans_prob, ans_target, mask, t)
    local gold_margin = torch.Tensor(ans_target:size())
    -- select probs of wrong answers
    local prob_mask = torch.ByteTensor(ans_prob:size(2)):fill(1)
    for i=1,ans_prob:size(1) do
        local gd = ans_target[i]
        prob_mask[gd] = 0
        -- use prob for reasonable threshold
        gold_margin[i] = ans_prob[i][gd] - ans_prob[i][prob_mask]:max()
        prob_mask[gd] = 1
    end

    local buzz = torch.IntTensor(ans_target:size()):fill(qb.WAIT)
    --if t > 30 then
    --    buzz:fill(qb.BUZZ)
    --end
    buzz[gold_margin:gt(self.threshold)] = qb.BUZZ
    buzz[mask:eq(0)] = qb.EOS
    return buzz
end

local BuzzCorrectRNN = torch.class('qb.BuzzCorrectRNN', 'qb.BuzzCorrect')    

function BuzzCorrectRNN:__init(rnn_type, rnn_size, n, batch_size, dropout, use_words, threshold)
    local net_params = {}
    net_params.rnn_type = rnn_type
    net_params.rnn_size = rnn_size
    net_params.num_layers = n
    net_params.batch_size = batch_size
    net_params.dropout = dropout or 0
    self.net_params = net_params
    -- 3: buzz, wait, eos for padding
    self.num_classes = 3
    -- default: use answer logprobs as only inputs
    self.use_words = use_words or false
    self.module = self:create_rnn_module(net_params)
    local weights = torch.Tensor(3):fill(1)
    weights[qb.WAIT] = 15
    self.criterion = nn.ClassNLLCriterion(weights)
    self.threshold = threshold or 0
end

-- TODO: put logprob feature to the last layer
function BuzzCorrectRNN:create_rnn_module(net_params)
    local nin = 1
    local inputs = {}
    -- answer logrobs
    table.insert(inputs, nn.Identity()()) 
    -- words
    if self.use_words then 
        table.insert(inputs, nn.Identity()()) -- x 
        nin = 2
    end
    -- previous hidden states
    for L = 1,net_params.num_layers do
        table.insert(inputs, nn.Identity()()) -- prev_h[L]
    end

    local x, input_size_L
    local outputs = {}
    for L = 1,net_params.num_layers do
        local prev_h = inputs[L+nin]
        -- the input to this layer
        if L == 1 then
            if self.use_words then
                --x = nn.JoinTable(2)({inputs[1], qb.word_embedding(inputs[2])})
                --input_size_L = qb.emb_size + qb.ans_size + 1
                x = nn.CAddTable()({nn.Linear(qb.ans_size+1, net_params.rnn_size)(inputs[1]), nn.Linear(qb.emb_size, net_params.rnn_size)(qb.word_embedding(inputs[2]))})
                input_size_L = net_params.rnn_size
            else
                x = inputs[1] 
                input_size_L = qb.ans_size + 1
            end
        else 
            x = outputs[(L-1)]
            -- apply dropout, if any
            if net_params.dropout > 0 then x = nn.Dropout(net_params.dropout)(x) end 
            input_size_L = net_params.rnn_size
        end
        -- rnn tick
        local next_h = recurrent[net_params.rnn_type](input_size_L, net_params.rnn_size, x, prev_h)
        table.insert(outputs, next_h)
    end

    -- set up the decoder
    local top_h = outputs[#outputs]
    if net_params.dropout > 0 then top_h = nn.Dropout(net_params.dropout)(top_h) end
    local proj = nn.Linear(net_params.rnn_size, self.num_classes)(top_h)
    local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, logsoft)

    return nn.gModule(inputs, outputs)
end

