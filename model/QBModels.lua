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

function BuzzCorrect:__init(rnn_type, rnn_size, n, batch_size, dropout, use_words, threshold)
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

function BuzzCorrect:create_rnn_module(net_params)
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
                x = nn.JoinTable(2)({inputs[1], qb.word_embedding(inputs[2])})
                -- TODO: record input feat dim
                input_size_L = qb.emb_size + qb.ans_size + 1
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

function BuzzCorrect:oracle_buzz(ans_prob, ans_target, mask, t)
    local gold_margin = torch.Tensor(ans_target:size())
    --local prob = ans_prob:clone():exp()
    -- select logprobs of wrong answers
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
