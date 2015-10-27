require 'util.misc'
local model_utils = require 'util.model_utils'

local RNNModel = torch.class('RNNModel')

function RNNModel:__init(rnn, seq_length, gpuid, init)
    local gpu = false 
    if gpuid > 0 then gpu = true end
    if init == nil then init = true end
    self.rnn = rnn.module
    self.criterion = rnn.criterion
    self.net_params = rnn.net_params

    -- ship the model to the GPU if desired
    if gpu then
        self.rnn:cuda()
        self.criterion:cuda()
    end

    -- put the above things into one flattened parameters tensor
    self.params, self.grad_params = model_utils.combine_all_parameters(self.rnn)
    print('number of parameters in the model: ' .. self.params:nElement())

    -- initialization
    if init then
        self.params:uniform(-0.08, 0.08) -- small numbers uniform
    end

    -- make a bunch of clones after flattening, as that reallocates memory
    self.clones = {}
    self.seq_length = seq_length
    print('cloning ' .. seq_length .. ' rnn units...')
    self.clones.rnn = model_utils.clone_many_times(self.rnn, seq_length)
    self.clones.criterion = model_utils.clone_many_times(self.criterion, seq_length)
    print('done')

    -- the initial state of the cell/hidden states
    self.init_state = {}
    for L=1,self.net_params.num_layers do
        local h_init = torch.zeros(self.net_params.batch_size, self.net_params.rnn_size)
        if gpu then h_init = h_init:cuda() end
        table.insert(self.init_state, h_init:clone())
        if self.net_params.rnn_type == 'lstm' then
            table.insert(self.init_state, h_init:clone())
        end
    end
end

function RNNModel:extract_input(input, t)
    local new_input = {}
    for k, v in ipairs(input) do 
        if type(v) == 'table' then
            table.insert(new_input, v[t])
        else
            table.insert(new_input, v[{{}, t}])
        end
    end
    return new_input
end

function RNNModel:forward(input, target, seq_length, eval)
    local eval = eval or false
    local rnn_state = {[0] = clone_list(self.init_state)}
    local output = {}
    local loss = 0
    for t=1,seq_length do
        if eval then
            self.clones.rnn[t]:evaluate()
        else
            self.clones.rnn[t]:training()
        end

        local inputs = self:extract_input(input, t)
        for k, v in ipairs(rnn_state[t-1]) do
            table.insert(inputs, v)
        end
        local lst = self.clones.rnn[t]:forward{unpack(inputs)}

        -- extract the state, without output
        rnn_state[t] = {}
        for i=1,#self.init_state do 
            table.insert(rnn_state[t], lst[i])
        end 

        -- extract outputs (assuming there's only one output--the last element) 
        output[t] = lst[#lst]

        loss = loss + self.clones.criterion[t]:forward(output[t], target[{{}, t}])
    end
    return output, rnn_state, loss / seq_length
end

function RNNModel:backward(input, target, output, rnn_state, seq_length)
    local drnn_state = {[seq_length] = clone_list(self.init_state, true)} -- true also zeros the clones
    for t=seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = self.clones.criterion[t]:backward(output[t], target[{{}, t}])
        table.insert(drnn_state[t], doutput_t)
        local inputs = self:extract_input(input, t)
        for k, v in ipairs(rnn_state[t-1]) do
            table.insert(inputs, v)
        end
        local dlst = self.clones.rnn[t]:backward(inputs, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in ipairs(dlst) do
            -- k == 1 is gradient on x, which we dont need
            if k > #input then 
                drnn_state[t-1][k-#input] = v
            end
        end
    end
end
