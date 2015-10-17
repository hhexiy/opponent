
local GRU = {}

--[[
based on Karparthy's implementation
input: a sequence of words
output: the answer at each word
]]--
function GRU.gru(input_embedding, output_size, rnn_size, n, dropout)
  dropout = dropout or 0 
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  local x, input_size_L
  local emb_size = input_embedding.weight[1]:size(1)
  local outputs = {}
  for L = 1,n do

    local prev_h = inputs[L+1]
    -- the input to this layer
    if L == 1 then 
      --x = OneHot(input_size)(inputs[1])
      --x = nn.LookupTable(input_size, rnn_size)(inputs[1])
      x = nn.Linear(emb_size, rnn_size)(input_embedding(inputs[1]))
      input_size_L = rnn_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(rnn_size, rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end
-- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, output_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

--[[
based on Karparthy's implementation
inputs: input_1, ..., input_nin, hidden_1, ..., hidden_L
output: hidden_1, ..., hidden_L, output_1, ..., output_nout 
]]--
function GRU.gru_new(input_embeddings, output_sizes, rnn_size, n, dropout)
  dropout = dropout or 0
  local nin = #input_embeddings
  local nout = #output_sizes
  -- there are n+nin inputs (hiddens on each layer and x)
  local inputs = {}
  for i=1,nin do
    table.insert(inputs, nn.Identity()()) -- x
  end
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  local x, input_size_L
  local emb_size = 0
  for i,input_embedding in ipairs(input_embeddings) do
    emb_size = emb_size + input_embedding.weight[1]:size(1)
  end
  local outputs = {}
  for L = 1,n do
    -- offset to hidden state is nin
    local prev_h = inputs[L+nin]
    -- the input to this layer
    if L == 1 then 
      local transform = {}
      for i,input_emb in ipairs(input_embeddings) do
        transform[i] = input_emb(inputs[i])
      end
      if nin == 1 then
        x = nn.Linear(emb_size, rnn_size)(transform)
      else
        -- JoinTable: dim=2 for mini-batch, dim=1 is batch_size
        x = nn.Linear(emb_size, rnn_size)(nn.JoinTable(2)(transform))
      end
      input_size_L = rnn_size
    else
      -- (multiple) outputs are after hidden state outputs, so this index is correct
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(rnn_size, rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end
-- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  --for i,output_size in ipairs(output_sizes) do
  --    table.insert(outputs, nn.LogSoftMax()(nn.Linear(rnn_size, output_size)(top_h)))
  --end
  for i,output_size in ipairs(output_sizes) do
    local proj = nn.Linear(rnn_size, output_sizes[1])(top_h)
    local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, logsoft)
  end
  return nn.gModule(inputs, outputs)
end
return GRU
