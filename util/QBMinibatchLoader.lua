
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

local QBMinibatchLoader = {}
QBMinibatchLoader.__index = QBMinibatchLoader

function QBMinibatchLoader.create(data_dir, input_file, batch_size)

    local self = {}
    setmetatable(self, QBMinibatchLoader)

    local input_file = path.join(data_dir, input_file)
    print('loading data ...')
    -- create mappings: vocab and ans 
    -- and tables: questions and buzzes
    self:text_to_tensor(input_file)

    -- make batches: {x, y, mask}
    -- train=1; dev=2; test=3
    print('making batches of size ' .. batch_size)
    self.batch_size = batch_size
    self.split_sizes = {}
    self.batches = {}
    for split_index=1,3 do
        self.batches[split_index] = self:make_batches(self.questions[split_index], batch_size)
        local nbatches = #self.batches[split_index][1]
        self.split_sizes[split_index] = nbatches
        print(string.format('split %d: %d batches', split_index, nbatches))
    end

    self.batch_ix = {0,0,0}

    print('data loading done')
    collectgarbage()
    return self
end

function QBMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function QBMinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    -- return x, y, m
    return self.batches[split_index][1][ix], self.batches[split_index][2][ix], self.batches[split_index][3][ix]
end

function QBMinibatchLoader:make_batches(data, batch_size)
    -- TODO: different model might need different data: may need new make_batches
    -- data is a tensor of one split
    local x_batches = {}
    local y_batches = {}
    local m_batches = {} -- mask for padded data
    local N = data:size(1)
    local num_batches = math.floor(N / batch_size)
    for i=1,num_batches do
        local from = (i-1) * batch_size + 1
        local to = from + batch_size - 1
        local residual = 0
        -- repeat example to make equal-sized batches
        if to > N then
           residual = N - to
           to = N
           from = from - residual
        end
        local seq_length = data[{{from,to}, 3}]:max() -- for padding
        x_batches[i] = data[{{from,to}, {4, 3+seq_length}}]
        -- use same labels for padded word
        --y_batches[i] = torch.IntTensor(batch_size, 1):copy(data[{{from,to}, 2}]):expand(batch_size, seq_length)
        -- use <eos> as label for padded word
        y_batches[i] = torch.IntTensor(batch_size, 1):copy(data[{{from,to}, 2}]):repeatTensor(1, seq_length)
        for j=from,to do
            if data[j][3]+1 < seq_length then
                y_batches[i][j-from+1]:sub(data[j][3]+1, -1):fill(self.ans_mapping['<eos>'])
            end
        end
        if opt.debug == 1 then
            assert(x_batches[i]:lt(1):sum() == 0)
            assert(x_batches[i]:gt(self.vocab_size):sum() == 0)
            assert(y_batches[i]:lt(1):sum() == 0)
            assert(y_batches[i]:gt(self.ans_size):sum() == 0)
            assert(y_batches[i]:isSameSizeAs(x_batches[i]))
        end
        m_batches[i] = torch.ByteTensor(x_batches[i]:size()):fill(1)
        -- mask repeated examples for evaluation
        if residual > 0 then
            m_batches[i]:narrow(1, 1, residual):fill(0)
        end
        -- mask padded words
        for j=from+residual,to do
            local len = data[j][3]
            if len < seq_length then
                m_batches[i][j-from+1]:sub(len+1, -1):fill(0)
            end
        end
    end
    return {x_batches, y_batches, m_batches}
end

function QBMinibatchLoader:text_to_tensor(in_textfile)
    local timer = torch.Timer()

    print('loading text file ' .. in_textfile)
    local f = io.open(in_textfile, "r")
    
    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    -- record all words to a set
    local vocab = {}
    local ans = {}
    local maxlen = 0
    local num_examples = 0
    local split_sizes = {0, 0, 0}
    while true do
        local line = f:read()
        if line == nil then break end
        local ss = string.split(line, ',')
        -- tok dict
        local toks = string.split(ss[4], ' ')
        for i, tok in pairs(toks) do
            if vocab[tok] == nil then vocab[tok] = true end
        end
        -- ans dict (count)
        if ans[ss[2]] == nil then 
            ans[ss[2]] = 1 
        else 
            ans[ss[2]] = ans[ss[2]] + 1
        end
        -- counts
        if ss[3] == 'train' then 
            split_sizes[1] = split_sizes[1] + 1
        elseif ss[3] == 'dev' then 
            split_sizes[2] = split_sizes[2] + 1
        else 
            split_sizes[3] = split_sizes[3] + 1
        end
        maxlen = math.max(maxlen, #toks)
        num_examples = num_examples + 1
    end
    f:close()

    self.vocab_size, self.vocab_mapping = QBMinibatchLoader.create_mapping(vocab, {'<eos>'})
    self.ans_size, self.ans_mapping = QBMinibatchLoader.create_mapping(ans, {'<eos>'})

    print(string.format('number of examples: %d(%d/%d/%d)', num_examples, split_sizes[1], split_sizes[2], split_sizes[3]))
    print('maximum sequence length: ' .. maxlen)
    self.max_seq_length = maxlen
    print('vocab size: ' .. self.vocab_size)
    print('number of answers/classes: ' .. self.ans_size)

    -- answer counts
    local ans_count = torch.IntTensor(self.ans_size)
    for a, n in pairs(ans) do
        ans_count[self.ans_mapping[a]] = n
    end
    -- sort answers from most questions to least
    _, self.sorted_ans = ans_count:sort(1, true) 

    -- construct a tensor with all the data
    print('putting data into tensor...')
    self.questions = {}
    self.buzzes = {}
    for split_index=1,3 do
        -- +3: qid, ans, length of question 
        self.questions[split_index] = torch.IntTensor(split_sizes[split_index], maxlen+3)
    end
    f = io.open(in_textfile, "r")
    local split_count = {0, 0, 0}
    local num_nobuzz = 0
    while true do
        local line = f:read()
        if line == nil then break end

        local ss = string.split(line, ',')
        local split_index = ss[3] == 'train' and 1 or ss[3] == 'dev' and 2 or 3
        split_count[split_index] = split_count[split_index] + 1

        -- question/answer data
        local t = self.questions[split_index][split_count[split_index]]
        local qid = tonumber(ss[1])
        t[1] = qid
        t[2] = self.ans_mapping[ss[2]] -- ans
        local toks = string.split(ss[4], ' ')
        t[3] = #toks -- len
        for i, tok in ipairs(toks) do
            t[i+3] = self.vocab_mapping[tok]
        end
        -- pad
        if #toks < maxlen then
            for i = #toks+1, maxlen do
                t[i+3] = self.vocab_mapping['<eos>']
            end
        end

        -- buzz data
        self.buzzes[qid] = {}
        -- lua string.split removes empty string :(
        if #ss == 4 then
            num_nobuzz = num_nobuzz + 1
        else
            local buzzes = string.split(ss[5], '|')
            for i, buzz in ipairs(buzzes) do
                --print(buzz)
                local ss = string.split(buzz, '-')
                --print(ss)
                local t = torch.IntTensor(3)
                t[1] = tonumber(ss[1]) -- user id
                t[2] = tonumber(ss[2]) -- buzz position
                t[3] = tonumber(ss[3]) -- correct or not
                self.buzzes[qid][i] = t
            end
        end
    end
    print('number of questions without buzz: ' .. num_nobuzz)
    f:close()
end

-- static function
function QBMinibatchLoader.create_mapping(vocab, special_toks)
    -- vocab is a table of tokens
    -- special toks is an array: e.g. <eos>, unk
    local size = 0
    local mapping = {}
    for tok in pairs(vocab) do 
        size = size + 1
        mapping[tok] = size
    end
    -- might want order for special toks
    for i, stok in ipairs(special_toks) do
        size = size + 1
        mapping[stok] = size
    end
    return size, mapping
end

function QBMinibatchLoader:load_embedding(emb_dir)
    print('loading embedding from ' .. emb_dir)
    -- read embedding vocabulary
    local f = io.open(emb_dir .. '/vocab.txt', 'r')
    local emb_vocab = {}
    local size = 0
    while true do
        local line = f:read()
        if not line then break end
        size = size + 1
        emb_vocab[line] = size
    end

    -- load embedding
    local emb_vec = torch.load(emb_dir .. '/vec.t7')
    assert(emb_vec:size(1) == size)
    local emb_size = emb_vec:size(2)

    -- embedding layer
    print('creating LookupTable...')
    local num_unks = 0
    local emb = nn.LookupTable(self.vocab_size, emb_size)
    for tok, i in pairs(self.vocab_mapping) do
        -- unk word
        if not emb_vocab[tok] then
            num_unks = num_unks + 1
            emb.weight[i]:uniform(-0.05, 0.05) 
            --emb.weight[i] = torch.randn(emb_size)
        else
            --emb.weight[i]:uniform(-0.05, 0.05) 
            --emb.weight[i] = torch.randn(emb_size)
            emb.weight[i] = torch.Tensor(emb_size):copy(emb_vec[emb_vocab[tok]])
        end
    end
    print('number of unk: ' .. num_unks)
    return emb
end

return QBMinibatchLoader

