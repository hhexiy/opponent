
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

require 'util.misc'

local MinibatchLoader = torch.class('qb.MinibatchLoader')

qb.UNK = '<unk>'
qb.PAD = '<eos>'

function MinibatchLoader:__init(data_dir, input_file, batch_size)
    self.input_file = path.join(data_dir, input_file)
    self.batch_size = batch_size
end

function MinibatchLoader:load_data(vocab_mapping, ans_mapping)
    -- create mappings: vocab and ans 
    -- and tables: questions and buzzes
    print('loading data from text file ' .. self.input_file)
    self:text_to_tensor(self.input_file, vocab_mapping, ans_mapping)

    -- make batches: {x, y, mask}
    -- train=1; dev=2; test=3
    print('making batches of size ' .. self.batch_size)
    self.split_sizes = {}
    self.batches = {}
    for split_index=1,3 do
        local nbatches
        if self.questions[split_index] == nil then
            nbatches = 0
        else
            self.batches[split_index] = self:make_batches(self.questions[split_index], self.batch_size)
            nbatches = #self.batches[split_index][1]
        end
        self.split_sizes[split_index] = nbatches
        print(string.format('split %d: %d batches', split_index, nbatches))
    end

    self.batch_ix = {0,0,0}

    print('data loading done')
    collectgarbage()
end

function MinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function MinibatchLoader:next_batch(split_index)
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
    -- return x, y, m, qid
    local x = self.batches[split_index][1][ix]
    local y = self.batches[split_index][2][ix] 
    local m = self.batches[split_index][3][ix] 
    local qid = self.batches[split_index][4][ix]
    -- ship the input arrays to GPU
    if opt.gpuid >= 0 then
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
        m = m:float():cuda()
        qid = qid:float():cuda()
    end
    return x, y, m, qid
end

function MinibatchLoader:create_mapping(vocab, special_toks)
    -- vocab is a table of tokens: {tok:true}
    -- special toks is an array: e.g. <eos>, unk
    local size = 0
    local mapping = {}
    for tok in pairs(vocab) do 
        size = size + 1
        mapping[tok] = size
    end
    -- might want order for special toks
    if special_toks ~= nil then
        for i, stok in ipairs(special_toks) do
            size = size + 1
            mapping[stok] = size
        end
    end
    return size, mapping
end

function MinibatchLoader:load_embedding(emb_dir)
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
        -- word without pretrained embedding
        if not emb_vocab[tok] then
            num_unks = num_unks + 1
            emb.weight[i]:uniform(-0.05, 0.05) 
            --emb.weight[i] = torch.randn(emb_size)
        else
            emb.weight[i] = torch.Tensor(emb_size):copy(emb_vec[emb_vocab[tok]])
        end
    end
    print('# words without pretrained embedding: ' .. num_unks)
    return emb
end

-- load questions and answers
local QAMinibatchLoader = torch.class('qb.QAMinibatchLoader', 'qb.MinibatchLoader')

function QAMinibatchLoader:__init(data_dir, input_file, batch_size)
    MinibatchLoader:__init(data_dir, input_file, batch_size)
end

function QAMinibatchLoader:text_to_tensor(in_textfile, vocab_mapping, ans_mapping)
    local timer = torch.Timer()

    local f = io.open(in_textfile, "r")
    
    -- create vocabulary if it doesn't exist yet
    local vocab, num_unks
    if vocab_mapping == nil then
        print('creating vocabulary mapping...')
        vocab = {}
    else
        print('using given vocabulary mapping...')
        num_unks = 0
        vocab = vocab_mapping
    end
    -- create answer set if it doesn't exist yet
    local ans = {}
    if ans_mapping == nil then
        print('creating answer mapping...')
    else
        print('using given answer mapping...')
    end
    local user = {}
    local maxlen = 0
    local num_examples = 0
    local num_buzzes = 0
    local split_sizes = {0, 0, 0}
    local split_buzz_sizes = {0, 0, 0}
    while true do
        local line = f:read()
        if line == nil then break end
        local ss = string.split(line, ' ||| ')
        -- tok dict
        local toks = string.split(ss[4], ' ')
        for i, tok in pairs(toks) do
            if vocab[tok] == nil then 
                if vocab_mapping == nil then
                    vocab[tok] = true 
                else
                    num_unks = num_unks + 1
                end
            end
        end
        -- check answer is in given answer mapping
        if ans_mapping ~= nil then
            assert(ans_mapping[ss[2]] ~= nil)
        end
        -- ans dict (count)
        if ans[ss[2]] == nil then 
            ans[ss[2]] = 1 
        else 
            ans[ss[2]] = ans[ss[2]] + 1
        end
        -- user dict
        local buzzes = {}
        if #ss > 4 then
            buzzes = string.split(ss[5], '|')
            for i, buzz in ipairs(buzzes) do
                local ss = string.split(buzz, '-')
                local uid = tonumber(ss[1]) -- user id
                user[uid] = true
            end
        end
        -- counts
        if ss[3] == 'train' then 
            split_sizes[1] = split_sizes[1] + 1
            split_buzz_sizes[1] = split_buzz_sizes[1] + #buzzes
        elseif ss[3] == 'dev' then 
            split_sizes[2] = split_sizes[2] + 1
            split_buzz_sizes[2] = split_buzz_sizes[2] + #buzzes
        else 
            split_sizes[3] = split_sizes[3] + 1
            split_buzz_sizes[3] = split_buzz_sizes[3] + #buzzes
        end
        maxlen = math.max(maxlen, #toks)
        num_examples = num_examples + 1
        num_buzzes = num_buzzes + #buzzes
    end
    f:close()

    if vocab_mapping == nil then
        self.vocab_size, self.vocab_mapping = self:create_mapping(vocab, {qb.PAD, qb.UNK})
    else
        self.vocab_mapping = vocab_mapping
        self.vocab_size = table_size(vocab_mapping)
        print('# unknown words: ' .. num_unks)
    end
    if ans_mapping == nil then
        self.ans_size, self.ans_mapping = self:create_mapping(ans, {qb.PAD})
    else
        self.ans_mapping = ans_mapping
        self.ans_size = table_size(ans_mapping)
    end
    self.user_size, self.user_mapping = self:create_mapping(user, {})

    self.num_questions = split_sizes
    self.num_buzzes = split_buzz_sizes
    print(string.format('# questions: %d(%d/%d/%d)', num_examples, split_sizes[1], split_sizes[2], split_sizes[3]))
    print(string.format('# buzzes: %d(%d/%d/%d)', num_buzzes, split_buzz_sizes[1], split_buzz_sizes[2], split_buzz_sizes[3]))
    print('maximum sequence length: ' .. maxlen)
    self.max_seq_length = maxlen
    print('vocab size: ' .. self.vocab_size)
    print('# answers/classes: ' .. self.ans_size)
    print('# users: ' .. self.user_size)

    -- answer counts
    local ans_count = torch.IntTensor(self.ans_size):zero()
    for a, n in pairs(ans) do
        ans_count[self.ans_mapping[a]] = n
    end
    -- sort answers from most questions to least
    _, self.sorted_ans = ans_count:sort(1, true) 
    print('majority class: ' .. self.sorted_ans[1], ans_count[self.sorted_ans[1]])
    for a, i in pairs(self.ans_mapping) do
        if i == self.sorted_ans[1] then
            print(a)
            break
        end
    end

    -- construct a tensor with all the data
    print('putting data into tensor...')
    self.questions = {}
    self.buzzes = {}
    for split_index=1,3 do
        -- +3: qid, ans, length of question 
        if split_sizes[split_index] > 0 then
            self.questions[split_index] = torch.IntTensor(split_sizes[split_index], maxlen+3)
        end
    end
    f = io.open(in_textfile, "r")
    local split_count = {0, 0, 0}
    local num_nobuzz = 0
    while true do
        local line = f:read()
        if line == nil then break end

        local ss = string.split(line, ' ||| ')
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
            if self.vocab_mapping[tok] == nil then
                t[i+3] = self.vocab_mapping[qb.UNK]
            else
                t[i+3] = self.vocab_mapping[tok]
            end
        end
        -- pad
        if #toks < maxlen then
            for i = #toks+1, maxlen do
                t[i+3] = self.vocab_mapping[qb.PAD]
            end
        end

        -- buzz data
        -- lua string.split removes empty string :(
        if #ss == 4 then
            num_nobuzz = num_nobuzz + 1
        else
            self.buzzes[qid] = {}
            local buzzes = string.split(ss[5], '|')
            for i, buzz in ipairs(buzzes) do
                --print(buzz)
                local ss = string.split(buzz, '-')
                --print(ss)
                local t = torch.IntTensor(3)
                t[1] = self.user_mapping[tonumber(ss[1])] -- user id
                t[2] = tonumber(ss[2]) -- buzz position
                t[3] = tonumber(ss[3]) -- correct or not
                self.buzzes[qid][i] = t
            end
        end
    end
    print('# questions without buzz: ' .. num_nobuzz)
    f:close()
end

function QAMinibatchLoader:make_batches(data, batch_size)
    -- data is a tensor of one split
    local x_batches = {}
    local y_batches = {}
    local m_batches = {} -- mask for padded data
    local qid_batches = {}
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
        qid_batches[i] = data[{{from,to}, 1}]
        -- use same labels for padded word
        --y_batches[i] = torch.IntTensor(batch_size, 1):copy(data[{{from,to}, 2}]):expand(batch_size, seq_length)
        -- use <eos> as label for padded word
        y_batches[i] = torch.IntTensor(batch_size, 1):copy(data[{{from,to}, 2}]):repeatTensor(1, seq_length)
        for j=from,to do
            if data[j][3]+1 < seq_length then
                y_batches[i][j-from+1]:sub(data[j][3]+1, -1):fill(self.ans_mapping[qb.PAD])
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
    return {x_batches, y_batches, m_batches, qid_batches}
end

-- load questions and buzzes
local QBMinibatchLoader = torch.class('qb.QBMinibatchLoader', 'qb.MinibatchLoader')

function QBMinibatchLoader:__init(data_dir, input_file, batch_size)
    MinibatchLoader:__init(data_dir, input_file, batch_size)
end

function QBMinibatchLoader:make_batches(data, batch_size)
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
                y_batches[i][j-from+1]:sub(data[j][3]+1, -1):fill(self.ans_mapping[qb.PAD])
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

