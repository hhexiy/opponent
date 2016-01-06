
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

function MinibatchLoader:filter_vocab(vocab, cutoff)
    for w, count in pairs(vocab) do
        if count < cutoff then
            vocab[w] = nil
        end
    end
    return vocab
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
    local mapping_file = in_textfile .. '.mapping.t7'

    if not path.exists(mapping_file) then
        print(mapping_file .. ' does not exist. Preprocessing...' )

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
        local category = {}
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
            local cat = ss[2]
            local answer = ss[3]
            local fold = ss[4]
            local toks = string.split(ss[5], ' ')
            local buzzes = #ss > 5 and string.split(ss[6], '|') or {}
            -- category mapping
            category[cat] = true
            -- tok dict
            for i, tok in pairs(toks) do
                -- no pre-load vocab
                if vocab_mapping == nil then
                    if vocab[tok] == nil then
                        vocab[tok] = 1
                    else
                        vocab[tok] = vocab[tok] + 1
                    end
                else
                    num_unks = num_unks + 1
                end
            end
            -- check answer is in given answer mapping
            if ans_mapping ~= nil then
                assert(ans_mapping[answer] ~= nil)
            end
            -- ans dict (count)
            if ans[answer] == nil then 
                ans[answer] = 1 
            else 
                ans[answer] = ans[answer] + 1
            end
            -- user dict
            for i, buzz in ipairs(buzzes) do
                local ss = string.split(buzz, '-')
                local uid = ss[1] -- user id
                user[uid] = true
            end
            -- counts
            if fold == 'train' then 
                split_sizes[1] = split_sizes[1] + 1
                split_buzz_sizes[1] = split_buzz_sizes[1] + #buzzes
            elseif fold == 'dev' then 
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
            local cutoff = 5
            print('Removing words appearing fewer than ' .. cutoff .. ' time from vocabulary')
            vocab = self:filter_vocab(vocab, cutoff)
        end

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
        self.cat_size, self.cat_mapping = self:create_mapping(category, {})

        self.num_questions = split_sizes
        self.num_buzzes = split_buzz_sizes
        self.max_seq_length = maxlen
        print(string.format('# questions: %d(%d/%d/%d)', num_examples, split_sizes[1], split_sizes[2], split_sizes[3]))
        print(string.format('# buzzes: %d(%d/%d/%d)', num_buzzes, split_buzz_sizes[1], split_buzz_sizes[2], split_buzz_sizes[3]))

        print('saving ' .. mapping_file)
        local mappings = {}
        mappings.vocab_size = self.vocab_size
        mappings.vocab_mapping = self.vocab_mapping
        mappings.ans_size = self.ans_size
        mappings.ans_mapping = self.ans_mapping
        mappings.user_size = self.user_size
        mappings.user_mapping = self.user_mapping
        mappings.cat_size = self.cat_size
        mappings.cat_mapping = self.cat_mapping
        mappings.num_questions = self.num_questions
        mappings.num_buzzes = self.num_buzzes
        mappings.max_seq_length = self.max_seq_length
        torch.save(mapping_file, mappings)
    else
        print('loading ' .. mapping_file)
        local mappings = torch.load(mapping_file)
        self.vocab_size = mappings.vocab_size
        self.vocab_mapping = mappings.vocab_mapping
        self.ans_size = mappings.ans_size
        self.ans_mapping = mappings.ans_mapping
        self.user_size = mappings.user_size
        self.user_mapping = mappings.user_mapping
        self.cat_size = mappings.cat_size
        self.cat_mapping = mappings.cat_mapping
        self.num_questions = mappings.num_questions
        self.num_buzzes = mappings.num_buzzes
        self.max_seq_length = mappings.max_seq_length
    end

    print('maximum sequence length: ' .. self.max_seq_length)
    print('vocab size: ' .. self.vocab_size)
    print('# answers/classes: ' .. self.ans_size)
    print('# users: ' .. self.user_size)
    print('# categories: ' .. self.cat_size)

    local time = timer:time().real
    print(string.format('creating mappings: %.2f', time))

    ---- answer counts
    --local ans_count = torch.IntTensor(self.ans_size):zero()
    --for a, n in pairs(ans) do
    --    ans_count[self.ans_mapping[a]] = n
    --end
    ---- sort answers from most questions to least
    --_, self.sorted_ans = ans_count:sort(1, true) 
    --print('majority class: ' .. self.sorted_ans[1], ans_count[self.sorted_ans[1]])
    --for a, i in pairs(self.ans_mapping) do
    --    if i == self.sorted_ans[1] then
    --        print(a)
    --        break
    --    end
    --end

    -- construct a tensor with all the data
    timer:reset()
    local tensor_file = in_textfile .. '.tensor.t7'
    if not path.exists(tensor_file) then
        print(tensor_file .. ' does not exist. putting data into tensor...' )
        self.questions = {}
        self.buzzes = {}
        self.categories = {}
        for split_index=1,3 do
            -- +3: qid, ans, length of question 
            if self.num_questions[split_index] > 0 then
                self.questions[split_index] = torch.IntTensor(self.num_questions[split_index], self.max_seq_length+3):zero()
            end
        end
        f = io.open(in_textfile, "r")
        local split_count = {0, 0, 0}
        local num_nobuzz = 0
        while true do
            local line = f:read()
            if line == nil then break end

            local ss = string.split(line, ' ||| ')
            local cat = ss[2]
            local answer = ss[3]
            local fold = ss[4]
            local toks = string.split(ss[5], ' ')
            local buzzes = #ss > 5 and string.split(ss[6], '|') or {}
            local split_index = fold == 'train' and 1 or fold == 'dev' and 2 or 3
            split_count[split_index] = split_count[split_index] + 1

            -- question/answer data
            local t = self.questions[split_index][split_count[split_index]]
            local qid = tonumber(ss[1])
            t[1] = qid
            t[2] = self.ans_mapping[answer] -- ans
            t[3] = #toks -- len
            for i, tok in ipairs(toks) do
                if self.vocab_mapping[tok] == nil then
                    t[i+3] = self.vocab_mapping[qb.UNK]
                else
                    t[i+3] = self.vocab_mapping[tok]
                end
            end
            -- pad
            if #toks < self.max_seq_length then
                for i = #toks+1, self.max_seq_length do
                    t[i+3] = self.vocab_mapping[qb.PAD]
                end
            end

            -- category
            self.categories[qid] = self.cat_mapping[cat]

            -- buzz data
            -- lua string.split removes empty string :(
            if #buzzes == 0 then
                num_nobuzz = num_nobuzz + 1
            else
                self.buzzes[qid] = {}
                for i, buzz in ipairs(buzzes) do
                    local ss = string.split(buzz, '-')
                    local t = torch.IntTensor(3)
                    t[1] = self.user_mapping[ss[1]] -- user id
                    t[2] = tonumber(ss[2]) -- buzz position
                    t[3] = tonumber(ss[3]) -- correct or not
                    self.buzzes[qid][i] = t
                end
            end
        end
        print('# questions without buzz: ' .. num_nobuzz)
        f:close()

        print('saving ' .. tensor_file)
        local tensors = {} 
        tensors.questions = self.questions
        tensors.buzzes = self.buzzes
        tensors.categories = self.categories
        torch.save(tensor_file, tensors)
    else
        print('loading ' .. tensor_file)
        local tensors = torch.load(tensor_file)
        self.questions = tensors.questions
        self.buzzes = tensors.buzzes
        self.categories = tensors.categories
    end

    -- get topk categories
    local cat_counts = torch.IntTensor(self.cat_size):zero()
    for qid, cat in pairs(self.categories) do
        cat_counts[cat] = cat_counts[cat] + 1
    end
    local sorted_cat_counts, sorted_cats = torch.sort(cat_counts, 1, true)
    local topk_cats = 0
    local top_cats = {}
    print('top categories:')
    for i=1,topk_cats do 
        print(self.cat_mapping[sorted_cats[i]])
        top_cats[sorted_cats[i]] = i
    end

    -- user stats on training set
    -- uid has been mapped
    -- TODO: topk cats
    self.user_stats = torch.FloatTensor(self.user_size, 3*(topk_cats+1)):zero()
    local questions = self.questions[1]
    for i=1,questions:size(1) do
        local qid = questions[i][1]
        local qlen = questions[i][3]
        local qcat = self.categories[qid]
        for i, buzz in ipairs(self.buzzes[qid]) do
            local uid = buzz[1]
            local buzz_pos = buzz[2]
            local correct = buzz[3]
            -- number of overall questions answered
            self.user_stats[uid][1] = self.user_stats[uid][1] + 1
            self.user_stats[uid][2] = self.user_stats[uid][2] + buzz_pos / qlen
            self.user_stats[uid][3] = self.user_stats[uid][3] + correct 
            -- number of questions answered of one category
            if top_cats[qcat] ~= nil then
                local offset = top_cats[qcat]*3
                self.user_stats[uid][offset+1] = self.user_stats[uid][offset+1] + 1
                self.user_stats[uid][offset+2] = self.user_stats[uid][offset+2] + buzz_pos / qlen
                self.user_stats[uid][offset+3] = self.user_stats[uid][offset+3] + correct 
            end
        end
    end
    -- get mean
    self.user_stats[{{},2}]:cdiv(self.user_stats[{{},1}])
    self.user_stats[{{},3}]:cdiv(self.user_stats[{{},1}])
    self.user_stats[{{},1}]:div(self.num_questions[1])
    for i=1,topk_cats do
        local offset = i*3
        local ind = self.user_stats[{{},offset+1}]:gt(0)
        self.user_stats[{{},offset+2}][ind] = self.user_stats[{{},offset+2}][ind]:cdiv(self.user_stats[{{},offset+1}][ind])
        self.user_stats[{{},offset+3}][ind] = self.user_stats[{{},offset+3}][ind]:cdiv(self.user_stats[{{},offset+1}][ind])
        self.user_stats[{{},offset+1}]:div(sorted_cat_counts[i])
    end
    assert(self.user_stats:gt(1):sum() == 0)
    assert(self.user_stats:lt(0):sum() == 0) 

    time = timer:time().real
    print(string.format('creating tensors: %.2f', time))
end

function QAMinibatchLoader:make_batches(data, batch_size)
    -- data is a tensor of one split
    local x_batches = {}
    local y_batches = {}
    local m_batches = {} -- mask for padded data
    local qid_batches = {}
    local N = data:size(1)
    local num_batches = math.ceil(N / batch_size)
    for i=1,num_batches do
        local from = (i-1) * batch_size + 1
        local to = from + batch_size - 1
        local residual = 0
        -- repeat example to make equal-sized batches
        if to > N then
           residual = to - N
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
            if data[j][3] < seq_length then
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
           residual = to - N
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
            local len = data[j][3]
            if len < seq_length then
                y_batches[i][j-from+1]:sub(len+1, -1):fill(self.ans_mapping[qb.PAD])
                m_batches[i][j-from+1]:sub(len+1, -1):fill(0)
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
    end
    return {x_batches, y_batches, m_batches}
end

