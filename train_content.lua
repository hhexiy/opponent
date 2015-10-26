
--[[

This file trains a quiz bowl player model. 

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local model_utils = require 'util.model_utils'
local eval = require 'util.eval'
include('model/RNNModel.lua')

torch.setheaptracking(true)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a question-answering model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','dat/qb','data directory. Should contain the file input.txt with input data')
cmd:option('-input_file','input.txt','data file name')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-rnn', 'lstm', 'lstm, gru or rnn')
cmd:option('-embedding', 'dat/glove', 'directory of pretrained word embeddings')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-finetune_after',10,'in number of epochs, when to start finetuning the content model')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',0,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', '/fs/clip-scratch/hhe/opponent/cv', 'output directory where checkpoints get written')
cmd:option('-savefile','','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
-- debug
cmd:option('-debug',0,'debug mode: printouts and assertions')
-- test
cmd:option('-test',0,'evaluate on test set')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
if opt.debug == 1 then nngraph.setDebug(true) end
if opt.savefile == '' then opt.savefile = 'content_' .. opt.rnn end

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

qb = {}
include('util/MinibatchLoader.lua')

-- create the data loader class
local loader = nil
loader = qb.QAMinibatchLoader(opt.data_dir, opt.input_file, opt.batch_size)
loader:load_data()
qb.vocab_size = loader.vocab_size  
qb.vocab = loader.vocab_mapping

-- word embedding
if string.len(opt.embedding) > 0 then
    qb.word_embedding = loader:load_embedding(opt.embedding)
    qb.emb_size = qb.word_embedding.weight[1]:size(1)
else
    qb.word_embedding = nn.LookupTable(qb.vocab_size, 300) 
    qb.emb_size = 300
end
print('word embedding size: ' .. qb.emb_size)

qb.ans_size = loader.ans_size
qb.max_seq_length = loader.max_seq_length

include('model/QBModels.lua')

function load_model(path)
    local checkpoint = torch.load(path)
    local vocab_compatible = check_vocab_compatible(checkpoint.vocab, qb.vocab)
    assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
    return checkpoint.rnn
end

local content_rnn, content_model, random_init
if string.len(opt.init_from) > 0 then
    print('loading content model from ' .. opt.init_from)
    content_rnn = load_model(opt.init_from)
    content_rnn.net_params.batch_size = opt.batch_size
    random_init = false
else
    content_rnn = qb.Content(opt.rnn, opt.rnn_size, opt.num_layers, opt.batch_size, opt.dropout)
    random_init = true
end
content_model = RNNModel(content_rnn, qb.max_seq_length, opt.gpuid, random_init)

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end
    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front

    local ans_loss = 0
    local ans_logprobs = torch.Tensor(n*opt.batch_size, loader.max_seq_length, qb.ans_size):fill(0)
    local ans_preds = torch.IntTensor(n*opt.batch_size, loader.max_seq_length)
    local ans_targets = torch.IntTensor(n*opt.batch_size, loader.max_seq_length)
    local qids = torch.IntTensor(n*opt.batch_size)
    local mask = torch.ByteTensor(n*opt.batch_size, loader.max_seq_length):fill(0)
    
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y, m, qid = loader:next_batch(split_index, opt.gpuid)
        -- seq_length is different for each batch (max length in *this* batch)
        local seq_length = x:size(2)
        -- starting id of this batch
        local from = (i-1)*opt.batch_size + 1
        local to = from+opt.batch_size-1
        -- TODO: instead of copy, try set
        ans_targets:sub(from, to, 1, seq_length):copy(y)
        mask:sub(from, to, 1, seq_length):copy(m)
        qids:sub(from, to):copy(qid)

        -- get content model predictions
        local ans_logprob, _, al = content_model:forward({x}, y, seq_length, true)
        ans_loss = ans_loss + al
        for t=1,seq_length do
            ans_logprobs:sub(from, to, t, t, 1, -1):copy(ans_logprob[t])
            local _, p = torch.max(ans_logprob[t], 2)
            ans_preds:sub(from, to, t, t):copy(p)
        end
    end

    ans_avg_acc = eval.accuracy(ans_preds, ans_targets, mask)
    ans_end_acc = eval.seq_accuracy(ans_preds, ans_targets, mask)
    ans_max_acc = eval.max_seq_accuracy(ans_preds, ans_targets, mask)
    mm_payoff, mm_mean_pos = eval.max_margin_buzz(ans_logprobs, ans_targets, mask, qids, loader.buzzes) 
    ans_loss = ans_loss / n
    print(string.format('loss = %.8f, avg acc = %.4f, end acc: %.4f, max acc: %.4f, mm payoff = (%.4f, %.4f)', ans_loss, ans_avg_acc, ans_end_acc, ans_max_acc, mm_payoff, mm_mean_pos))
    return ans_loss
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= content_model.params then
        content_model.params:copy(x)
    end
    content_model.grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y, m = loader:next_batch(1, opt.gpuid) -- discard mask
    local seq_length = x:size(2)
    ------------------- forward pass -------------------
    local ans_logprob, ans_rnn_state, ans_loss = content_model:forward({x}, y, seq_length)
    ------------------ backward pass -------------------
    content_model:backward({x}, y, ans_logprob, ans_rnn_state, seq_length)
    -- TODO: double check and save checkpoint
    ------------------------ misc ----------------------
    -- clip gradient element-wise
    content_model.grad_params:clamp(-opt.grad_clip, opt.grad_clip)

    return ans_loss, content_model.grad_params
end

-- test only
if opt.test == 1 then
    local test_loss = eval_split(3)
    os.exit()
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
--local optim_state = {learningRate = opt.learning_rate}
local ntrain = loader.split_sizes[1]
if opt.eval_val_every == 0 then
    opt.eval_val_every = ntrain
end
local iterations = opt.max_epochs * ntrain
local iterations_per_epoch = ntrain
local loss0 = nil
local epoch = 0
for i = 1, iterations do
    epoch = i / ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, content_model.params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.rnn = content_rnn
        --checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, content_model.grad_params:norm() / content_model.params:norm(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end


