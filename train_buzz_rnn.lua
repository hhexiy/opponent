
--[[

This file trains a quiz bowl player model. 

]]--

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
cmd:option('-rnn', 'gru', 'lstm, gru or rnn')
cmd:option('-model', 'buzz_correct_rnn', 'buzz_correct')
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
cmd:option('-init_content', '', 'pretrained content model parameters from checkpoint at this path')
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
if opt.savefile == '' then opt.savefile = opt.model .. '_' .. opt.rnn end

require 'setup'
env_setup()
local loader, content_model = qb_setup()
local eval = require 'util.eval'

local buzz_rnn, random_init
if string.len(opt.init_from) > 0 then
    print('loading buzz model from ' .. opt.init_from)
    buzz_rnn = load_model(opt.init_from)
    buzz_rnn.net_params.batch_size = opt.batch_size
    random_init = false
else
    if opt.model == 'buzz_correct_rnn' then
        buzz_rnn = qb.BuzzCorrectRNN(opt.rnn, opt.rnn_size, opt.num_layers, opt.batch_size, opt.dropout, false, 0)
    end
    random_init = true
end
local buzz_model = RNNModel(buzz_rnn, qb.max_seq_length, opt.gpuid, random_init)

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end
    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front

    local ans_loss, buzz_loss = 0, 0
    local ans_logprobs = torch.Tensor(n*opt.batch_size, loader.max_seq_length, qb.ans_size):fill(0)
    local ans_preds = torch.IntTensor(n*opt.batch_size, loader.max_seq_length)
    local ans_targets = torch.IntTensor(n*opt.batch_size, loader.max_seq_length)
    local buzz_preds = torch.IntTensor(n*opt.batch_size, loader.max_seq_length)
    local buzz_targets = torch.IntTensor(n*opt.batch_size, loader.max_seq_length)
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
        local buzz_feat = torch.Tensor(opt.batch_size, seq_length, qb.ans_size+1) 
        for t=1,seq_length do
            ans_logprob[t] = ans_logprob[t]:exp()
            ans_logprobs:sub(from, to, t, t, 1, -1):copy(ans_logprob[t])
            buzz_feat:sub(1, -1, t, t, 1, -2):copy(ans_logprob[t]:clone():sort(2, true))
            buzz_feat:sub(1, -1, t, t, -1, -1):fill(t)
            local _, p = torch.max(ans_logprob[t], 2)
            p = p:type('torch.IntTensor'):squeeze(2)
            ans_preds:sub(from, to, t, t):copy(p)
            local buzz_target = buzz_rnn:oracle_buzz(ans_logprob[t], y[{{}, t}], m[{{}, t}], t)
            --local buzz_target = y[{{}, t}]
            buzz_targets:sub(from, to, t, t):copy(buzz_target)
        end

        -- run buzz model
        --local buzz_logprob, _, bl = buzz_model:forward({ans_logprob, x}, buzz_targets:narrow(1, from, opt.batch_size), seq_length, true) 
        --buzz_feat:fill(7)
        --print(buzz_feat)
        local input
        if buzz_rnn.use_words then 
            input = {buzz_feat, x} 
        else 
            input = {buzz_feat}
        end
        local buzz_logprob, _, bl = buzz_model:forward(input, buzz_targets:narrow(1, from, opt.batch_size), seq_length, true) 
        buzz_loss = buzz_loss + bl
        for t=1,seq_length do
            local _, p = torch.max(buzz_logprob[t], 2)
            buzz_preds:sub(from, to, t, t):copy(p)
        end
        --print('prediction:')
        --print(buzz_preds:sub(from, to, 1, seq_length))
        --print('target:')
        --print(buzz_targets:sub(from, to, 1, seq_length))
    end

    buzz_acc = eval.accuracy(buzz_preds, buzz_targets, mask)
    --ans_avg_acc = eval.accuracy(ans_preds, ans_targets, mask)
    --ans_end_acc = eval.seq_accuracy(ans_preds, ans_targets, mask)
    --ans_max_acc = eval.max_seq_accuracy(ans_preds, ans_targets, mask)
    mm_payoff, mm_mean_pos = eval.max_margin_buzz(ans_logprobs, ans_targets, mask, qids, loader.buzzes) 
    pred_payoff, pred_mean_pos = eval.predicted_buzz(ans_preds, buzz_preds, ans_targets, mask, qids, loader.buzzes) 
    oracle_payoff, oracle_mean_pos = eval.predicted_buzz(ans_preds, buzz_targets, ans_targets, mask, qids, loader.buzzes) 
    ans_loss = ans_loss / n
    buzz_loss = buzz_loss / n
    print(string.format('ans loss = %.8f, buzz loss = %.8f, buzz_acc = %.4f, mm payoff = (%.4f, %.4f), pred payoff = (%.4f, %.4f), oracle payoff = (%.4f, %.4f)', ans_loss, buzz_loss, buzz_acc, mm_payoff, mm_mean_pos, pred_payoff, pred_mean_pos, oracle_payoff, oracle_mean_pos))
    return buzz_loss
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= buzz_model.params then
        buzz_model.params:copy(x)
    end
    buzz_model.grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y, m = loader:next_batch(1, opt.gpuid) -- discard mask
    local seq_length = x:size(2)
    ------------------- forward pass -------------------
    local buzz_preds = {}
    local buzz_targets = torch.IntTensor(buzz_model.net_params.batch_size, seq_length)

    -- get content model predictions
    local ans_logprob, _, ans_loss = content_model:forward({x}, y, seq_length, true)
    local buzz_feat = torch.Tensor(opt.batch_size, seq_length, qb.ans_size+1) 
    for t=1,seq_length do
        --local _, p = torch.max(ans_logprob[t], 2)
        --p = p:type('torch.IntTensor'):squeeze(2)
        --print(p)
        ans_logprob[t] = ans_logprob[t]:exp()
        local buzz_target = buzz_rnn:oracle_buzz(ans_logprob[t], y[{{}, t}], m[{{}, t}], t)
        --local buzz_target = y[{{}, t}]
        buzz_targets:narrow(2, t, 1):copy(buzz_target)
        buzz_feat:sub(1, -1, t, t, 1, -2):copy(ans_logprob[t]:clone():sort(2, true))
        buzz_feat:sub(1, -1, t, t, -1, -1):fill(t)
    end

    -- run buzz model
    --local input = {ans_logprob, x}
    --buzz_feat:fill(7)
    local input 
    if buzz_rnn.use_words then
        input = {buzz_feat, x}
    else
        input = {buzz_feat}
    end
    local buzz_logprob, buzz_rnn_state, buzz_loss = buzz_model:forward(input, buzz_targets, seq_length) 
    --print(buzz_targets)
    --print('buzz:')
    --for t=1,seq_length do
    --    local _, p = torch.max(buzz_logprob[t], 2)
    --    print('prediction:')
    --    print(p)
    --    print('target:')
    --    print(buzz_targets[{{},t}])
    --end

    ------------------ backward pass -------------------
    buzz_model:backward(input, buzz_targets, buzz_logprob, buzz_rnn_state, seq_length)
    -- TODO: double check and save checkpoint

    ------------------------ misc ----------------------
    -- clip gradient element-wise
    buzz_model.grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    --print(buzz_model.grad_params)
    return buzz_loss, buzz_model.grad_params
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
    local _, loss = optim.rmsprop(feval, buzz_model.params, optim_state)
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

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, buzz_model.grad_params:norm() / buzz_model.params:norm(), time))
    end
   
    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.rnn = rnn
        --checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
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


