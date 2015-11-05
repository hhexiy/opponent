
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
cmd:option('-hidden_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-rnn', 'lstm', 'lstm, gru or rnn')
cmd:option('-model', 'buzz_correct', 'buzz models')
cmd:option('-embedding', 'dat/glove', 'directory of pretrained word embeddings')
cmd:option('-use_content_state', 0, 'whether to use hidden states from content model as a feature')
cmd:option('-history', 0, 'whether to use previous state features')
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
cmd:option('-save',1,'save or not')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
-- debug
cmd:option('-debug',0,'debug mode: printouts and assertions')
-- test
cmd:option('-test',0,'evaluate on test set')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
if opt.savefile == '' then opt.savefile = opt.model .. '_mlp' end

require 'setup'
env_setup()
local loader, content_model = qb_setup()

local eval = require 'util.eval'

-- create content and buzz model
local buzz_rnn, input_sizes
if string.len(opt.init_from) > 0 then
    print('loading buzz model from ' .. opt.init_from)
    local checkpoint = load_model(opt.init_from)
    buzz_model = checkpoint.model 
    opt.use_content_state = checkpoint.opt.use_content_state
    opt.history = checkpoint.opt.history
end
if opt.use_content_state == 1 then
    input_sizes = {qb.ans_size*(opt.history+1)+1, content_model.net_params.rnn_size}
else
    input_sizes = {qb.ans_size*(opt.history+1)+1}
end
print(input_sizes)
if not buzz_model then
    if opt.model == 'buzz_correct' then
        buzz_model = qb.BuzzCorrect(input_sizes, opt.hidden_size, 0)
    end
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end
    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front

    local ans_loss, buzz_loss, total_length = 0, 0, 0
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
        total_length = total_length + seq_length
        -- starting id of this batch
        local from = (i-1)*opt.batch_size + 1
        local to = from+opt.batch_size-1
        -- TODO: instead of copy, try set
        ans_targets:sub(from, to, 1, seq_length):copy(y)
        mask:sub(from, to, 1, seq_length):copy(m)
        qids:sub(from, to):copy(qid)

        -- get content model predictions
        local ans_logprob, ans_rnn_state, al = content_model:forward({x}, y, seq_length, true)
        ans_loss = ans_loss + al
        --local buzz_feat = torch.Tensor(opt.batch_size, seq_length, qb.ans_size+1) 
        local loss = 0
        for t=1,seq_length do
            ans_logprob[t] = ans_logprob[t]:exp()
            ans_logprobs:sub(from, to, t, t, 1, -1):copy(ans_logprob[t])
            local buzz_feat = torch.Tensor(opt.batch_size, qb.ans_size*(opt.history+1)+1):zero()
            for i=0,opt.history do
                if t-i > 0 then
                    local sorted_logprob = ans_logprob[t-i]:clone():sort(2, true)
                    buzz_feat:narrow(2, i*qb.ans_size+1, qb.ans_size):copy(sorted_logprob)
                end
            end
            buzz_feat:narrow(2, qb.ans_size*(opt.history+1)+1, 1):fill(t)
            local _, p = torch.max(ans_logprob[t], 2)
            p = p:type('torch.IntTensor'):squeeze(2)
            ans_preds:sub(from, to, t, t):copy(p)
            local buzz_target = buzz_model:oracle_buzz(ans_logprob[t], y[{{}, t}], m[{{}, t}], t)
            buzz_targets:sub(from, to, t, t):copy(buzz_target)
            -- run buzz model
            local input
            if opt.use_content_state == 1 then
                input = {buzz_feat, ans_rnn_state[t][1]}
            else
                input = buzz_feat
            end
            local buzz_logprob, bl = buzz_model:forward(input, buzz_target, true)
            loss = loss + bl
            _, p = torch.max(buzz_logprob, 2)
            p = p:squeeze(2)
            buzz_preds:sub(from, to, t, t):copy(p)
        end
        --print(i .. ': ' .. loss/seq_length)
        buzz_loss = buzz_loss + loss / seq_length
    end

    buzz_acc = eval.accuracy(buzz_preds, buzz_targets, mask)
    mm_payoff, mm_mean_pos = eval.max_margin_buzz(ans_logprobs, ans_targets, mask, qids, loader.buzzes) 
    pred_payoff, pred_mean_pos = eval.predicted_buzz(ans_preds, buzz_preds, ans_targets, mask, qids, loader.buzzes) 
    static_payoff, static_mean_pos = eval.static_buzz(math.ceil(pred_mean_pos), ans_preds, ans_targets, mask, qids, loader.buzzes) 
    oracle_payoff, oracle_mean_pos = eval.predicted_buzz(ans_preds, buzz_targets, ans_targets, mask, qids, loader.buzzes) 
    ans_loss = ans_loss / n
    buzz_loss = buzz_loss / n
    print(string.format('ans loss = %.8f, buzz loss = %.8f, buzz_acc = %.4f, mm payoff = (%.4f, %.4f), pred payoff = (%.4f, %.4f), static payoff = (%.4f, %.4f), oracle payoff = (%.4f, %.4f)', ans_loss, buzz_loss, buzz_acc, mm_payoff, mm_mean_pos, pred_payoff, pred_mean_pos, static_payoff, static_mean_pos, oracle_payoff, oracle_mean_pos))
    return buzz_loss
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

    -- process one batch of questions
    local timer = torch.Timer()
    ------------------ get minibatch -------------------
    local x, y, m = loader:next_batch(1, opt.gpuid) -- discard mask
    local seq_length = x:size(2)
    ------------------- run content model --------------
    local ans_logprob, ans_rnn_state, _ = content_model:forward({x}, y, seq_length, true)
    local loss = 0
    local ans_logprobs = torch.Tensor(opt.batch_size*seq_length, qb.ans_size)
    local ans_rnn_states = torch.Tensor(opt.batch_size*seq_length, content_model.net_params.rnn_size)
    local ans_targets = torch.Tensor(opt.batch_size*seq_length)
    local buzz_targets = torch.Tensor(opt.batch_size*seq_length)
    local buzz_feats = torch.Tensor(opt.batch_size*seq_length, qb.ans_size*(opt.history+1)+1)
    local num_ex = 0
    for t=1,seq_length do
        ans_logprob[t] = ans_logprob[t]:exp()
        ans_logprobs:narrow(1, num_ex+1, opt.batch_size):copy(ans_logprob[t])
        ans_rnn_states:narrow(1, num_ex+1, opt.batch_size):copy(ans_rnn_state[t][1])
        ans_targets:narrow(1, num_ex+1, opt.batch_size):copy(y[{{}, t}])

        local buzz_target = buzz_model:oracle_buzz(ans_logprob[t], y[{{}, t}], m[{{}, t}], t)
        buzz_targets:narrow(1, num_ex+1, opt.batch_size):copy(buzz_target)
        local buzz_feat = torch.Tensor(opt.batch_size, qb.ans_size*(opt.history+1)+1):zero()
        for i=0,opt.history do
            if t-i > 0 then
                local sorted_logprob = ans_logprob[t-i]:clone():sort(2, true)
                buzz_feat:narrow(2, i*qb.ans_size+1, qb.ans_size):copy(sorted_logprob)
            end
        end
        buzz_feat:narrow(2, qb.ans_size*(opt.history+1)+1, 1):fill(t)
        buzz_feats:narrow(1, num_ex+1, opt.batch_size):copy(buzz_feat)
        num_ex = num_ex + opt.batch_size
    end

    local shuf = torch.randperm(opt.batch_size*seq_length)
    for t=1,seq_length do
        local from = (t-1)*opt.batch_size + 1
        local to = from + opt.batch_size - 1
        local ind = shuf:sub(from, to):type('torch.LongTensor')

        -- do fwd/bwd and return loss, grad_params
        local feval = function (x)
            if x ~= buzz_model.params then
                buzz_model.params:copy(x)
            end
            buzz_model.grad_params:zero()
       
            local input
            if opt.use_content_state == 1 then
                input = {buzz_feats:index(1, ind), ans_rnn_states:index(1, ind)}
            else
                input = buzz_feats:index(1, ind)
            end
            local buzz_target = buzz_targets:index(1, ind)
            local buzz_logprob, buzz_loss = buzz_model:forward(input, buzz_target)
            buzz_model:backward(input, buzz_target, buzz_logprob)
        
            -- clip gradient element-wise
            buzz_model.grad_params:clamp(-opt.grad_clip, opt.grad_clip)
            return buzz_loss, buzz_model.grad_params
        end
        local _, l = optim.rmsprop(feval, buzz_model.params, optim_state)
        loss = loss + l[1]
    end
    loss = loss / seq_length 
    local time = timer:time().real

    local train_loss = loss -- the loss is inside a list, pop it
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
        
        if opt.save == 1 then
            local savefile = string.format('%s/%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
            print('saving checkpoint to ' .. savefile)
            local checkpoint = {}
            checkpoint.model = buzz_model
            checkpoint.opt = opt
            checkpoint.train_losses = train_losses
            checkpoint.val_loss = val_loss
            checkpoint.val_losses = val_losses
            checkpoint.i = i
            checkpoint.epoch = epoch
            checkpoint.vocab = loader.vocab_mapping
            checkpoint.ans = loader.ans_mapping
            torch.save(savefile, checkpoint)
        end
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss ~= loss then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss end
    --if loss > loss0 * 3 then
    --    print('loss is exploding, aborting.')
    --    break -- halt
    --end
end


