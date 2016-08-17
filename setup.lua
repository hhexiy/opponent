require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.misc'
require 'model.MultiHot'
require 'model.OneHot'
require 'model.MaskedLookupTable'
include('model/RNNModel.lua')

function env_setup()
    ------------------- general setup ------------------------
    torch.setheaptracking(true)
    torch.manualSeed(opt.seed)
    if opt.debug == 1 then nngraph.setDebug(true) end
    if opt.threads == 0 then
        opt.threads = 6
    end
    torch.setnumthreads(opt.threads)

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
    if opt.checkpoint_dir and not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end
end

function soccer_setup(h, w)
    if not soccer then
        soccer = {}
    end
    -- actions
    soccer.UP, soccer.DOWN, soccer.LEFT, soccer.RIGHT, soccer.NOOP = 1, 2, 3, 4, 5
    -- size: h x w (h should be even and w should be odd)
    assert(h > 0 and h % 2 == 0)
    assert(w > 0 and (w+1) % 2 == 0)
    soccer.HEIGHT, soccer.WIDTH = h, w
    -- player A and B
    soccer.A = 0
    soccer.B = 1
end

function qb_setup()
    if not qb then
        qb = {}
        include('model/QBModels.lua')
        include('util/MinibatchLoader.lua')
        qb.eval = require 'util.eval'
    end
    -- buzz class label
    qb.BUZZ, qb.WAIT, qb.EOS = 1, 2, 3

    -- load content model
    print('loading content model from ' .. opt.init_content)
    local checkpoint = torch.load(opt.init_content)
    qb.vocab_mapping = checkpoint.vocab_mapping
    qb.ans_mapping = checkpoint.ans_mapping
    
    -- create the data loader class
    local loader
    loader = qb.QAMinibatchLoader(opt.data_dir, opt.input_file, opt.batch_size)
    loader:load_data(qb.vocab_mapping, qb.ans_mapping)
    qb.vocab_size = loader.vocab_size  
    qb.ans_size = loader.ans_size
    qb.max_seq_length = loader.max_seq_length
    
    local content_rnn = checkpoint.model
    content_rnn.net_params.batch_size = opt.batch_size
    local content_model = RNNModel(content_rnn, qb.max_seq_length, opt.gpuid, false)
    --local content_model = RNNModel(content_rnn, 1, opt.gpuid, false)
    
    -- word embedding
    if string.len(opt.embedding) > 0 then
        qb.word_embedding = loader:load_embedding(opt.embedding)
        qb.emb_size = qb.word_embedding.weight[1]:size(1)
    else
        qb.word_embedding = nn.LookupTable(qb.vocab_size, 300) 
        qb.emb_size = 300
    end
    print('word embedding size: ' .. qb.emb_size)
    return loader, content_model
end

function soccer_dqn_setup()
    if not dqn then
        dqn = {}
        require 'dqn.nnutils'
        require 'dqn.Scale'
        require 'dqn.RandomAgent'
        require 'dqn.SoccerRuleAgent'
        require 'dqn.SoccerNeuralQLearner'
        require 'dqn.SoccerNeuralQLearner_multitask'
        require 'dqn.SoccerONeuralQLearner'
        require 'dqn.SoccerONeuralQLearner_multitask_group'
        require 'dqn.SoccerONeuralQLearner_multitask_action'
        require 'dqn.TransitionTable'
        require 'dqn.Rectifier'
    end
    soccer_setup(opt.height, opt.width)
    include('soccer_framework.lua')
    local framework = soccer.Framework(opt)

    opt.agent_params = str_to_table(opt.agent_params)
    opt.agent_params.gpu       = opt.gpu
    opt.agent_params.best      = opt.best
    opt.agent_params.verbose   = opt.verbose
    if opt.network ~= '' then
        opt.agent_params.network = opt.network
    end
    if not opt.agent_params.state_dim then
        -- ans probs and position
        opt.agent_params.state_dim = framework.state_dim
        print('state dim: ', opt.agent_params.state_dim)
    end
    opt.agent_params.actions = framework:get_actions()
    opt.agent_params.feat_groups = framework:get_feat_groups()
    return dqn[opt.agent](opt.agent_params), dqn[opt.opponent](opt.agent_params), framework 
end

function qb_dqn_setup()
    if not dqn then
        dqn = {}
        require 'dqn.nnutils'
        require 'dqn.Scale'
        require 'dqn.QBNeuralQLearner'
        require 'dqn.QBONeuralQLearner'
        require 'dqn.QBONeuralQLearner2'
        require 'dqn.QBONeuralQLearner_cheat'
        require 'dqn.QBONeuralQLearner_multitask_group'
        require 'dqn.QBONeuralQLearner_multitask_action'
        require 'dqn.TransitionTable'
        require 'dqn.Rectifier'
    end

    local loader, content_model = qb_setup()
    include('qb_framework.lua')
    local framework = qb.Framework(loader, content_model, opt)
    -- TODO: using words is messy now. fix this if it works
    if framework.use_words then
        print 'using words as policy features'
        --framework.word_embedding = loader:load_embedding('dat/glove', framework.word_padding)
        framework.word_embedding = nn.LookupTable(qb.vocab_size, 128)
    end
    if opt.simulate > 0 then
        print('using simulated player buzzes: ' .. opt.simulate .. ' per question')
    end

    opt.agent_params = str_to_table(opt.agent_params)
    opt.agent_params.gpu       = opt.gpu
    opt.agent_params.best      = opt.best
    opt.agent_params.verbose   = opt.verbose
    opt.agent_params.ans_size = qb.ans_size
    if opt.network ~= '' then
        opt.agent_params.network = opt.network
    end
    if not opt.agent_params.state_dim then
        -- ans probs and position
        opt.agent_params.state_dim = framework.state_dim
        print('state dim: ', opt.agent_params.state_dim)
    end
    opt.agent_params.actions = framework:get_actions()
    opt.agent_params.feat_groups = framework:get_feat_groups()
    opt.agent_params.num_players = framework:get_num_players()
    return dqn[opt.agent](opt.agent_params), framework 
end

--- other functions

function str_to_table(str)
    if type(str) == 'table' then
        return str
    end
    if not str or type(str) ~= 'string' then
        if type(str) == 'table' then
            return str
        end
        return {}
    end
    local ttr
    if str ~= '' then
        local ttx=tt
        loadstring('tt = {' .. str .. '}')()
        ttr = tt
        tt = ttx
    else
        ttr = {}
    end
    return ttr
end

