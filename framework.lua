local QBFramework = torch.class('qb.Framework')

function QBFramework:__init(loader, content_model, args)
    self.loader = loader
    self.content_model = content_model
    self.ans_state_size = self.content_model.net_params.rnn_size
    self.actions = {qb.BUZZ, qb.WAIT}
    self.hist_len = args.hist_len or 1
    self.cat_size = self.loader.topk_cats + 1 -- number of top categories (and others)
    self.max_seq_length = self.loader.max_seq_length
    self.use_words = false
    self.word_padding = loader.vocab_mapping[qb.PAD]
    -- mode
    self.debug = false
    self.simulate = args.simulate or 0
    self.supervise = args.supervise or false  
    -- number of questions in each split
    self.num_questions = loader.num_questions
    -- number of buzzes in each split
    if self.simulate > 0 then
        self.num_buzzes = {}
        for i,n in ipairs(loader.num_questions) do
            self.num_buzzes[i] = n * self.simulate
        end
    else
        self.num_buzzes = loader.num_buzzes
    end
    -- user stats
    self.user_stats = loader.user_stats
    self.user_stats_size = self.user_stats[1]:size(1)
    self.num_users = loader.user_size
    -- game batch
    self.game_pointer = 0
    self.batch_ans_probs = nil
    self.batch_ans_targets = nil
    self.batch_masks = nil
    self.batch_qids = nil
    self.batch_size = 0
    -- current game
    self.category = nil
    self.buzz_pointer = 0   -- for iterating through buzzes
    self.buzzes = {}
    self.ans_prob = nil 
    self.ans_target = nil
    self.ans_pred = nil
    self.player_buzz_pos = nil
    self.player_correct = nil
    self.player_id = nil
    -- player groups: binning based on answer position (%)
    self.player_group = 1
    self.num_player_groups = 4
    self.player_bins = torch.linspace(0, 1, self.num_player_groups + 1)
    self.step_count = 0
    self.max_step = nil
    self.buzzed = nil
    -- set up dynamci functions
    self.get_state = self.get_state_test
    self.feat_groups = nil
    self.topk = 1   -- include topk ans preds in feature
    self:set_feat_map(args.agent)
    -- error analysis
    self.buzz_early_correct = torch.zeros(self.num_player_groups)
    self.buzz_early_wrong = torch.zeros(self.num_player_groups)
    self.miss_can_help = torch.zeros(self.num_player_groups)
    self.miss_cant_help = torch.zeros(self.num_player_groups)
    self.buzz_late_correct = torch.zeros(self.num_player_groups)
    self.buzz_late_wrong = torch.zeros(self.num_player_groups)
    self.total_num_game = torch.zeros(self.num_player_groups)
end

function QBFramework:reset_analysis_stats()
    self.buzz_early_correct:zero()
    self.buzz_early_wrong:zero()
    self.miss_can_help:zero()
    self.miss_cant_help:zero()
    self.buzz_late_correct:zero()
    self.buzz_late_wrong:zero()
    self.total_num_game:zero()
end

function QBFramework:report_error_analysis(group)
    if group ~= nil then
        print(string.format('early correct = %.4f, early wrong = %.4f, miss can help = %.4f, miss cant help = %.4f, late correct = %.4f, late wrong = %.4f', 
        self.buzz_early_correct[group] / self.total_num_game[group],
        self.buzz_early_wrong[group] / self.total_num_game[group],
        self.miss_can_help[group] / self.total_num_game[group],
        self.miss_cant_help[group] / self.total_num_game[group],
        self.buzz_late_correct[group] / self.total_num_game[group],
        self.buzz_late_wrong[group] / self.total_num_game[group]
        ))
    else
        print(string.format('early correct = %.4f, early wrong = %.4f, miss can help = %.4f, miss cant help = %.4f, late correct = %.4f, late wrong = %.4f', 
        self.buzz_early_correct:sum() / self.total_num_game:sum(),
        self.buzz_early_wrong:sum() / self.total_num_game:sum(),
        self.miss_can_help:sum() / self.total_num_game:sum(),
        self.miss_cant_help:sum() / self.total_num_game:sum(),
        self.buzz_late_correct:sum() / self.total_num_game:sum(),
        self.buzz_late_wrong:sum() / self.total_num_game:sum()
        ))
    end
end

function QBFramework:get_actions()
    return self.actions
end

function QBFramework:get_num_players()
    return self.loader.user_size
end

-- offset and length of each feature group
function QBFramework:get_feat_groups()
    return self.feat_groups
end

function QBFramework:set_feat_map(agent_type)
    -- default network features: prediction, position, rnn state
    --local state_dim_default = self.hist_len*qb.ans_size + 1 + self.ans_state_size 
    local state_dim_default = self.hist_len*qb.ans_size + 1 + 3
    local feat_groups_default = {pred={offset=1, size=self.hist_len*qb.ans_size + 1 + 3}}
    --feat_groups_default.state = {offset=feat_groups_default.pred.size+1, size=self.ans_state_size}
    -- TODO: set use_words
    if self.use_words then
        feat_groups_default.words = {offset=state_dim_default+1, size=self.max_seq_length}
        state_dim_default = state_dim_default + self.max_seq_length
    end
    -- TODO: make functions to assign feat groups
    self.feat_groups = feat_groups_default
    if agent_type == 'QBNeuralQLearner' then
        self._fill_state = self._fill_default
        self.state_dim = state_dim_default 
    elseif agent_type == 'QBONeuralQLearner' then
        self._fill_state = self._fill_opponent
        --self.state_dim = state_dim_default + 3 + self.ans_state_size
        --self.feat_groups.opponent = {offset=state_dim_default+1, size=3+self.ans_state_size}
        --local opp_size = self.user_stats_size + 1
        local opp_size = 4
        self.state_dim = state_dim_default + opp_size
        self.feat_groups.opponent = {offset=state_dim_default+1, size=opp_size}
    elseif agent_type == 'QBONeuralQLearner_multitask_action' then
        self._fill_state = self._fill_multitask_action
        --local opp_size = self.user_stats_size + 1
        local opp_size = 4
        self.state_dim = state_dim_default + opp_size
        self.feat_groups.opponent = {offset=state_dim_default+1, size=opp_size}
        self.state_dim = self.state_dim + 1
        self.feat_groups.opp_action = {offset=self.feat_groups.opponent.offset+self.feat_groups.opponent.size, size=1}
    elseif agent_type == 'QBONeuralQLearner_multitask_group' then
        self._fill_state = self._fill_multitask_group
        local opp_size = 4
        self.state_dim = state_dim_default + opp_size
        self.feat_groups.opponent = {offset=state_dim_default+1, size=opp_size}
        self.state_dim = self.state_dim + 1
        self.feat_groups.opp_group = {offset=self.feat_groups.opponent.offset+self.feat_groups.opponent.size, size=1}
    elseif agent_type == 'QBONeuralQLearner_cheat' then
        self._fill_state = self._fill_cheat
        local cheat_size = self.num_player_groups
        self.state_dim = state_dim_default + cheat_size
        self.feat_groups.cheat = {offset=state_dim_default+1, size=cheat_size}
    end
    if self.supervise then
        self._fill_state = self._fill_supervised
        self.state_dim = state_dim_default - 3
        self.feat_groups = {pred={offset=1, size=self.hist_len*qb.ans_size + 1}}
    end
end

function QBFramework:state_feat(t)
    local state = torch.Tensor(self.state_dim):zero()
    local from = 1
    from = self:_fill_state(t, state, from)
    assert(from == state:size()[1]+1)
    return state
end

function QBFramework:_fill_cheat(t, state, from)
    from = self:_fill_default(t, state, from)
    -- player group
    state[from+self.player_group-1] = 1
    -- if the player will buzz next
    --if self.player_buzz_pos == t + 1 then
    --state[from] = t / self.player_buzz_pos
    --state[from+1] = self.player_correct and t / self.player_buzz_pos or 0
    --state[from+2] = self.ans_pred(t)[1] == self.ans_target and 1 or 0
    --end
    from = from + self.num_player_groups
    return from
end

function QBFramework:_fill_opponent(t, state, from)
    from = self:_fill_default(t, state, from)
    -- all category
    --state:sub(from, from+self.user_stats_size-1):copy(self.user_stats[self.player_id])
    --from = from + self.user_stats_size
    -- dynamic category
    --state:sub(from, from+2):copy(self.user_stats[self.player_id]:narrow(1, self.category*3+1, 3))
    --from = from + 3
    -- average category
    state:sub(from, from+2):copy(self.user_stats[self.player_id]:narrow(1, 1, 3))
    from = from + 3
    -- current category
    --state[from+self.category-1] = 1
    --from = from + self.cat_size
    state[from] = t/300
    from = from + 1
    --state:narrow(1, from, self.ans_state_size):copy(self.ans_state(t))
    --from = from + self.ans_state_size
    --state[from+self.player_id-1] = 1
    --from = from + self.num_users
    return from
end

function QBFramework:_fill_multitask_action(t, state, from)
    from = self:_fill_opponent(t, state, from)
    --if self.step_count >= self.player_buzz_pos then
    --    if self.player_correct then
    --        state[from] = 1
    --    else
    --        state[from] = 2
    --    end
    --else
    --    state[from] = 3
    --end
    state[from] = math.min(1, t / self.player_buzz_pos)
    from = from + 1
    return from
end

function QBFramework:_fill_multitask_group(t, state, from)
    from = self:_fill_opponent(t, state, from)
    --if self.step_count >= self.player_buzz_pos then
    --    if self.player_correct then
    --        state[from] = 1
    --    else
    --        state[from] = 2
    --    end
    --else
    --    state[from] = 3
    --end
    state[from] = self.player_group
    from = from + 1
    return from
end

function QBFramework:_fill_supervised(t, state, from)
    for i=t,t-self.hist_len+1,-1 do
        if i > 0 then
            state:narrow(1, from, qb.ans_size):copy(self.ans_prob(i))
        end
        from = from + qb.ans_size
    end
    -- normalization for t 
    state[from] = t/300
    from = from + 1
    return from
end

function QBFramework:_fill_default(t, state, from)
    for i=t,t-self.hist_len+1,-1 do
        if i > 0 then
            state:narrow(1, from, qb.ans_size):copy(self.ans_prob(i))
        end
        from = from + qb.ans_size
    end
    -- normalization for t 
    state[from] = t/300
    from = from + 1
    -- player's action in the last round (observed)
    if self.player_buzzed then
        if self.player_correct then
            state[from] = 1
        else
            state[from+1] = 1
        end
    else
        state[from+2] = 1
    end
    from = from + 3
    -- hiddent state of content
    --state:narrow(1, from, self.ans_state_size):copy(self.ans_state(t))
    --from = from + self.ans_state_size
    -- category 
    --state[from+self.category-1] = 1
    --from = from + self.cat_size 
    -- top-k predictions
    --local k = self.topk
    --state:narrow(1, from, k):copy(self.ans_pred(t):narrow(1, 1, k))
    --from = from + k
    --state[from+self.ans_pred(t)-1] = 1
    --from = from + qb.ans_size
    -- bag of words
    if self.use_words then
        state:narrow(1, from, self.max_seq_length):fill(self.word_padding)
        if t > 0 then
            state:narrow(1, from , t):copy(self.words:narrow(1, 1, t))
        end
        from = from + self.max_seq_length 
    end
    return from
end

-- reset for a split (used during evaluation)
function QBFramework:reset(split_index)
    torch.manualSeed(opt.seed)
    self.loader:reset_batch_pointer(split_index)
    self:reset_analysis_stats()
end

function QBFramework:new_game(split_index, test)
    self.split_index = split_index or 1
    self.test = test or false
    if not self.test then
        if self.supervise then
            self.get_state = self.get_state_oracle
        else
            self.get_state = self.get_state_train
        end
    else
        self.get_state = self.get_state_test
    end
    self:load_next_buzz()
    --if not self.test then
    --    while self.player_group ~= 3 do
    --        self:load_next_buzz()
    --    end
    --end
    -- starting state is the first word
    self.step_count = 1
    self.buzzed = false
    self.player_buzzed = false
    if self.debug then 
        print(string.format('ans=%d, max_step=%d, player_id=%d, player_buzz_pos=%d, player_correct=%s', self.ans_target, self.max_step, self.player_id, self.player_buzz_pos, self.player_correct))
    end
    return self:get_state()
end

function QBFramework:load_next_batch()
    if self.debug then print('loading a new batch from split ' .. self.split_index) end
    ------------------ get minibatch -------------------
    local x, y, m, qids = self.loader:next_batch(self.split_index)
    local seq_length = x:size(2)
    ------------------- run content model --------------
    local ans_logprob, ans_rnn_state, _ = self.content_model:forward({x}, y, seq_length, true)
    self.batch_ans_probs = {}
    self.batch_ans_preds = {}
    self.batch_ans_states = {}
    for t=1,seq_length do
        _, self.batch_ans_preds[t] = torch.max(ans_logprob[t], 2)
        -- NOTE: must squeeze! otherwize ans_preds(t) is a tensor instead of a number.
        self.batch_ans_preds[t] = self.batch_ans_preds[t]:squeeze(2)
        self.batch_ans_probs[t] = ans_logprob[t]:exp():sort(2, true)
        local num_states = #ans_rnn_state[t]
        self.batch_ans_states[t] = ans_rnn_state[t][num_states]
    end
    self.batch_ans_targets = y
    self.batch_words = x
    self.batch_masks = m
    self.batch_size = m:size(1) 
    local true_batch_size = m:narrow(2, 1, 1):sum()
    -- skip examples padded at the beginning
    self.game_pointer = self.batch_size - true_batch_size
    self.batch_qids = qids
end

function QBFramework:load_next_question()
    if self.game_pointer >= self.batch_size then
        self:load_next_batch()
    end
    self.game_pointer = self.game_pointer + 1
    if self.debug then print(string.format('-------- game %s --------', self.batch_qids[self.game_pointer])) end
    self.ans_prob = function (t) return self.batch_ans_probs[t][self.game_pointer] end
    self.ans_pred = function (t) return self.batch_ans_preds[t][self.game_pointer] end
    self.ans_state = function (t) return self.batch_ans_states[t][self.game_pointer] end
    self.ans_target = self.batch_ans_targets[self.game_pointer][1] 
    self.max_step = self.batch_masks[self.game_pointer]:sum()
    local qid = self.batch_qids[self.game_pointer]
    self.qid = qid
    self.words = self.batch_words[self.game_pointer]
    -- only use top categories
    self.category = self.loader.categories[qid]
    if self.loader.top_cats[self.category] == nil then
        self.category = self.cat_size
    else
        self.category = self.loader.top_cats[self.category]
    end
    if self.simulate > 0 then
        local buzz_pos = torch.Tensor({0.6, 0.9})
        self.buzzes = self:simulate_buzzes(self.simulate, buzz_pos, self.max_step)
    else
        self.buzzes = self.loader.buzzes[qid]
    end
    assert(#self.buzzes > 0)
    self.buzz_pointer = 0
end

function QBFramework:simulate_buzzes(n, buzz_pos, max_len)
    -- buzz_pos are different buzz positions to be sampled uniformly
    local buzzes = {}
    local num_buzz_pos = buzz_pos:size(1)
    for i=1,n do 
        local buzz = torch.IntTensor(3)
        buzz[1] = torch.random(1,num_buzz_pos)
        local pos = buzz_pos[buzz[1]]
        buzz[2] = math.max(1, math.floor(torch.uniform(pos-0.08, pos+0.08)*max_len))
        buzz[3] = torch.uniform() < 0.8 and 1 or 0 
        buzzes[i] = buzz
    end
    return buzzes
end

function QBFramework:load_next_buzz()
    if self.buzz_pointer >= #self.buzzes then
        self:load_next_question()
    end
    local player_buzz
    if self.test then
        self.buzz_pointer = self.buzz_pointer + 1
        player_buzz = self.buzzes[self.buzz_pointer]
    else
        -- randomly sample a player
        player_buzz = self.buzzes[torch.random(1, #self.buzzes)]
        --player_buzz = self.buzzes[1]
        -- move to next question
        self.buzz_pointer = #self.buzzes
    end
    -- player id is the id after user_mapping
    self.player_id = player_buzz[1]
    self.player_buzz_pos = player_buzz[2]
    -- bin players
    self.player_group = self:_bin_player(self.player_buzz_pos / self.max_step)
    self.player_correct = player_buzz[3] == 1 and true or false
end

function QBFramework:_bin_player(buzz_pos)
    for i=2,self.player_bins:size(1) do
        if buzz_pos <= self.player_bins[i] then
            return i-1
        end
    end
    print(self.player_buzz_pos, self.max_step)
    error('cannot bin player ' .. buzz_pos)
end

function QBFramework:step(action)
    -- always buzz at the final word
    if action == qb.BUZZ or self.step_count == self.max_step then self.buzzed = true end
    if self.debug then print(action == qb.BUZZ and 'buzz' or 'wait') end
    self.step_count = self.step_count + 1
    return self:get_state()
end

-- use immediate reward (same as in supervised oracle setting) 
function QBFramework:get_state_oracle()
    -- agent cannot take any action in terminal states
    local terminal, reward, state
    if self.step_count == 1 then
        reward = 0
    else
        local correct = self.ans_pred(self.step_count - 1) == self.ans_target
        if correct then
            if self.buzzed then
                reward = 10
            else
                reward = -10
            end
        else
            if self.buzzed then
                reward = -15
            else
                reward = 15
            end
        end
    end

    if self.step_count > self.max_step then
        terminal = true
    else
        terminal = false
    end

    if self.debug then
        local correct = terminal and 0 or (self.ans_pred(self.step_count) == self.ans_target and 1 or -1)
        print(string.format('%s at %d. Correct %d. Reward %d.', self.buzzed and 'Buzzed' or 'Wait', self.step_count-1, correct, reward))
    end

    if self.buzzed then
        self.buzzed = false
    end

    local t = math.min(self.step_count, self.max_step)
    state = self:state_feat(t)
    return state, terminal, reward
end

-- TODO: don't terminate until the end
-- TODO: recored buzzing position without player terminating
-- use actual reward in quizbowl
function QBFramework:get_state_test()
    -- agent cannot take any action in terminal states
    local terminal, reward, state
    if self.buzzed then
        local buzz_pos = self.step_count - 1 
        local correct = self.ans_pred(buzz_pos) == self.ans_target
        assert(buzz_pos <= self.player_buzz_pos or not self.player_correct)
        terminal = true
        if buzz_pos <= self.player_buzz_pos then
            if correct then
                self.buzz_early_correct[self.player_group] = self.buzz_early_correct[self.player_group] + 1
            else
                self.buzz_early_wrong[self.player_group] = self.buzz_early_wrong[self.player_group] + 1
            end
        else
            if correct then
                self.buzz_late_correct[self.player_group] = self.buzz_late_correct[self.player_group] + 1
            else
                self.buzz_late_wrong[self.player_group] = self.buzz_late_wrong[self.player_group] + 1
            end
        end
        reward = qb.eval.get_one_payoff(buzz_pos, correct, self.player_buzz_pos, self.player_correct)
        if self.debug then
            print(string.format('Buzzed at %d. %s. Reward %d.', buzz_pos, correct and 'Correct' or 'Wrong', reward))
        end
    elseif self.step_count > self.player_buzz_pos then
        self.player_buzzed = true
        if self.player_correct then
            terminal = true
            reward = -10
            local can_help = false
            for t=1,self.player_buzz_pos do
                if self.ans_pred(t) == self.ans_target then
                    can_help = true
                    break
                end
            end
            if can_help then
                self.miss_can_help[self.player_group] = self.miss_can_help[self.player_group] + 1
            else
                self.miss_cant_help[self.player_group] = self.miss_cant_help[self.player_group] + 1
            end
            if self.debug then
                print(string.format('Player buzzed. Reward -10.'))
            end
        elseif self.step_count <= self.max_step then
            terminal = false
            reward = 0
            if self.debug then
                print(string.format('Player buzzed wrong. Waiting. Reward 0.'))
            end
        else
            error('agent should always buzz at the final word')
        end
    else
        terminal = false
        reward = 0
        if self.debug then
            print(string.format('No one buzzed yet. Reward 0.'))
        end
    end
    if self.debug then
        print(string.format('At word %d. terminal=%s', self.step_count, terminal))
        if terminal then print ('---------------------------') end
    end

    if self.debug then
        local correct = terminal and 0 or (self.ans_pred(self.step_count) == self.ans_target and 1 or -1)
        print('correct: ' .. correct)
    end

    if terminal then
        self.total_num_game[self.player_group] = self.total_num_game[self.player_group] + 1
    end

    local t = math.min(self.step_count, self.max_step)
    state = self:state_feat(t)
    return state, terminal, reward
end

function QBFramework:get_state_train()
    -- agent cannot take any action in terminal states
    local terminal, reward, state
    if self.buzzed then
        local buzz_pos = self.step_count - 1 
        local correct = self.ans_pred(buzz_pos) == self.ans_target
        assert(buzz_pos <= self.player_buzz_pos or not self.player_correct)
        terminal = true
        reward = qb.eval.get_one_payoff(buzz_pos, correct, self.player_buzz_pos, self.player_correct)
        -- adjust reward during training to encourage buzzing at high confidence
        --if correct then
        --    reward = reward * self.ans_prob(buzz_pos)[self.ans_target] * 5
        --end
        if self.debug then
            print(string.format('Buzzed at %d. %s. Reward %d.', buzz_pos, correct and 'Correct' or 'Wrong', reward))
        end
    elseif self.step_count > self.player_buzz_pos then
        self.player_buzzed = true
        if self.player_correct then
            terminal = true
            reward = -10
            if self.debug then
                print(string.format('Player buzzed. Reward -10.'))
            end
        elseif self.step_count <= self.max_step then
            terminal = false
            local correct = self.ans_pred(self.step_count-1) == self.ans_target
            if correct then
                reward = -1
            else
                reward = 1
            end
            if self.debug then
                print(string.format('Player buzzed wrong. Waiting. Reward 0.'))
            end
        else
            error('agent should always buzz at the final word')
        end
    else
        terminal = false
        if self.step_count == 1 then
            reward = 0
        else
            local correct = self.ans_pred(self.step_count-1) == self.ans_target
            if correct then
                reward = -1
            else
                reward = 1
            end
        end
        if self.debug then
            print(string.format('No one buzzed yet. Reward 0.'))
        end
    end
    if self.debug then
        print(string.format('At word %d. terminal=%s', self.step_count, terminal))
        if terminal then print ('---------------------------') end
    end

    if self.debug then
        local correct = terminal and 0 or (self.ans_pred(self.step_count) == self.ans_target and 1 or -1)
        print('correct: ' .. correct)
    end

    local t = math.min(self.step_count, self.max_step)
    state = self:state_feat(t)
    return state, terminal, reward
end
