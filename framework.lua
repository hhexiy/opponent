local QBFramework = torch.class('qb.Framework')

function QBFramework:__init(loader, content_model, hist_len, agent_type, simulate)
    self.loader = loader
    self.content_model = content_model
    self.actions = {qb.BUZZ, qb.WAIT}
    self.hist_len = hist_len or 1
    -- mode
    self.debug = false
    self.simulate = simulate or 0
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
    -- game batch
    self.game_pointer = 0
    self.batch_ans_probs = nil
    self.batch_ans_targets = nil
    self.batch_masks = nil
    self.batch_qids = nil
    self.batch_size = 0
    -- current game
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
    self:set_feat_map(agent_type)
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
    -- default network
    -- TODO: don't need the function 
    local state_dim_default = self.hist_len*qb.ans_size + 1
    local feat_groups_default = {default={offset=1, size=state_dim_default}}
    if agent_type == 'QBNeuralQLearner' then
        self._fill_state = self._fill_default
        self.state_dim = state_dim_default 
        self.feat_groups = feat_groups_default
    -- QB+OpponentId
    -- TODO: fix state_dim
    elseif agent_type == 'QBONeuralQLearner1' or agent_type == 'QBONeuralQLearner2' then
        self._fill_state = self._fill_opponent_id
        self.state_dim = self:_state_dim_default() + 1
        self.feat_groups = {default={offset=1, size=self.state_dim-1},
                            id={offset=self.state_dim, size=1}}
    elseif agent_type == 'QBONeuralQLearner_cheat' then
        self._fill_state = self._fill_cheat
        self.state_dim = state_dim_default + 3
        feat_groups_default.cheat = {offset=state_dim_default+1, size=3}
        self.feat_groups = feat_groups_default
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
    -- if the player will buzz next
    --if self.player_buzz_pos == t + 1 then
    state[from] = t / self.player_buzz_pos
    --state[from+1] = self.player_correct and 1 or 0
    --state[from+2] = self.ans_pred(t)[1] == self.ans_target and 1 or 0
    --end
    from = from + 3
    return from
end

function QBFramework:_fill_opponent_id(t, state, from)
    from = self:_fill_default(t, state, from)
    state[from] = self.player_id 
    from = from + 1
    return from
end

function QBFramework:_state_dim_default()
    return self.hist_len*qb.ans_size + 1 + self.topk
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
    -- top-k predictions
    --local k = self.topk
    --state:narrow(1, from, k):copy(self.ans_pred(t):narrow(1, 1, k))
    --from = from + k
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
    self:load_next_buzz()
    -- starting state is the first word
    self.step_count = 1
    self.buzzed = false
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
    for t=1,seq_length do
        _, self.batch_ans_preds[t] = torch.max(ans_logprob[t], 2)
        -- NOTE: must squeeze! otherwize ans_preds(t) is a tensor instead of a number.
        self.batch_ans_preds[t] = self.batch_ans_preds[t]:squeeze(2)
        self.batch_ans_probs[t] = ans_logprob[t]:exp():sort(2, true)
    end
    self.batch_ans_targets = y
    self.batch_masks = m
    self.game_pointer = 0
    self.batch_size = m:narrow(2, 1, 1):sum()
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
    self.ans_target = self.batch_ans_targets[self.game_pointer][1] 
    self.max_step = self.batch_masks[self.game_pointer]:sum()
    if self.simulate > 0 then
        local buzz_pos = torch.Tensor({0.6, 0.9})
        self.buzzes = self:simulate_buzzes(self.simulate, buzz_pos, self.max_step)
    else
        self.buzzes = self.loader.buzzes[self.batch_qids[self.game_pointer]]
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
                reward = 1
            else
                reward = -1
            end
        else
            if self.buzzed then
                reward = -5
            else
                reward = 1
            end
        end
    end

    if self.step_count > self.max_step then
        terminal = true
    else
        terminal = false
    end

    local correct = terminal and 0 or (self.ans_pred(self.step_count) == self.ans_target and 1 or -1)

    if self.debug then
        print(string.format('%s at %d. Correct %d. Reward %d.', self.buzzed and 'Buzzed' or 'Wait', self.step_count-1, correct, reward))
    end

    if self.buzzed then
        self.buzzed = false
    end

    local t = math.min(self.step_count, self.max_step)
    state = self:state_feat(t)
    return state, terminal, reward
end

-- use actual reward in quizbowl
function QBFramework:get_state_test()
    -- agent cannot take any action in terminal states
    local terminal, reward, state
    if self.buzzed then
        local buzz_pos = self.step_count - 1
        assert(buzz_pos <= self.player_buzz_pos or not self.player_correct)
        terminal = true
        local correct = self.ans_pred(buzz_pos) == self.ans_target
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
        -- adjust reward during training to encourage buzzing at high confidence
        if not self.test then
            if correct then
                reward = reward * self.ans_prob(buzz_pos)[self.ans_target] * 5
            end
        end
        if self.debug then
            print(string.format('Buzzed at %d. %s. Reward %d.', buzz_pos, correct and 'Correct' or 'Wrong', reward))
        end
    elseif self.step_count > self.player_buzz_pos then
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

