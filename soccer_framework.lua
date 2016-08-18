local SoccerFramework = torch.class('soccer.Framework')
DEBUG=0

-- action atoms
ADVANCE=1
DEFEND=2
INTERCEPT=3
AVOID=4

function SoccerFramework:__init(args)
    self.actions = {soccer.UP, soccer.DOWN, soccer.LEFT, soccer.RIGHT, soccer.NOOP}
    self.n_actions = #self.actions
    self.action_str = {[soccer.UP]='UP', [soccer.DOWN]='DOWN', [soccer.LEFT]='LEFT', [soccer.RIGHT]='RIGHT', [soccer.NOOP]='NOOP'}
    self.hist_len = args.hist_len or 1
    self.max_x = soccer.WIDTH
    self.min_x = 1
    self.mid_x = (soccer.WIDTH+1) / 2
    self.max_y = soccer.HEIGHT
    self.min_y = 1
    self.mid_y = {soccer.HEIGHT/2, soccer.HEIGHT/2 + 1}
    -- A wants to shoot to the right
    self.goal_area_a = {x=self.max_x, y={self.mid_y[1], self.mid_y[2]}}
    -- B wants to shoot to the left
    self.goal_area_b = {x=self.min_x, y={self.mid_y[1], self.mid_y[2]}}
    -- who has the ball
    self.has_ball = nil
    self.max_steps = 100
    self.step_count = 0
    -- build-in agent
    self.weights = torch.FloatTensor(4):fill(1)
    self.defend_percentage = args.defend
    print('defensive agent:', self.defend_percentage)
    self.defensiveness = {0.1, 0.9}
    --self.defensiveness = {0.1}
    -- opponent features
    self.opp_pos_stat = torch.IntTensor(2, 4):zero()
    self.opp_pos_stat_len = -1 
    self.opp_pos = torch.IntTensor(5):zero()
    self.opp_pos_seq_len = 2
    self.opp_pos_seq = torch.IntTensor(self.opp_pos_seq_len, 5):zero()
    self.opp_action_stat = torch.IntTensor(self.n_actions):zero()
    self.opp_action_stat_len = 3
    self.opp_action = torch.IntTensor(self.n_actions):zero()
    self.opp_action_seq_len = 1
    self.opp_action_seq = torch.IntTensor(self.opp_action_seq_len, self.n_actions):zero()
    self.lost_ball = 0
    -- features
    self.feat_groups = nil
    self:set_feat_map(args.agent)
end

function SoccerFramework:set_builtin_agent(id)
    self.player_id = id
    --self.defensive = self.defensiveness[id]
    local defensive = self.defensiveness[id] 
    local offensive = 1 - defensive
    self.weights[ADVANCE] = offensive
    self.weights[INTERCEPT] = offensive
    self.weights[AVOID] = defensive 
    self.weights[DEFEND] = defensive
end

function SoccerFramework:print_state()
    local s = torch.IntTensor(self.max_y, self.max_x):zero()
    -- direction of y axis is reversed in Tensor
    s[self.max_y-self.coord_a.y+1][self.coord_a.x] = self.has_ball == soccer.A and 11 or 1
    s[self.max_y-self.coord_b.y+1][self.coord_b.x] = self.has_ball == soccer.B and 22 or 2
    self:print_coord()
    print(s)
end

function SoccerFramework:get_actions()
    return self.actions
end

function SoccerFramework:new_game(test)
    self.test = test or false
    local r = torch.rand(1)[1]
    if r < self.defend_percentage then
        self:set_builtin_agent(2)
    else
        self:set_builtin_agent(1)
    end
    -- boarders at left the right are for goals
    -- A: random position in the left half
    self.coord_a = {x=torch.random(self.min_x+1, self.mid_x), y=torch.random(self.min_y, self.max_y)}
    -- B: random position in the right half
    self.coord_b = {x=torch.random(self.mid_x, self.max_x-1), y=torch.random(self.min_y, self.max_y)}
    self.has_ball = torch.random(0, 1) == 0 and soccer.A or soccer.B
    self.step_count = 0
    -- opponent stats
    -- (has_ball, relative positions to me and the goals)
    self.opp_pos_stat:zero()
    self.prev_action_a = nil
    self.prev_action_b = nil
    self.lost_ball = 0
    if DEBUG == 1 then
        print('============ start new game ============')
        print('has ball:', self.has_ball == soccer.A and 'A' or 'B')
        self:print_state()
    end
    return self:state_feat(self.step_count), false, 0 
end

function SoccerFramework:print_coord(player)
    print(string.format('A position: x=%d, y=%d', self.coord_a.x, self.coord_a.y))
    print(string.format('B position: x=%d, y=%d', self.coord_b.x, self.coord_b.y))
end

function SoccerFramework:reset(seed)
    if seed == nil then seed = opt.seed end
    torch.manualSeed(seed)
end

function SoccerFramework:update_coord(player, curr_coord, action)
    if action == soccer.UP then
        curr_coord.y = math.min(self.max_y, curr_coord.y+1)
    elseif action == soccer.DOWN then
        curr_coord.y = math.max(self.min_y, curr_coord.y-1)
    -- A cannot step into its goal area (left)
    -- B cannot step into its goal area (right)
    elseif action == soccer.LEFT then
        curr_coord.x = math.max(player == soccer.A and self.min_x+1 or self.min_x, curr_coord.x-1)
    elseif action == soccer.RIGHT then
        curr_coord.x = math.min(player == soccer.B and self.max_x-1 or self.max_x, curr_coord.x+1)
    end
end

function SoccerFramework:reset_coord(player, curr_coord, action)
    if action == soccer.UP then
        self:update_coord(player, curr_coord, soccer.DOWN)
    elseif action == soccer.DOWN then
        self:update_coord(player, curr_coord, soccer.UP)
    elseif action == soccer.LEFT then
        self:update_coord(player, curr_coord, soccer.RIGHT)
    elseif action == soccer.RIGHT then
        self:update_coord(player, curr_coord, soccer.LEFT)
    end
end

function SoccerFramework:is_goal(has_ball, coord, goal_area)
    if has_ball and 
        coord.x == goal_area.x and
        coord.y >= goal_area.y[1] and coord.y <= goal_area.y[2] then
        return true
    end
    return false
end

function SoccerFramework:set_feat_map(agent_type)
    local state_dim_default = 15
    local feat_groups_default = {state={offset=1, size=state_dim_default}}
    self.feat_groups = feat_groups_default
    self.state_dim = state_dim_default
    local opp_size = self.opp_pos_stat:nElement() + self.opp_pos_seq:nElement() + self.opp_action_seq:nElement() + 2

    if string.starts(agent_type, 'SoccerONeuralQLearner') then
        self._fill_state = self._fill_opponent
        self.feat_groups.opponent = {offset=self.state_dim+1, size=opp_size}
        self.state_dim = self.state_dim + opp_size
        if agent_type == 'SoccerONeuralQLearner_multitask_action' then
            self._fill_state = self._fill_multitask_action
            self.feat_groups.supervision = {offset=self.state_dim+1, size=1}
            self.state_dim = self.state_dim + 1
        elseif agent_type == 'SoccerONeuralQLearner_multitask_group' then
            self._fill_state = self._fill_multitask_group
            self.feat_groups.supervision = {offset=self.state_dim+1, size=1}
            self.state_dim = self.state_dim + 1
        end
    else
        self._fill_state = self._fill_default
        self.state_dim = state_dim_default 
    end
end

function SoccerFramework:_fill_default(t, state, from)
    state[from] = self.coord_a.x
    state[from+1] = self.coord_a.y
    state[from+2] = self.coord_b.x
    state[from+3] = self.coord_b.y
    state[from+4] = self.min_x
    state[from+5] = self.max_x
    state[from+6] = self.min_y
    state[from+7] = self.max_y
    state[from+8] = self.goal_area_a.x
    state[from+9] = self.goal_area_a.y[1]
    state[from+10] = self.goal_area_a.y[2]
    state[from+11] = self.goal_area_b.x
    state[from+12] = self.goal_area_b.y[1]
    state[from+13] = self.goal_area_b.y[2]
    state[from+14] = self.has_ball == soccer.A and 1 or 0
    from = from + 15
    return from
end

function SoccerFramework:_fill_opponent(t, state, from)
    --print(self.opp_pos_stat)
    from = self:_fill_default(t, state, from)

    state:sub(from, from+self.opp_pos_stat:nElement()-1):copy(self.opp_pos_stat)
    from = from + self.opp_pos_stat:nElement()

    state:sub(from, from+self.opp_pos_seq:nElement()-1):copy(self.opp_pos_seq)
    from = from + self.opp_pos_seq:nElement()

    --state:sub(from, from+self.opp_action_stat:nElement()-1):copy(self.opp_action_stat)
    --from = from + self.opp_action_stat:nElement()

    state:sub(from, from+self.opp_action_seq:nElement()-1):copy(self.opp_action_seq)
    from = from + self.opp_action_seq:nElement()

    if self.opp_pos_stat_len > 0 and t % self.opp_pos_stat_len == 0 then
        self.opp_pos_stat:zero()
    end
    if self.opp_action_stat_len > 0 and t % self.opp_action_stat_len == 0 then
        self.opp_action_stat:zero()
    end

    state[from] = self.lost_ball
    from = from + 1
    state[from] = t/100
    from = from + 1

    return from
end

function SoccerFramework:_fill_multitask_action(t, state, from)
    from = self:_fill_opponent(t, state, from)
    state[from] = self:act(self.has_ball == soccer.B, self.coord_b, self.goal_area_a, self.coord_a, self.goal_area_b, self.prev_action_b)
    from = from + 1
    return from
end

function SoccerFramework:_fill_multitask_group(t, state, from)
    from = self:_fill_opponent(t, state, from)
    state[from] = self.player_id 
    from = from + 1
    return from
end

-- offset and length of each feature group
function SoccerFramework:get_feat_groups()
    return self.feat_groups
end

function SoccerFramework:state_feat(t)
    local state = torch.Tensor(self.state_dim):zero()
    local from = 1
    from = self:_fill_state(t, state, from)
    assert(from == state:size(1)+1)
    return state
end

-- within one move
function SoccerFramework:is_adjacent()
    if math.abs(self.coord_a.x - self.coord_b.x) == 1 and 
        math.abs(self.coord_a.y - self.coord_b.y) == 0 or 
        math.abs(self.coord_a.x - self.coord_b.x) == 0 and 
        math.abs(self.coord_a.y - self.coord_b.y) == 1 then
        return true
    end
    return false
end

-- go toward the opponent's goal area
function SoccerFramework:act_advance(actions, coord, goal_area)
    local score = self.weights[ADVANCE]
    if coord.x < goal_area.x then
        actions[soccer.RIGHT] = actions[soccer.RIGHT] + score
    elseif coord.x > goal_area.x then 
        actions[soccer.LEFT] = actions[soccer.LEFT] + score
    end
    if coord.y < goal_area.y[1] then
        actions[soccer.UP] = actions[soccer.UP] + score
    elseif coord.y > goal_area.y[2] then
        actions[soccer.DOWN] = actions[soccer.DOWN] + score
    end
end

-- go toward my goal area
function SoccerFramework:act_defend(actions, coord, goal_area)
    local score = self.weights[DEFEND]
    -- if in front of goal, then moving within the goal
    if math.abs(coord.x - goal_area.x) == 1 then
        if coord.y <= goal_area.y[1] then
            actions[soccer.UP] = actions[soccer.UP] + score
        elseif coord.y >= goal_area.y[2] then
            actions[soccer.DOWN] = actions[soccer.DOWN] + score
        else
            actions[soccer.UP] = actions[soccer.UP] + score
            actions[soccer.DOWN] = actions[soccer.DOWN] + score
        end
    -- moving toward the goal
    else
        if coord.x < goal_area.x then
            actions[soccer.RIGHT] = actions[soccer.RIGHT] + score
        else
            actions[soccer.LEFT] = actions[soccer.LEFT] + score
        end
        if coord.y <= goal_area.y[1] then
            actions[soccer.UP] = actions[soccer.UP] + score
        elseif coord.y >= goal_area.y[2] then
            actions[soccer.DOWN] = actions[soccer.DOWN] + score
        end
    end
end

-- intercept the opponent
function SoccerFramework:act_intercept(actions, coord, opp_coord)
    local score = self.weights[INTERCEPT]
    if self:is_adjacent(coord, opp_coord) then
        actions[soccer.NOOP] = actions[soccer.NOOP] + score
    else
        if coord.x < opp_coord.x then
            actions[soccer.RIGHT] = actions[soccer.RIGHT] + score
        elseif coord.x > opp_coord.x then 
            actions[soccer.LEFT] = actions[soccer.LEFT] + score
        end
        if coord.y < opp_coord.y then
            actions[soccer.UP] = actions[soccer.UP] + score
        elseif coord.y > opp_coord.y then
            actions[soccer.DOWN] = actions[soccer.DOWN] + score
        end
    end
end

-- avoid the opponent
function SoccerFramework:act_avoid(actions, coord, opp_coord)
    local score = self.weights[AVOID]
    if coord.x <= opp_coord.x then
        actions[soccer.LEFT] = actions[soccer.LEFT] + score
    end
    if coord.x >= opp_coord.x then 
        actions[soccer.RIGHT] = actions[soccer.RIGHT] + score
    end
    if coord.y <= opp_coord.y then
        actions[soccer.DOWN] = actions[soccer.DOWN] + score
    end
    if coord.y >= opp_coord.y then
        actions[soccer.UP] = actions[soccer.UP] + score
    end
end

function SoccerFramework:act(has_ball, coord, goal_area, opp_coord, opp_goal_area, prev_action)
    local actions = torch.FloatTensor(self.n_actions):zero()
    local r = torch.rand(1)[1]
    if has_ball then
        --if r < self.defensive then
        --    --print('avoid')
        --    self:act_avoid(actions, coord, opp_coord)
        --else
        --    --print('advance')
        --    self:act_advance(actions, coord, opp_goal_area)
        --end
        self:act_advance(actions, coord, opp_goal_area)
        self:act_avoid(actions, coord, opp_coord)
    else
        --if r < self.defensive then
        --    --print('defend')
        --    self:act_defend(actions, coord, goal_area)
        --else
        --    --print('intercept')
        --    self:act_intercept(actions, coord, opp_coord)
        --end
        self:act_defend(actions, coord, goal_area)
        self:act_intercept(actions, coord, opp_coord)
    end
    local scores, sorted_actions = torch.sort(actions, true) 
    local n = 1
    for i=2,scores:size(1) do
        if scores[i] == scores[i-1] then
            n = n + 1
        else break end
    end
    for i=1,n do
        if sorted_actions[i] == prev_action then
            return prev_action
        end
    end
    return sorted_actions[torch.random(1,n)]
end

function SoccerFramework:observe_opponent(has_ball, coord, old_opp_coord, new_opp_coord, goal_area, opp_goal_area)
    --print('my old coord:', coord)
    --print('opp old coord:', old_opp_coord)
    --print('opp new coord:', new_opp_coord)
    local i = has_ball and 1 or 2
    local towards_me, towards_my_goal, towards_opp_goal, noop
    -- if it's going towards p when changing from p1 to p2
    function is_towards(p, p1, p2)
        if math.abs(p2.x - p.x) < math.abs(p1.x - p.x) or
            math.abs(p2.y - p.y) < math.abs(p1.y - p.y) then
            return 1
        elseif math.abs(p2.x - p.x) > math.abs(p1.x - p.x) or
            math.abs(p2.y - p.y) > math.abs(p1.y - p.y) then
            return -1
        else
            return 0
        end
    end
    towards_me = is_towards(coord, old_opp_coord, new_opp_coord)
    towards_my_goal = is_towards({x=goal_area.x, y=(goal_area.y[1]+goal_area.y[2])/2}, old_opp_coord, new_opp_coord)
    towards_opp_goal = is_towards({x=opp_goal_area.x, y=(opp_goal_area.y[1]+opp_goal_area.y[2])/2}, old_opp_coord, new_opp_coord)
    if old_opp_coord.x == new_opp_coord.x and old_opp_coord.y == new_opp_coord.y then noop = 1 else noop = 0 end
    --print('towards me:', towards_me)
    --print('towards my goal:', towards_my_goal)
    --print('towards opp goal:', towards_opp_goal)
    self.opp_pos_stat[i][1] = self.opp_pos_stat[i][1]*0.8 + towards_me
    self.opp_pos_stat[i][2] = self.opp_pos_stat[i][2]*0.8 + towards_my_goal
    self.opp_pos_stat[i][3] = self.opp_pos_stat[i][3]*0.8 + towards_opp_goal
    self.opp_pos_stat[i][4] = self.opp_pos_stat[i][4]*0.8 + noop
    self.opp_pos[1] = i
    self.opp_pos[2] = towards_me
    self.opp_pos[3] = towards_my_goal
    self.opp_pos[4] = towards_opp_goal
    self.opp_pos[5] = noop 
end

function SoccerFramework:step(action_a, action_b)
    local state, terminal, reward
    self.step_count = self.step_count + 1
    if self.step_count > self.max_steps then
        return self:state_feat(self.max_steps), true, 0
    end
    self.prev_action_a = action_a
    self.prev_action_b = action_b
    -- use built-in rule-based agent
    if action_a == nil then
        action_a = self:act(self.has_ball == soccer.A, self.coord_a, self.goal_area_b, self.coord_b, self.goal_area_a, self.prev_action_a)
    elseif action_b == nil then
        action_b = self:act(self.has_ball == soccer.B, self.coord_b, self.goal_area_a, self.coord_a, self.goal_area_b, self.prev_action_b)
    end
    -- update position
    local coord_a_old = {x=self.coord_a.x, y=self.coord_a.y}
    local coord_b_old = {x=self.coord_b.x, y=self.coord_b.y}
    self:update_coord(soccer.A, self.coord_a, action_a)
    self:update_coord(soccer.B, self.coord_b, action_b)

    -- update opponent history
    self:observe_opponent(self.has_ball == soccer.B, coord_a_old, coord_b_old, self.coord_b, self.goal_area_a, self.goal_area_b)
    function add_history(array, new_data)
        local n = array:size(1)
        if self.step_count > n then
            for i=1,n-1 do
                array[i]:copy(array[i+1])
            end
        end
        array[n]:copy(new_data)
    end
    add_history(self.opp_pos_seq, self.opp_pos)
    -- update opponent action history
    self.opp_action[action_b] = 1
    self.opp_action_stat[action_b] = self.opp_action_stat[action_b] + 1
    add_history(self.opp_action_seq, self.opp_action)

    if DEBUG == 1 then
        assert(self.coord_a.x > self.min_x and self.coord_a.x <= self.max_x)
        assert(self.coord_a.y >= self.min_y and self.coord_a.y <= self.max_y)
        assert(self.coord_b.x >= self.min_x and self.coord_b.x < self.max_x)
        assert(self.coord_b.y >= self.min_y and self.coord_b.y <= self.max_y)
    end
    -- exchange ball when hit
    if self.coord_a.x == self.coord_b.x and self.coord_a.y == self.coord_b.y then
        if self.has_ball == soccer.A then
            self.has_ball = soccer.B
            self.lost_ball = self.lost_ball + 1
            --if not self.test then reward = -1 end
        elseif self.has_ball == soccer.B then
            self.has_ball = soccer.A
            --if not self.test then reward = 1 end
        end
        -- reset position
        self.coord_a.x = coord_a_old.x
        self.coord_a.y = coord_a_old.y
        self.coord_b.x = coord_b_old.x
        self.coord_b.y = coord_b_old.y
    end
    -- assuming A is the agent we want to control
    if self:is_goal(self.has_ball == soccer.A, self.coord_a, self.goal_area_a) then
        reward = 1
        terminal = true
    elseif self:is_goal(self.has_ball == soccer.B, self.coord_b, self.goal_area_b) then
        reward = -1
        terminal = true
    else
        reward = 0
        terminal = false
    end
    if DEBUG == 1 then
        print('action_a:', self.action_str[action_a])
        print('action_b:', self.action_str[action_b])
        print('reward:', reward, 'terminal:', terminal)
        self:print_state()
    end
    return self:state_feat(self.step_count), terminal, reward
end
