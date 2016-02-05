local nql = torch.class('dqn.SoccerRuleAgent')

function nql:__init(args)
    self.actions = args.actions
end

function nql:perceive(args)
    -- the game env should decide the action when input action is nil
    return nil
end
