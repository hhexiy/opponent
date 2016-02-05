local nql = torch.class('dqn.RandomAgent')

function nql:__init(args)
    self.actions = args.actions
    self.n_actions = #args.actions
end

function nql:perceive(args)
    return torch.random(1, self.n_actions)
end
