local nql = torch.class('dqn.SoccerONeuralQLearner_multitask_action', 'dqn.SoccerNeuralQLearner_multitask')

function nql:__init(args)
    self.n_experts = args.n_experts
    self.feat_groups = args.feat_groups
    print('number of experts: ', self.n_experts)
    self.criterion = nn.ClassNLLCriterion() 
    self.num_classes = #args.actions
    print('network model:', args.model)
    if args.model == 'fc2' then
        self.createNetwork = self.createNetwork_fc2
    else
        self.createNetwork = self.createNetwork_moe
    end
    dqn.SoccerNeuralQLearner_multitask.__init(self, args)
end

