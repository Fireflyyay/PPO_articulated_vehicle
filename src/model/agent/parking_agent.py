
class ParkingAgent(object):
    def __init__(
        self, rl_agent
    ) -> None:
        self.agent = rl_agent

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.agent, name)
    
    def reset(self,):
        pass

    def choose_action(self, obs, deterministic=False):
        '''
        Get the decision from the agent.

        Params:
            obs(dict): the observation of the environment

        Return:
            action(np.array): the decision
            other: the other information, such as the log_prob of the action in case of PPO
        '''
        return self.agent.choose_action(obs, deterministic)
        
    def get_action(self, obs):
        '''
        Get the decision from the agent.
        '''
        return self.agent.get_action(obs)
