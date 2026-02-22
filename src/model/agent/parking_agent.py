
class PrimitivePlanner(object):
    """A simple plan executor for motion primitives.

    It holds a queue of primitive IDs to execute (macro-actions).
    """

    def __init__(self) -> None:
        self.plan = []

    def reset(self):
        self.plan.clear()

    def set_plan(self, primitive_id_list, forced: bool = False):
        if primitive_id_list is None:
            return
        if forced or len(self.plan) == 0:
            self.plan = list(primitive_id_list)

    @property
    def executing(self) -> bool:
        return len(self.plan) > 0

    def get_action(self) -> int:
        pid = int(self.plan.pop(0))
        return pid


class ParkingAgent(object):
    def __init__(
        self, rl_agent, planner: PrimitivePlanner = None
    ) -> None:
        self.agent = rl_agent
        self.planner = planner

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.agent, name)
    
    def reset(self,):
        if self.planner is not None:
            self.planner.reset()

    def set_planner_path(self, path=None, forced: bool = False):
        if self.planner is None:
            return
        self.planner.set_plan(path, forced=forced)

    @property
    def executing_plan(self):
        return not (self.planner is None or not self.planner.executing)

    def choose_action(self, obs, deterministic=False, action_mask=None):
        '''
        Get the decision from the agent.

        Params:
            obs(dict): the observation of the environment

        Return:
            action(np.array): the decision
            other: the other information, such as the log_prob of the action in case of PPO
        '''
        if not self.executing_plan:
            return self.agent.choose_action(obs, deterministic=deterministic, action_mask=action_mask)

        primitive_id = self.planner.get_action()
        log_prob = self.agent.get_log_prob(obs, primitive_id, action_mask=action_mask)
        return primitive_id, log_prob
        
    def get_action(self, obs):
        '''
        Get the decision from the agent.
        '''
        return self.agent.get_action(obs)
