
import numpy as np

DEFAULT_UPDATE_MODAL = {'img':False, 'lidar':True, 'target':True, 'action_mask':False}

class StateNorm():
    def __init__(self, observation_shape, update_modal=None) -> None:
        if not isinstance(observation_shape, dict):
            # Handle flat observation shape (e.g. tuple or list)
            self.flat_obs = True
            self.observation_shape = {'default': observation_shape}
            if update_modal is None:
                self.update_modal = {'default': True}
            else:
                self.update_modal = {'default': True} # Ignore passed modal if flat
        else:
            self.flat_obs = False
            self.observation_shape = observation_shape
            self.update_modal = update_modal if update_modal is not None else DEFAULT_UPDATE_MODAL
            
        self.n_state = 0
        self.state_mean = np.zeros(1, dtype=np.float32) # Dummy init
        self.state_std = np.zeros(1, dtype=np.float32) # Dummy init
        self.S = np.zeros(1, dtype=np.float32) # Dummy init
        
        # Re-init proper shapes
        self.state_mean, self.S, self.state_std = {}, {}, {}
        for obs_type in self.observation_shape.keys():
            shape = self.observation_shape[obs_type]
            self.state_mean[obs_type] = np.zeros(shape, dtype=np.float32)
            self.S[obs_type] = np.zeros(shape, dtype=np.float32)
            self.state_std[obs_type] = np.sqrt(self.S[obs_type])
            
    def fix_parameters(self,):
        self.fixed = True
    
    def init_state_norm(self, mean, std, S, n_state):
        self.n_state = n_state
        self.state_mean, self.state_std, self.S = mean, std, S

    def state_norm(self, observation, update=False):
        # Handle flat observation input
        if self.flat_obs:
            if isinstance(observation, dict):
                # Should not happen if initialized as flat
                pass
            else:
                # Wrap
                obs_dict = {'default': observation}
                norm_dict = self._state_norm_dict(obs_dict, update)
                return norm_dict['default']
        else:
            return self._state_norm_dict(observation, update)

    def _state_norm_dict(self, observation: dict, update=False):
        if self.n_state == 0:
            self.n_state += 1
            for obs_type in self.observation_shape.keys():
                if self.update_modal.get(obs_type, False):
                    self.state_mean[obs_type] = observation[obs_type]
                    # Stable init: first observation becomes 0 after normalization.
                    self.S[obs_type] = np.zeros_like(observation[obs_type], dtype=np.float32)
                    self.state_std[obs_type] = np.ones_like(observation[obs_type], dtype=np.float32)
                    observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (self.state_std[obs_type] + 1e-8)
                    
        elif update==False or getattr(self, 'fixed', False):
            for obs_type in self.observation_shape.keys():
                if self.update_modal.get(obs_type, False):
                    std = np.maximum(self.state_std[obs_type], 1e-3)
                    observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (std + 1e-8)
        elif update==True:
            self.n_state += 1
            for obs_type in self.observation_shape.keys():
                if self.update_modal.get(obs_type, False):
                    old_mean = self.state_mean[obs_type].copy()
                    
                    # Welford's algorithm update usually:
                    # mean = old_mean + (x - old_mean) / n
                    # S = S + (x - old_mean) * (x - new_mean)
                    # std = sqrt(S / (n-1)) or similar.
                    # Original code:
                    # mean = old_mean + (obs - old_mean) / n
                    # S = S + (obs - old_mean) * (obs - mean)
                    # std = sqrt(S / n)
                    
                    self.state_mean[obs_type] = old_mean + (observation[obs_type] - old_mean) / self.n_state
                    self.S[obs_type] = self.S[obs_type] + (observation[obs_type] - old_mean) *\
                        (observation[obs_type] - self.state_mean[obs_type])

                    # Use an unbiased-ish variance estimate when possible; clamp to avoid zero-variance blow-ups.
                    denom = max(self.n_state - 1, 1)
                    self.state_std[obs_type] = np.sqrt(self.S[obs_type] / denom)
                    std = np.maximum(self.state_std[obs_type], 1e-3)

                    observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (std + 1e-8)
        return observation
