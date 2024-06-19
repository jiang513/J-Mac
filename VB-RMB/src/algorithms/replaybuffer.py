import torch
import numpy as np
import utils


class ReplayBuffer:
    def __init__(self, env, critic, args):
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape

        obs_dtype = env.observation_space.dtype.name
        action_dtype = env.action_space.dtype.name

        self.critic = critic

        self.obses = np.zeros((args.horizon, *obs_shape), dtype = obs_dtype)
        self.actions = np.zeros((args.horizon, *action_shape), dtype= action_dtype)
        self.action_logps = np.zeros((args.horizon, *action_shape), dtype= action_dtype)
        self.rewards = np.zeros((args.horizon, 1), dtype = np.float32)
        self.dones = np.zeros((args.horizon, 1), dtype = np.float32)
        self.next_obses = np.zeros((args.horizon, *obs_shape), dtype = obs_dtype)
        self.dones_w = np.zeros((args.horizon, 1), dtype = np.float32)
        self.returns = np.zeros((args.horizon, 1), dtype = np.float32)
        self.values = np.zeros((args.horizon, 1), dtype = np.float32)
        self.next_values = np.zeros((args.horizon, 1), dtype = np.float32)
        self.advs = np.zeros((args.horizon, 1), dtype = np.float32)
        

        self.idx = 0
        self.max_size = args.horizon
        self.gamma = args.gamma
        self.lam = args.lam
        self.full = False


    def store(self, obs, action, action_logb, reward, next_obs, value, done_w, done):
        assert self.idx < self.max_size

        self.obses[self.idx] = obs
        self.next_obses[self.idx] = next_obs
        self.rewards[self.idx] = reward
        self.actions[self.idx] = action
        self.values[self.idx] = value.cpu()
        self.action_logps[self.idx] = action_logb
        self.dones_w[self.idx] = done_w
        self.dones[self.idx] = done
        self.idx += 1

    
    def _encode_obses(self, obses):
        np_obses = []
        for obs in obses:
            np_obses.append(np.array(obs, copy=False))
        
        return np.array(np_obses)
    

    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames):
            _obs = np.array(obs)
        else:
            _obs = obs
        _obs = torch.FloatTensor(_obs).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs


    def get(self):
        assert self.idx == self.max_size

        self.next_values[:self.idx - 1] = self.values[1:self.idx]
        last_next_obs = self.next_obses[self.idx - 1]
        # print(last_next_obs.shape)
        with torch.no_grad():
            _last_next_obs = self._obs_to_input(last_next_obs)
            self.next_values[self.idx - 1] = self.critic(_last_next_obs).cpu()

        lastgaelam = 0.0
        for t in reversed(range(0, self.idx)):
            nondone = 1.0 - self.dones[t]
            nondone_w = 1.0 -self.dones_w[t]
            delta = self.rewards[t] + self.gamma * nondone_w * self.next_values[t] - self.values[t]
            self.advs[t] = delta + self.gamma * self.lam * nondone * lastgaelam
            lastgaelam = self.advs[t]
        
        self.returns[:] = self.advs + self.values

        self.obses = self._encode_obses(self.obses)

        self.idx = 0

        obses = torch.as_tensor(self.obses).cuda().float()
        actions = torch.as_tensor(self.actions).cuda().float()
        advs = torch.as_tensor(self.advs).cuda().float()
        returns = torch.as_tensor(self.returns).cuda().float()
        action_logps = torch.as_tensor(self.action_logps).cuda().float()

        return obses, actions, advs, returns, action_logps
    
    def reset(self):
        self.idx = 0