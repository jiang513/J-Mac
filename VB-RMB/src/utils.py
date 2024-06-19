import torch
import numpy as np
import os
import glob
import json
import random
import augmentations
import subprocess
from datetime import datetime
import copy
import torch.nn as nn
import kornia


def center_crop_images(image, output_size):
	h, w = image.shape[2:]
	new_h, new_w = output_size, output_size

	top = (h - new_h)//2
	left = (w - new_w)//2

	image = image[:, :, top:top + new_h, left:left + new_w]
	return image

def center_crop_image(image, output_size):
	h, w = image.shape[1:]
	new_h, new_w = output_size, output_size

	top = (h - new_h)//2
	left = (w - new_w)//2

	image = image[:, top:top + new_h, left:left + new_w]
	return image


class eval_mode(object):
	def __init__(self, *models):
		self.models = models

	def __enter__(self):
		self.prev_states = []
		for model in self.models:
			self.prev_states.append(model.training)
			model.train(False)

	def __exit__(self, *args):
		for model, state in zip(self.models, self.prev_states):
			model.train(state)
		return False


def soft_update_params(net, target_net, tau):
	for param, target_param in zip(net.parameters(), target_net.parameters()):
		target_param.data.copy_(
			tau * param.data + (1 - tau) * target_param.data
		)


def cat(x, y, axis=0):
	return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def write_info(args, fp):
	data = {
		'timestamp': str(datetime.now()),
		'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
		'args': vars(args)
	}
	with open(fp, 'w') as f:
		json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
	path = os.path.join('setup', 'config.cfg')
	with open(path) as f:
		data = json.load(f)
	if key is not None:
		return data[key]
	return data


def make_dir(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def listdir(dir_path, filetype='jpg', sort=True):
	fpath = os.path.join(dir_path, f'*.{filetype}')
	fpaths = glob.glob(fpath, recursive=True)
	if sort:
		return sorted(fpaths)
	return fpaths


def prefill_memory(obses, capacity, obs_shape):
	"""Reserves memory for replay buffer"""
	c,h,w = obs_shape
	for _ in range(capacity):
		frame = np.ones((3,h,w), dtype=np.uint8)
		obses.append(frame)
	return obses


class ReplayBuffer(object):
	"""Buffer to store environment transitions"""
	def __init__(self, obs_shape, action_shape, capacity, batch_size, prefill=True, image_pad = 4):
		self.capacity = capacity
		self.batch_size = batch_size

		self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

		self._obses = []
		if prefill:
			self._obses = prefill_memory(self._obses, capacity, obs_shape)
		self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
		self.rewards = np.empty((capacity, 1), dtype=np.float32)
		self.not_dones = np.empty((capacity, 1), dtype=np.float32)

		self.idx = 0
		self.full = False

	def add(self, obs, action, reward, next_obs, done):
		obses = (obs, next_obs)
		if self.idx >= len(self._obses):
			self._obses.append(obses)
		else:
			self._obses[self.idx] = (obses)
		np.copyto(self.actions[self.idx], action)
		np.copyto(self.rewards[self.idx], reward)
		np.copyto(self.not_dones[self.idx], not done)

		self.idx = (self.idx + 1) % self.capacity
		self.full = self.full or self.idx == 0

	def _get_idxs(self, n=None):
		if n is None:
			n = self.batch_size
		return np.random.randint(
			0, self.capacity if self.full else self.idx, size=n
		)

	def _encode_obses(self, idxs):
		obses, next_obses = [], []
		for i in idxs:
			obs, next_obs = self._obses[i]
			obses.append(np.array(obs, copy=False))
			next_obses.append(np.array(next_obs, copy=False))
		return np.array(obses), np.array(next_obses)

	def sample_soda(self, n=None):
		idxs = self._get_idxs(n)
		obs, _ = self._encode_obses(idxs)
		return torch.as_tensor(obs).cuda().float()

	def sample_curl(self, n=None):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		pos = augmentations.random_crop(obs.clone())
		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones, pos

	def sample_drq(self, n=None, pad=4):
		idxs = self._get_idxs(n)
		obs, next_obs = self._encode_obses(idxs)
		obs_aug, next_obs_aug = obs.copy(), next_obs.copy()

		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		obs_aug = torch.as_tensor(obs_aug).cuda().float()
		next_obs_aug = torch.as_tensor(next_obs_aug).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		# obs = augmentations.random_shift(obs, pad)
		# next_obs = augmentations.random_shift(next_obs, pad)

		obs = self.aug_trans(obs)
		next_obs = self.aug_trans(next_obs)

		obs_aug = self.aug_trans(obs_aug)
		obs_aug = augmentations.random_conv(obs_aug.clone())
		next_obs_aug = self.aug_trans(next_obs_aug)
		next_obs_aug = augmentations.random_conv(next_obs_aug.clone())

		return obs, actions, rewards, next_obs, not_dones, obs_aug, next_obs_aug
	
	def sample_drq_jacobian(self, n=None, pad=4):
		idxs = np.random.randint(
			0, self.capacity if self.full else self.idx, size=self.batch_size*2
		)

		not_dones=torch.as_tensor(self.not_dones[idxs]).cuda() # (2*B,1)

		valid_idxs=torch.where(not_dones==1)[0].cpu().numpy()
		idxs=idxs[valid_idxs]
		idxs = idxs[:self.batch_size] if idxs.shape[0] >= self.batch_size else idxs

		obs_drq, next_obs_drq = self._encode_obses(idxs)
		obs_drq_aug, next_obs_drq_aug = obs_drq.copy(), next_obs_drq.copy()
		obs_jacobian, next_obs_jacobian = copy.deepcopy(obs_drq), copy.deepcopy(next_obs_drq)

		obs_drq = torch.as_tensor(obs_drq).cuda().float()
		next_obs_drq = torch.as_tensor(next_obs_drq).cuda().float()
		obs_drq_aug = torch.as_tensor(obs_drq_aug).cuda().float()
		next_obs_drq_aug = torch.as_tensor(next_obs_drq_aug).cuda().float()
		obs_jacobian = torch.as_tensor(obs_jacobian).cuda().float()
		next_obs_jacobian = torch.as_tensor(next_obs_jacobian).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		# obs = augmentations.random_shift(obs, pad)
		# next_obs = augmentations.random_shift(next_obs, pad)

		obs_drq = self.aug_trans(obs_drq)
		next_obs_drq = self.aug_trans(next_obs_drq)

		obs_drq_aug = self.aug_trans(obs_drq_aug)
		obs_drq_aug = augmentations.random_conv(obs_drq_aug.clone())
		next_obs_drq_aug = self.aug_trans(next_obs_drq_aug)
		next_obs_drq_aug = augmentations.random_conv(next_obs_drq_aug.clone())

		return obs_drq, actions, rewards, next_obs_drq, not_dones, obs_drq_aug, next_obs_drq_aug, obs_jacobian, next_obs_jacobian
	
	def sample_sacai(self, n=None, pad=4):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()


		return obs, actions, rewards, next_obs, not_dones

	def sample(self, n=None):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones
	
	def sample_rad(self,aug_funcs,n=None):
		
		# augs specified as flags
		# curl_sac organizes flags into aug funcs
		# passes aug funcs into sampler

		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)

		if aug_funcs:
			for aug,func in aug_funcs.items():
				# apply crop and cutout first
				if 'crop' in aug or 'cutout' in aug:
					obs = func(obs)
					next_obs = func(next_obs)
				elif 'translate' in aug: 
					og_obs = center_crop_images(obs, self.pre_image_size)
					og_next_obs = center_crop_images(next_obs, self.pre_image_size)
					obs, rndm_idxs = func(og_obs, self.image_size, return_random_idxs=True)
					next_obs = func(og_next_obs, self.image_size, **rndm_idxs)                     

		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		# augmentations go here
		if aug_funcs:
			for aug,func in aug_funcs.items():
				# skip crop and cutout augs
				if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
					continue
				obs = func(obs)
				next_obs = func(next_obs)

		return obs, actions, rewards, next_obs, not_dones
	
	def sample_rad_jacobian(self,aug_funcs,n=None,pad=4):
		idxs = np.random.randint(
			0, self.capacity if self.full else self.idx, size=self.batch_size*2
		)

		not_dones=torch.as_tensor(self.not_dones[idxs]).cuda() # (2*B,1)

		valid_idxs=torch.where(not_dones==1)[0].cpu().numpy()
		idxs=idxs[valid_idxs]
		idxs = idxs[:self.batch_size] if idxs.shape[0] >= self.batch_size else idxs

		obs, next_obs = self._encode_obses(idxs)

		obs_aug = copy.deepcopy(obs)
		next_obs_aug = copy.deepcopy(next_obs)

		if aug_funcs:
			for aug,func in aug_funcs.items():
				# apply crop and cutout first
				if 'crop' in aug or 'cutout' in aug:
					obs_aug = func(obs_aug)
					next_obs_aug = func(next_obs_aug)
				elif 'translate' in aug: 
					obs_aug = center_crop_images(obs_aug, self.pre_image_size)
					next_obs_aug = center_crop_images(next_obs_aug, self.pre_image_size)
					obs_aug, rndm_idxs = func(obs_aug, self.image_size, return_random_idxs=True)
					next_obs_aug = func(next_obs_aug, self.image_size, **rndm_idxs)

		obs_aug = torch.as_tensor(obs_aug).cuda().float()
		next_obs_aug = torch.as_tensor(next_obs_aug).cuda().float()

		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		# augmentations go here
		if aug_funcs:
			for aug,func in aug_funcs.items():
				# skip crop and cutout augs
				if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
					continue
				obs_aug = func(obs_aug)
				next_obs_aug = func(next_obs_aug)


		return obs, obs_aug, actions, rewards, next_obs, next_obs_aug, not_dones


class LazyFrames(object):
	def __init__(self, frames, extremely_lazy=True):
		self._frames = frames
		self._extremely_lazy = extremely_lazy
		self._out = None

	@property
	def frames(self):
		return self._frames

	def _force(self):
		if self._extremely_lazy:
			return np.concatenate(self._frames, axis=0)
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=0)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		if self._extremely_lazy:
			return len(self._frames)
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]

	def count(self):
		if self.extremely_lazy:
			return len(self._frames)
		frames = self._force()
		return frames.shape[0]//3

	def frame(self, i):
		return self._force()[i*3:(i+1)*3]


def count_parameters(net, as_int=False):
	"""Returns total number of params in a network"""
	count = sum(p.numel() for p in net.parameters())
	if as_int:
		return count
	return f'{count:,}'