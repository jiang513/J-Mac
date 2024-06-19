import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from copy import deepcopy
import copy
import math
import utils
import algorithms.modules as m
from algorithms.sac import SAC
import augmentations
from algorithms.modules import DeterministicTransitionModel, \
encoder_logits_contractive, compute_mask_new
import algorithms.data_augs as rad 
from torch.autograd.functional import jacobian, hessian
from algorithms.encoder import make_encoder


def squash(mu, pi, log_pi):
	"""Apply squashing function.
	See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
	"""
	mu = torch.tanh(mu)
	if pi is not None:
		pi = torch.tanh(pi)
	if log_pi is not None:
		log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
	return mu, pi, log_pi


def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability."""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
	"""MLP actor network."""
	def __init__(
		self, obs_shape, action_shape, hidden_dim, encoder_type,
		encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
	):
		super().__init__()

		self.encoder = make_encoder(
			encoder_type, obs_shape, encoder_feature_dim, num_layers,
			num_filters, output_logits=True
		)

		self.log_std_min = log_std_min
		self.log_std_max = log_std_max

		self.trunk = nn.Sequential(
			nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 2 * action_shape[0])
		)

		self.outputs = dict()
		self.apply(weight_init)

	def forward(
		self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
	):
		obs = self.encoder(obs, detach=detach_encoder)

		mu, log_std = self.trunk(obs).chunk(2, dim=-1)

		# constrain log_std inside [log_std_min, log_std_max]
		log_std = torch.tanh(log_std)
		log_std = self.log_std_min + 0.5 * (
			self.log_std_max - self.log_std_min
		) * (log_std + 1)

		self.outputs['mu'] = mu
		self.outputs['std'] = log_std.exp()

		if compute_pi:
			std = log_std.exp()
			noise = torch.randn_like(mu)
			pi = mu + noise * std
		else:
			pi = None
			entropy = None

		if compute_log_pi:
			log_pi = gaussian_logprob(noise, log_std)
		else:
			log_pi = None

		mu, pi, log_pi = squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std


class QFunction(nn.Module):
	"""MLP for q-function."""
	def __init__(self, obs_dim, action_dim, hidden_dim):
		super().__init__()

		self.trunk = nn.Sequential(
			nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)

	def forward(self, obs, action):
		assert obs.size(0) == action.size(0)

		obs_action = torch.cat([obs, action], dim=1)
		return self.trunk(obs_action)


class Critic(nn.Module):
	"""Critic network, employes two q-functions."""
	def __init__(
		self, obs_shape, action_shape, hidden_dim, encoder_type,
		encoder_feature_dim, num_layers, num_filters
	):
		super().__init__()


		self.encoder = make_encoder(
			encoder_type, obs_shape, encoder_feature_dim, num_layers,
			num_filters, output_logits=True
		)

		self.Q1 = QFunction(
			self.encoder.feature_dim, action_shape[0], hidden_dim
		)
		self.Q2 = QFunction(
			self.encoder.feature_dim, action_shape[0], hidden_dim
		)

		self.outputs = dict()
		self.apply(weight_init)

	def forward(self, obs, action, detach_encoder=False):
		# detach_encoder allows to stop gradient propogation to encoder
		obs = self.encoder(obs, detach=detach_encoder)

		q1 = self.Q1(obs, action)
		q2 = self.Q2(obs, action)

		self.outputs['q1'] = q1
		self.outputs['q2'] = q2

		return q1, q2


class RAD(object):
	"""RAD with SAC."""
	def __init__(
		self,
		obs_shape,
		action_shape,
		args,
		device='cuda',
		hidden_dim=1024,
		discount=0.99,
		init_temperature=0.1,
		alpha_lr=1e-3,
		alpha_beta=0.9,
		actor_lr=1e-3,
		actor_beta=0.9,
		actor_log_std_min=-10,
		actor_log_std_max=2,
		actor_update_freq=2,
		critic_lr=1e-3,
		critic_beta=0.9,
		critic_tau=0.01,
		critic_target_update_freq=2,
		encoder_type='pixel',
		encoder_feature_dim=50,
		encoder_lr=1e-3,
		encoder_tau=0.05,
		num_layers=11,
		num_filters=32,
		log_interval=100,
		detach_encoder=False,
		data_augs = 'crop-rotate-flip',
	):
		self.device = device
		self.discount = discount
		self.critic_tau = critic_tau
		self.encoder_tau = encoder_tau
		self.actor_update_freq = actor_update_freq
		self.critic_target_update_freq = critic_target_update_freq
		self.log_interval = log_interval
		# self.image_size = obs_shape[-1]

		self.detach_encoder = detach_encoder
		self.encoder_type = encoder_type
		self.data_augs = data_augs
		self.use_aux = args.use_aux
		self.encoder_loss_type = args.encoder_loss_type
		self.transition_loss_type = args.transition_loss_type
		self.jacobin_update_freq = args.jacobin_update_freq
		self.use_jacobian = args.use_jacobian
		self.image_crop_size = args.image_crop_size
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		self.updata_transition_model_freq = args.updata_transition_model_freq

		self.augs_funcs = {}

		aug_to_func = {
				'crop':rad.random_crop,
				'grayscale':rad.random_grayscale,
				'cutout':rad.random_cutout,
				'cutout_color':rad.random_cutout_color,
				'flip':rad.random_flip,
				'rotate':rad.random_rotation,
				'rand_conv':rad.random_convolution,
				'color_jitter':rad.random_color_jitter,
				'translate':rad.random_translate,
				'no_aug':rad.no_aug,
			}

		for aug_name in self.data_augs.split('-'):
			assert aug_name in aug_to_func, 'invalid data aug string'
			self.augs_funcs[aug_name] = aug_to_func[aug_name]

		self.actor = Actor(
			obs_shape, action_shape, hidden_dim, encoder_type,
			encoder_feature_dim, actor_log_std_min, actor_log_std_max,
			num_layers, num_filters
		).to(device)

		self.critic = Critic(
			obs_shape, action_shape, hidden_dim, encoder_type,
			encoder_feature_dim, num_layers, num_filters
		).to(device)

		self.critic_target = Critic(
			obs_shape, action_shape, hidden_dim, encoder_type,
			encoder_feature_dim, num_layers, num_filters
		).to(device)

		self.critic_target.load_state_dict(self.critic.state_dict())

		# tie encoders between actor and critic
		self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

		self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
		self.log_alpha.requires_grad = True
		# set target entropy to -|A|
		self.target_entropy = -np.prod(action_shape)


		# begin
		self.global_classifier = nn.Sequential(
			nn.Linear(encoder_feature_dim, args.latent_dim), nn.ReLU(),
			nn.Linear(args.latent_dim, encoder_feature_dim)).cuda()
		self.global_target_classifier = nn.Sequential(
			nn.Linear(encoder_feature_dim, args.latent_dim), nn.ReLU(),
			nn.Linear(args.latent_dim, encoder_feature_dim)).cuda()
		self.global_final_classifier = nn.Sequential(
			nn.Linear(encoder_feature_dim, args.latent_dim), nn.ReLU(),
			nn.Linear(args.latent_dim, encoder_feature_dim)).cuda()



		if self.encoder_loss_type=="contractive":
			self.encoder_logits_contra=encoder_logits_contractive(critic=self.critic).cuda()

		if self.transition_loss_type=="contractive":
			self.transition_logits_contra=encoder_logits_contractive(critic=self.critic).cuda()

		self.transition_model=DeterministicTransitionModel(
			self.critic.encoder.out_dim,self.critic.encoder.out_dim,action_shape,1024).cuda()
		# end


		
		# optimizers
		self.actor_optimizer = torch.optim.Adam(
			self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
		)

		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
		)

		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
		)

		if self.encoder_type == 'pixel':
			self.encoder_optimizer = torch.optim.Adam(
				self.critic.encoder.parameters(), lr=encoder_lr
			)
		

		# begin
		if self.encoder_loss_type=="contractive":
			self.encoder_logits_W_optimizer=torch.optim.Adam(
				self.encoder_logits_contra.parameters(), lr=args.encoder_lr
			)

		if self.transition_loss_type=="contractive":
			self.transition_logits_W_optimizer=torch.optim.Adam(
				self.transition_logits_contra.parameters(), lr=args.transition_model_lr
			)

		self.transition_model_optimizer=torch.optim.Adam(
			[{"params":self.transition_model.parameters()},
				{"params":self.global_classifier.parameters()},
				{"params":self.global_final_classifier.parameters()}],lr=args.transition_model_lr)
		# end


		self.cross_entropy_loss = nn.CrossEntropyLoss()

		self.train()
		self.critic_target.train()

	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.critic.train(training)
		self.transition_model.train(training)
		# if self.encoder_type == 'pixel':
		# 	self.CURL.train(training)

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def _obs_to_input(self, obs):
		if isinstance(obs, utils.LazyFrames):
			_obs = np.array(obs)
		else:
			_obs = obs
		_obs = torch.FloatTensor(_obs).to(self.device)
		_obs = _obs.unsqueeze(0)
		return _obs

	def select_action(self, obs):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			mu, _, _, _ = self.actor(
				_obs, compute_pi=False, compute_log_pi=False
			)
			return mu.cpu().data.numpy().flatten()

	def sample_action(self, obs):
		_obs = self._obs_to_input(obs)

		with torch.no_grad():
			mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
			return pi.cpu().data.numpy().flatten()

	def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		# get current Q estimates
		current_Q1, current_Q2 = self.critic(
			obs, action, detach_encoder=self.detach_encoder)
		critic_loss = F.mse_loss(current_Q1,
								target_Q) + F.mse_loss(current_Q2, target_Q)
		if step % self.log_interval == 0:
			L.log('train_critic/loss', critic_loss, step)


		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# self.critic.log(L, step)

	def update_actor_and_alpha(self, obs, L, step):
		# detach encoder, so we don't update it with the actor loss
		_, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
		actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

		if step % self.log_interval == 0:
			L.log('train_actor/loss', actor_loss, step)
			L.log('train_actor/target_entropy', self.target_entropy, step)
		entropy = 0.5 * log_std.shape[1] * \
			(1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
		if step % self.log_interval == 0:                                    
			L.log('train_actor/entropy', entropy.mean(), step)

		# optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# self.actor.log(L, step)

		self.log_alpha_optimizer.zero_grad()
		alpha_loss = (self.alpha *
					(-log_pi - self.target_entropy).detach()).mean()
		if step % self.log_interval == 0:
			L.log('train_alpha/loss', alpha_loss, step)
			L.log('train_alpha/value', self.alpha, step)
		alpha_loss.backward()
		self.log_alpha_optimizer.step()

	# 更改 begin

	##### original
	def batch_jacobian(self, f, obs, action):
		f_sum = lambda x, y : torch.sum(f(x, y), axis=0)
		return jacobian(f_sum, (obs, action))
	##### original

	##### original
	def jacobian_func(self, x, y):
		x = self.critic.encoder(x)
		x = self.transition_model.sample_prediction(torch.cat([x,y],dim=1))
		x = self.global_classifier(x)
		x = self.global_final_classifier(x)
		return x
	##### original

	def update_transition_model(self, obs, next_obs, action, step):
		obs = utils.center_crop_images(obs, self.image_crop_size)
		next_obs = utils.center_crop_images(next_obs, self.image_crop_size)
		enc = self.critic.encoder(obs)
		next_obs_latent_pre = self.transition_model.sample_prediction(torch.cat([enc,action],dim=1))
		next_obs_latent_pre = self.global_classifier(next_obs_latent_pre)
		next_obs_latent_pre = self.global_final_classifier(next_obs_latent_pre)
		with torch.no_grad():
			next_obs_latent = self.critic_target.encoder(next_obs)
			next_obs_latent = self.global_target_classifier(next_obs_latent)

		if self.transition_loss_type=="contractive":
			logits=self.transition_logits_contra.compute_encoder_logits_contractive(new_obs_latent=next_obs_latent,
			obs_latent=next_obs_latent_pre)
			labels = torch.arange(logits.shape[0]).long().cuda()
			transition_model_loss = self.cross_entropy_loss(logits, labels)
		else:
			# transition_model_loss = F.mse_loss(next_obs_latent_pre,next_obs_latent)
			next_obs_latent = F.normalize(next_obs_latent.float(), p=2., dim=-1, eps=1e-3)
			next_obs_latent_pre = F.normalize(next_obs_latent_pre.float(), p=2., dim=-1, eps=1e-3)
			transition_model_loss = F.mse_loss(next_obs_latent, next_obs_latent_pre, reduction="none").sum(-1).mean(0)

		wandb.log({"transition_model_loss":transition_model_loss},step=step)

		self.transition_model_optimizer.zero_grad()
		self.encoder_optimizer.zero_grad()
		if self.transition_loss_type=="contractive":
			self.transition_logits_W_optimizer.zero_grad()
		transition_model_loss.backward()
		self.transition_model_optimizer.step()
		self.encoder_optimizer.step()
		if self.transition_loss_type=="contractive":
			self.transition_logits_W_optimizer.step()
	
	def update_encoder_jacobin(self, obs, action, step):
		obs = utils.center_crop_images(obs, self.image_crop_size)

		##### depart 3 masks
		J=self.batch_jacobian(self.jacobian_func,obs,action)
		partial_grad_all=J[0].permute(1,0,2,3,4)
		partial_grad_all = torch.sum(torch.abs(partial_grad_all), dim=1)
		masks = compute_mask_new(partial_grad_all)
		new_obs=obs * masks
		with torch.no_grad():
			new_obs_latent=self.critic_target.encoder(new_obs)
		##### depart 3 masks

		##### jacobian general new loss
		obs_latent=self.critic.encoder(obs)
		obs_aug = augmentations.random_overlay(obs.clone())
		new_obs_aug = obs_aug * masks
		with torch.no_grad():
			new_obs_aug_latent = self.critic_target.encoder(new_obs_aug)
		obs_aug_latent=self.critic.encoder(obs_aug)
		##### jacobian general new loss

		if self.encoder_loss_type=="contractive":

			##### jacobina general new loss
			logits=self.encoder_logits_contra.compute_encoder_logits_contractive(new_obs_latent=new_obs_latent,
			obs_latent=obs_latent)
			logits_aug=self.encoder_logits_contra.compute_encoder_logits_contractive(new_obs_latent=new_obs_aug_latent,
			obs_latent=obs_aug_latent)
			labels = torch.arange(logits.shape[0]).long().cuda()
			encoder_loss = self.cross_entropy_loss(logits, labels) + self.cross_entropy_loss(logits_aug, labels)
			##### jacobian general new loss
		else:

			##### jacobina general new loss
			encoder_loss=F.mse_loss(new_obs_latent,obs_latent) + F.mse_loss(new_obs_aug_latent,obs_aug_latent)
			##### jacobian general new loss

		wandb.log({"encoder_loss":encoder_loss},step=step)

		self.encoder_optimizer.zero_grad()
		if self.encoder_loss_type=="contractive":
			self.encoder_logits_W_optimizer.zero_grad()
		encoder_loss.backward()
		self.encoder_optimizer.step()
		if self.encoder_loss_type=="contractive":
			self.encoder_logits_W_optimizer.step()

	# 更改 end

	def soft_update_critic_target(self):
		utils.soft_update_params(
			self.critic.Q1, self.critic_target.Q1, self.critic_tau
		)
		utils.soft_update_params(
			self.critic.Q2, self.critic_target.Q2, self.critic_tau
		)
		utils.soft_update_params(
			self.critic.encoder, self.critic_target.encoder,
			self.encoder_tau
		)


	def update(self, replay_buffer, L, step):
		if self.encoder_type == 'pixel' and self.use_aux:
			obs_jacobian, obs_rad, action, reward, next_obs_jacobian, next_obs_rad, not_done = replay_buffer.sample_rad_jacobian(self.augs_funcs)
		elif self.encoder_type == 'pixel' and (not self.use_aux):
			obs_rad, action, reward, next_obs_rad, not_done = replay_buffer.sample_rad(self.augs_funcs)
		# else:
		# 	obs_rad, action, reward, next_obs_rad, not_done = replay_buffer.sample_proprio()
	
		if step % self.log_interval == 0:
			L.log('train/batch_reward', reward.mean(), step)

		self.update_critic(obs_rad, action, reward, next_obs_rad, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs_rad, L, step)
		
		# 更改 begin
		if self.use_aux and step % self.updata_transition_model_freq == 0:
			#### jacobian general 
			self.update_transition_model(obs_jacobian, next_obs_jacobian, action, step)
			#### jacobian general


		if self.use_aux and step % self.jacobin_update_freq == 0:
			#### jacobian general
			self.update_encoder_jacobin(obs_jacobian, action, step)
			#### jacobian general

		# 更改 end

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()
			utils.soft_update_params(
			self.global_classifier, self.global_target_classifier,
			self.encoder_tau
		)

	def save(self, model_dir, step):
		torch.save(
			self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
		)
		torch.save(
			self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
		)

	# def load(self, model_dir, step):
	# 	self.actor.load_state_dict(
	# 		torch.load('%s/actor_%s.pt' % (model_dir, step))
	# 	)
	# 	self.critic.load_state_dict(
	# 		torch.load('%s/critic_%s.pt' % (model_dir, step))
	# 	)