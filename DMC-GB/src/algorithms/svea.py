import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.modules import DeterministicTransitionModel, \
encoder_logits_contractive, compute_mask_new
from algorithms.sac import SAC
import wandb
from torch.autograd.functional import jacobian, hessian


class SVEA(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		self.encoder_loss_type = args.encoder_loss_type
		self.transition_loss_type = args.transition_loss_type
		self.jacobin_update_freq = args.jacobin_update_freq
		self.use_aux = args.use_aux
		self.use_jacobian = args.use_jacobian
		self.quantile = args.quantile
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		# self.quantile = args.quantile
		self.projection_dim = args.projection_dim
		self.updata_transition_model_freq = args.updata_transition_model_freq

		self.global_classifier = nn.Sequential(
			nn.Linear(self.critic.encoder.out_dim, args.latent_dim), nn.ReLU(),
			nn.Linear(args.latent_dim, self.critic.encoder.out_dim)).cuda()
		self.global_target_classifier = nn.Sequential(
			nn.Linear(self.critic.encoder.out_dim, args.latent_dim), nn.ReLU(),
			nn.Linear(args.latent_dim, self.critic.encoder.out_dim)).cuda()
		self.global_final_classifier = nn.Sequential(
			nn.Linear(self.critic.encoder.out_dim, args.latent_dim), nn.ReLU(),
			nn.Linear(args.latent_dim, self.critic.encoder.out_dim)).cuda()


		if self.encoder_loss_type=="contractive":
			self.encoder_logits_contra=encoder_logits_contractive(critic=self.critic).cuda()

		if self.transition_loss_type=="contractive":
			self.transition_logits_contra=encoder_logits_contractive(critic=self.critic).cuda()
			

		self.transition_model=DeterministicTransitionModel(
			self.critic.encoder.out_dim,self.critic.encoder.out_dim,action_shape,1024).cuda()

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
	

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if self.svea_alpha == self.svea_beta:
			obs = utils.cat(obs, augmentations.random_overlay(obs.clone()))
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_aug = augmentations.random_overlay(obs.clone())
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
		wandb.log({"critic_loss":critic_loss},step=step)
			
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

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
		
		J=self.batch_jacobian(self.jacobian_func,obs,action)
		partial_grad_all=J[0].permute(1,0,2,3,4)
		partial_grad_all = torch.sum(torch.abs(partial_grad_all), dim=1)
		masks = compute_mask_new(partial_grad_all, self.quantile)
		new_obs=obs * masks
		with torch.no_grad():
			new_obs_latent=self.critic_target.encoder(new_obs)

		obs_latent=self.critic.encoder(obs)
		obs_aug = augmentations.random_overlay(obs.clone())
		new_obs_aug = obs_aug * masks
		with torch.no_grad():
			new_obs_aug_latent = self.critic_target.encoder(new_obs_aug)
		obs_aug_latent=self.critic.encoder(obs_aug)

		if self.encoder_loss_type=="contractive":
			
			logits=self.encoder_logits_contra.compute_encoder_logits_contractive(new_obs_latent=new_obs_latent,
			obs_latent=obs_latent)
			logits_aug=self.encoder_logits_contra.compute_encoder_logits_contractive(new_obs_latent=new_obs_aug_latent,
			obs_latent=obs_aug_latent)
			labels = torch.arange(logits.shape[0]).long().cuda()
			encoder_loss = self.cross_entropy_loss(logits, labels) + self.cross_entropy_loss(logits_aug, labels)
			
		else:
			encoder_loss=F.mse_loss(new_obs_latent,obs_latent) + F.mse_loss(new_obs_aug_latent,obs_aug_latent)
			
		wandb.log({"encoder_loss":encoder_loss},step=step)

		self.encoder_optimizer.zero_grad()
		if self.encoder_loss_type=="contractive":
			self.encoder_logits_W_optimizer.zero_grad()
		encoder_loss.backward()
		self.encoder_optimizer.step()
		if self.encoder_loss_type=="contractive":
			self.encoder_logits_W_optimizer.step()

	def update(self, replay_buffer, L, step):
		if self.use_aux:
			obs, action, reward, next_obs, not_done = replay_buffer.sample_svea_jacobin()
		else:
			obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if self.use_aux and step % self.updata_transition_model_freq == 0:		 
			self.update_transition_model(obs, next_obs, action, step)
			

		if self.use_aux and self.use_jacobian and step % self.jacobin_update_freq == 0:
			self.update_encoder_jacobin(obs, action, step)
			
		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()
			utils.soft_update_params(
			self.global_classifier, self.global_target_classifier,
			self.encoder_tau
		)
