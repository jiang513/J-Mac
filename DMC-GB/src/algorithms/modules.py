import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from functools import partial
# from captum.attr import GuidedBackprop
# from functorch.compile import aot_function
# from functorch import jacrev


def _get_out_shape_cuda(in_shape, layers):
	x = torch.randn(*in_shape).cuda().unsqueeze(0)
	return layers(x).squeeze(0).shape


def _get_out_shape(in_shape, layers):
	x = torch.randn(*in_shape).unsqueeze(0)
	return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability"""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
	"""Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
	mu = torch.tanh(mu)
	if pi is not None:
		pi = torch.tanh(pi)
	if log_pi is not None:
		log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
	return mu, pi, log_pi


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
	"""Truncated normal distribution, see https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf"""
	def norm_cdf(x):
		return (1. + math.erf(x / math.sqrt(2.))) / 2.
	with torch.no_grad():
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)
		tensor.uniform_(2 * l - 1, 2 * u - 1)
		tensor.erfinv_()
		tensor.mul_(std * math.sqrt(2.))
		tensor.add_(mean)
		tensor.clamp_(min=a, max=b)
		return tensor


def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers"""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CenterCrop(nn.Module):
	def __init__(self, size):
		super().__init__()
		assert size in {84, 100}, f'unexpected size: {size}'
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		assert x.size(3) == 100, f'unexpected size: {x.size(3)}'
		if self.size == 84:
			p = 8
		return x[:, :, p:-p, p:-p]


class NormalizeImg(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x/255.


class Flatten(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


class RLProjection(nn.Module):
	def __init__(self, in_shape, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.projection = nn.Sequential(
			nn.Linear(in_shape[0], out_dim),
			nn.LayerNorm(out_dim),
			nn.Tanh()
		)
		self.apply(weight_init)
	
	def forward(self, x):
		return self.projection(x)


class SODAMLP(nn.Module):
	def __init__(self, projection_dim, hidden_dim, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.mlp = nn.Sequential(
			nn.Linear(projection_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, out_dim)
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(x)


class SharedCNN(nn.Module):
	def __init__(self, obs_shape, num_layers=11, num_filters=32):
		super().__init__()
		assert len(obs_shape) == 3
		self.num_layers = num_layers
		self.num_filters = num_filters

		self.layers = [CenterCrop(size=84), NormalizeImg(), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		for _ in range(1, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(obs_shape, self.layers)
		self.apply(weight_init)

	def forward(self, x):
		return self.layers(x)


class HeadCNN(nn.Module):
	def __init__(self, in_shape, num_layers=0, num_filters=32):
		super().__init__()
		self.layers = []
		for _ in range(0, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers.append(Flatten())
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(in_shape, self.layers)
		self.apply(weight_init)

	def forward(self, x):
		return self.layers(x)


class Encoder(nn.Module):
	def __init__(self, shared_cnn, head_cnn, projection):
		super().__init__()
		self.shared_cnn = shared_cnn
		self.head_cnn = head_cnn
		self.projection = projection
		self.out_dim = projection.out_dim

	def forward(self, x, detach=False):
		x = self.shared_cnn(x)
		x = self.head_cnn(x)
		if detach:
			x = x.detach()
		return self.projection(x)


class Actor(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max):
		super().__init__()
		self.encoder = encoder
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.mlp = nn.Sequential(
			nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 2 * action_shape[0])
		)
		self.mlp.apply(weight_init)

	def forward(self, x, compute_pi=True, compute_log_pi=True, detach=False):
		x = self.encoder(x, detach)
		mu, log_std = self.mlp(x).chunk(2, dim=-1)
		log_std = torch.tanh(log_std)
		log_std = self.log_std_min + 0.5 * (
			self.log_std_max - self.log_std_min
		) * (log_std + 1)

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
	def __init__(self, obs_dim, action_dim, hidden_dim):
		super().__init__()
		self.trunk = nn.Sequential(
			nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		self.apply(weight_init)

	def forward(self, obs, action):
		assert obs.size(0) == action.size(0)
		return self.trunk(torch.cat([obs, action], dim=1))


class Critic(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.Q1 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)
		self.Q2 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)

	def forward(self, x, action, detach=False):
		x = self.encoder(x, detach)
		return self.Q1(x, action), self.Q2(x, action)


class CURLHead(nn.Module):
	def __init__(self, encoder):
		super().__init__()
		self.encoder = encoder
		self.W = nn.Parameter(torch.rand(encoder.out_dim, encoder.out_dim))

	def compute_logits(self, z_a, z_pos):
		"""
		Uses logits trick for CURL:
		- compute (B,B) matrix z_a (W z_pos.T)
		- positives are all diagonal elements
		- negatives are all other elements
		- to compute loss use multiclass cross entropy with identity matrix for labels
		"""
		Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
		logits = torch.matmul(z_a, Wz)  # (B,B)
		logits = logits - torch.max(logits, 1)[0][:, None]
		return logits


class InverseDynamics(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = nn.Sequential(
			nn.Linear(2*encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, action_shape[0])
		)
		self.apply(weight_init)

	def forward(self, x, x_next):
		h = self.encoder(x)
		h_next = self.encoder(x_next)
		joint_h = torch.cat([h, h_next], dim=1)
		return self.mlp(joint_h)


class SODAPredictor(nn.Module):
	def __init__(self, encoder, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = SODAMLP(
			encoder.out_dim, hidden_dim, encoder.out_dim
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(self.encoder(x))


class encoder_logits_contractive(torch.nn.Module):
	def __init__(self,critic):
		super().__init__()
		self.W=torch.nn.Parameter(torch.rand(critic.encoder.out_dim,critic.encoder.out_dim))
	
	def compute_encoder_logits_contractive(self,new_obs_latent,obs_latent):
		Wz = torch.matmul(self.W, new_obs_latent.T)  # (z_dim,B)
		logits = torch.matmul(obs_latent, Wz)  # (B,B)
		logits = logits - torch.max(logits, 1)[0][:, None]
		return logits


class DeterministicTransitionModel(nn.Module):
	def __init__(self, encoder_feature_dim, output_dim, action_shape, layer_width, act_fn=nn.ELU, out_act=nn.Identity):
		super().__init__()
		self.fc = nn.Linear(encoder_feature_dim + action_shape[0], layer_width)
		self.ln = nn.LayerNorm(layer_width)
		self.fc_mu = nn.Linear(layer_width, output_dim)
		print("Deterministic transition model chosen.")

		##### new model
		#layers = [nn.Linear(encoder_feature_dim + action_shape[0], layer_width[0]), act_fn()]
		#for i in range(len(layer_width)-1):
		#	layers += [nn.Linear(layer_width[i], layer_width[i+1]), act_fn()]

		#layers += [nn.Linear(layer_width[-1], output_dim), out_act()]
		#self.model = nn.Sequential(*layers)
		##### new model
	
	def forward(self, x):
		x = self.fc(x)
		x = self.ln(x)
		x = torch.relu(x)
		
		mu = self.fc_mu(x)
		sigma = None
		return mu, sigma

		##### new model
		#x = self.model(x)
		#return x
		##### new model
	
	def sample_prediction(self, x):
		mu, sigma = self(x)

		return mu

		##### new model
		#x = self(x)
		#return x
		##### new model


# class ModelWrapper(torch.nn.Module):
# 	def __init__(self, model1, model2, action=None):
# 		super(ModelWrapper, self).__init__()
# 		self.model1 = model1
# 		self.model2 = model2
# 		self.action = action

# 	def forward(self, obs):
# 		if self.action is None:
# 			enc = self.model(obs)
# 			output=self.model2.sample_prediction(enc)
# 			return output
# 		enc = self.model1(obs)
# 		output = self.model2.sample_prediction(enc, self.action)
# 		# output = output.sum(dim=1)
# 		return output


# class func_warpper(torch.nn.Module):
# 	def __init__(self, function, obs):
# 		super(func_warpper, self).__init__()
# 		self.obs = obs
# 		self.function = function
	
# 	def forward(self, target):
# 		print("target:",target)
# 		attribution = self.function(self.obs, target).unsqueeze(1)
# 		return attribution


# def func_warpper(function, obs, target):
# 		print("target:",target)
# 		attribution = function(obs, target).unsqueeze(1)
# 		return attribution


# def compute_attribution(model1, model2, obs, action=None):
# 		model = ModelWrapper(model1, model2, action=action)
# 		gbp = GuidedBackprop(model)
# 		attribution = gbp.attribute(obs)
# 		return attribution

class ModelWrapper(torch.nn.Module):
	def __init__(self, model1, model2, action=None):
		super(ModelWrapper, self).__init__()
		self.model1 = model1
		self.model2 = model2
		self.action = action

	def forward(self, obs):
		enc = self.model1(obs)
		output = self.model2.sample_prediction(torch.cat([enc,self.action], dim=1))
		return output



# def batch_jacobian(self, f, enc,action):
# 	f_sum = lambda x,y: torch.sum(f(x,y), axis=0)
# 	return jacobian(f_sum, (enc,action))

# def jacobian_func(self,x,y):
# 	x=self.critic.encoder(x)
# 	x=self.transition_model.sample_prediction(x,y)
# 	return x



# def compute_mask(obs_grad, quantile=0.95):
# 	# obs_grad (B, C, H, W)
# 	mask = []
# 	for i in [0, 3, 6]:
# 		attributions = obs_grad[:, i : i + 3].max(dim=1)[0]
# 		q = torch.quantile(attributions.flatten(1), quantile, 1)
# 		mask.append((attributions >= q[:, None, None]).unsqueeze(1).repeat(1, 3, 1, 1))
# 	return torch.cat(mask, dim=1)


def compute_mask_new(obs_grad, quantile = 0.65):
	# obs_grad (B, C, H, W)
	mask = []
	for i in [0, 3, 6]:
		attributions = obs_grad[:, i : i + 3].max(dim=1)[0]
		# threshold = torch.sum(obs_grad,dim=1)
		# threshold=torch.mean(attributions.flatten(1),dim=1)
		threshold = torch.quantile(attributions.flatten(1), quantile, 1)
		mask.append((attributions >= threshold[:, None, None]).unsqueeze(1).repeat(1, 3, 1, 1))
	return torch.cat(mask, dim=1)


##### speed
def ts_compile(fx_g):
	f = torch.jit.script(fx_g)
	f = torch.jit.freeze(f)
	return f

def ts_compiler(f):
	return aot_function(f, ts_compile, ts_compile)
##### speed

# def batch_jacobian_speed(model, obs):
# 		"""calculate a jacobian tensor along a batch of inputs. returns something of size
# 		`batch_size` x `output_dim` x `input_dim`"""
# 		def _func_sum(obs):
# 			return model(obs).sum(dim=0)
# 		return jacrev(_func_sum)(obs).permute(1,0,2,3,4)
