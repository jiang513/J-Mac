import argparse
import numpy as np


def parse_args():
	parser = argparse.ArgumentParser()

	# correlation
	# parser.add_argument('--correlated_with_colour', default="False", type=str)
	# parser.add_argument('--correlation_probability', default=0.95, type=float) # sum of probabilities on leading diagonal, see paper
	# parser.add_argument('--test_correlation_probability', default=0.05, type=float) # sum of probabilities on leading diagonal, see paper


	# CMID
	# parser.add_argument('--cmid_encoder_lr', default=1e-3, type=float)
	# parser.add_argument('--cmid_discriminator_lr', default=1e-2, type=float)
	# parser.add_argument('--adversarial_loss_coef', default=0.1, type=float)
	# parser.add_argument('--cmid_knn', default=5, type=int)
	# parser.add_argument('--feature_dim', default=56, type=int)
	# parser.add_argument('--cmid', default=False, type=bool)
	# parser.add_argument('--num_conv_layers', default=11, type=int)
	# parser.add_argument('--hidden_depth', default=2, type=int)


	# environment
	parser.add_argument('--domain_name', default='walker')
	parser.add_argument('--task_name', default='walk')
	parser.add_argument('--frame_stack', default=3, type=int)
	parser.add_argument('--action_repeat', default=4, type=int)
	parser.add_argument('--episode_length', default=1000, type=int)
	parser.add_argument('--train_mode', default='train', type=str)
	parser.add_argument('--data_augs', default='crop-rotate-flip', type=str)
	# parser.add_argument('--eval_mode_color_hard', default='color_hard', type=str)
	parser.add_argument('--eval_mode_video_easy', default='video_easy', type=str)
	parser.add_argument('--eval_mode_video_hard', default='video_hard', type=str)
	
	# agent
	parser.add_argument('--algorithm', default='sac', type=str)
	parser.add_argument('--train_steps', default='300k', type=str)
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--init_steps', default=1000, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)

	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	parser.add_argument('--critic_target_update_freq', default=2, type=int)

	# architecture
	parser.add_argument('--num_shared_layers', default=11, type=int)
	parser.add_argument('--num_head_layers', default=0, type=int)
	parser.add_argument('--num_filters', default=32, type=int)
	parser.add_argument('--projection_dim', default=100, type=int)
	parser.add_argument('--latent_dim', default=100, type=int)
	parser.add_argument('--encoder_tau', default=0.05, type=float)
	
	# entropy maximization
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# auxiliary tasks
	parser.add_argument('--quantile', default=0.65, type=float)
	parser.add_argument('--encoder_lr', default=1e-3, type=float)
	parser.add_argument('--encoder_loss_type', default='contractive', type=str)
	parser.add_argument('--transition_loss_type', default='mse', type=str)
	parser.add_argument('--updata_transition_model_freq', default=1, type=int)
	parser.add_argument('--jacobin_update_freq', default=2, type=int)
	parser.add_argument('--transition_model_lr', default=1e-3, type=float)
	parser.add_argument('--use_aux', default = False, action="store_true")
	parser.add_argument('--use_jacobian', default = False, action="store_true")

	# soda
	parser.add_argument('--soda_batch_size', default=256, type=int)
	parser.add_argument('--soda_tau', default=0.005, type=float)

	# svea
	parser.add_argument('--svea_alpha', default=0.5, type=float)
	parser.add_argument('--svea_beta', default=0.5, type=float)

	# eval
	parser.add_argument('--save_freq', default='50k', type=str)
	parser.add_argument('--save_model', default=True, type=bool)
	parser.add_argument('--eval_freq', default='10k', type=str)
	parser.add_argument('--eval_episodes', default=10, type=int)
	parser.add_argument('--distracting_cs_intensity', default=0., type=float)

	# misc
	parser.add_argument('--seed', default=1, type=int)
	parser.add_argument('--log_dir', default='logs', type=str)
	parser.add_argument('--save_video', default=False, action='store_true')
	parser.add_argument('--transition_model_general', default=False, type=bool)
	parser.add_argument('--jacobin_general', default=True, type=bool)
	parser.add_argument('--reflect', default=False, type=bool)
	parser.add_argument('--new_way', default=False, type=bool)
	parser.add_argument('--depart_3_masks', default=True, type=bool)

	

	args = parser.parse_args()

	assert args.algorithm in {'sac', 'rad', 'curl', 'pad', 'soda', 'drq', 'svea', 'cmid'}, f'specified algorithm "{args.algorithm}" is not supported'

	# assert args.eval_mode in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs', 'none'}, f'specified mode "{args.eval_mode}" is not supported'
	assert args.seed is not None, 'must provide seed for experiment'
	assert args.log_dir is not None, 'must provide a log directory for experiment'

	intensities = {0., 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
	assert args.distracting_cs_intensity in intensities, f'distracting_cs has only been implemented for intensities: {intensities}'

	args.train_steps = int(args.train_steps.replace('k', '000'))
	args.save_freq = int(args.save_freq.replace('k', '000'))
	args.eval_freq = int(args.eval_freq.replace('k', '000'))

	# if args.eval_mode == 'none':
	# 	args.eval_mode = None

	if args.algorithm in {'rad', 'curl', 'pad', 'soda'}:
		args.image_size = 100
		args.image_crop_size = 84
	else:
		args.image_size = 84
		args.image_crop_size = 84
	
	return args
