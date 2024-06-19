import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
import wandb
import copy


def evaluate(env, agent, video, num_episodes, L, step, test_color_hard=False,test_video_easy=False,test_video_hard=False):
	episode_rewards = []
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			obs=np.array(obs)
			if (args.algorithm == "rad") and 'crop' in args.data_augs:
				obs = utils.center_crop_image(obs,args.image_crop_size)
			if (args.algorithm == "rad") and 'translate' in args.data_augs:
				# first crop the center with pre_image_size
				obs = utils.center_crop_image(obs, args.image_size)
				# then translate cropped to center
				obs = utils.center_translate(obs, args.image_crop_size)
			with utils.eval_mode(agent):
				action = agent.select_action(obs)
			obs, reward, done, _ = env.step(action)
			video.record(env)
			episode_reward += reward

		if L is not None:
			if test_color_hard:
				_test_env = '_test_env_color_hard'
			elif test_video_easy:
				_test_env = '_test_env_video_easy'
			elif test_video_hard:
				_test_env = '_test_env_video_hard'
			else:
				_test_env = ''
			video.save(f'{step}{_test_env}.mp4')
			L.log(f'eval/episode_reward{_test_env}', episode_reward, step)    #存的是平均值
		episode_rewards.append(episode_reward)
	wandb.log({f"eval_episode_reward_average{_test_env}":np.mean(episode_rewards)},step=step)
	
	return np.mean(episode_rewards)


def main(args):
	# Set seed
	argsDict = args.__dict__
	# args.seed=np.random.randint(1000,10000)
	print("seed is:",args.seed)
	print("use_aux:",args.use_aux)
	print("use_jacobian:",args.use_jacobian)
	print("cmid:", args.cmid)
	utils.set_seed_everywhere(args.seed)
	wandb.init(project="jacobian-overlay-open-test", entity="jiangyi",name=argsDict["log_dir"],config=argsDict)

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='train'
	)
	# test_env_color_hard = make_env(
	# 	domain_name=args.domain_name,
	# 	task_name=args.task_name,
	# 	seed=args.seed+42,
	# 	episode_length=args.episode_length,
	# 	action_repeat=args.action_repeat,
	# 	image_size=args.image_size,
	# 	mode=args.eval_mode_color_hard
	# )
	test_env_video_easy = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode=args.eval_mode_video_easy
	)
	test_env_video_hard = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode=args.eval_mode_video_hard
	)

	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, str(args.seed))
	print('Working directory:', work_dir)
	assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
	# utils.write_info(args, os.path.join(work_dir, 'info.log'))
	args.action_range_low=float(env.action_space.low.min())
	args.action_range_high=float(env.action_space.high.max())

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	replay_buffer = utils.ReplayBuffer(
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		capacity=args.train_steps,
		batch_size=args.batch_size,
		cmid = args.cmid
	)
	cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
	print('Observations:', env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)

	start_step, episode, episode_reward, done = 0, 0, 0, True
	L = Logger(work_dir)
	start_time = time.time()
	for step in range(start_step, args.train_steps+1):
		if done:
			if step > start_step:
				L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				L.dump(step)

			# Evaluate agent periodically
			if step % args.eval_freq == 0 :
				print('Evaluating:', work_dir)
				L.log('eval/episode', episode, step)
				# evaluate(env, agent, video, args.eval_episodes, L, step)
				# evaluate(test_env_color_hard, agent, video, args.eval_episodes, L, step, test_color_hard=True)
				evaluate(test_env_video_easy, agent, video, args.eval_episodes, L, step, test_video_easy=True)
				evaluate(test_env_video_hard, agent, video, args.eval_episodes, L, step, test_video_hard=True)
				L.dump(step)

			# Save agent periodically
			if step > start_step and args.save_model and step % args.save_freq == 0:
				torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

			L.log('train/episode_reward', episode_reward, step)
			wandb.log({"train_episode_reward": episode_reward}, step=step)

			obs = env.reset()

			prev_obs = copy.deepcopy(obs)

			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1

			L.log('train/episode', episode, step)

		# Sample action for data collection
		if step < args.init_steps:
			action = env.action_space.sample()
		else:
			with utils.eval_mode(agent):
				if args.algorithm == "rad":
					obs_temp = np.array(obs).copy()
					if obs_temp.shape[-1] != args.image_crop_size:
						obs_temp = utils.center_crop_image(obs_temp, args.image_crop_size)
					action = agent.sample_action(obs_temp)
				else:
					action = agent.sample_action(obs)

		# Run training update
		if step >= args.init_steps:
			num_updates = args.init_steps if step == args.init_steps else 1
			for _ in range(num_updates):
				agent.update(replay_buffer, L, step)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		replay_buffer.add(obs, action, reward, next_obs, done_bool, prev_obs)
		episode_reward += reward

		prev_obs = obs

		obs = next_obs

		episode_step += 1

	print('Completed training for', work_dir)


if __name__ == '__main__':
	args = parse_args()
	main(args)
