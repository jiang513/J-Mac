import numpy as np
import torch
import torch.nn as nn
from env.wrappers import make_env
import argparse
import utils
import wandb
import matplotlib.pyplot as plt
import os
from algorithms.factory import make_agent
from arguments import parse_args
from algorithms.modules import compute_mask_new


# def load(self, model_dir, step):
#         self.actor.load_state_dict(
#             torch.load('%s/actor_%s.pt' % (model_dir, step))
#         )
#         self.critic.load_state_dict(
#             torch.load('%s/critic_%s.pt' % (model_dir, step))
#         )
#         if self.transition_model is not None:
#             self.transition_model.load_state_dict(
#                 torch.load('%s/transition_model_%s.pt' % (model_dir, step))
#             )


def main(args):
    argsDict = args.__dict__
    utils.set_seed_everywhere(args.seed)
    # wandb.init(project="jacobin", entity="jiangyi",name=argsDict["domain_name"],config=argsDict)
    device="cuda"
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.train_mode
    )
    # test_env_color_hard = make_env(
    #     domain_name=args.domain_name,
    #     task_name=args.task_name,
    #     seed=args.seed+42,
    #     episode_length=args.episode_length,
    #     action_repeat=args.action_repeat,
    #     image_size=args.image_size,
    #     mode=args.eval_mode_color_hard
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

    cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
    print('Observations:', env.observation_space.shape)
    print('Cropped observations:', cropped_obs_shape)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=args
    )
    # rous = [0.75, 0.7, 0.6]
    rous = [0.65]
    for rou in rous:
        base_file = "/aiarena/nas/svea-jacobian/svea-overaly-jacobian-freq-2-{}-max-jacobian-general-trans-mse-pro-pre-freq-1-1e-3-new-loss/walker_walk/svea/".format(rou)
        for dirs in os.listdir(base_file):
            print(dirs)
            for j in dirs:
                file = base_file + j
                print(file)
                for i in range(50000,310000,50000): 
                    try:
                        os.makedirs(r"/aiarena/nas/svea-jacobin/visulize_images/{}/{}_step".format(file+"_video_hard_computer_v0",i))
                    except OSError:
                        pass
                    agent=torch.load(file + "/model/{}.pt".format(i))
                    obs=test_env_video_hard.reset()
                    done=0  
                    for k in range(50):
                        if done==0:
                            obs=np.array(obs)
                            action=agent.select_action(obs / 255.)
                            next_obs, reward, done, _ = test_env_video_hard.step(action)
                            obs = agent._obs_to_input(obs)
                            # print("obs:",obs)
                            # plt.imsave(r"/data/jiangy/pytorch_jacobin_online/visulize_images/{}_step/{}_1.png".format(i,j),
                            # obs[0].permute(1,2,0)[:,:,0:3].cpu().numpy()/255.)
                            # plt.imsave(r"/data/jiangy/pytorch_jacobin_online/visulize_images/{}_step/{}_2.png".format(i,j),
                            # obs[0].permute(1,2,0)[:,:,0:3].cpu().numpy()/255.)
                            # plt.imsave(r"/data/jiangy/pytorch_jacobin_online/visulize_images/{}_step/{}_3.png".format(i,j),
                            # obs[0].permute(1,2,0)[:,:,0:3].cpu().numpy()/255.)
                            # print(obs.shape)
                            # print(type(obs[0]))
                            action=torch.FloatTensor(action).cuda()
                            action=action.unsqueeze(0)
                            # J=agent.batch_jacobian(agent.jacobian_func,obs,action)
                            # # print("J[0].shape:",J[0].shape)
                            # partial_grad_all=J[0].permute(1,0,2,3,4).flatten(1,2)
                            # partial_grad_all = torch.sum(torch.abs(partial_grad_all), dim=1)
                            # # partial_grad_all = torch.sum(partial_grad_all, dim=1)  # (B,84,84)
                            # threshold = torch.mean(partial_grad_all.view(partial_grad_all.shape[0],-1),dim=1)
                            # threshold=threshold.reshape(partial_grad_all.shape[0],1,1).repeat(1,84,84)
                            # chose_region=torch.where(partial_grad_all>threshold,1.,0.)[:,None,:,:].repeat(1,9,1,1)
                            J=agent.batch_jacobian(agent.jacobian_func, obs / 255.,action)
                            partial_grad_all=J[0].permute(1,0,2,3,4)
                            partial_grad_all = torch.sum(torch.abs(partial_grad_all), dim=1)
                            masks = compute_mask_new(partial_grad_all)
                            # print(masks)
                            new_obs=obs * masks
                            # new_obs_latent=agent.critic_target.encoder(new_obs)
                            # obs_latent=agent.critic.encoder(obs)
                            # # print("obs_shape:",obs.shape)
                            # new_obs=torch.mul(obs,chose_region)



                            plt.imsave(r"/aiarena/nas/svea-jacobin/visulize_images/{}/{}_step/{}.png".format(file+"_video_hard_computer_v0",i,k),
                            new_obs[0].permute(1,2,0)[:,:,0:3].cpu().numpy()/255.)
                            # print(masks.shape)
                            # mask = torch.ones((1,9,84,84)).cuda()
                            # mask = mask * masks
                            # plt.imsave(r"/aiarena/nas/svea-jacobin/visulize_images/{}_step/{}.png".format(i,j),
                            # mask[0].permute(1,2,0)[:,:,0:3].cpu().numpy())



                            # print("new_obs_shape:",new_obs.shape)
                            # plt.imsave(r"/data/jiangy/pytorch_jacobin_online/visulize_images/{}_step/{}.png".format(i,j),
                            # new_obs.cpu())
                            # print("obs:",obs)
                            obs=next_obs
                        else: 
                            break

            
if __name__ == '__main__':
    args = parse_args()
    main(args)
