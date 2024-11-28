import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter
from typing import Optional
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

import sys
import os

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 5
REWARD_STEPS = 10

#------------Import and Configure Env --------

env = gym.make("CartPole-v1")
#----------------------------------------------

class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def smooth(old: Optional[float], val: float, alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--name", required=True, help="Name of the run")

    parser.add_argument("-r", "--stop_reward",type=int, help="Stop mean renward",default=100_000_000)
    parser.add_argument("-s", "--stop_steps",type=int, help="Stop step renward",default=100_000_000)

    parser.add_argument("-b", "--batch_size",type=int, help="Batch Size",default=BATCH_SIZE)
    parser.add_argument("-rs", "--renward_steps",type=int, help="Rewards Steps",default=REWARD_STEPS)

    def list_of_ints(arg): return list(map(int, arg.split(',')))

    parser.add_argument("-R","--safe_net_reward",type=list_of_ints ,help="Renward to safe net")
    parser.add_argument("-S","--safe_net_steps",type=list_of_ints ,help="Steps to safe net")

    args = parser.parse_args()
    print("")
    print("-----------------RESUME-------------")
    print(" - BATCH_SIZE : " + str(args.batch_size)  )
    print(" - RENWARD_STEPS : " + str(args.renward_steps) )
    print(" - GAMMA ( descount factor ) : " + str(GAMMA) )
    print(" - ALFA ( learning rate ) : " + str(LEARNING_RATE) )
    print(" - BETTA ( entropy factor ) : " + str(ENTROPY_BETA) )
    print("---------------------------------------")
    print("")

    BATCH_SIZE = args.batch_size
    RENWARD_STEPS = args.renward_steps

    writer = SummaryWriter(comment="REINFORCE_")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=RENWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0
    bs_smoothed = entropy = l_entropy = l_policy = l_total = None

    batch_states, batch_actions, batch_scales = [], [], []

    for step_idx, exp in enumerate(exp_source):
        reward_sum += exp.reward
        baseline = reward_sum / (step_idx + 1)
        writer.add_scalar("baseline", baseline, step_idx)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline) # <--- escalares

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))

            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)

            # safe nets
            if not (args.safe_net_reward == None) :
              for rs in args.safe_net_reward :
                if mean_rewards >= rs :
                    print("Safe Net at Renward "+str(rs) )
                    args.safe_net_reward.remove(rs)
                    torch.save(net.state_dict(),"./"+args.name+"REINFORCE_Renward_"+str(rs))


            if not (args.safe_net_steps == None) :
              for s in args.safe_net_steps :
                if step_idx >= s :
                    print("Safe Net at Step "+str(s) )
                    args.safe_net_steps.remove(s)
                    torch.save(net.state_dict(),"./"+args.name+"REINFORCE_Step_"+str(s))

            # stop train
            if mean_rewards >= args.stop_reward or step_idx >= args.stop_steps :
                print(f"End of the Train!")
                break


        if len(batch_states) < BATCH_SIZE:
            continue

        #print("update")
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)

        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        loss_v = loss_policy_v + entropy_loss_v

        loss_v.backward()
        optimizer.step()

        # calc KL-div
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        writer.add_scalar("kl", kl_div_v.item(), step_idx)

        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1

        bs_smoothed = smooth(bs_smoothed, np.mean(batch_scales))
        entropy = smooth(entropy, entropy_v.item())
        l_entropy = smooth(l_entropy, entropy_loss_v.item())
        l_policy = smooth(l_policy, loss_policy_v.item())
        l_total = smooth(l_total, loss_v.item())

        writer.add_scalar("baseline", baseline, step_idx)
        writer.add_scalar("entropy", entropy, step_idx)
        writer.add_scalar("loss_entropy", l_entropy, step_idx)
        writer.add_scalar("loss_policy", l_policy, step_idx)
        writer.add_scalar("loss_total", l_total, step_idx)
        writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
        writer.add_scalar("grad_max", grad_max, step_idx)
        writer.add_scalar("batch_scales", bs_smoothed, step_idx)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()
