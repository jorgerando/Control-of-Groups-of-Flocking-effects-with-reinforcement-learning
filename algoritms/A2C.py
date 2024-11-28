import gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

import sys
import os

GAMMA = 0.99
LEARNING_RATE = 0.01
ENTROPY_BETA = 0.01
BATCH_SIZE = 10
REWARD_STEPS = 15

CLIP_GRAD = 1000

class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()

        self.in_ = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
        )

        self.policy = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        fx = x.float()
        out = self.in_(fx)
        return self.policy(out), self.value(out)

def unpack_batch(batch, net, device='cpu'):
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= GAMMA ** REWARD_STEPS
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v

def track_rewards(reward_history, reward, step_idx, writer, done_episodes):
    reward_history.append(reward)
    mean_reward = np.mean(reward_history[-100:])  # Promedio móvil de las últimas 100 recompensas
    writer.add_scalar("reward_100", mean_reward, step_idx)
    print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (step_idx, reward, mean_reward, done_episodes))
    return mean_reward

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
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

    device = torch.device("cuda" if args.cuda else "cpu")
    env = gym.make("CartPole-v1")
    envs = [env]
    writer = SummaryWriter(comment="A2C_" + args.name)

    net = Net(envs[0].observation_space.shape[0], envs[0].action_space.n).to(device)
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=args.renward_steps)

    #optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, eps=1e-5)

    batch = []
    reward_history = []
    done_episodes = 0

    with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
        for step_idx, exp in enumerate(exp_source):
            batch.append(exp)

            # Procesar recompensas nuevas
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                done_episodes += 1
                mean_reward = track_rewards(reward_history, new_rewards[0], step_idx, writer, done_episodes)

                # safe nets
                if not (args.safe_net_reward == None) :
                  for rs in args.safe_net_reward :
                    if mean_reward >= rs :
                        print("Safe Net at Renward "+str(rs) )
                        args.safe_net_reward.remove(rs)
                        torch.save(net.state_dict(),"./"+args.name+"A2C_Renward_"+str(rs))


                if not (args.safe_net_steps == None) :
                  for s in args.safe_net_steps :
                    if step_idx >= s :
                        print("Safe Net at Step "+str(s) )
                        args.safe_net_steps.remove(s)
                        torch.save(net.state_dict(),"./"+args.name+"A2C_Step_"+str(s))

                # stop train
                if mean_reward >= args.stop_reward or step_idx >= args.stop_steps :
                    print(f"End of the Train!")
                    break

            if len(batch) < args.batch_size:
                continue

            states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
            batch.clear()

            optimizer.zero_grad()
            logits_v, value_v = net(states_v)
            loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

            log_prob_v = F.log_softmax(logits_v, dim=1)
            adv_v = vals_ref_v - value_v.detach()
            log_prob_actions_v = adv_v * log_prob_v[range(args.batch_size), actions_t]
            loss_policy_v = -log_prob_actions_v.mean()

            prob_v = F.softmax(logits_v, dim=1)
            entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

            # Calcular gradientes de política
            loss_policy_v.backward(retain_graph=True)
            grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                    for p in net.parameters() if p.grad is not None])

            # Aplicar gradientes de entropía y valor
            loss_v = entropy_loss_v + loss_value_v
            loss_v.backward()
            nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            optimizer.step()
            # Obtener pérdida total
            loss_v += loss_policy_v

            tb_tracker.track("advantage", adv_v, step_idx)
            tb_tracker.track("values", value_v, step_idx)
            tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
            tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
            tb_tracker.track("loss_policy", loss_policy_v, step_idx)
            tb_tracker.track("loss_value", loss_value_v, step_idx)
            tb_tracker.track("loss_total", loss_v, step_idx)
            tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
            tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
            tb_tracker.track("grad_var", np.var(grads), step_idx)

    writer.close()
