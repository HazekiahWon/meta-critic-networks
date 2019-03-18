#-------------------------------------
# Project: Meta Value Network
# Date: 2017.5.25
# All Rights Reserved
#-------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
from cartpole import CartPoleEnv

# CartPole Task Parameter
L_MIN = 0.5   # min length
L_MAX = 5    # max length

# Hyper Parameters
TASK_NUMS = 10
STATE_DIM = 4 # cont
ACTION_DIM = 2 # cat
task_nlayer = 3

Z_DIM = 16
actor_dim = 64
task_dim = 64
value_dim = 64
TASK_CONFIG_DIM = 3
EPISODE = 1000
STEP = 500
SAMPLE_NUMS = 30 #5,10,20
TEST_SAMPLE_NUMS = 5


class ActorNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))
        return out

class MetaValueNetwork(nn.Module):
    """
    given state,action,z, output reward
    """
    def __init__(self,input_size,hidden_size,output_size):
        super(MetaValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class DynamicsEmb(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, k, timestep):
        super(DynamicsEmb, self).__init__()
        pad = k//2
        self.conv1 = nn.Conv1d(input_size, hidden_size, k, padding=pad)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, k, padding=pad)
        self.conv3 = nn.Conv1d(hidden_size, output_size, k)
        self.pool = nn.AvgPool1d(timestep-pad*2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.pool(out) # 1,output_size
        return out.squeeze(dim=-1)

class VAE(nn.Module):
    def __init__(self, encoder, decoder, emb_dim, z_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(emb_dim, z_dim)
        self._enc_log_sigma = torch.nn.Linear(emb_dim, z_dim)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)

class TaskConfigNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TaskConfigNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

def roll_out(actor_network,task,sample_nums, z):
    """
    sample a sequence of time steps
    :param actor_network:
    :param task:
    :param sample_nums:
    :return:
    """
    states = []
    actions = []
    rewards = []
    is_done = False
    state = task.state # ?
    actions_logp = list()
    result = 0
    for j in range(sample_nums):
        states.append(state)
        if actor_network is None: action = task.action_space.sample()
        else:
            log_softmax_action = get_action_logp(torch.Tensor([state]).cuda(), z, actor_network, do_grad=True)
            actions_logp.append(log_softmax_action)
            softmax_action = torch.exp(log_softmax_action)
            action = np.random.choice(ACTION_DIM,p=softmax_action.cpu().data.numpy()[0]) # sample from cat distri
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        next_state,reward,done,_ = task.step(action)
        #task.result += reward
        fix_reward = -10 if done else 1
        actions.append(one_hot_action)
        rewards.append(fix_reward)
        final_state = next_state
        state = next_state
        '''
        if task.result >= 250:
            task.reset()
            break
        '''
        if done:
            is_done = True 
            task.reset()
            #print("result:",result)
            break

    if actor_network is None:
        return torch.Tensor(states),torch.Tensor(actions),rewards,is_done,torch.Tensor(final_state)
    else: return torch.Tensor(states),torch.Tensor(actions),rewards,is_done,torch.Tensor(final_state), \
                 torch.cat(actions_logp, dim=0)

def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(list(range(0, len(r)))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def dis_reward(rs, gamma):
    return np.sum(rs*np.cumproduct([1]+[gamma]*(len(rs)-1)))

def seq_reward2go(rs, gamma):
    return [dis_reward(rs[i:], gamma) for i in range(len(rs))]

def get_dyn_embedding(s,a,sp, network):
    """
    return with batch size
    :param s:
    :param a:
    :param sp:
    :param network:
    :return:
    """
    pre_data_samples = torch.cat(  # s,a,s
        (s,a,sp), dim=1).transpose(0,1).unsqueeze(0)  # 1,10,t-1
    # sas - z
    return network(Variable(pre_data_samples).cuda())

def get_predicted_rewards(s,a,z, network, do_grad=False):
    value_inputs = torch.cat(
        (s,a,z), dim=-1)  # t,22
    if do_grad: value_inputs = Variable(value_inputs)
    return network(value_inputs)  # for vae loss

def get_action_logp(s,z, network, do_grad=False):
    actor_inputs = torch.cat((s,z), dim=1)  # sz - a
    if do_grad: actor_inputs = Variable(actor_inputs)
    return network(actor_inputs)  # why input the state sequence

def opt_loss(optim, network, max_norm=0.5):
    torch.nn.utils.clip_grad_norm(network.parameters(), max_norm)
    optim.step()

def one_hot_action_sample(task):
    a = np.zeros((ACTION_DIM,))
    sam = task.action_space.sample()
    a[sam] = 1
    return a

def main():

    # task_embedding + dynamics
    meta_value_network = MetaValueNetwork(input_size = STATE_DIM+ACTION_DIM+Z_DIM,hidden_size = value_dim,output_size = 1)
    task_config_network = DynamicsEmb(input_size=STATE_DIM*2+ACTION_DIM,
                                      hidden_size=task_dim,
                                      output_size=Z_DIM,
                                      k=5,
                                      timestep=SAMPLE_NUMS)
    meta_value_network.cuda()
    task_config_network.cuda()
    # load params
    # if os.path.exists("meta_value_network_cartpole.pkl"):
    #     meta_value_network.load_state_dict(torch.load("meta_value_network_cartpole.pkl"))
    #     print("load meta value network success")
    # if os.path.exists("task_config_network_cartpole.pkl"):
    #     task_config_network.load_state_dict(torch.load("task_config_network_cartpole.pkl"))
    #     print("load task config network success")
    # optim instances for two nets
    meta_value_network_optim = torch.optim.Adam(meta_value_network.parameters(),lr=0.001)
    task_config_network_optim = torch.optim.Adam(task_config_network.parameters(),lr=0.001) # not used?

    # multiple tasks, can randomizing environments by sending randomized args, initialize all tasks
    task_list = [CartPoleEnv(np.random.uniform(L_MIN,L_MAX)) for task in range(TASK_NUMS)] # a list of env
    [task.reset() for task in task_list]

    task_lengths = [task.length for task in task_list]
    print(("task length:",task_lengths))

    for episode in range(EPISODE):
        # ----------------- Training ------------------

        if (episode+1) % 10 ==0 : # sample another batch of tasks
            # renew the tasks
            task_list = [CartPoleEnv(np.random.uniform(L_MIN,L_MAX)) for task in range(TASK_NUMS)]
            task_lengths = [task.length for task in task_list]
            print(("task length:",task_lengths))
            [task.reset() for task in task_list]

        actor_network = ActorNetwork(STATE_DIM+Z_DIM,actor_dim,ACTION_DIM)
        actor_network.cuda()
        actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 0.01)

        # a list of num_task items
        pre_states = []
        pre_actions = []
        pre_rewards = []
        # can use multiprocessing
        for i in range(TASK_NUMS):
            # currently we sample from the action space, because the actor requires dynamics embedding which we for now does not know
            states,actions,rewards,_,_ = roll_out(None,task_list[i],SAMPLE_NUMS, None)
            pre_states.append(states)
            pre_actions.append(actions)
            pre_rewards.append(rewards)


        for step in range(STEP):
            actor_loss = list()
            value_loss = list()
            for i in range(TASK_NUMS):

                task_config = get_dyn_embedding(pre_states[i][:-1], pre_actions[i][:-1], pre_states[i][1:],
                                                task_config_network) # 1,z_dim

                pred_rs = get_predicted_rewards(torch.Tensor(pre_states[i]).cuda(), torch.Tensor(pre_actions[i]).cuda(),
                                                task_config.repeat(pre_states[i].shape[0], 1),
                                                meta_value_network, do_grad=True)
                # ===> should be resampled
                states,actions,rewards,is_done,final_state, log_softmax_actions = roll_out(actor_network,task_list[i],SAMPLE_NUMS, task_config)
                # logp_action 30,2

                n_sample = 2
                gamma = .99
                # random_actions = [one_hot_action_sample(task_list[i]) for _ in range(n_sample)]
                st = np.concatenate([states]*n_sample,axis=0) # t*n,stat_dim
                ac = np.asarray([[0,1]]*len(states)+[[1,0]]*len(states)) # t*n,2
                # t*n,1
                baseline_ = get_predicted_rewards(torch.from_numpy(st).cuda(),
                                                 torch.Tensor(ac).cuda(), # must use Tensor to transform to float
                                                 task_config.repeat(len(states)*n_sample, 1), meta_value_network, do_grad=False)
                baseline = baseline_.detach().view(-1, n_sample)
                baseline = torch.mean(baseline, dim=-1) # t
                # calculate qs
                # qs = Variable(torch.Tensor(discount_reward(rewards,0.99,final_r))).cuda()
                ret = torch.Tensor(seq_reward2go(rewards, gamma)).cuda()
                # baseline = torch.mean(baseline).detach()
                advantages = ret-baseline.cuda()
                # why is there multiplied by actions_var
                actions_var = Variable(actions).cuda()
                # take the mean over time steps
                actions_logp = torch.sum(log_softmax_actions*actions_var,dim=1) # n,1
                actor_loss.append(- torch.sum(actions_logp*advantages, dim=0)) #+ entropy #+ actor_criterion(actor_y_samples,target_y)

                target_values = torch.Tensor(seq_reward2go(pre_rewards[i], gamma)).view(-1,1).cuda() #[dis_reward(rewards[i:], gamma) for i in range(len(rewards))]
                # values = meta_value_network(torch.cat((states_var,task_configs),1))
                criterion = nn.MSELoss()
                value_loss.append(criterion(pred_rs,target_values))

            ac_loss = torch.mean(torch.stack(actor_loss))
            tmp = torch.stack(value_loss)
            rec_loss = torch.mean(tmp)
            actor_network_optim.zero_grad()
            meta_value_network_optim.zero_grad()
            task_config_network_optim.zero_grad()
            ac_loss.backward()
            rec_loss.backward()
            opt_loss(actor_network_optim, actor_network)
            opt_loss(meta_value_network_optim, meta_value_network)
            opt_loss(task_config_network_optim, task_config_network)

            if (step + 1) % 100 == 0:
                for i in range(TASK_NUMS):
                    result = 0
                    test_task = CartPoleEnv(length = task_list[i].length)
                    for test_epi in range(10): # test for 10 epochs and takes the average
                        state = test_task.reset()
                        for test_step in range(200): # rollout for 200 steps
                            softmax_action = torch.exp(actor_network(Variable(torch.from_numpy([state])).cuda()))
                            action = np.argmax(softmax_action.cpu().data.numpy()[0])
                            next_state,reward,done,_ = test_task.step(action)
                            result += reward
                            state = next_state
                            if done:
                                break
                    print(("episode:",episode,"task:",i,"step:",step+1,"test result:",result/10.0))

        
        if (episode+1) % 10 == 0 :
            # Save meta value network
            torch.save(meta_value_network.state_dict(),"meta_value_network_cartpole.pkl")
            torch.save(task_config_network.state_dict(),"task_config_network_cartpole.pkl")
            print(("save networks for episode:",episode))

if __name__ == '__main__':
    main()
