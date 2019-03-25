#-------------------------------------
# Project: Meta Value Network
# Date: 2017.5.25
# All Rights Reserved
#-------------------------------------

import time
from collections import deque
from modules import *
from algos import *
from tensorboardX import SummaryWriter
import os
import config
from rlkit.envs.half_cheetah_dir import HalfCheetahDirEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch import pytorch_util as ptu
import gym

class BaseModel():
    def __init__(self, env, algo, single=False):
        self.env = env
        self.task_ids = (0,) if single else env.get_all_task_idx()
        self.single = single
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.algo = algo['cls'](self.state_dim, self.action_dim, **algo['params'])
        max_size = 100000
        self.max_buffer_size = max_size
        self.replay_buffer = MultiTaskReplayBuffer(max_replay_buffer_size=max_size, env=env, tasks=self.task_ids)
        self.max_path_length = 200
        self.batch_size = 256

    def reset_cur_task(self, idx):
        # idx = np.random.randint(len(self.task_ids))
        self.env.reset_task(idx)

    def defualt_setup(self, memo=None):
        expname = os.path.join(logdir, time.strftime('exp_%y%m%d_%H%M%S', time.localtime()))
        os.makedirs(expname, exist_ok=True)
        self.save_meta(expname, config.memo if memo is None else memo)
        self.writer = SummaryWriter(log_dir=expname)
        self.expname = expname
        ptu.set_gpu_mode(True)


    def collect_data(self, task_id, sample_nums):
        # print(f'collecting {sample_nums} steps.')
        state = None
        current_cnt = 0
        returns = list()
        ret = 0
        for i in range(sample_nums):
            if state is None: state = self.task.reset()
            action, logp = self.algo.get_action(ptu.from_numpy(np.expand_dims(state, 0)))  # return cpu version
            action = ptu.get_numpy(action)[0]
            next_state, reward, done = single_step_rollout(action, self.task)
            reward *= reward_scale
            ret += reward
            self.replay_buffer.add_sample(task=task_id, # single task
                                          observation=state,
                                          action=action,
                                          reward=reward,
                                          terminal=done,
                                          next_observation=next_state
                                          )
            current_cnt += 1
            if done or current_cnt==self.max_path_length:
                self.replay_buffer.terminate_episode(self.task_ids[0])
                state = None
                current_cnt = 0
                returns.append(ret)
                ret = 0
            else: state = next_state
        return np.mean(returns)

    def get_batch(self, idx):
        batch = self.replay_buffer.random_batch(idx, self.batch_size)

        o = batch['observations']
        a = batch['actions']
        r = batch['rewards']
        no = batch['next_observations']
        t = batch['terminals']

        o,a,no = [ptu.from_numpy(x) for x in [o,a,no]]

        return o,a,r,no,t

    def roll_out(self, state, task, sample_nums, to_cuda=True):
        """
        sample a sequence of time steps
        :param actor_network:
        :param task:
        :param sample_nums:
        :return:
        """
        if state is None:
            state = task.reset()
        states = []
        actions = []
        rewards = []
        is_done = False
        # state = task.state  # ?
        actions_logp = list()
        result = 0
        for j in range(sample_nums):
            states.append(state)

            action, logp = self.algo.get_action(ptu.from_numpy(np.expand_dims(state, 0))) # return cpu version
            action = ptu.get_numpy(action)[0]
            next_state, reward, done = single_step_rollout(action, task)
            # one_hot_action = [int(k == action) for k in range(self.action_dim)]
            actions_logp.append(logp)
            # fix_reward = -10 if done else 1  # this will cause total reward to be smaller than 0
            actions.append(action)
            rewards.append(reward)
            final_state = next_state
            state = next_state

            if done:
                is_done = True
                task.reset()
                # print("result:",result)
                break
        states.append(final_state)
        s, a, r = torch.Tensor(states), torch.Tensor(actions), reward_scale*np.asarray(rewards)
        if to_cuda: s, a = s.cuda(), a.cuda()

        dones = np.zeros((len(actions),1))
        if is_done:
            dones[-1,0] = 1

        dones = torch.Tensor(dones).cuda()

        logp = torch.cat(actions_logp, dim=0)
        if to_cuda: logp = logp.cuda()
        return s, a, r, dones, logp, final_state

    @staticmethod
    def load_model(net, model_file_path):
        if os.path.exists(model_file_path):
            net.load_state_dict(torch.load(model_file_path))
            print(f'{model_file_path} loaded.')

    @staticmethod
    def save_model(net, model_file_path):
        torch.save(net.state_dict(), model_file_path)

    @staticmethod
    def save_meta(save_dir, memo):
        with open(os.path.join(save_dir, 'meta.txt'), 'w') as f:
            for k in dir(config):
                if k.startswith('_'): continue
                f.write(f'{k}={getattr(config, k)}\n')

    @staticmethod
    def step_optimizer(optim, network, max_norm=0.5):
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm)
        optim.step()
        optim.zero_grad()

    def step_optimizers(self, model_names):
        for n in model_names:
            self.step_optimizer(self.model_opt[n], self.model_dict[n])

class Multtask(BaseModel):
    def __init__(self, env, algo, meta_batch=4, val_batch=4, single=True):
        super(Multtask, self).__init__(env, algo, single)
        self.meta_batch = meta_batch
        self.val_batch = val_batch

    def setup_modules(self):
        self.algo.setup()

        print(f'tasks totaling {len(self.task_ids)}')
        self.defualt_setup(memo=config.memo)
        self.task = self.env

    def train_policy(self):
        print('Train policy.')
        loss_buffer = deque(maxlen=actor_report_freq)

        val_cnt = 0
        horizon = HORIZON  # to continuously increase the hardship of the trajectories.
        gamma = 0.99#pow(0.05, 1. / horizon)
        double_check = 0
        done = True
        s = None
        # collect some data
        for idx in self.task_ids:
            self.reset_cur_task(idx)
            self.collect_data(idx, self.max_path_length*20)
        for step in range(STEP):
            # collect some data
            meta_train_idx = np.random.choice(self.task_ids, min(len(self.task_ids), self.meta_batch))
            s,a,r,ns,dones = [],[],[],[],[]
            r_tasks = list()
            for idx in meta_train_idx:
                self.reset_cur_task(idx)
                if step % self.max_path_length == 0:
                    rewards = self.collect_data(idx, self.max_path_length)
                    r_tasks.append(rewards)

                ret = self.get_batch(idx)
                for x,y in zip([s,a,r,ns,dones],ret):
                    x.append(y)
            states,actions,rewards,nstates,dones = torch.stack(s),torch.stack(actions),np.stack(rewards),torch.cat(ns),np.stack(done)
            # train for several steps
            self.algo.train(states, actions, rewards, nstates, dones, gamma, self.writer, step)

            avg_ret = np.mean(r_tasks) / reward_scale
            self.writer.add_scalar('actor/avg_return', avg_ret, step)

            if (step + 1) % policy_task_resample_freq == 0:
                self.val_and_save(horizon, val_cnt, step)
                val_cnt += 1

    def val_and_save(self, horizon, val_cnt, step):
        print('=' * 25 + ' validation ' + '=' * 25)
        results = list()
        val_idx = np.random.choice(self.task_ids, min(len(self.task_ids), self.val_batch))
        for idx in val_idx:
            self.reset_cur_task(idx)
            for test_epi in range(10):  # test for 10 epochs and takes the average
                states, actions, rewards, is_done, log_softmax_actions, fin = self.roll_out(None, self.task, 200)
                results.append(np.sum(rewards))

        val_res = np.mean(results)/reward_scale
        self.writer.add_scalar('actor/val_return', val_res, val_cnt)
        print(f'average return {val_res} for 10 tests.')
        print('=' * 25 + ' validation ' + '=' * 25)

    def deploy(self):
        self.setup_modules()
        self.train_policy()

class Singletask(BaseModel):
    def __init__(self, env, algo, single=True):
        super(Singletask, self).__init__(env, algo, single)

    def setup_modules(self):
        self.algo.setup()

        # random_choice = np.random.uniform(L_MIN, L_MAX)
        random_choice = np.random.randint(len(self.task_ids))
        print(f'tasks totaling {len(self.task_ids)}, chose {random_choice}')
        self.defualt_setup(memo=config.memo + f'/n{random_choice}')
        # self.task = CartPoleEnv(random_choice)
        if not self.single:
            self.reset_cur_task(random_choice)
        self.task = self.env

    def train_policy(self):
        print('Train policy.')
        loss_buffer = deque(maxlen=actor_report_freq)

        val_cnt = 0
        horizon = HORIZON  # to continuously increase the hardship of the trajectories.
        gamma = 0.99#pow(0.05, 1. / horizon)
        double_check = 0
        done = True
        s = None
        # collect some data
        self.collect_data(self.task_ids[0], self.max_path_length*20)
        for step in range(STEP):
            # collect some data
            if step%self.max_path_length==0:
                total_rewards = self.collect_data(self.task_ids[0], self.max_path_length)
                avg_ret = total_rewards / reward_scale
                self.writer.add_scalar('actor/avg_return', avg_ret, step)
            states, actions, rewards,nstates,dones = self.get_batch(self.task_ids[0])
            # train for several steps
            self.algo.train(states, actions, rewards, nstates, dones, gamma, self.writer, step)

            if (step + 1) % policy_task_resample_freq == 0:
                self.val_and_save(horizon, val_cnt, step)
                val_cnt += 1

    def val_and_save(self, horizon, val_cnt, step):
        # validation

        print('=' * 25 + ' validation ' + '=' * 25)
        results = list()
        for test_epi in range(10):  # test for 10 epochs and takes the average
            states, actions, rewards, is_done, log_softmax_actions, fin = self.roll_out(None, self.task, 200)
            results.append(np.sum(rewards))

        val_res = np.mean(results)/reward_scale
        self.writer.add_scalar('actor/val_return', val_res, val_cnt)
        print(f'average return {val_res} for 10 tests.')
        print('=' * 25 + ' validation ' + '=' * 25)

    def deploy(self):
        self.setup_modules()
        self.train_policy()


def main():
    env = NormalizedBoxEnv(HalfCheetahDirEnv(), reward_scale=0.1)
    env2 = NormalizedActions(gym.make("Pendulum-v0"))
    model = Singletask(env2, algo=dict(cls=SAC,
                                       params=dict(modules=('value_baseline', 'target_v', 'actor_network', 'q1', 'q2'),
                                                  model_lr=[3e-4]*5)))
    model.deploy()

if __name__ == '__main__':
    main()
