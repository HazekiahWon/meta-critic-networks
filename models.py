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

class BaseModel():
    def __init__(self, env, algo):
        self.env = env
        self.task_ids = env.get_all_task_idx()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.algo = algo['cls'](self.state_dim, self.action_dim, **algo['params'])

    def reset_cur_task(self, idx):
        # idx = np.random.randint(len(self.task_ids))
        self.env.reset_task(idx)

    def defualt_setup(self, memo=None):


        expname = os.path.join(logdir, time.strftime('exp_%y%m%d_%H%M%S', time.localtime()))
        os.makedirs(expname, exist_ok=True)
        self.save_meta(expname, config.memo if memo is None else memo)
        self.writer = SummaryWriter(log_dir=expname)
        self.expname = expname

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

class Multitask(BaseModel):
    def __init__(self, env, algo, model_lr=(5e-3,5e-3,5e-3,5e-3)):
        modules = ['nstate_decoder','dyn_encoder','value_baseline','actor_network']
        super(Multitask, self).__init__(env, modules, model_lr)
        self.algo = algo

    def setup_modules(self):
        self.modules = ['nstate_decoder','dyn_encoder','value_baseline','actor_network']
        self.nstate_decoder = Trans_with_latent(self.state_dim, self.action_dim)
        self.dyn_encoder = DynEmb(self.state_dim, self.action_dim)
        self.value_baseline = VBase_with_latent(self.state_dim)
        self.actor_network = GaussianActor_with_latent(self.state_dim, self.action_dim)
        self.vae = VAE(self.dyn_encoder, self.nstate_decoder)
        # send to cuda
        self.value_baseline.cuda()
        self.nstate_decoder.cuda()
        self.dyn_encoder.cuda()
        self.actor_network.cuda()
        self.vae.cuda()

        self.defualt_setup()

        self.task = self.env

    def roll_out(self, state, actor_network, task, sample_nums, z, reset=False, to_cuda=True):
        """
        sample a sequence of time steps
        :param actor_network:
        :param task:
        :param sample_nums:
        :return:
        """
        if reset:
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

            if actor_network is not None:
                action, _, logp = actor_network(torch.Tensor([state]).cuda(), z)
                action = action.cpu().data.numpy()[0]
                next_state, reward, done = single_step_rollout(action, task)
                actions_logp.append(logp)
            else:
                action = task.action_space.sample()
                # one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
                next_state, reward, done, _ = task.step(action)

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
        if actor_network is None:
            return s, a, r, is_done
        else:
            logp = torch.cat(actions_logp, dim=0)
            if to_cuda: logp = logp.cuda()
            return s, a, r, is_done, logp

    def train_vae(self):
        print(f'train VAE.')
        loss_buffer = deque(maxlen=vae_report_freq)

        step = 0
        start = time.time()
        while True:
            # actor_loss = list()
            state_loss = list()
            value_loss = list()

            # if step % task_resample_freq == 0:  # sample another batch of tasks
            #     self.task_list = resample_task()

            for i in range(TASK_NUMS):
                self.reset_cur_task(np.random.randint(len(self.task_ids)))
                states, actions, rewards, _ = self.roll_out(None, None, self.task, HORIZON, None, reset=True,
                                                       to_cuda=True)  # cuda
                # states = states[:-1]
                pred_nstate, latent_z = self.vae(states[:-2], actions[:-1])  # cuda
                target_nstate = states[1:-1]
                criterion = nn.MSELoss()
                state_loss.append(criterion(pred_nstate, target_nstate))  # cuda
                # # if train baseline while training the vae.
                # latent_z = latent_z.detach()
                # cur_return_pred = get_predicted_rewards(states, latent_z.repeat(states.size(0), 1), return_baseline) #cuda
                # ########## bellman backup
                # nex_return_pred = cur_return_pred[1:]  # t,1
                # rewards = torch.Tensor(rewards).view(-1, 1).cuda()
                # td_target = rewards + gamma * nex_return_pred  # t
                # ##########################
                # # td_target = torch.Tensor(rewards).cuda()#torch.Tensor(seq_reward2go(rewards, gamma)).cuda()
                # value_loss.append(criterion(td_target, cur_return_pred[:-1]))
                # if i == 0:
                #     writer.add_histogram('VAE/target_return', td_target, step)
                #     writer.add_histogram('VAE/pred_return', cur_return_pred[:-1], step)

            ######## if train baseline while training vae
            # if use_baseline:
            #     bl_loss = torch.mean(torch.stack(value_loss))
            #     overall_loss = rec_loss+bl_loss
            #     writer.add_scalar('VAE/baseline_loss', bl_loss, step)
            # else: overall_loss = rec_loss
            ######## optimize
            rec_loss = torch.mean(torch.stack(state_loss))
            rec_loss.backward()
            if rec_loss.cpu().data.numpy()==np.nan:
                print(state_loss)
                break
            self.writer.add_scalar('VAE/reconstruction_loss', rec_loss, step)
            self.step_optimizers(('dyn_encoder', 'nstate_decoder'))

            if len(loss_buffer) == vae_report_freq: loss_buffer.popleft()
            loss_buffer.append(rec_loss.item())
            m = np.mean(list(loss_buffer))
            # print(f'step {step} takes {time.time() - start} sec, with recloss={rec_loss}.')
            if step % vae_report_freq == 0:
                print(f'step {step} takes {time.time() - start} sec, with recloss={m}.')
                start = time.time()
            if m < vae_thresh:
                break
            else:
                step += 1
        print('Finish VAE.')

    def train_policy(self):
        print('Train policy.')
        loss_buffer = deque(maxlen=actor_report_freq)
        # task_list = resample_task()
        val_cnt = 0
        horizon = HORIZON  # to continuously increase the hardship of the trajectories.
        gamma = pow(0.05, 1. / horizon)
        double_check = 0
        for step in range(STEP):
            actor_loss = list()
            value_loss = list()
            rets = list()

            for i in range(TASK_NUMS):
                # obtain z and experience data to train the actor
                task = self.reset_cur_task(np.random.randint(len(self.task_ids)))
                states, actions, _, _ = self.roll_out(None, None, self.task, horizon, None, reset=True)
                states = states[:-1]  # to remove the final state
                latent_z = get_dyn_embedding(states[:-1], actions[:-1], states[1:], self.dyn_encoder)  # 1,z_dim
                states, actions, rewards, is_done, log_softmax_actions = self.roll_out(None, self.actor_network, self.task, horizon,
                                                                                  latent_z, reset=True)

                cstate_value = self.value_baseline(states, latent_z.repeat(states.size(0), 1))
                sudo_loss, bellman_error, total_rewards, actions_logp, advantages = self.algo.train(states, rewards,
                                                                                               cstate_value, log_softmax_actions,
                                                                                               gamma)
                actor_loss.append(sudo_loss)  # + entropy #+ actor_criterion(actor_y_samples,target_y)
                value_loss.append(bellman_error)
                rets.append(total_rewards)
                if i == 0:
                    self.writer.add_histogram('actor/actions_logp', actions_logp, step)
                    self.writer.add_histogram('actor/advantages', advantages, step)

            ac_loss = torch.mean(torch.stack(actor_loss))
            bl_loss = torch.mean(torch.stack(value_loss))
            self.algo.optimize(ac_loss, bl_loss, self.model_dict, self.model_opt, ('actor_network', 'value_baseline'), self.writer, step)

            avg_ret = np.mean(rets)/reward_scale
            if len(loss_buffer) == actor_report_freq: loss_buffer.popleft()
            loss_buffer.append(avg_ret.item())
            m = np.mean(list(loss_buffer))
            self.writer.add_scalar('actor/avg_return', avg_ret, step)

            if (step + 1) % actor_report_freq == 0:
                print(f'step {step} with avg return {m}.')
                if m > double_horizon_threshold * horizon:
                    double_check += 1
                    if double_check == 3:
                        horizon += HORIZON
                        print(f'step {step} horizon={horizon}')
                        double_check = 0

            if (step + 1) % policy_task_resample_freq == 0:
                self.val_and_save(horizon, val_cnt, step)
                val_cnt += 1

    def val_and_save(self, horizon, val_cnt, step):
        # validation

        print('=' * 25 + ' validation ' + '=' * 25)
        results = list()
        for i in range(TASK_NUMS):
            result = 0
            # test_task = CartPoleEnv(
            #     length=self.task_list[i].length)  # simply set this to random can test the generalizability
            self.reset_cur_task(np.random.randint(len(self.task_ids)))
            for test_epi in range(10):  # test for 10 epochs and takes the average
                states, actions, _, _ = self.roll_out(None, None, self.task, horizon, None, reset=True)
                states = states[:-1]
                z = get_dyn_embedding(states[:-1], actions[:-1], states[1:],
                                      self.dyn_encoder)  # 1,z_dim
                state = self.task.reset()
                for test_step in range(200):  # rollout for 200 steps
                    action, _, logp = self.actor_network(torch.Tensor([state]).cuda(), z)
                    next_state, reward, done, action = single_step_rollout(action, self.task)
                    # test_task.render()
                    result += reward
                    state = next_state
                    if done:
                        break
            results.append(result / 10.)  # average for 10 tests
        val_res = np.mean(results) # this does not require scaling because I use single-step-rollout
        self.writer.add_scalar('actor/val_return', val_res, val_cnt)
        print(f'average return {val_res} for {TASK_NUMS} tasks.')
        print('=' * 25 + ' validation ' + '=' * 25)

        torch.save(self.nstate_decoder.state_dict(), os.path.join(self.expname, f"meta_value_network_cartpole_{step}.pkl"))
        torch.save(self.dyn_encoder.state_dict(), os.path.join(self.expname, f"task_config_network_cartpole_{step}.pkl"))
        print("save networks")
        self.task_list = resample_task()  # resample

    def deploy(self):
        self.setup_modules()
        self.train_vae()
        self.train_policy()

class Singletask(BaseModel):
    def __init__(self, env, algo):
        super(Singletask, self).__init__(env, algo)


    def setup_modules(self):
        self.algo.setup()

        # random_choice = np.random.uniform(L_MIN, L_MAX)
        random_choice = np.random.randint(len(self.task_ids))
        print(f'tasks totaling {len(self.task_ids)}, chose {random_choice}')
        self.defualt_setup(memo=config.memo + f'/n{random_choice}')
        # self.task = CartPoleEnv(random_choice)
        self.reset_cur_task(random_choice)
        self.task = self.env

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

            action, logp = self.algo.get_action(torch.Tensor([state]).cuda()) # return cpu version
            action = action.cpu().data.numpy()[0]
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

    def train_policy(self):
        print('Train policy.')
        loss_buffer = deque(maxlen=actor_report_freq)

        val_cnt = 0
        horizon = HORIZON  # to continuously increase the hardship of the trajectories.
        gamma = pow(0.05, 1. / horizon)
        double_check = 0
        done = True
        s = None
        for step in range(STEP):
            # obtain z and experience data to train the actor
            states, actions, rewards, is_done, log_softmax_actions, fin = self.roll_out(s, self.task, horizon)

            if is_done[-1,0]: s = None
            else: s = fin

            # sudo_loss, bellman_error, actions_logp, advantages = self.algo.train(states, rewards,
            #                                                                                     is_done,
            #                                                                                     log_softmax_actions,
            #                                                                                     gamma)
            # states, actions, rewards, dones, gamma
            qf_loss, vf_loss, actor_loss, actions_logp, advantages = self.algo.train(states, actions, rewards, is_done, gamma)
            losses = (qf_loss, vf_loss, actor_loss)

            self.writer.add_histogram('actor/actions_logp', actions_logp, step)
            self.writer.add_histogram('actor/advantages', advantages, step)

            self.algo.optimize(losses, self.writer, step)

            total_rewards = np.sum(rewards)
            avg_ret = total_rewards/reward_scale
            if len(loss_buffer) == actor_report_freq: loss_buffer.popleft()
            loss_buffer.append(avg_ret.item())
            m = np.mean(list(loss_buffer))
            self.writer.add_scalar('actor/avg_return', avg_ret, step)
            # if print_every_step:
            #     print(f'step {step} with return {avg_ret}, bellman={bellman_error}, sudo={sudo_loss}')
            if (step + 1) % actor_report_freq == 0:
                print(f'step {step} with avg return {m}.')
                if m > double_horizon_threshold * horizon:
                    double_check += 1
                    if double_check == 3:
                        horizon += HORIZON
                        print(f'step {step} horizon={horizon}')
                        double_check = 0

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
    env = NormalizedBoxEnv(HalfCheetahDirEnv(), reward_scale=1.)
    # model = Multitask(env, algo=A2C(1), model_lr=(10e-3,10e-3,10e-3,5e-3))
    # model.deploy()
    # model = Singletask(env, algo=dict(cls=A2C,
    #                                   params=dict(be_w=1., model_lr=(1e-3, 1e-3), modules=('value_baseline','actor_network')))) # value actor
    model = Singletask(env, algo=dict(cls=SAC,
                                      params=dict(modules=('value_baseline', 'target_v', 'actor_network', 'q1', 'q2'),
                                                  model_lr=[1e-3]*5)))
    model.deploy()

if __name__ == '__main__':
    main()
