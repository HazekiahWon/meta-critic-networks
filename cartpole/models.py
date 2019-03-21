#-------------------------------------
# Project: Meta Value Network
# Date: 2017.5.25
# All Rights Reserved
#-------------------------------------

from cartpole import CartPoleEnv
import time
from collections import deque
from common_imports import *
from modules import *
from utils import *
from algos import *
from tensorboardX import SummaryWriter
import os

class BaseModel():
    def __init__(self, modules, model_lr):
        self.modules = modules
        self.model_lr = {name:lr for name,lr in zip(self.modules,model_lr)}

    def defualt_setup(self, memo=None):
        # for the convenience of saving
        model_dict = {name: self.__getattribute__(name) for name in self.modules}
        if resume_model_dir is not None:
            for n, m in model_dict.items():
                self.load_model(m, os.path.join(resume_model_dir, n, '.pkl'))  # remember to add {step}

        model_opt = {name: torch.optim.Adam(net.parameters(), lr=self.model_lr[name]) for name, net in
                     model_dict.items()}

        self.model_dict, self.model_opt = model_dict, model_opt

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
        torch.nn.utils.clip_grad_norm(network.parameters(), max_norm)
        optim.step()
        optim.zero_grad()

    def step_optimizers(self, model_names):
        for n in model_names:
            self.step_optimizer(self.model_opt[n], self.model_dict[n])

class Multitask(BaseModel):
    def __init__(self, algo, model_lr=(5e-3,5e-3,5e-3,5e-3)):
        modules = ['nstate_decoder','dyn_encoder','value_baseline','actor_network']
        super(Multitask, self).__init__(modules, model_lr)
        self.algo = algo

    def setup_modules(self):
        self.modules = ['nstate_decoder','dyn_encoder','value_baseline','actor_network']
        self.nstate_decoder = Trans_with_latent()
        self.dyn_encoder = DynEmb()
        self.value_baseline = VBase_with_latent()
        self.actor_network = Actor_with_latent()
        self.vae = VAE(self.dyn_encoder, self.nstate_decoder)
        # send to cuda
        self.value_baseline.cuda()
        self.nstate_decoder.cuda()
        self.dyn_encoder.cuda()
        self.actor_network.cuda()
        self.vae.cuda()

        self.defualt_setup()

    def roll_out(self, actor_network, task, sample_nums, z, reset=False, to_cuda=True):
        """
        sample a sequence of time steps
        :param actor_network:
        :param task:
        :param sample_nums:
        :return:
        """
        if reset: task.reset()
        states = []
        actions = []
        rewards = []
        is_done = False
        state = task.state  # ?
        actions_logp = list()
        result = 0
        for j in range(sample_nums):
            states.append(state)

            if actor_network is not None:
                logp = actor_network(torch.Tensor([state]), z)
                next_state, reward, done, action= single_step_rollout(logp, task)
                one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
                actions_logp.append(logp)
            else:
                action = task.action_space.sample()
                one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
                next_state, reward, done, _ = task.step(action)

            fix_reward = -10 if done else 1  # this will cause total reward to be smaller than 0
            actions.append(one_hot_action)
            rewards.append(fix_reward)
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

            if step % task_resample_freq == 0:  # sample another batch of tasks
                self.task_list = resample_task()

            for i in range(TASK_NUMS):
                states, actions, rewards, _ = self.roll_out(None, self.task_list[i], HORIZON, None, reset=True,
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
        task_list = resample_task()
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
                states, actions, _, _ = self.roll_out(None, task_list[i], horizon, None, reset=True)
                states = states[:-1]  # to remove the final state
                latent_z = get_dyn_embedding(states[:-1], actions[:-1], states[1:], self.dyn_encoder)  # 1,z_dim
                states, actions, rewards, is_done, log_softmax_actions = self.roll_out(self.actor_network, task_list[i], horizon,
                                                                                  latent_z, reset=True)

                cstate_value = self.value_baseline(states, latent_z.repeat(states.size(0), 1))
                sudo_loss, bellman_error, total_rewards, actions_logp, advantages = self.algo.train(states, actions, rewards,
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
            test_task = CartPoleEnv(
                length=self.task_list[i].length)  # simply set this to random can test the generalizability
            for test_epi in range(10):  # test for 10 epochs and takes the average
                states, actions, _, _ = self.roll_out(None, test_task, horizon, None, reset=True)
                states = states[:-1]
                z = get_dyn_embedding(states[:-1], actions[:-1], states[1:],
                                      self.dyn_encoder)  # 1,z_dim
                state = test_task.reset()
                for test_step in range(200):  # rollout for 200 steps
                    logp = self.actor_network(torch.Tensor(state), z)
                    next_state, reward, done, action= single_step_rollout(logp, test_task)
                    # test_task.render()
                    result += reward
                    state = next_state
                    if done:
                        break
            results.append(result / 10.)  # average for 10 tests
        val_res = np.mean(results)/reward_scale
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
    def __init__(self, algo, model_lr=(5e-3,5e-3)):
        modules = ['value_baseline', 'actor_network']
        super(Singletask, self).__init__(modules, model_lr)
        self.algo = algo

    def setup_modules(self):
        self.value_baseline = VBase()
        self.actor_network = Actor()
        # send to cuda
        self.value_baseline.cuda()
        self.actor_network.cuda()
        # algo
        # self.algo = A2C(5)


        random_choice = np.random.uniform(L_MIN, L_MAX)
        self.defualt_setup(memo=config.memo + f'/n{random_choice}')
        self.task = CartPoleEnv(random_choice)

    def roll_out(self, actor_network, task, sample_nums, reset=False, to_cuda=True):
        """
        sample a sequence of time steps
        :param actor_network:
        :param task:
        :param sample_nums:
        :return:
        """
        if reset: task.reset()
        states = []
        actions = []
        rewards = []
        is_done = False
        state = task.state  # ?
        actions_logp = list()
        result = 0
        for j in range(sample_nums):
            states.append(state)

            if actor_network is not None:
                logp = actor_network(torch.Tensor([state]).cuda())
                next_state, reward, done, action = single_step_rollout(logp, task)
                one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
                actions_logp.append(logp)
            else:
                action = task.action_space.sample()
                one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
                next_state, reward, done, _ = task.step(action)

            fix_reward = -10 if done else 1  # this will cause total reward to be smaller than 0
            actions.append(one_hot_action)
            rewards.append(fix_reward)
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

    def train_policy(self):
        print('Train policy.')
        loss_buffer = deque(maxlen=actor_report_freq)

        val_cnt = 0
        horizon = HORIZON  # to continuously increase the hardship of the trajectories.
        gamma = pow(0.05, 1. / horizon)
        double_check = 0
        for step in range(STEP):
            # obtain z and experience data to train the actor
            states, actions, rewards, is_done, log_softmax_actions = self.roll_out(self.actor_network, self.task,
                                                                              horizon, reset=True)
            cstate_value = self.value_baseline(states)
            sudo_loss, bellman_error, total_rewards, actions_logp, advantages = self.algo.train(states, actions,
                                                                                                rewards,
                                                                                                cstate_value,
                                                                                                log_softmax_actions,
                                                                                                gamma)

            self.writer.add_histogram('actor/actions_logp', actions_logp, step)
            self.writer.add_histogram('actor/advantages', advantages, step)

            self.algo.optimize(sudo_loss, bellman_error, self.model_dict, self.model_opt,
                               ('actor_network', 'value_baseline'), self.writer, step)

            avg_ret = total_rewards/reward_scale
            if len(loss_buffer) == actor_report_freq: loss_buffer.popleft()
            loss_buffer.append(avg_ret.item())
            m = np.mean(list(loss_buffer))
            self.writer.add_scalar('actor/avg_return', avg_ret, step)
            print(f'step {step} with return {avg_ret}, bellman={bellman_error}, sudo={sudo_loss}')

            # if (step + 1) % actor_report_freq == 0:
            #     print(f'step {step} with avg return {m}.')
            #     if m > double_horizon_threshold * horizon:
            #         double_check += 1
            #         if double_check == 3:
            #             horizon += HORIZON
            #             print(f'step {step} horizon={horizon}')
            #             double_check = 0

            if (step + 1) % policy_task_resample_freq == 0:
                self.val_and_save(horizon, val_cnt, step)
                val_cnt += 1

    def val_and_save(self, horizon, val_cnt, step):
        # validation

        print('=' * 25 + ' validation ' + '=' * 25)
        results = list()
        for test_epi in range(10):  # test for 10 epochs and takes the average
            states, actions, rewards, is_done, log_softmax_actions = self.roll_out(self.actor_network, self.task,
                                                                                   200, reset=True)
            results.append(np.sum(rewards))

        val_res = np.mean(results)/reward_scale
        self.writer.add_scalar('actor/val_return', val_res, val_cnt)
        print(f'average return {val_res} for 10 tests.')
        print('=' * 25 + ' validation ' + '=' * 25)

    def deploy(self):
        self.setup_modules()
        self.train_policy()


def main():
    # model = Multitask(algo=A2C(5))
    # model.deploy()
    model = Singletask(algo=A2C(1), model_lr=(0.001, 5e-3)) # value actor
    model.deploy()

if __name__ == '__main__':
    main()
