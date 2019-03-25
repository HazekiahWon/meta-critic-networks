from common_imports import *
from utils import *
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from modules import *
from test import PolicyNetwork

class BaseAlgo():
    def __init__(self, state_dim, action_dim, modules, model_lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.modules = modules
        self.model_lr = {name: lr for name, lr in zip(self.modules, model_lr)}

    def setup(self):
        # for the convenience of saving
        model_dict = {name: self.__getattribute__(name) for name in self.modules}
        # if resume_model_dir is not None:
        #     for n, m in model_dict.items():
        #         self.load_model(m, os.path.join(resume_model_dir, n, '.pkl'))  # remember to add {step}

        model_opt = {name: torch.optim.Adam(net.parameters(), lr=self.model_lr[name]) for name, net in
                     model_dict.items()}

        self.model_dict, self.model_opt = model_dict, model_opt


class A2C(BaseAlgo):
    def __init__(self, state_dim, action_dim, modules, model_lr, be_w, ent_w=0.1):
        super(A2C, self).__init__(state_dim, action_dim, modules, model_lr)
        self.bellman_weight = be_w
        self.ent_weight = ent_w

    def setup(self):

        # self.value_baseline = VBase(self.state_dim, nonlin=F.elu)
        # self.actor_network = GaussianActor(self.state_dim, self.action_dim, nonlin=F.tanh)
        net_size = 300
        self.value_baseline = FlattenMlp(  # state value
            hidden_sizes=[net_size, net_size, net_size],
            input_size=self.state_dim,
            output_size=1,
        )
        self.actor_network = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=self.state_dim, action_dim=self.action_dim,)
        super(A2C, self).setup()
        for net in self.model_dict.values():
            net.cuda()

    def get_action(self, *inp, full=False):
        ret = self.actor_network(*inp)
        if full: return ret
        else:
            action, pre_a, logp, mean, logstd = ret
            return action,logp

    def train(self, states, rewards, dones, log_softmax_actions, gamma, entropy_fn=None):
        """
        do make sure logp and adv be of shape (n,1)
        :param states:
        :param rewards:
        :param cstate_value:
        :param log_softmax_actions:
        :param gamma:
        :param entropy_fn:
        :return:
        """
        n_t = len(states)
        # cstate_value = cstate_value.detach()
        total_rewards = np.sum(rewards)  # no disc
        cstate_value = self.value_baseline(states)
        # disc_return = torch.Tensor(seq_reward2go(rewards, gamma)).cuda()
        # value[0:t+1], including the final state
        # cstate_value = get_predicted_values(states, latent_z.repeat(states.size(0), 1), value_baseline)
        # value[1:t+1]
        nstate_value = cstate_value[1:]  # t,1
        rewards = torch.Tensor(rewards).view(-1, 1).cuda()
        # R = r[0:t]+gamma*value[1:t+1]
        td_target = rewards + (1.-dones) * gamma * nstate_value  # t
        # A = R - v = R - value[0:t]
        adv = td_target - cstate_value[:-1]  # bellman error
        # adv = adv.squeeze(-1) # i am not sure but i guess critic should not be involved in differentiation
        criterion = nn.MSELoss()
        bellman_error = criterion(td_target, cstate_value[:-1])

        # this is required only by discrete action space
        # actions_logp = -torch.sum(log_softmax_actions * actions_var, dim=1)[:n_t] + 1e-4  # n,1
        actions_logp = -log_softmax_actions
        sudo_loss = torch.sum(actions_logp * adv.detach(), dim=(0,1)) # n,1

        if entropy_fn is not None:
            sudo_loss += self.ent_weight*entropy_fn(log_softmax_actions)

        return sudo_loss, bellman_error, actions_logp, adv

    def optimize(self, losses, writer, step):
        # ac_loss.backward()
        # step_optimizers(self.model_dict, self.model_opt, ('actor_network',))
        # bl_loss.backward()
        # step_optimizers(self.model_dict, self.model_opt, ('value_baseline',))
        ac_loss, bl_loss = losses
        self.value_baseline.zero_grad()
        bl_loss.backward()
        step_optimizers(self.model_dict, self.model_opt, ('value_baseline',))
        ac_loss.backward()
        step_optimizers(self.model_dict, self.model_opt, ('actor_network',))
        # overall_loss = ac_loss + self.bellman_weight * bl_loss
        # overall_loss.backward()
        # step_optimizers(self.model_dict, self.model_opt, self.modules)
        writer.add_scalar('actor/critic_loss', bl_loss, step)
        writer.add_scalar('actor/actor_loss', ac_loss, step)


class SAC(BaseAlgo):
    def __init__(self, state_dim, action_dim, modules, model_lr, a_mean_w=1e-3, a_std_w=1e-3, pre_a_w=0., soft_update_w=0.005):
        super(SAC, self).__init__(state_dim, action_dim, modules, model_lr)
        # self.bellman_weight = be_w
        # self.ent_weight = ent_w
        self.a_mean_w = a_mean_w
        self.a_std_w = a_std_w
        self.pre_a_w = pre_a_w
        self.soft_update_weight = soft_update_w


    def setup(self):

        # self.value_baseline = VBase(self.state_dim, nonlin=F.elu)
        # self.target_v = self.value_baseline.copy()
        # self.actor_network = GaussianActor(self.state_dim, self.action_dim, nonlin=F.tanh)
        # self.q1 = QBase(self.state_dim, self.action_dim, nonlin=F.elu)
        # self.q2 = QBase(self.state_dim, self.action_dim, nonlin=F.elu)
        netsize = 256
        self.value_baseline = FlattenMlp(hidden_sizes=[netsize]*3,
                                         input_size=self.state_dim,
                                         output_size=1)
        self.target_v = self.value_baseline.copy()
        self.actor_network = TanhGaussianPolicy(hidden_sizes=[netsize]*3,
                                                obs_dim=self.state_dim,
                                                action_dim=self.action_dim)
        # self.actor_network = PolicyNetwork(self.state_dim, self.action_dim, netsize)
        self.q1 = FlattenMlp(hidden_sizes=[netsize]*3,
                             input_size=self.state_dim+self.action_dim,
                             output_size=1)
        self.q2 = FlattenMlp(hidden_sizes=[netsize]*3,
                             input_size=self.state_dim+self.action_dim,
                             output_size=1)
        super(SAC, self).setup()
        # send to cuda
        # self.value_baseline.cuda()
        # self.actor_network.cuda()
        for net in self.model_dict.values():
            net.cuda()

    def get_action(self, *inp, full=False):
        ret = self.actor_network(*inp)
        if full: return ret
        else:
            action, logp, pre_a, mean, logstd = ret
            return action,logp

    def min_q(self, states, actions):
        q1 = self.q1(states, actions) # z detach
        q2 = self.q2(states, actions) # z detach
        min_q = torch.min(q1, q2)
        return min_q, q1

    def train(self, states, actions, rewards, nstates, dones, gamma, writer, step, reparameterize=reparameterize, double_q=double_q, entropy_fn=None):
        """
        do make sure logp and adv be of shape (n,1)
        :param states: n+1,20
        :param actions: n,6
        :param rewards:
        :param cstate_value:
        :param log_softmax_actions:
        :param gamma:
        :param entropy_fn:
        :return:
        """
        soft_q_net, value_net, policy_net, target_value_net = self.q1, self.value_baseline, self.actor_network, self.target_v
        soft_q_criterion = nn.MSELoss()
        value_criterion = nn.MSELoss()
        mean_lambda,std_lambda,z_lambda = self.a_mean_w,self.a_std_w,self.pre_a_w
        soft_q_optimizer,value_optimizer,policy_optimizer = [self.model_opt[x] for x in ['q1','value_baseline','actor_network']]

        expected_q_value = soft_q_net(states, actions)
        expected_value = value_net(states)
        new_action, log_prob, z, mean, log_std = policy_net(states)
        # q
        target_value = target_value_net(nstates)
        next_q_value = rewards + (1 - dones) * gamma * ptu.get_numpy(target_value)
        q_value_loss = soft_q_criterion(expected_q_value, ptu.from_numpy(next_q_value))
        if double_q:
            q2 = self.q2(states, actions)
            q_value_loss += soft_q_criterion(q2, ptu.from_numpy(next_q_value))
        # v
        expected_new_q_value = soft_q_net(states, new_action)
        if double_q:
            nq2 = self.q2(states, new_action)
            expected_new_q_value = torch.min(nq2, expected_new_q_value)
        next_value = expected_new_q_value - log_prob
        value_loss = value_criterion(expected_value, next_value.detach())
        # p
        log_prob_target = expected_new_q_value - expected_value
        adv = log_prob - log_prob_target# oyster uses this
        if reparameterize:
            policy_loss = (log_prob - expected_new_q_value+expected_value.detach()).mean()
        else:
            policy_loss = (log_prob * adv.detach()).mean()

        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss = std_lambda * log_std.pow(2).mean()
        z_loss = z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        soft_q_optimizer.zero_grad()
        if double_q:
            self.model_opt['q2'].zero_grad()
        q_value_loss.backward()
        soft_q_optimizer.step()
        if double_q: self.model_opt['q2'].step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        ptu.soft_update_from_to(self.value_baseline, self.target_v, self.soft_update_weight)

        writer.add_scalar('vloss', value_loss, step)
        writer.add_scalar('qloss', q_value_loss, step)
        writer.add_scalar('ploss', policy_loss, step)
        writer.add_histogram('logp', log_prob, step)
        writer.add_histogram('adv', log_prob - log_prob_target, step)
        writer.add_histogram('q1', expected_q_value, step)
        if double_q:
            writer.add_histogram('q2', q2, step)
        writer.add_histogram('qt', next_q_value, step)


class Entropy:
    @staticmethod
    def cat_ent(logp):
        p = torch.exp(logp) # n,2
        ent = -p*logp # n,2
        ent = torch.sum(ent, dim=-1) # n,
        return torch.mean(ent, dim=-1) # average ent for actions

    @staticmethod
    def gauss_ent(logstd):
        return torch.sum(logstd + .5 * np.log(2.0 * np.pi * np.e), dim=-1)