from common_imports import *
from utils import *
from modules import *
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
        self.value_baseline = FlattenMlp(hidden_sizes=[300]*3,
                                         input_size=self.state_dim,
                                         output_size=1)
        self.target_v = self.value_baseline.copy()
        self.actor_network = TanhGaussianPolicy(hidden_sizes=[300]*3,
                                                obs_dim=self.state_dim,
                                                action_dim=self.action_dim)
        self.q1 = FlattenMlp(hidden_sizes=[300]*3,
                             input_size=self.state_dim+self.action_dim,
                             output_size=1)
        self.q2 = FlattenMlp(hidden_sizes=[300]*3,
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
            action, pre_a, logp, mean, logstd = ret
            return action,logp

    def min_q(self, states, actions):
        q1 = self.q1(states, actions) # z detach
        q2 = self.q2(states, actions) # z detach
        min_q = torch.min(q1, q2)
        return min_q

    def train(self, states, actions, rewards, dones, gamma, entropy_fn=None):
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
        # ? where to get actions

        q1 = self.q1(states[:-1], actions)
        q2 = self.q2(states[:-1], actions)
        v = self.value_baseline(states[:-1]) # z detach

        new_actions, pre_tanh_a, logp, mean_action, logstd_action = self.get_action(states[:-1], full=True)

        with torch.no_grad():
            target_v = self.target_v(states[1:])

        q_target = torch.Tensor(rewards).cuda() + (1.-dones)*gamma*target_v.detach()
        qf_loss = torch.mean((q1-q_target)**2)+torch.mean((q2-q_target)**2) # q1,q2,contextenc

        # min_q
        min_q = self.min_q(states[:-1], new_actions)
        v_target = min_q - logp
        criterion = nn.MSELoss()
        vf_loss = criterion(v, v_target.detach()) # value_baseline, target_v (soft update, see line189 sac.py)

        logp_target = min_q
        adv = logp - logp_target + v
        # reparameterize
        policy_loss = (logp - logp_target.detach()).mean()
        actor_loss = policy_loss + self.a_mean_w*(mean_action**2).mean() + self.a_std_w*(logstd_action**2).mean()\
                     + self.pre_a_w*(pre_tanh_a**2).sum(dim=1).mean()

        return qf_loss, vf_loss, actor_loss, logp, adv

    def optimize(self, losses, writer, step):

        qf_loss,vf_loss,actor_loss = losses
        self.q1.zero_grad()
        self.q2.zero_grad()
        qf_loss.backward()
        step_optimizers(self.model_dict, self.model_opt, ('q1','q2'))
        self.value_baseline.zero_grad()
        vf_loss.backward()
        step_optimizers(self.model_dict, self.model_opt, ('value_baseline',))
        ptu.soft_update_from_to(self.value_baseline, self.target_v, self.soft_update_weight)
        self.actor_network.zero_grad()
        actor_loss.backward()
        step_optimizers(self.model_dict, self.model_opt, ('actor_network',))

        writer.add_scalar('actor/qf_loss', qf_loss, step)
        writer.add_scalar('actor/vf_loss', vf_loss, step)
        writer.add_scalar('actor/actor_loss', actor_loss, step)


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