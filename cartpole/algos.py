from common_imports import *
from utils import *

class A2C():
    def __init__(self, be_w, ent_w=0.1):
        self.bellman_weight = be_w
        self.ent_weight = ent_w

    def train(self, states, actions, rewards, cstate_value, log_softmax_actions, gamma, entropy_fn=None):
        n_t = len(states)
        # cstate_value = cstate_value.detach()
        total_rewards = np.sum(rewards)  # no disc
        # disc_return = torch.Tensor(seq_reward2go(rewards, gamma)).cuda()
        # value[0:t+1], including the final state
        # cstate_value = get_predicted_values(states, latent_z.repeat(states.size(0), 1), value_baseline)
        # value[1:t+1]
        nstate_value = cstate_value[1:]  # t,1
        rewards = torch.Tensor(rewards).view(-1, 1).cuda()
        # R = r[0:t]+gamma*value[1:t+1]
        td_target = rewards + gamma * nstate_value  # t
        # A = R - v = R - value[0:t]
        adv = td_target - cstate_value[:-1]  # bellman error
        adv = adv.squeeze(-1) # i am not sure but i guess critic should not be involved in differentiation
        criterion = nn.MSELoss()
        bellman_error = criterion(td_target, cstate_value[:-1])

        actions_var = Variable(actions).cuda()
        # if this drops to 0, might well lead to some problem, e.g., the return drop drastically, but is multiplied by 0
        actions_logp = -torch.sum(log_softmax_actions * actions_var, dim=1)[:n_t] + 1e-4  # n,1

        sudo_loss = torch.sum(actions_logp * adv.detach(), dim=0)

        if entropy_fn is not None:
            sudo_loss += self.ent_weight*entropy_fn(log_softmax_actions)

        return sudo_loss, bellman_error, total_rewards, actions_logp, adv

    def optimize(self, ac_loss, bl_loss, model_dict, model_opt, model_names, writer, step):
        overall_loss = ac_loss + self.bellman_weight * bl_loss
        overall_loss.backward()
        step_optimizers(model_dict, model_opt, model_names)
        writer.add_scalar('actor/critic_loss', bl_loss, step)
        writer.add_scalar('actor/actor_loss', ac_loss, step)

class Entropy:
    @staticmethod
    def cat_ent(logp):
        p = torch.exp(logp) # n,2
        ent = -p*logp # n,2
        ent = torch.sum(ent, dim=-1) # n,
        return torch.mean(ent, dim=-1) # average ent for actions