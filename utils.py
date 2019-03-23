from common_imports import *
from config import *
import numpy as np
from cartpole import CartPoleEnv



def resample_task():
    task_list = [CartPoleEnv(np.random.uniform(L_MIN, L_MAX)) for task in range(TASK_NUMS)]
    task_lengths = [task.length for task in task_list]
    print(("task length:", task_lengths))
    [task.reset() for task in task_list]
    return task_list



def single_step_rollout(logp, action, test_task):
    # softmax_action = torch.exp(logp).cuda()
    # action = np.argmax(softmax_action.cpu().data.numpy()[0]
    next_state,reward,done,_ = test_task.step(action)
    return next_state, reward, done

def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(list(range(0, len(r)))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def dis_reward(rs, gamma):
    raw_ret = np.sum(rs*np.cumproduct([1]+[gamma]*(len(rs)-1)))
    if use_baseline: return raw_ret
    else: return max(0., raw_ret)

def seq_reward2go(rs, gamma):
    return [dis_reward(rs[i:], gamma) for i in range(len(rs))]

def get_dyn_embedding(s,a,sp, network):
    """
    return with batch size
    :param s: b,s_dim
    :param a: b,a_dim
    :param sp: b,s_dim
    :param network:
    :return:
    """
    # pre_data_samples = torch.cat(  # s,a,s
    #     (s,a,sp), dim=1).transpose(0,1).unsqueeze(0)  # 1,10,t-1
    # sas - z

    out = network(s, a, sp)
    return out

def get_predicted_values(s, z, network, do_grad=False):

    # value_inputs = torch.cat(
    #     (s,z), dim=-1)  # t,22
    # if do_grad: value_inputs = Variable(value_inputs)
    return network(s, z)  # for vae loss

def get_predicted_nstate(s,a,z, network, do_grad=False):

    # value_inputs = torch.cat(
    #     (s,a,z), dim=-1)  # t,22

    return network(s,a,z)  # for vae loss

def get_action_logp(s,z, network, do_grad=False):
    # actor_inputs = torch.cat((s,z), dim=1)  # sz - a
    # if do_grad: actor_inputs = Variable(actor_inputs)
    return network(s,z)  # why input the state sequence

def step_optimizer(optim, network, max_norm=0.5):
    torch.nn.utils.clip_grad_norm(network.parameters(), max_norm)
    optim.step()
    optim.zero_grad()

def step_optimizers(model_dict, model_optim, model_names):
    for n in model_names:
        step_optimizer(model_optim[n],model_dict[n])

def one_hot_action_sample(task):
    a = np.zeros((ACTION_DIM,))
    sam = task.action_space.sample()
    a[sam] = 1
    return a
