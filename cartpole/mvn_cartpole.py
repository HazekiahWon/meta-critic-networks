#-------------------------------------
# Project: Meta Value Network
# Date: 2017.5.25
# All Rights Reserved
#-------------------------------------

from cartpole import CartPoleEnv
import time
from collections import deque
from utils import *
from config import *
from tensorboardX import SummaryWriter
import os
def main():

    # task_embedding + dynamics
    reward_decoder = MetaValueNetwork(input_size = STATE_DIM+ACTION_DIM+Z_DIM,hidden_size = value_dim,output_size = 1)
    dyn_encoder = DynamicsEmb(input_size=STATE_DIM*2+ACTION_DIM,
                                      hidden_size=task_dim,
                                      output_size=vae_dim,
                              emb_dim=vae_dim, z_dim=Z_DIM,
                                      k=5,
                                      stoch=stochastic_encoder)
    actor_network = ActorNetwork(STATE_DIM + Z_DIM, actor_dim, ACTION_DIM)

    reward_decoder.cuda()
    dyn_encoder.cuda()
    actor_network.cuda()

    # load params
    # if os.path.exists("meta_value_network_cartpole.pkl"):
    #     meta_value_network.load_state_dict(torch.load("meta_value_network_cartpole.pkl"))
    #     print("load meta value network success")
    # if os.path.exists("task_config_network_cartpole.pkl"):
    #     task_config_network.load_state_dict(torch.load("task_config_network_cartpole.pkl"))
    #     print("load task config network success")
    # optim instances for two nets
    meta_value_network_optim = torch.optim.Adam(reward_decoder.parameters(),lr=vae_lr)
    task_config_network_optim = torch.optim.Adam(dyn_encoder.parameters(),lr=vae_lr) # not used?
    actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=0.01)
    os.makedirs(logdir, exist_ok=True)
    expname = os.path.join(logdir, time.strftime('exp_%y%m%d_%H%M%S', time.localtime()))
    writer = SummaryWriter(log_dir=expname)
    # task_list = resample_task()

    # ----------------- Training ------------------
    print(f'train VAE.')
    loss_buffer = deque(maxlen=vae_report_freq)

    step = 0
    start = time.time()
    while True:
        # actor_loss = list()
        value_loss = list()


        if step % task_resample_freq == 0:  # sample another batch of tasks
            # renew the tasks
            task_list = resample_task()

        for i in range(TASK_NUMS):
            states, actions, rewards, _, _ = roll_out(None, task_list[i], HORIZON, None, reset=True)

            # latent_z = get_dyn_embedding(states[:-1], actions[:-1], states[1:],
            #                                 task_config_network) # 1,z_dim
            #
            # pred_return = get_predicted_rewards(torch.Tensor(states).cuda(), torch.Tensor(actions).cuda(),
            #                                 latent_z.repeat(states.shape[0], 1),
            #                                 meta_value_network, do_grad=True)
            vae = VAE(dyn_encoder, reward_decoder)
            vae.cuda()
            pred_return = vae(torch.Tensor(states).cuda(), torch.Tensor(actions).cuda())
            target_values = torch.Tensor(seq_reward2go(rewards, gamma)).view(-1,1).cuda() #[dis_reward(rewards[i:], gamma) for i in range(len(rewards))]
            # values = meta_value_network(torch.cat((states_var,task_configs),1))
            criterion = nn.MSELoss()
            value_loss.append(criterion(pred_return,target_values))

        meta_value_network_optim.zero_grad()
        task_config_network_optim.zero_grad()
        rec_loss = torch.mean(torch.stack(value_loss))
        rec_loss.backward()
        opt_loss(meta_value_network_optim, reward_decoder)
        opt_loss(task_config_network_optim, dyn_encoder)

        writer.add_scalar('VAE/reconstruction_loss', rec_loss, step)
        # step += 1
        if len(loss_buffer)==vae_report_freq: loss_buffer.popleft()
        loss_buffer.append(rec_loss.item())
        m = np.mean(list(loss_buffer))
        if step%vae_report_freq==0:
            print(f'step {step} takes {time.time() - start} sec, with recloss={m}.')
            start = time.time()
        if m <vae_thresh: break
        else: step += 1
    print('Finish VAE.')

    print('Train policy.')
    loss_buffer = deque(maxlen=actor_report_freq)
    task_list = resample_task()
    val_cnt = 0
    horizon = HORIZON # to continuously increase the hardship of the trajectories.
    for step in range(STEP):
        actor_loss = list()
        rets = list()
        # value_loss = list()
        start = time.time()

        for i in range(TASK_NUMS):
            states, actions, rewards, _, _ = roll_out(None, task_list[i], horizon, None, reset=True)
            latent_z = get_dyn_embedding(states[:-1], actions[:-1], states[1:],
                                            dyn_encoder) # 1,z_dim
            states,actions,rewards,is_done,final_state, log_softmax_actions = roll_out(actor_network, task_list[i], horizon, latent_z, reset=True)
            n_t = len(states)
            ret = torch.Tensor(seq_reward2go(rewards[:n_t], gamma)).cuda()
            rets.append(np.sum(rewards))
            if use_baseline:
                st = np.concatenate([states[:n_t]]*n_sample,axis=0) # t*n,stat_dim
                ac = np.asarray([[0,1]]*n_t+[[1,0]]*n_t) # t*n,2
                # t*n,1
                baseline_ = get_predicted_rewards(torch.from_numpy(st).cuda(),
                                                 torch.Tensor(ac).cuda(), # must use Tensor to transform to float
                                                 latent_z.repeat(n_t*n_sample, 1), reward_decoder, do_grad=False)
                baseline = baseline_.detach().view(-1, n_sample)
                baseline = torch.mean(baseline, dim=-1) # t
                advantages = ret - baseline.cuda()
                # writer.add_scalar('actor/baseline', baseline[0], step)
                # writer.add_scalar('actor/real_return', ret[0], step)
                # writer.add_scalar('actor/advantage', advantages[0], step)
                # writer.add_histogram('actor/baseline', step)
            else:
                advantages = ret
                # writer.add_scalar('actor/real_return', ret[0], step)

            # why is there multiplied by actions_var
            actions_var = Variable(actions).cuda()
            # take the mean over time steps
            # if this drops to 0, might well lead to some problem, e.g., the return drop drastically, but is multiplied by 0
            actions_logp = -torch.sum(log_softmax_actions*actions_var,dim=1)[:n_t]+1e-4 # n,1
            if i == 0:
                writer.add_histogram('actor/actions_logp', actions_logp, step)
                writer.add_histogram('actor/advantages', advantages, step)
            sudo_loss = torch.sum(actions_logp*advantages.detach(), dim=0)
            actor_loss.append(sudo_loss) #+ entropy #+ actor_criterion(actor_y_samples,target_y)


        ac_loss = torch.mean(torch.stack(actor_loss))
        avg_ret = np.mean(rets)
        actor_network_optim.zero_grad()
        ac_loss.backward()
        # rec_loss.backward()
        opt_loss(actor_network_optim, actor_network)

        if len(loss_buffer)==actor_report_freq: loss_buffer.popleft()
        loss_buffer.append(avg_ret.item())
        m = np.mean(list(loss_buffer))
        writer.add_scalar('actor/actor_loss', ac_loss, step)
        writer.add_scalar('actor/avg_return', avg_ret, step)
        # print(f'step {step} takes {time.time() - start} sec, with return={avg_ret}, acloss={ac_loss}.')
        if (step+1) % actor_report_freq == 0:
            print(f'step {step} with avg return {m}.')
            if m>double_horizon_threshold*horizon:
                horizon += HORIZON
                print(f'step {step} horizon={horizon}')


        if (step+1) % task_resample_freq == 0:
            print('='*25+' validation '+'='*25)
            results = list()
            for i in range(TASK_NUMS):
                result = 0
                test_task = CartPoleEnv(length = task_list[i].length)
                for test_epi in range(10): # test for 10 epochs and takes the average
                    states, actions, rewards, _, _ = roll_out(None, test_task, horizon, None, reset=True)
                    z = get_dyn_embedding(states[:-1], actions[:-1], states[1:],
                                                    dyn_encoder)  # 1,z_dim
                    state = test_task.reset()
                    for test_step in range(200): # rollout for 200 steps

                        logp = get_action_logp(torch.Tensor([state]).cuda(), z, actor_network, do_grad=True)
                        softmax_action = torch.exp(logp).cuda()
                        action = np.argmax(softmax_action.cpu().data.numpy()[0])
                        next_state,reward,done,_ = test_task.step(action)
                        result += reward
                        state = next_state
                        if done:
                            break
                results.append(result/10.)
            val_res = np.mean(results)
            writer.add_scalar('actor/val_return', val_res, val_cnt)
            print(f'average return {val_res} for {TASK_NUMS} tasks.')
            print('=' * 25 + ' validation ' + '=' * 25)

            torch.save(reward_decoder.state_dict(),os.path.join(expname, f"meta_value_network_cartpole_{step}.pkl"))
            torch.save(dyn_encoder.state_dict(),os.path.join(expname, f"task_config_network_cartpole_{step}.pkl"))
            print("save networks")
            task_list = resample_task() # resample
            val_cnt += 1

if __name__ == '__main__':
    main()
