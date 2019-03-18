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
def main():

    # task_embedding + dynamics
    meta_value_network = MetaValueNetwork(input_size = STATE_DIM+ACTION_DIM+Z_DIM,hidden_size = value_dim,output_size = 1)
    task_config_network = DynamicsEmb(input_size=STATE_DIM*2+ACTION_DIM,
                                      hidden_size=task_dim,
                                      output_size=Z_DIM,
                                      k=5,
                                      timestep=HORIZON)
    actor_network = ActorNetwork(STATE_DIM + Z_DIM, actor_dim, ACTION_DIM)

    meta_value_network.cuda()
    task_config_network.cuda()
    actor_network.cuda()

    # load params
    # if os.path.exists("meta_value_network_cartpole.pkl"):
    #     meta_value_network.load_state_dict(torch.load("meta_value_network_cartpole.pkl"))
    #     print("load meta value network success")
    # if os.path.exists("task_config_network_cartpole.pkl"):
    #     task_config_network.load_state_dict(torch.load("task_config_network_cartpole.pkl"))
    #     print("load task config network success")
    # optim instances for two nets
    meta_value_network_optim = torch.optim.Adam(meta_value_network.parameters(),lr=vae_lr)
    task_config_network_optim = torch.optim.Adam(task_config_network.parameters(),lr=vae_lr) # not used?
    actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=0.01)
    # task_list = resample_task()
    for episode in range(EPISODE):
        # ----------------- Training ------------------
        print(f'train VAE.')
        loss_buffer = deque(maxlen=vae_report_freq)

        step = 0
        while True:
            # actor_loss = list()
            value_loss = list()
            start= time.time()

            if step % task_resample_freq == 0:  # sample another batch of tasks
                # renew the tasks
                task_list = resample_task()

            for i in range(TASK_NUMS):
                states, actions, rewards, _, _ = roll_out(None, task_list[i], HORIZON, None, reset=True)

                dyn_encoder = get_dyn_embedding(states[:-1], actions[:-1], states[1:],
                                                task_config_network) # 1,z_dim

                reward_decoder = get_predicted_rewards(torch.Tensor(states).cuda(), torch.Tensor(actions).cuda(),
                                                dyn_encoder.repeat(states.shape[0], 1),
                                                meta_value_network, do_grad=True)
                vae = VAE(dyn_encoder, reward_decoder, vae_dim, Z_DIM)
                target_values = torch.Tensor(seq_reward2go(rewards, gamma)).view(-1,1).cuda() #[dis_reward(rewards[i:], gamma) for i in range(len(rewards))]
                # values = meta_value_network(torch.cat((states_var,task_configs),1))
                criterion = nn.MSELoss()
                value_loss.append(criterion(reward_decoder,target_values))

            meta_value_network_optim.zero_grad()
            task_config_network_optim.zero_grad()
            rec_loss = torch.mean(torch.stack(value_loss))
            rec_loss.backward()
            opt_loss(meta_value_network_optim, meta_value_network)
            opt_loss(task_config_network_optim, task_config_network)

            # step += 1
            if len(loss_buffer)==vae_report_freq: loss_buffer.popleft()
            loss_buffer.append(rec_loss.item())
            m = np.mean(list(loss_buffer))
            if step%vae_report_freq==0:
                print(f'step {step} takes {time.time() - start} sec, with recloss={m}.')
            if m <15.: break
            else: step += 1
        print('Finish VAE.')
        print('Train policy.')
        loss_buffer = deque(maxlen=actor_report_freq)
        for step in range(STEP):
            actor_loss = list()
            # value_loss = list()
            start = time.time()

            if (step + 1) % task_resample_freq == 0:  # sample another batch of tasks
                # renew the tasks
                task_list = resample_task()

            for i in range(TASK_NUMS):
                states, actions, rewards, _, _ = roll_out(None, task_list[i], HORIZON, None, reset=True)
                dyn_encoder = get_dyn_embedding(states[:-1], actions[:-1], states[1:],
                                                task_config_network) # 1,z_dim

                # pred_rs = get_predicted_rewards(torch.Tensor(pre_states[i]).cuda(), torch.Tensor(pre_actions[i]).cuda(),
                #                                 task_config.repeat(pre_states[i].shape[0], 1),
                #                                 meta_value_network, do_grad=True)
                # # ===> should be resample
                states,actions,rewards,is_done,final_state, log_softmax_actions = roll_out(actor_network, task_list[i], HORIZON, dyn_encoder, reset=True)
                # logp_action 30,2

                # random_actions = [one_hot_action_sample(task_list[i]) for _ in range(n_sample)]
                n_t = len(states)//2
                st = np.concatenate([states[:n_t]]*n_sample,axis=0) # t*n,stat_dim
                ac = np.asarray([[0,1]]*n_t+[[1,0]]*n_t) # t*n,2
                # t*n,1
                baseline_ = get_predicted_rewards(torch.from_numpy(st).cuda(),
                                                 torch.Tensor(ac).cuda(), # must use Tensor to transform to float
                                                 dyn_encoder.repeat(n_t*n_sample, 1), meta_value_network, do_grad=False)
                baseline = baseline_.detach().view(-1, n_sample)
                baseline = torch.mean(baseline, dim=-1) # t
                # calculate qs
                # qs = Variable(torch.Tensor(discount_reward(rewards,0.99,final_r))).cuda()
                ret = torch.Tensor(seq_reward2go(rewards[:n_t], gamma)).cuda()
                # baseline = torch.mean(baseline).detach()
                advantages = ret-baseline.cuda()
                # why is there multiplied by actions_var
                actions_var = Variable(actions).cuda()
                # take the mean over time steps
                actions_logp = torch.sum(log_softmax_actions*actions_var,dim=1)[:n_t] # n,1
                # print(actions_logp, advantages)
                actor_loss.append(- torch.sum(actions_logp*advantages.detach(), dim=0)) #+ entropy #+ actor_criterion(actor_y_samples,target_y)


            ac_loss = torch.mean(torch.stack(actor_loss))
            # rec_loss = torch.mean(torch.stack(value_loss))
            actor_network_optim.zero_grad()
            # meta_value_network_optim.zero_grad()
            # task_config_network_optim.zero_grad()
            ac_loss.backward()
            # rec_loss.backward()
            opt_loss(actor_network_optim, actor_network)

            if len(loss_buffer)==actor_report_freq: loss_buffer.popleft()
            loss_buffer.append(ac_loss.item())
            m = np.mean(list(loss_buffer))

            if (step + 1) % task_resample_freq == 0:
                print(f'step {step} takes {time.time() - start} sec, with acloss={m}.')
                print('='*25+' validation '+'='*25)
                results = list()
                for i in range(TASK_NUMS):
                    result = 0
                    test_task = CartPoleEnv(length = task_list[i].length)
                    for test_epi in range(10): # test for 10 epochs and takes the average
                        states, actions, rewards, _, _ = roll_out(None, test_task, HORIZON, None, reset=True)
                        z = get_dyn_embedding(states[:-1], actions[:-1], states[1:],
                                                        task_config_network)  # 1,z_dim
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
                print(f'episode {episode} average return {np.mean(results)} for {TASK_NUMS} tasks.')
                print('=' * 25 + ' validation ' + '=' * 25)

        
        if (episode+1) % 10 == 0 :
            # Save meta value network
            torch.save(meta_value_network.state_dict(),"meta_value_network_cartpole.pkl")
            torch.save(task_config_network.state_dict(),"task_config_network_cartpole.pkl")
            print(("save networks for episode:",episode))

if __name__ == '__main__':
    main()
