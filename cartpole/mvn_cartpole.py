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
                                      timestep=SAMPLE_NUMS)
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



        print(f'train VAE.')
        loss_buffer = deque(maxlen=20)

        step = 0
        while True:
            # actor_loss = list()
            value_loss = list()
            start= time.time()

            for i in range(TASK_NUMS):
                states, actions, rewards, _, _ = roll_out(None, task_list[i], SAMPLE_NUMS, None, reset=True)

                task_config = get_dyn_embedding(states[:-1], actions[:-1], states[1:],
                                                task_config_network) # 1,z_dim

                pred_rs = get_predicted_rewards(torch.Tensor(states).cuda(), torch.Tensor(actions).cuda(),
                                                task_config.repeat(states.shape[0], 1),
                                                meta_value_network, do_grad=True)
                target_values = torch.Tensor(seq_reward2go(rewards, gamma)).view(-1,1).cuda() #[dis_reward(rewards[i:], gamma) for i in range(len(rewards))]
                # values = meta_value_network(torch.cat((states_var,task_configs),1))
                criterion = nn.MSELoss()
                value_loss.append(criterion(pred_rs,target_values))

            meta_value_network_optim.zero_grad()
            task_config_network_optim.zero_grad()
            rec_loss = torch.mean(torch.stack(value_loss))
            rec_loss.backward()
            opt_loss(meta_value_network_optim, meta_value_network)
            opt_loss(task_config_network_optim, task_config_network)

            # step += 1
            if len(loss_buffer)==20: loss_buffer.popleft()
            loss_buffer.append(rec_loss.item())
            m = np.mean(list(loss_buffer))
            if step%20==0:
                print(f'step {step} takes {time.time() - start} sec, with recloss={m}.')
            if m <1.: break
            else: step += 1
        print('Finish VAE.')

        for step in range(STEP):
            actor_loss = list()
            # value_loss = list()
            start = time.time()
            for i in range(TASK_NUMS):
                states, actions, rewards, _, _ = roll_out(None, task_list[i], SAMPLE_NUMS, None, reset=True)
                task_config = get_dyn_embedding(states[:-1], actions[:-1], states[1:],
                                                task_config_network) # 1,z_dim

                # pred_rs = get_predicted_rewards(torch.Tensor(pre_states[i]).cuda(), torch.Tensor(pre_actions[i]).cuda(),
                #                                 task_config.repeat(pre_states[i].shape[0], 1),
                #                                 meta_value_network, do_grad=True)
                # # ===> should be resample
                states,actions,rewards,is_done,final_state, log_softmax_actions = roll_out(actor_network,task_list[i],SAMPLE_NUMS, task_config, reset=True)
                # logp_action 30,2

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
                actor_loss.append(- torch.sum(actions_logp*advantages.detach(), dim=0)) #+ entropy #+ actor_criterion(actor_y_samples,target_y)

                # target_values = torch.Tensor(seq_reward2go(pre_rewards[i], gamma)).view(-1,1).cuda() #[dis_reward(rewards[i:], gamma) for i in range(len(rewards))]
                # # values = meta_value_network(torch.cat((states_var,task_configs),1))
                # criterion = nn.MSELoss()
                # value_loss.append(criterion(pred_rs,target_values))

            ac_loss = torch.mean(torch.stack(actor_loss))
            # rec_loss = torch.mean(torch.stack(value_loss))
            actor_network_optim.zero_grad()
            # meta_value_network_optim.zero_grad()
            # task_config_network_optim.zero_grad()
            ac_loss.backward()
            # rec_loss.backward()
            opt_loss(actor_network_optim, actor_network)


            print(f'step {step} takes {time.time()-start} sec, with actloss={ac_loss}.')
            if (step + 1) % 20 == 0:
                print('='*25+' validation '+'='*25)
                results = list()
                for i in range(TASK_NUMS):
                    result = 0
                    test_task = CartPoleEnv(length = task_list[i].length)
                    for test_epi in range(10): # test for 10 epochs and takes the average
                        states, actions, rewards, _, _ = roll_out(None, test_task, SAMPLE_NUMS, None, reset=True)
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
