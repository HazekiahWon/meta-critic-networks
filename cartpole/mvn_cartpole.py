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

    # models
    nstate_decoder = Trans()
    dyn_encoder = DynEmb()
    return_baseline = RBase()
    actor_network = Actor()
    vae = VAE(dyn_encoder, nstate_decoder)
    # send to cuda
    return_baseline.cuda()
    nstate_decoder.cuda()
    dyn_encoder.cuda()
    actor_network.cuda()
    vae.cuda()
    # for the convenience of saving
    model_dict = dict(dyn_encoder=dyn_encoder, nstate_decoder=nstate_decoder,
                      return_baseline=return_baseline, actor=actor_network)
    if resume_model_dir is not None:
        for n,m in model_dict.items():
            load_model(m, os.path.join(resume_model_dir, n, '.pkl')) # remember to add {step}
    # for the convenience of optimization?
    model_lr = dict(nstate_decoder=vae_lr, dyn_encoder=vae_lr, actor=0.001, return_baseline=0.001)
    model_opt = {name:torch.optim.Adam(net.parameters(), lr=model_lr[name])  for name, net in model_dict.items()}

    # algo
    algo = AC(5)

    expname = os.path.join(logdir, time.strftime('exp_%y%m%d_%H%M%S', time.localtime()))
    os.makedirs(expname, exist_ok=True)
    save_meta(expname)
    writer = SummaryWriter(log_dir=expname)

    # ----------------- Training ------------------
    print(f'train VAE.')
    loss_buffer = deque(maxlen=vae_report_freq)
    horizon = HORIZON  # to continuously increase the hardship of the trajectories.
    gamma = pow(0.05, 1. / horizon)
    step = 0
    start = time.time()
    while True:
        # actor_loss = list()
        state_loss = list()
        value_loss = list()

        if step % task_resample_freq == 0:  # sample another batch of tasks
            task_list = resample_task()

        for i in range(TASK_NUMS):
            states, actions, rewards, _ = roll_out(None, task_list[i], HORIZON, None, reset=True, to_cuda=True) # cuda
            # states = states[:-1]
            pred_nstate,latent_z = vae(states[:-2], actions[:-1]) # cuda
            target_nstate = states[1:-1]
            criterion = nn.MSELoss()
            state_loss.append(criterion(pred_nstate,target_nstate)) # cuda
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
        writer.add_scalar('VAE/reconstruction_loss', rec_loss, step)
        step_optimizers(model_dict, model_opt, ('dyn_encoder','nstate_decoder'))

        if len(loss_buffer)==vae_report_freq: loss_buffer.popleft()
        loss_buffer.append(rec_loss.item())
        m = np.mean(list(loss_buffer))
        # print(f'step {step} takes {time.time() - start} sec, with recloss={rec_loss}.')
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

    double_check = 0
    for step in range(STEP):
        actor_loss = list()
        value_loss = list()
        rets = list()
        # value_loss = list()
        start = time.time()

        for i in range(TASK_NUMS):
            # obtain z and experience data to train the actor
            states, actions, _, _ = roll_out(None, task_list[i], horizon, None, reset=True)
            states = states[:-1] # to remove the final state
            latent_z = get_dyn_embedding(states[:-1], actions[:-1], states[1:], dyn_encoder) # 1,z_dim
            states,actions,rewards,is_done,log_softmax_actions = roll_out(actor_network, task_list[i], horizon, latent_z, reset=True)

            sudo_loss, bellman_error, total_rewards, actions_logp, advantages = algo.train(states, actions, rewards,
                                                                                             latent_z, return_baseline,
                                                                                             log_softmax_actions)
            actor_loss.append(sudo_loss) #+ entropy #+ actor_criterion(actor_y_samples,target_y)
            value_loss.append(bellman_error)
            rets.append(total_rewards)
            if i == 0:
                writer.add_histogram('actor/actions_logp', actions_logp, step)
                writer.add_histogram('actor/advantages', advantages, step)

        ac_loss = torch.mean(torch.stack(actor_loss))
        bl_loss = torch.mean(torch.stack(value_loss))
        algo.optimize(ac_loss, bl_loss, model_dict, model_opt, ('actor','return_baseline'), writer, step)

        avg_ret = np.mean(rets)
        if len(loss_buffer)==actor_report_freq: loss_buffer.popleft()
        loss_buffer.append(avg_ret.item())
        m = np.mean(list(loss_buffer))
        writer.add_scalar('actor/avg_return', avg_ret, step)

        if (step+1) % actor_report_freq == 0:
            print(f'step {step} with avg return {m}.')
            if m>double_horizon_threshold*horizon:
                double_check += 1
                if double_check==3:
                    horizon += HORIZON
                    print(f'step {step} horizon={horizon}')
                    double_check = 0

        # if we set validation to cross over task resample, we can test generalization ability?
        if (step+1) % policy_task_resample_freq == 0:
            print('='*25+' validation '+'='*25)
            results = list()
            for i in range(TASK_NUMS):
                result = 0
                test_task = CartPoleEnv(length = task_list[i].length)
                for test_epi in range(10): # test for 10 epochs and takes the average
                    states, actions, _, _ = roll_out(None, test_task, horizon, None, reset=True)
                    states = states[:-1]
                    z = get_dyn_embedding(states[:-1], actions[:-1], states[1:],
                                                    dyn_encoder)  # 1,z_dim
                    state = test_task.reset()
                    for test_step in range(200): # rollout for 200 steps
                        next_state, reward, done, action, logp = single_step_rollout(state, z, actor_network, test_task)
                        # test_task.render()
                        result += reward
                        state = next_state
                        if done:
                            break
                results.append(result/10.)
            val_res = np.mean(results)
            writer.add_scalar('actor/val_return', val_res, val_cnt)
            print(f'average return {val_res} for {TASK_NUMS} tasks.')
            print('=' * 25 + ' validation ' + '=' * 25)

            torch.save(nstate_decoder.state_dict(),os.path.join(expname, f"meta_value_network_cartpole_{step}.pkl"))
            torch.save(dyn_encoder.state_dict(),os.path.join(expname, f"task_config_network_cartpole_{step}.pkl"))
            print("save networks")
            task_list = resample_task() # resample
            val_cnt += 1

if __name__ == '__main__':
    main()
