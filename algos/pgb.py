import numpy as np
import torch
from torch.optim import Adam
import gym

from utils.logx import EpochLogger  # For visualization purposes
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import num_procs

from algos.agent import AC
from misc import HyperParams as hp
from misc import Memory


def PGB(bullet_env):

    def get_actor_loss(dat):
        state, action, advantage, _ = dat['states'], dat['actions'], \
            dat['advantages'], dat['logp']

        action, logp = agent.actor_net(state, action)
        actor_loss = -(logp * advantage).mean()

        return actor_loss

    def get_critic_loss(dat):
        state, returns = dat['states'], dat['returns']
        return ((agent.target_net(state) - returns) ** 2).mean()

    def update():
        dat = memory.recall()

        # Prep for update
        actor_loss_prev = get_actor_loss(dat)
        actor_loss_prev = actor_loss_prev.item()

        # Train actor
        actor_optim.zero_grad()
        actor_loss = get_actor_loss(dat)
        actor_loss.backward()
        mpi_avg_grads(agent.actor_net)
        actor_optim.step()

        # Train critic
        for i in range(hp.n_iters_target):
            critic_optim.zero_grad()
            critic_loss = get_critic_loss(dat)
            critic_loss.backward()
            mpi_avg_grads(agent.target_net)
            critic_optim.step()

    # Config
    setup_pytorch_for_mpi()  # Speed up mpi calculations
    env = gym.make(bullet_env)  # Instantiate environment
    n_inputs = env.observation_space.shape[0]
    n_outputs = env.action_space.shape[0]
    epc_steps = int(hp.steps_per_epoch / num_procs())

    # Logger
    logger = EpochLogger(output_fname='pgb_data.txt')
    torch.manual_seed(500)
    np.random.seed(500)
    memory = Memory(n_inputs, n_outputs, epc_steps)

    # Create actor-critic
    agent = AC(n_inputs, n_outputs)
    sync_params(agent)

    # Set up optimizers for actor and target functions
    actor_optim = Adam(agent.actor_net.parameters(), lr=hp.actor_lr)
    critic_optim = Adam(agent.target_net.parameters(), lr=hp.critic_lr)
    logger.setup_pytorch_saver(agent)

    # Prepare for interaction with environment
    state, ep_returns, n_steps = env.reset(), 0, 0

    # Loop through epochs
    for epoch in range(hp.epochs):
        for t in range(epc_steps):
            action, target, logp = agent.step(torch.as_tensor(state, dtype=torch.float32))

            next_state, reward, done, _ = env.step(action)
            ep_returns += reward
            n_steps += 1

            memory.save(state, action, reward, target, logp)

            # Update state
            state = next_state

            epoch_ends = (t == epc_steps - 1)
            if done or epoch_ends:
                if epoch_ends and not(done):
                    print('Ep cut off at %d steps.' % n_steps)
                if epoch_ends:  # Bootstrap
                    _, Vs_T, _ = agent.step(torch.as_tensor(state, dtype=torch.float32))
                else:
                    Vs_T = 0

                memory.finish(Vs_T)

                if done:
                    logger.store(EpReturns=ep_returns, EpLength=n_steps)

                state, ep_returns, n_steps = env.reset(), 0, 0

        # Save
        if (epoch % hp.save_freq == 0) or (epoch == hp.epochs - 1):
            logger.save_state({'env': env}, None)

        # Update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpReturns')
        logger.log_tabular('EpLength', average_only=True)
        logger.dump_tabular()

    return agent
