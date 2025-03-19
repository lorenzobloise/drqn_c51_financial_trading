import os
from agents.trading_agent import TradingAgent
from agents.agent_c51 import Agent_C51
from agents.agent import Agent
from data_preprocessing import Data
import argparse
from common import create_json_params
import torch
import numpy as np
import random
from trading_environment import TradingEnvironment
from tqdm import tqdm

class RunAgent:

    def __init__(self, env: TradingEnvironment, agent: TradingAgent):
        self.env = env
        self.agent = agent

    def run(self, episodes, args):
        progress_bar = tqdm(total=episodes, desc="Train")  # Initialize a progress bar
        state = self.env.reset() # initial_state
        for step in range(episodes):
            action = self.agent.act(state) # select greedy action, exploration is done in step-method
            actions, rewards, new_states, state, done = self.env.step(action, step)
            if done:
                break
            self.agent.store(state, actions, new_states, rewards, action, step)
            self.agent.optimize(step)
            progress_bar_dict = dict(episode=step)
            progress_bar.set_postfix(progress_bar_dict)
            progress_bar.update()
        progress_bar.close()
        self.env.store_test_result(args)

def training(args, agent: TradingAgent):
    T = args.T
    M = args.batch_size  # minibatch size
    alpha = args.lr  # Learning rate
    gamma = args.gamma  # Discount factor
    theta = args.para_target  # Target network
    n_units = args.n_units  # number of units in a hidden layer
    n_episodes = 10000
    closing_path = os.path.join(args.root_path, args.dataset, 'closing/' + args.stock + '-closing.json')
    states_path = os.path.join(args.root_path, args.dataset, 'states/' + args.stock + '-states.json')
    RunAgent(
        TradingEnvironment(Data(closing_path, states_path, T)),
        agent
    ).run(n_episodes,args)

def cycle(args):
    file_path = './dataset/closing/'
    datafile_list = os.listdir(file_path)
    for s in range(0, len(datafile_list)):
        args.stock = datafile_list[s][0:3]
        if args.stock == 'AXP':
            continue
        print(args.stock)
        SEED = args.seed
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        agent = Agent(args, epsilon_greedy=False)
        training(args, agent)


if __name__ == "__main__":

    gamma = 0.001
    lr = 0.00025
    replay_memory_size = 25000
    batch_size = 32

    output = './output'
    os.mkdir(output)
    dir1 = os.path.join(output, 'agent')
    dir2 = os.path.join(output, 'agent_c51')
    os.mkdir(dir1)
    os.mkdir(dir2)
    action_path_1 = os.path.join(dir1, 'action')
    portfolio_path_1 = os.path.join(dir1, 'portfolio')
    os.mkdir(action_path_1)
    os.mkdir(portfolio_path_1)
    action_path_2 = os.path.join(dir2, 'action')
    portfolio_path_2 = os.path.join(dir2, 'portfolio')
    os.mkdir(action_path_2)
    os.mkdir(portfolio_path_2)

    parser = argparse.ArgumentParser(description='DRQN_C51_Stock_Trading')
    parser.add_argument('--root_path', type=str, default='./', help="root path")
    parser.add_argument('--dataset', type=str, default='dataset', help="dataset directory")
    parser.add_argument('--test_path', type=str, default=output)
    parser.add_argument('--action_path',type=str,default=action_path_1)
    parser.add_argument('--portfolio_path', type=str, default=portfolio_path_1)
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--gamma', type=float, default=gamma, help="the discount factor of Q learning")
    parser.add_argument('--lr', type=float, default=lr, help="training learning rate")
    parser.add_argument('--replay_memory_size', type=int, default=replay_memory_size,
                        help="replay memory size")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="training batch size")
    parser.add_argument('--n_units', type=int, default=32, help="the number of units in a hidden layer")
    parser.add_argument('--T', type=int, default=96, help="the length of series data")
    parser.add_argument('--stock', type=str, default='AIG', help="determine which stock")
    parser.add_argument('--seed', type=int, default=2037, help="random seed")
    parser.add_argument('--para_target', type=float, default=0.001,
                        help="the parameter which controls the soft update")

    args = parser.parse_args()

    create_json_params(args)

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # DRQN Standard
    cycle(args)

    # DRQN C51
    args.action_path = action_path_2
    args.portfolio_path = portfolio_path_2
    cycle(args)