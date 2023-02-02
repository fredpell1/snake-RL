from utils.controller import user_mode, agent_mode
from utils.util import *
from agents import heuristic_agent, mc_agent, td_agent
from argparse import ArgumentParser
import yaml

def main(new, filename, n_episodes):
    agent = load_config_file(td_agent.TDLambdaNN, 'configs/TD-V2.yaml', 'saved_agent')
    
    train_and_save(agent, n_episodes, 500, filename)
    test(agent, 1, 100)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--new', action='store_true')
    parser.add_argument('-f', '--file')
    parser.add_argument('-e', '--episodes', type=int)
    args = parser.parse_args()
    main(args.new, args.file, args.episodes)