from utils.controller import user_mode, agent_mode
from utils.util import *
from agents import heuristic_agent, mc_agent, td_agent
from argparse import ArgumentParser
import yaml
import sys


def main(config_file, agent_folder, n_episodes, output_file, user, verbose):
    if user:
        user_mode(verbose)
    else:
        agent = get_agent_type(config_file)
        agent, file = load_config_file(agent, config_file, agent_folder, verbose)
        train_and_save(agent, n_episodes, 500, file, output_file, verbose)
        test(agent, 10, 50, verbose)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-e", "--episodes", type=int)
    parser.add_argument("-f", "--folder", default="saved_agent")
    parser.add_argument("-o", "--output")
    parser.add_argument("-u", "--user", action='store_true')
    parser.add_argument("-v", "--verbose", action='store_true')
    args,unknown = parser.parse_known_args()
    main(
        config_file=args.config.strip(),
        n_episodes=args.episodes,
        agent_folder=args.folder.strip(),
        output_file=args.output.strip(),
        user=args.user,
        verbose=args.verbose
    )
