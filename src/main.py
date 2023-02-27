from utils.controller import user_mode, agent_mode
from utils.util import *
from agents import heuristic_agent, mc_agent, td_agent
from argparse import ArgumentParser
import yaml


def main(config_file, agent_folder, n_episodes, output_file, user, verbose):
    if user:
        user_mode(verbose)
    else:
        agent = get_agent_type(config_file)
        agent, file = load_config_file(agent, config_file, agent_folder)
        train_and_save(agent, n_episodes, 500, file, output_file)
        test(agent, 10, 50)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-e", "--episodes", type=int)
    parser.add_argument("-f", "--folder", default="saved_agent")
    parser.add_argument("-o", "--output")
    parser.add_argument("-u", "--user", action='store_true')
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()
    main(
        config_file=args.config,
        n_episodes=args.episodes,
        agent_folder=args.folder,
        output_file=args.output,
        user=args.user,
        verbose=args.verbose
    )
