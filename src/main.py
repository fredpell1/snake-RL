from utils.controller import user_mode, agent_mode
from utils.util import *
from agents import heuristic_agent, mc_agent, td_agent
from argparse import ArgumentParser
import yaml


def main(config_file, agent_folder, n_episodes, output_file):
    agent, file = load_config_file(td_agent.TDLambdaNN, config_file, agent_folder)

    train_and_save(agent, n_episodes, 500, file, output_file)
    test(agent, 10, 100)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-e", "--episodes", type=int)
    parser.add_argument("-f", "--folder", default="saved_agent")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    main(
        config_file=args.config,
        n_episodes=args.episodes,
        agent_folder=args.folder,
        output_file=args.output,
    )
