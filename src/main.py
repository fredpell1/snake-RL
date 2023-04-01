from utils.controller import user_mode, agent_mode
from utils.util import *
from agents import heuristic_agent, mc_agent, td_agent
from argparse import ArgumentParser
import yaml
import sys


def main(
    config_file,
    agent_folder,
    n_episodes,
    output_file,
    mode,
    verbose,
    periodic_save,
    frequency,
    results_file,
):
    if mode == "user":
        user_mode(verbose)
    else:
        agent = get_agent_type(config_file)
        agent, file = load_config_file(agent, config_file, agent_folder, verbose)
        if mode == "train":
            train_and_save(
                agent,
                n_episodes,
                500,
                file,
                output_file,
                verbose,
                periodic_save,
                frequency,
            )
        elif mode == "demo":
            test(agent, n_episodes, 5000, verbose, True)
        elif mode == "test":
            test(agent, n_episodes, 5000, verbose, False, results_file)
        else:
            raise ValueError("unrecognized mode. Please use: user,train,demo or test.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-e", "--episodes", type=int)
    parser.add_argument("-f", "--folder", default="saved_agent")
    parser.add_argument("-o", "--output")
    parser.add_argument("-m", "--mode")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("-r", "--results_file", type=str, default=None)
    args, unknown = parser.parse_known_args()
    main(
        config_file=args.config.strip(),
        n_episodes=args.episodes,
        agent_folder=args.folder.strip(),
        output_file=args.output.strip(),
        mode=args.mode,
        verbose=args.verbose,
        periodic_save=args.save,
        frequency=args.frequency,
        results_file=args.results_file,
    )
